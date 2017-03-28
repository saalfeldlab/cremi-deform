/**
 *
 */
package org.janelia.saalfeldlab.deform;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Random;

import com.beust.jcommander.JCommander;
import com.beust.jcommander.Parameter;

import bdv.bigcat.label.FragmentSegmentAssignment;
import bdv.bigcat.ui.GoldenAngleSaturatedARGBStream;
import bdv.img.h5.H5LabelMultisetSetupImageLoader;
import bdv.img.h5.H5UnsignedByteSetupImageLoader;
import bdv.img.h5.H5Utils;
import bdv.labels.labelset.Label;
import bdv.labels.labelset.LabelMultisetType;
import bdv.util.Bdv;
import bdv.util.BdvFunctions;
import bdv.util.BdvStackSource;
import bdv.util.LocalIdService;
import ch.systemsx.cisd.hdf5.HDF5Factory;
import ch.systemsx.cisd.hdf5.IHDF5Reader;
import ch.systemsx.cisd.hdf5.IHDF5Writer;
import mpicbg.spim.data.generic.sequence.ImgLoaderHints;
import net.imglib2.FinalInterval;
import net.imglib2.Interval;
import net.imglib2.RandomAccessible;
import net.imglib2.RandomAccessibleInterval;
import net.imglib2.RealRandomAccessible;
import net.imglib2.converter.Converter;
import net.imglib2.converter.Converters;
import net.imglib2.img.array.ArrayImg;
import net.imglib2.img.array.ArrayImgs;
import net.imglib2.img.array.ArrayRandomAccess;
import net.imglib2.img.basictypeaccess.array.DoubleArray;
import net.imglib2.interpolation.InterpolatorFactory;
import net.imglib2.interpolation.randomaccess.ClampingNLinearInterpolatorFactory;
import net.imglib2.interpolation.randomaccess.NearestNeighborInterpolatorFactory;
import net.imglib2.realtransform.AffineTransform2D;
import net.imglib2.realtransform.DeformationFieldTransform;
import net.imglib2.realtransform.RealTransform;
import net.imglib2.realtransform.RealTransformRandomAccessible;
import net.imglib2.realtransform.RealTransformSequence;
import net.imglib2.realtransform.Scale3D;
import net.imglib2.type.numeric.ARGBType;
import net.imglib2.type.numeric.integer.LongType;
import net.imglib2.type.numeric.integer.UnsignedByteType;
import net.imglib2.type.numeric.real.DoubleType;
import net.imglib2.view.Views;

/**
 * @author Stephan Saalfeld &lt;saalfelds@janelia.hhmi.org&gt;
 *
 */
public class Deform {

	public static class Parameters {

		@Parameter(names = { "--infile", "-i" }, description = "input CREMI-format HDF5 file name")
		public String inFile;

		@Parameter(names = { "--outfile", "-o" }, description = "output CREMI-format HDF5 file name")
		public String outFile;

		@Parameter(names = { "--spacing", "-s" }, description = "control point spacing in world units")
		public double controlPointSpacing = 512;

		@Parameter(names = { "--jitter", "-j" }, description = "jitter radius in world units")
		public double jitterRadius = 32;

		@Parameter(names = { "--jitter3d", "-3" }, description = "perform the jitter in 3D")
		public boolean jitter3d = false;

		@Parameter(names = { "--subsampleFactor" }, description = "approximate the exact spline transform by subsampling, which will be faster")
		public double subsampleFactor = 0;

		@Parameter(names = { "--rotate", "-r" }, description = "rotate each section with the given angle in radians")
		public List<Double> rotate = new ArrayList<Double>();

		@Parameter(names = { "--mirror", "-m" }, description = "a string like 'xy' indicating which axises to mirror")
		public String mirror = "";

		@Parameter(names = { "--num", "-n" }, description = "number of outputs")
		public double n = 1;

		@Parameter(names = { "--onlyLut", "-l" }, description = "only store a transformation LUT int the output HDF, don't transform")
		public boolean onlyLut = false;

		// anisotropic

		@Parameter(names = { "--jitterchance",
				"-c" }, description = "chance for each section to get jittered relative to its predecessor, in other words, probability of an alignment error")
		public double jitterChance = 0.5;
	}

	final static private int[] cellDimensions = new int[] { 64, 64, 8 };

	final static private RandomAccessibleInterval<LabelMultisetType> loadLabels(
			final IHDF5Reader reader,
			final String dataset,
			final double[] rawResolution) throws IOException {
		final RandomAccessibleInterval<LabelMultisetType> fragmentsPixels;
		if (reader.exists(dataset)) {
			final double[] resolution = readResolution(reader, dataset, rawResolution);
			final H5LabelMultisetSetupImageLoader fragments = new H5LabelMultisetSetupImageLoader(reader, null, dataset,
					1, cellDimensions, resolution, new double[]{0, 0, 0});
			fragmentsPixels = fragments.getImage(0, 0);
		} else {
			System.out.println("no labels found cooresponding to requested dataset '" + dataset + "'");
			fragmentsPixels = null;
		}
		return fragmentsPixels;
	}

	final static public void display(
			final RealRandomAccessible<UnsignedByteType> raw,
			final RealRandomAccessible<LongType> labels,
			final Interval interval) {
		final FragmentSegmentAssignment assignment = new FragmentSegmentAssignment(new LocalIdService());
		final GoldenAngleSaturatedARGBStream argbStream = new GoldenAngleSaturatedARGBStream(assignment);
		final BdvStackSource<UnsignedByteType> source = BdvFunctions.show(raw, interval, "raw", Bdv.options());
		BdvFunctions.show(
				Converters.convert(
						labels,
						new Converter<LongType, ARGBType>() {

							@Override
							public void convert(final LongType input, final ARGBType output) {
								final long id = input.get();
								if (id == Label.TRANSPARENT || id == Label.INVALID)
									output.set(0);
								else {
									final int argb = argbStream.argb(input.get());
									final int r = ((argb >> 16) & 0xff) / 4;
									final int g = ((argb >> 8) & 0xff) / 4;
									final int b = (argb & 0xff) / 4;

									output.set(((((r << 8) | g) << 8) | b) | 0xff000000);
								}
							}
						},
						new ARGBType()),
				interval, "labels", Bdv.options().addTo(source.getBdvHandle()));
	}

	/**
	 * Creates the inverse thin plate spline transform for jittered points on a
	 * grid.
	 *
	 * @param interval
	 * @param controlPointSpacing
	 * @param jitterRadius
	 * @param baseTransform
	 *            A transormation to apply to each grid point before jittering.
	 * @param mirror
	 *            Whether to mirror the x and/or y axis.
	 * @return
	 */
	static public ThinplateSplineTransform make2DSectionJitterTransform(
			final Random rnd,
			final Interval interval,
			final double[] controlPointSpacing,
			final double[] jitterRadius,
			final AffineTransform2D baseTransform,
			final boolean[] mirror) {

		final ArrayList<double[]> p = new ArrayList<>();
		final ArrayList<double[]> q = new ArrayList<>();

		final double[] center = new double[2];
		center[0] = (double) (interval.dimension(0)) / 2;
		center[1] = (double) (interval.dimension(1)) / 2;

		for (double y = 0; y <= interval.dimension(1); y += controlPointSpacing[1]) {
			for (double x = 0; x <= interval.dimension(0); x += controlPointSpacing[0]) {
				p.add(new double[] { x, y });
				final double[] transformed = new double[2];
				baseTransform.apply(new double[] { x, y }, transformed);
				q.add(new double[] {
						transformed[0] + jitterRadius[0] * (2 * rnd.nextDouble() - 1),
						transformed[1] + jitterRadius[1] * (2 * rnd.nextDouble() - 1) });
			}
		}

		final double[][] ps = new double[2][p.size()];
		final double[][] qs = new double[2][q.size()];

		for (int i = 0; i < p.size(); ++i) {
			final double[] pi = p.get(i);
			final double[] qi = q.get(i);
			for (int d = 0; d < 2; d++) {
				ps[d][i] = pi[d];
				qs[d][i] = (mirror[d] ? interval.dimension(d) - (qi[d] + 1) : qi[d]);
			}
		}
		return new ThinplateSplineTransform(qs, ps);
	}

	/**
	 * Get a stack of 2D transforms, to be applied to each section individually.
	 * @param mirror
	 *            Whether to mirror the x, and/or y axis.
	 */
	static public ArrayList<RealTransform> make2DSectionJitterTransforms(
			final Random rnd,
			final Interval interval,
			final double[] controlPointSpacing,
			final double[] jitterRadius,
			final double jitterChance,
			final double rotationAngle,
			final boolean[] mirror) {

		final ArrayList<RealTransform> sliceTransforms = new ArrayList<>();

		final AffineTransform2D r = makeRotation(interval, rotationAngle);
		RealTransform t = r.inverse();
		for (int z = 0; z < interval.dimension(2); ++z) {

			if (rnd.nextDouble() < jitterChance)
				t = make2DSectionJitterTransform(
						rnd,
						interval,
						controlPointSpacing,
						jitterRadius,
						r,
						mirror);

			sliceTransforms.add(t);
		}

		return sliceTransforms;
	}

	/**
	 * Creates the inverse thin plate spline transform for jittered points on a
	 * 3D grid.
	 *
	 * @param interval
	 * @param controlPointSpacing
	 * @param jitterRadius
	 * @param baseTransform
	 *            A transormation to apply to each grid point before jittering.
	 * @param mirror
	 *            Whether to mirror the x, y, and/or z axis.
	 * @return
	 */
	static public RealTransform make3DJitterTransform(
			final Random rnd,
			final Interval interval,
			final double[] controlPointSpacing,
			final double[] jitterRadius,
			final double rotationAngle,
			final boolean[] mirror,
			final double subsampleFactor) {

		System.out.println("Creating 3D jitter for interval " + interval);

		final AffineTransform2D baseTransform = makeRotation(interval, rotationAngle);

		final ArrayList<double[]> p = new ArrayList<>();
		final ArrayList<double[]> q = new ArrayList<>();

		final double[] center = new double[3];
		center[0] = (double) (interval.dimension(0)) / 2;
		center[1] = (double) (interval.dimension(1)) / 2;
		center[2] = (double) (interval.dimension(2)) / 2;

		for (double z = 0; z <= interval.dimension(2); z += controlPointSpacing[2]) {
			for (double y = 0; y <= interval.dimension(1); y += controlPointSpacing[1]) {
				for (double x = 0; x <= interval.dimension(0); x += controlPointSpacing[0]) {
					p.add(new double[] { x, y, z });
					final double[] transformed = new double[2];
					baseTransform.apply(new double[] { x, y }, transformed);
					q.add(new double[] {
							transformed[0] + jitterRadius[0] * (2 * rnd.nextDouble() - 1),
							transformed[1] + jitterRadius[1] * (2 * rnd.nextDouble() - 1),
							z              + jitterRadius[2] * (2 * rnd.nextDouble() - 1) });
				}
			}
		}

		final double[][] ps = new double[3][p.size()];
		final double[][] qs = new double[3][q.size()];

		for (int i = 0; i < p.size(); ++i) {
			final double[] pi = p.get(i);
			final double[] qi = q.get(i);
			for (int d = 0; d < 3; d++) {
				ps[d][i] = pi[d];
				qs[d][i] = (mirror[d] ? interval.dimension(d) - (qi[d] + 1) : qi[d]);
			}
		}

		final ThinplateSplineTransform tpst = new ThinplateSplineTransform(qs, ps);

		if (subsampleFactor == 0)
			return tpst;

		// subsample the spline transform for faster transformation

		// create a vector field
		final int width  = (int)(interval.dimension(0)/subsampleFactor) + 1;
		final int height = (int)(interval.dimension(1)/subsampleFactor) + 1;
		final int depth  = (int)(interval.dimension(2)/subsampleFactor) + 1;
		final ArrayImg<DoubleType,DoubleArray> vectorField = ArrayImgs.doubles(width, height, depth, 3);

		// let each location in the vector field point to the spline transformed location
		final ArrayRandomAccess<DoubleType> access = vectorField.randomAccess();
		final int[] position = new int[4]; position[3] = 0;
		final double[] target = new double[3];
		final double[] offset = new double[3];
		for (int z = 0; z < depth; z++)
			for (int y = 0; y < height; y++)
				for (int x = 0; x < width; x++) {

					position[0] = x;
					position[1] = y;
					position[2] = z;
					access.setPosition(position);

					// transform (x*f,y*f,z*f) -> (x',y',z'), where f is scaling factor
					tpst.apply(
							new double[]{
									position[0]*subsampleFactor,
									position[1]*subsampleFactor,
									position[2]*subsampleFactor
							},
							target);

					// get offset to target, downscaled
					for (int d = 0; d < 3; d++)
						offset[d] = target[d]/subsampleFactor - position[d];

					// set offset as the vector at (x,y,z)
					for (int d = 0; d < 3; d++) {
						access.get().set(offset[d]);
						access.fwd(3);
					}
				}

		// create a transform from the vector field
		final DeformationFieldTransform<DoubleType> dft =
				new DeformationFieldTransform<DoubleType>(
						Views.interval(
								Views.extendZero(
										vectorField
								),
								new long[]{0,0,0,0},
								new long[]{width,height,depth,3}
						)
				);

		// create a sequence of transformations:
		// 		1. scale down
		//		2. apply vector field transform
		//		3. scale up
		final RealTransformSequence subsampleTransform = new RealTransformSequence();
		subsampleTransform.add(new Scale3D(1.0/subsampleFactor, 1.0/subsampleFactor, 1.0/subsampleFactor));
		subsampleTransform.add(dft);
		subsampleTransform.add(new Scale3D(subsampleFactor, subsampleFactor, subsampleFactor));

		return subsampleTransform;
	}

	static public <T> RandomAccessibleInterval<T> jitterSlices(
			final RandomAccessible<T> source,
			final Interval interval,
			final ArrayList<? extends RealTransform> sliceTransforms,
			final InterpolatorFactory<T, RandomAccessible<T>> interpolatorFactory,
			final boolean mirrorZ) {

		final ArrayList<RandomAccessibleInterval<T>> slices = new ArrayList<>();
		for (int z = 0; z < sliceTransforms.size(); ++z) {
			final RandomAccessible<T> slice = Views.hyperSlice(source, 2, (mirrorZ ? sliceTransforms.size() - (z+1) : z));
			slices.add(
					Views.interval(
							new RealTransformRandomAccessible<T, RealTransform>(
									Views.interpolate(slice, interpolatorFactory),
									sliceTransforms.get(z)),
							new FinalInterval(interval.dimension(0), interval.dimension(1))));
		}
		return Views.stack(slices);
	}

	static public <T> RandomAccessibleInterval<T> jitterVolume(
			final RandomAccessible<T> source,
			final Interval interval,
			final RealTransform transform,
			final InterpolatorFactory<T, RandomAccessible<T>> interpolatorFactory) {

		return Views.interval(
				new RealTransformRandomAccessible<T, RealTransform>(
						Views.interpolate(source, interpolatorFactory),
						transform),
				new FinalInterval(interval.dimension(0), interval.dimension(1), interval.dimension(2)));
	}

	static public <T> AffineTransform2D makeRotation(
			final Interval interval,
			final double angle) {

		final long width = interval.dimension(0);
		final long height = interval.dimension(1);

		final AffineTransform2D t = new AffineTransform2D();
		t.translate(-width / 2, -height / 2);
		t.rotate(angle);
		t.translate(width / 2, height / 2);

		return t;
	}

	final static double[] readResolution(final IHDF5Reader reader, final String dataset) {

		return readResolution(reader, dataset, null);
	}

	final static double[] readResolution(final IHDF5Reader reader, final String dataset, final double[] defaultResolution) {

		if (!reader.object().hasAttribute(dataset, "resolution")) {

			if (defaultResolution == null)
				return new double[]{1, 1, 1};
			System.out.println("Dataset " + dataset + " does not have resolution" +
					" set, using previous read resolution.");
			return defaultResolution;
		}

		final double[] data = reader.getDoubleArrayAttribute(dataset, "resolution");
		final double[] resolution = { data[2], data[1], data[0] };

		System.out.println("Using resolution of (" + resolution[0] + ", " +
				resolution[1] + ", " + resolution[2] + ") found in dataset " +
				dataset + ".");

		if (defaultResolution != null)
			for (int d = 0; d < 3; d++)
				if (resolution[d] != defaultResolution[d]) {
					System.out.println("Warning: dataset " + dataset + " has different resolution than others.");
					break;
				}

		return resolution;
	}

	/**
	 * @param args
	 * @throws IOException
	 */
	public static void main(final String[] args) throws IOException {

		final Parameters params = new Parameters();
		new JCommander(params, args);

		final String labelsDataset = "neuron_ids";
		final String rawDataset = "raw";

		System.out.println("Opening " + params.inFile);
		final IHDF5Reader reader = HDF5Factory.openForReading(params.inFile);

		// support both file_format 0.0 and >=0.1
		final String volumesPath = reader.isGroup("/volumes") ? "/volumes" : "";
		final String labelsPath = reader.isGroup(volumesPath + "/labels") ? volumesPath + "/labels" : "";

		/* raw pixels */
		final String rawPath = volumesPath + "/" + rawDataset;
		final double[] resolution = readResolution(reader, rawPath);
		final H5UnsignedByteSetupImageLoader raw = new H5UnsignedByteSetupImageLoader(
				reader,
				rawPath,
				0,
				cellDimensions,
				resolution,
				new double[]{0, 0, 0});
		final RandomAccessibleInterval<UnsignedByteType> rawPixels = raw.getImage(0, ImgLoaderHints.LOAD_COMPLETELY);

		/* labels */
		final String fragmentsPath = labelsPath + "/" + labelsDataset;
		final RandomAccessibleInterval<LabelMultisetType> labels = loadLabels(reader, fragmentsPath, resolution);

		final RandomAccessibleInterval<LongType> longLabels = Converters.convert(labels,
				new Converter<LabelMultisetType, LongType>() {
					@Override
					public void convert(final LabelMultisetType a, final LongType b) {
						b.set(a.entrySet().iterator().next().getElement().id());
					}
				}, new LongType());

		final Random rnd = new Random();

		/* deform */
		for (int i = 0; i < params.n; ++i) {

			double rotate;
			if (params.rotate.size() == 0)
				rotate = 0.0;
			else
				rotate = params.rotate.get(i % params.rotate.size());

			RandomAccessibleInterval<UnsignedByteType> deformedRawPixels = null;
			RandomAccessibleInterval<LongType> deformedLongLabels = null;

			final boolean anisotropic = (resolution[2] != resolution[0]);
			if (anisotropic)
				System.out.println("Volume is anisotropic in z -- will not jitter in z-direction!");

			final double[] jitterRadius = {
					params.jitterRadius/resolution[0],
					params.jitterRadius/resolution[1],
					(anisotropic ? 0 : params.jitterRadius/resolution[2])
			};
			final double[] controlPointSpacing = {
					params.controlPointSpacing/resolution[0],
					params.controlPointSpacing/resolution[1],
					params.controlPointSpacing/resolution[2]
			};

			final boolean[] mirror = {
					params.mirror.contains("x"),
					params.mirror.contains("y"),
					params.mirror.contains("z")
			};
			System.out.println("Mirroring (x,y,z): " + Arrays.toString(mirror));

			if (params.jitter3d) {

				final RealTransform transform = make3DJitterTransform(
						rnd,
						rawPixels,
						controlPointSpacing,
						jitterRadius,
						rotate,
						mirror,
						params.subsampleFactor);

				if (params.onlyLut) {

					final long start = System.currentTimeMillis();
					int size = 1;
					for (int d = 0; d < 3; d++)
						size *= rawPixels.dimension(d);
					final double[][] lut = new double[size][3];

					System.out.println("Creating reverse transform LUT (target to source)...");
					int j = 0;
					for (int z = 0; z < rawPixels.dimension(2); z++)
						for (int y = 0; y < rawPixels.dimension(1); y++)
							for (int x = 0; x < rawPixels.dimension(0); x++) {
								transform.apply(new double[]{x,y,z}, lut[j]);
								j++;
							}
					final double time = System.currentTimeMillis() - start;
					System.out.println("Done in " + time/1000 + "s (" + (time/size) + "ms per voxel).");

					final IHDF5Writer writer = HDF5Factory.open(params.outFile);
					writer.float64().writeMatrix("/reverse_transform_lut", lut);
					writer.close();

					System.exit(0);
				}

				deformedRawPixels = jitterVolume(
						Views.extendValue(rawPixels, new UnsignedByteType(0)),
						rawPixels,
						transform,
						new ClampingNLinearInterpolatorFactory<UnsignedByteType>());
				deformedLongLabels = jitterVolume(
						Views.extendValue(longLabels, new LongType(Label.TRANSPARENT)),
						longLabels,
						transform,
						new NearestNeighborInterpolatorFactory<>());

			} else {

				final ArrayList<RealTransform> jitterTransforms = make2DSectionJitterTransforms(
						rnd,
						rawPixels,
						controlPointSpacing,
						jitterRadius,
						params.jitterChance,
						rotate,
						mirror);

				deformedRawPixels = jitterSlices(
						Views.extendValue(rawPixels, new UnsignedByteType(0)),
						rawPixels,
						jitterTransforms,
						new ClampingNLinearInterpolatorFactory<UnsignedByteType>(),
						mirror[2]);
				deformedLongLabels = jitterSlices(
						Views.extendValue(longLabels, new LongType(Label.TRANSPARENT)),
						longLabels,
						jitterTransforms,
						new NearestNeighborInterpolatorFactory<>(),
						mirror[2]);
			}

			String rawDatasetName = rawPath;
			String fragmentsDatasetName = fragmentsPath;

			if (params.n > 1) {

				rawDatasetName = rawDatasetName + "_" + i;
				fragmentsDatasetName = fragmentsDatasetName + "_" + i;
			}

			System.out.println("writing " + params.outFile);

			final File outFile = new File(params.outFile);
			System.out.println("  " + rawDatasetName);
			H5Utils.saveUnsignedByte(
					deformedRawPixels,
					outFile,
					rawDatasetName,
					cellDimensions);

			System.out.println("  " + fragmentsDatasetName);
			H5Utils.saveUnsignedLong(
					deformedLongLabels,
					outFile,
					fragmentsDatasetName,
					cellDimensions);

			final IHDF5Writer writer = HDF5Factory.open(params.outFile);
			final double[] resData = { resolution[2], resolution[1], resolution[0] };
			writer.float64().setArrayAttr(rawDatasetName, "resolution", resData);
			writer.float64().setArrayAttr(fragmentsDatasetName, "resolution", resData);
			writer.string().setAttr(rawDatasetName, "comment", "jittered with " + Arrays.toString(args));
			writer.string().setAttr(fragmentsDatasetName, "comment", "jittered with " + Arrays.toString(args));
			writer.close();

			// display(
			// RealViews.affine(
			// Views.interpolate(
			// Views.extendZero(deformedRawPixels),
			// new NearestNeighborInterpolatorFactory<>()),
			// new Scale3D(1, 1, 10)),
			// RealViews.affine(
			// Views.interpolate(
			// Views.extendZero(deformedLongLabels),
			// new NearestNeighborInterpolatorFactory<>()),
			// new Scale3D(1, 1, 10)),
			// new FinalInterval(
			// rawPixels.dimension(0),
			// rawPixels.dimension(1),
			// rawPixels.dimension(2) * 10));

		}
	}
}
