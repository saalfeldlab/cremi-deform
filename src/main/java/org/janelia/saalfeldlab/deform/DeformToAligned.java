/**
 *
 */
package org.janelia.saalfeldlab.deform;

import java.io.File;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Random;

import com.beust.jcommander.JCommander;
import com.beust.jcommander.Parameter;
import com.google.gson.Gson;

import bdv.img.h5.H5Utils;
import bdv.labels.labelset.Label;
import bdv.labels.labelset.LabelMultisetType;
import ch.systemsx.cisd.hdf5.HDF5Factory;
import ch.systemsx.cisd.hdf5.IHDF5Reader;
import ch.systemsx.cisd.hdf5.IHDF5Writer;
import mpicbg.trakem2.transform.CoordinateTransform;
import mpicbg.trakem2.transform.CoordinateTransformList;
import net.imglib2.FinalInterval;
import net.imglib2.Interval;
import net.imglib2.RandomAccessible;
import net.imglib2.RandomAccessibleInterval;
import net.imglib2.RealLocalizable;
import net.imglib2.RealPositionable;
import net.imglib2.RealRandomAccessible;
import net.imglib2.converter.Converter;
import net.imglib2.converter.Converters;
import net.imglib2.img.planar.PlanarImgs;
import net.imglib2.interpolation.InterpolatorFactory;
import net.imglib2.interpolation.randomaccess.ClampingNLinearInterpolatorFactory;
import net.imglib2.interpolation.randomaccess.NearestNeighborInterpolatorFactory;
import net.imglib2.realtransform.RealTransform;
import net.imglib2.realtransform.RealTransformRandomAccessible;
import net.imglib2.realtransform.Translation2D;
import net.imglib2.type.Type;
import net.imglib2.type.numeric.integer.LongType;
import net.imglib2.type.numeric.integer.UnsignedByteType;
import net.imglib2.view.Views;

/**
 * @author Stephan Saalfeld &lt;saalfelds@janelia.hhmi.org&gt;
 *
 */
public class DeformToAligned {

	public static class Parameters {

		@Parameter(names = { "--infile", "-i" }, description = "input CREMI-format HDF5 file name")
		public String inFile;

		@Parameter(names = { "--infile_labels", "-j" }, description = "input CREMI-format HDF5 file name")
		public String inFileLabels = null;

		@Parameter( names = { "--label", "-l" }, description = "label dataset" )
		public List<String> labels = Arrays.asList( new String[]{"/volumes/labels/clefts", "/volumes/labels/neuron_ids"});

		@Parameter(names = { "--outfile", "-o" }, description = "output CREMI-format HDF5 file name")
		public String outFile;

		@Parameter(names = { "--intransformations", "-t" }, description = "input JSON export of alignment transofomations, formatted as a list of lists")
		public String inTransformations;

		@Parameter(names = { "--meshcellsize", "-c" }, description = "mesh cell size for rendering the output")
		public long meshCellSize;
	}

	final static private int[] cellDimensions = new int[] { 64, 64, 8 };

	/**
	 * Creates the inverse thin plate spline transform for jittered points on a
	 * grid.
	 *
	 * @param interval
	 * @param controlPointSpacing
	 * @param jitterRadius
	 * @return
	 */
	static public ThinplateSplineTransform make2DSectionJitterTransform(
			final Random rnd,
			final Interval interval,
			final double controlPointSpacing,
			final double jitterRadius) {

		final ArrayList<double[]> p = new ArrayList<>();
		final ArrayList<double[]> q = new ArrayList<>();

		for (double y = 0; y <= interval.dimension(1); y += controlPointSpacing) {
			for (double x = 0; x <= interval.dimension(0); x += controlPointSpacing) {
				p.add(new double[] { x, y });
				q.add(new double[] {
						x + jitterRadius * (2 * rnd.nextDouble() - 1),
						y + jitterRadius * (2 * rnd.nextDouble() - 1) });
			}
		}

		final double[][] ps = new double[2][p.size()];
		final double[][] qs = new double[2][q.size()];

		for (int i = 0; i < p.size(); ++i) {
			final double[] pi = p.get(i);
			ps[0][i] = pi[0];
			ps[1][i] = pi[1];
			final double[] qi = q.get(i);
			qs[0][i] = qi[0];
			qs[1][i] = qi[1];
		}
		return new ThinplateSplineTransform(qs, ps);
	}

	static public ArrayList<RealTransform> make2DSectionJitterTransforms(
			final Random rnd,
			final Interval interval,
			final double controlPointSpacing,
			final double jitterRadius,
			final double jitterChance) {

		final ArrayList<RealTransform> sliceTransforms = new ArrayList<>();

		RealTransform t = new Translation2D();
		for (int z = 0; z < interval.dimension(2); ++z) {

			if (rnd.nextDouble() < jitterChance)
				t = make2DSectionJitterTransform(
						rnd,
						interval,
						controlPointSpacing,
						jitterRadius);

			sliceTransforms.add(t);
		}

		return sliceTransforms;
	}

	static public <T> RandomAccessibleInterval<T> jitterSlices(
			final RandomAccessible<T> source,
			final Interval interval,
			final ArrayList<? extends RealTransform> sliceTransforms,
			final InterpolatorFactory<T, RandomAccessible<T>> interpolatorFactory) {

		final ArrayList<RandomAccessibleInterval<T>> slices = new ArrayList<>();
		for (int z = 0; z < sliceTransforms.size(); ++z) {
			final RandomAccessible<T> slice = Views.hyperSlice(source, 2, z);
			slices.add(
					Views.interval(
							new RealTransformRandomAccessible<T, RealTransform>(
									Views.interpolate(slice, interpolatorFactory),
									sliceTransforms.get(z)),
							new FinalInterval(interval.dimension(0), interval.dimension(1))));
		}
		return Views.stack(slices);
	}

	/**
	 * Creates a transformed version of a 3D volume by mapping each slice with
	 * a mesh transform, mpicbg style.
	 *
	 * @param source extended source
	 * @param interval input interval
	 * @param sliceTransforms
	 * @param interpolatorFactory
	 * @return
	 */
	static public <T extends Type<T>> void mapSlices(
			final RandomAccessible<T> source,
			final Interval sourceInterval,
			final List<CoordinateTransform> sliceTransforms,
			final InterpolatorFactory<T, RandomAccessible<T>> interpolatorFactory,
			final RandomAccessibleInterval<T> target,
			final long meshCellSize) {

		final long meshRes = Math.max(2, target.dimension(0) / meshCellSize);
		for (int z = 0; z < Math.min(sliceTransforms.size(), target.dimension(2)); ++z) {

			System.out.println( z + " " + interpolatorFactory.getClass().getSimpleName() );

			final RenderTransformMesh mesh =
					new RenderTransformMesh(
							sliceTransforms.get(z),
							(int)meshRes,
							sourceInterval.dimension(0),
							sourceInterval.dimension(1));
			mesh.updateAffines();

			final RenderTransformMeshMappingWithMasks<T> mapping = new RenderTransformMeshMappingWithMasks<>(mesh);

			final RealRandomAccessible<T> sourceSlice =
					Views.interpolate(
							Views.hyperSlice(source, 2, z),
							interpolatorFactory);

			final RandomAccessibleInterval<T> targetSlice = Views.hyperSlice(target, 2, z);

			mapping.map(sourceSlice, targetSlice, Runtime.getRuntime().availableProcessors());
		}
	}

	/**
	 * Generates an interval view of slices deformed by the inverse of a series
	 * of transforms.  This is supposed to be the inverse of
	 * {@link #mapSlices(RandomAccessible, Interval, List, InterpolatorFactory, RandomAccessibleInterval, long)}
	 *
	 * @param source
	 * @param interval
	 * @param sliceTransforms
	 * @param interpolatorFactory
	 * @return
	 */
	static public <T> RandomAccessibleInterval<T> mapInverseSlices(
			final RandomAccessible<T> source,
			final Interval interval,
			final List<CoordinateTransform> sliceTransforms,
			final InterpolatorFactory<T, RandomAccessible<T>> interpolatorFactory) {

		final ArrayList<RandomAccessibleInterval<T>> slices = new ArrayList<>();
		for (int z = 0; z < sliceTransforms.size(); ++z) {
			final RandomAccessible<T> slice = Views.hyperSlice(source, 2, z);
			slices.add(
					Views.interval(
							new RealTransformRandomAccessible<T, RealTransform>(
									Views.interpolate(slice, interpolatorFactory),
									new CoordinateTransformRealTransform( sliceTransforms.get(z), 2)),
							new FinalInterval(interval.dimension(0), interval.dimension(1))));
		}
		return Views.stack(slices);
	}




	/**
	 * @param args
	 * @throws IOException
	 * @throws IllegalAccessException
	 * @throws InstantiationException
	 * @throws ClassNotFoundException
	 */
	public static void main(final String... args) throws IOException, ClassNotFoundException, InstantiationException, IllegalAccessException {

		final Parameters params = new Parameters();
		new JCommander(params, args);

		try (final FileReader transformsReader = new FileReader(new File(params.inTransformations))) {

			System.out.println("Opening " + params.inFile);
			final IHDF5Reader rawReader = HDF5Factory.openForReading(params.inFile);
			final IHDF5Reader labelsReader = params.inFileLabels == null ? rawReader : HDF5Factory.openForReading(params.inFileLabels);

			/* raw pixels */
			System.out.println("Loading raw pixels " + "/volumes/raw");
			final String rawPath = "/volumes/raw";
			final RandomAccessibleInterval<UnsignedByteType> rawSource = Util.loadRaw(rawReader, rawPath, cellDimensions);

			System.out.println("Opening " + params.outFile + " for writing");
			final File outFile = new File(params.outFile);
			final IHDF5Writer writer = HDF5Factory.open(params.outFile);

			final ArrayList<CoordinateTransform> transforms = new ArrayList<>();
			final TrakEM2Export trakem2Export = new Gson().fromJson(transformsReader, TrakEM2Export.class);
			for (final ArrayList<TransformSpec> transformList : trakem2Export.transforms) {
				final CoordinateTransformList<CoordinateTransform> ctl = new CoordinateTransformList<>();
				for (final TransformSpec t : transformList)
					ctl.add(t.createTransform());

				transforms.add(ctl);
			}

			/* deform */
			final RandomAccessibleInterval<UnsignedByteType> rawTarget = PlanarImgs.unsignedBytes(
					trakem2Export.width,
					trakem2Export.height,
					rawSource.dimension(2));

			mapSlices(
					Views.extendValue(rawSource, new UnsignedByteType(0)),
					rawSource,
					transforms,
					new ClampingNLinearInterpolatorFactory<>(),
					rawTarget,
					params.meshCellSize);

			/* save */
			System.out.println("writing " + params.outFile);

			System.out.println("  " + rawPath);
			H5Utils.saveUnsignedByte(
					rawTarget,
					outFile,
					rawPath,
					cellDimensions);

			writer.float64().setArrayAttr(rawPath, "resolution", new double[] { 40.0, 4.0, 4.0 });

			/* labels */
			for (final String labelsPath : params.labels) {

				final RandomAccessibleInterval<LabelMultisetType> labelsSource = Util.loadLabels(labelsReader, labelsPath, cellDimensions);

				final RandomAccessibleInterval<LongType> longLabelsSource = Converters.convert(labelsSource,
						new Converter<LabelMultisetType, LongType>() {
							@Override
							public void convert(final LabelMultisetType a, final LongType b) {
								b.set(a.entrySet().iterator().next().getElement().id());
							}
						}, new LongType());

				/* deform */
				final RandomAccessibleInterval<LongType> labelsTarget = PlanarImgs.longs(
						trakem2Export.width,
						trakem2Export.height,
						rawSource.dimension(2));

				/* clear canvas by filling with outside */
				for (final LongType t : Views.iterable(labelsTarget))
					t.set(Label.OUTSIDE);

				mapSlices(
						Views.extendValue(longLabelsSource, new LongType(Label.OUTSIDE)),
						rawSource, //!< as interval
						transforms,
						new NearestNeighborInterpolatorFactory<>(),
						labelsTarget,
						params.meshCellSize);

				/* save */
				System.out.println("writing " + labelsPath);
				H5Utils.saveUnsignedLong(
						labelsTarget,
						outFile,
						labelsPath,
						cellDimensions);

				writer.float64().setArrayAttr(labelsPath, "resolution", new double[] { 40.0, 4.0, 4.0 });
			}

			writer.close();

//			Util.display(
//					RealViews.affine(
//							Views.interpolate(
//									Views.extendZero(rawTarget),
//									new NearestNeighborInterpolatorFactory<>()),
//							new Scale3D(1, 1, 10)),
//					RealViews.affine(
//							Views.interpolate(
//									Views.extendValue(labelsTarget, new LongType(Label.OUTSIDE)),
//									new NearestNeighborInterpolatorFactory<>()),
//							new Scale3D(1, 1, 10)),
//					new FinalInterval(
//							trakem2Export.width,
//							trakem2Export.height,
//							rawTarget.dimension(2) * 10));
		}
	}
}
