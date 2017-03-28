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
import net.imglib2.RealRandomAccessible;
import net.imglib2.converter.Converter;
import net.imglib2.converter.Converters;
import net.imglib2.img.planar.PlanarImgs;
import net.imglib2.interpolation.InterpolatorFactory;
import net.imglib2.interpolation.randomaccess.ClampingNLinearInterpolatorFactory;
import net.imglib2.interpolation.randomaccess.NearestNeighborInterpolatorFactory;
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
		public List<String> labels = Arrays.asList( new String[]{"/volumes/labels/neuron_ids","/volumes/labels/clefts","/volumes/labels/clefts_corrected"});

		@Parameter(names = { "--outfile", "-o" }, description = "output CREMI-format HDF5 file name")
		public String outFile;

		@Parameter(names = { "--intransformations", "-t" }, description = "input JSON export of alignment transofomations, formatted as a list of lists")
		public String inTransformations;

		@Parameter(names = { "--meshcellsize", "-c" }, description = "mesh cell size for rendering the output")
		public long meshCellSize;
	}

	final static private int[] cellDimensions = new int[] { 64, 64, 8 };

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

		final long meshRes = Math.max(2, sourceInterval.dimension(0) / meshCellSize);
		for (long z = target.min(2); z < Math.min(sliceTransforms.size(), target.max(2) + 1); ++z) {

			System.out.println( z + " " + interpolatorFactory.getClass().getSimpleName() );

			final TransformMesh mesh =
					new TransformMesh(
							sliceTransforms.get((int)z),
							(int)meshRes,
							sourceInterval.dimension(0),
							sourceInterval.dimension(1));
			mesh.updateAffines();

			final TransformMeshMapping<T> mapping = new TransformMeshMapping<>(mesh);

			final RealRandomAccessible<T> sourceSlice =
					Views.interpolate(
							Views.hyperSlice(source, 2, z),
							interpolatorFactory);

			final RandomAccessibleInterval<T> targetSlice = Views.hyperSlice(target, 2, z);

			mapping.map(sourceSlice, targetSlice, Runtime.getRuntime().availableProcessors());
		}
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

			final FinalInterval sourceInterval = new FinalInterval(trakem2Export.width, trakem2Export.height, trakem2Export.transforms.size());

			/* deform */
			RandomAccessibleInterval<UnsignedByteType> rawTarget = PlanarImgs.unsignedBytes(
					sourceInterval.dimension(0),
					sourceInterval.dimension(1),
					sourceInterval.dimension(2));

			mapSlices(
					Views.extendValue(rawSource, new UnsignedByteType(0)),
					sourceInterval,
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

			H5Utils.saveAttribute(new double[] { 40.0, 4.0, 4.0 }, writer, rawPath, "resolution");
			rawTarget = null;

			/* labels */
			for (final String labelsPath : params.labels) {

                System.out.println("transforming: " + labelsPath);
				final RandomAccessibleInterval<LabelMultisetType> labelsSource = Util.loadLabels(labelsReader, labelsPath, cellDimensions);

				final RandomAccessibleInterval<LongType> longLabelsSource = Converters.convert(labelsSource,
						new Converter<LabelMultisetType, LongType>() {
							@Override
							public void convert(final LabelMultisetType a, final LongType b) {
								b.set(a.entrySet().iterator().next().getElement().id());
							}
						}, new LongType());

				H5Utils.createUnsignedLong(writer, labelsPath, sourceInterval, cellDimensions);

				/* process in mellSize[2] thick slices to save some memory */
				for (int zOffset = 0; zOffset < sourceInterval.dimension(2); zOffset += cellDimensions[2]) {

					/* deform */
					final RandomAccessibleInterval<LongType> labelsTarget =
							Views.translate(
									PlanarImgs.longs(
											sourceInterval.dimension(0),
											sourceInterval.dimension(1),
											Math.min(sourceInterval.dimension(2), cellDimensions[2])),
									0,
									0,
									zOffset);

					/* clear canvas by filling with outside */
					for (final LongType t : Views.iterable(labelsTarget))
						t.set(Label.OUTSIDE);

					mapSlices(
							Views.extendValue(longLabelsSource, new LongType(Label.OUTSIDE)),
							sourceInterval,
							transforms,
							new NearestNeighborInterpolatorFactory<>(),
							labelsTarget,
							params.meshCellSize);

					/* save */
                    // FIXME This threw the following exception
                    // Exception in thread "main" ncsa.hdf.hdf5lib.exceptions.HDF5FunctionArgumentException: Invalid arguments to routine:Bad value ["H5Dio.c line 686 in H5D__write(): src dataspace has invalid selection"]
                    // which I have 'fixed' by now with replacing 'saveUnsignedLong' with 'saveLong'
                    // maybe different HDF5 versions?
					System.out.println("writing " + labelsPath);
					H5Utils.saveLong(
							labelsTarget,
							outFile,
							labelsPath,
							cellDimensions);
					System.out.println("done!");
				}

				H5Utils.saveAttribute(new double[] { 40.0, 4.0, 4.0 }, writer, labelsPath, "resolution");
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
