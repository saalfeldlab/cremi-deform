/**
 *
 */
package org.janelia.saalfeldlab.deform;

import java.io.File;
import java.io.FileReader;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;
import java.util.List;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

import org.janelia.saalfeldlab.n5.DataType;
import org.janelia.saalfeldlab.n5.GzipCompression;
import org.janelia.saalfeldlab.n5.N5FSReader;
import org.janelia.saalfeldlab.n5.N5FSWriter;
import org.janelia.saalfeldlab.n5.N5Reader;
import org.janelia.saalfeldlab.n5.N5Writer;
import org.janelia.saalfeldlab.n5.hdf5.N5HDF5Reader;
import org.janelia.saalfeldlab.n5.hdf5.N5HDF5Writer;
import org.janelia.saalfeldlab.n5.imglib2.N5Utils;

import com.google.gson.Gson;

import bdv.bigcat.annotation.Annotation;
import bdv.bigcat.annotation.Annotations;
import bdv.bigcat.annotation.AnnotationsHdf5Store;
import bdv.labels.labelset.Label;
import bdv.util.LocalIdService;
import ch.systemsx.cisd.hdf5.HDF5Factory;
import ch.systemsx.cisd.hdf5.IHDF5Writer;
import mpicbg.models.InvertibleCoordinateTransform;
import mpicbg.models.NoninvertibleModelException;
import mpicbg.trakem2.transform.CoordinateTransform;
import mpicbg.trakem2.transform.CoordinateTransformList;
import net.imglib2.FinalInterval;
import net.imglib2.Interval;
import net.imglib2.RandomAccessible;
import net.imglib2.RandomAccessibleInterval;
import net.imglib2.RealPoint;
import net.imglib2.RealRandomAccessible;
import net.imglib2.converter.Converters;
import net.imglib2.img.basictypeaccess.array.ByteArray;
import net.imglib2.img.planar.PlanarImg;
import net.imglib2.img.planar.PlanarImgFactory;
import net.imglib2.img.planar.PlanarImgs;
import net.imglib2.interpolation.InterpolatorFactory;
import net.imglib2.interpolation.randomaccess.ClampingNLinearInterpolatorFactory;
import net.imglib2.interpolation.randomaccess.NearestNeighborInterpolatorFactory;
import net.imglib2.type.Type;
import net.imglib2.type.numeric.RealType;
import net.imglib2.type.numeric.integer.UnsignedByteType;
import net.imglib2.type.numeric.integer.UnsignedLongType;
import net.imglib2.util.Intervals;
import net.imglib2.view.Views;
import picocli.CommandLine;
import picocli.CommandLine.Option;

/**
 * @author Stephan Saalfeld &lt;saalfelds@janelia.hhmi.org&gt;
 *
 */
public class DeformFromAligned {

	public static class Parameters {

		@Option(names = { "--infile", "-i" }, description = "input CREMI-format HDF5 or N5 file name")
		public String inFile = null;

		@Option(names = { "--infile_labels", "-j" }, description = "input CREMI-format HDF5 file name")
		public String inFileLabels = null;

		@Option(names = { "--infile_annotations" }, description = "input for annotaions CREMI-format HDF5 file name (default == --infile")
		public String inFileAnnotations = null;

		@Option(names = { "--raw" }, description = "raw datasets, multiple entries possible" )
		public List<String> raws = new ArrayList<>();

		@Option( names = { "--label", "-l" }, description = "label dataset" )
		public List<String> labels = Arrays.asList( new String[]{"/volumes/labels/clefts", "/volumes/labels/neuron_ids"});

		@Option( names = { "--labelexportname", "-e" }, description = "label dataset export name" )
		public List<String> labelExportNames = new ArrayList<>();

		@Option(names = { "--outfile", "-o" }, required = true, description = "output CREMI-format HDF5 or N5 file name")
		public String outFile;

		@Option(names = { "--intransform", "-t" }, required = true, description = "input JSON export of alignment transofomations, formatted as a list of lists")
		public String inTransformations;

		@Option(names = { "--targetoffset", "-m" }, description = "offset (min coordinate of the output interval, CSV in numpy order)")
		public String targetOffset = "0,0,0";

		@Option(names = { "--targetsize", "-s" }, description = "size (dimensions of the output interval, CSV in numpy order)")
		public String targetDimensions = "200,3072,3072";

		@Option(names = { "--transformsize", "-k" }, description = "transformation size (dimensions of the interval in which the transformations are defined, CSV in numpy order)")
		public String transformDimensions = "0,0,0";

		@Option(names = { "--labelssourceoffset", "-n" }, description = "label source offset (min coordinates of label sources in px, CSV in numpy order, overrides offset attribute in label datasets)")
		public String labelsSourceOffset = null;

		@Option(names = { "--resolution", "-r" }, description = "resolution (pixel size, CSV in numpy order)")
		public String resolution = "40,4,4";

		@Option(names = { "--meshcellsize", "-c" }, required = true, description = "mesh cell size for rendering the output")
		public long meshCellSize;

		@Option(names = { "--threshold", "-x" }, description = "threshold for label input")
		public Double threshold = null;


		public static long[] getReorderedLongArray(final String csv) {
			final String[] valueStrings = csv.split(",");
			final long[] values = new long[ valueStrings.length ];
			for (int i = 0; i < valueStrings.length; ++i)
				values[values.length - i - 1] = Long.parseLong(valueStrings[i]);

			return values;
		}

		public static double[] getReorderedDoubleArray(final String csv) {
			final String[] valueStrings = csv.split(",");
			final double[] values = new double[ valueStrings.length ];
			for (int i = 0; i < valueStrings.length; ++i)
				values[values.length - i - 1] = Double.parseDouble(valueStrings[i]);

			return values;
		}

		public long[] getTargetMin() {
			return getReorderedLongArray(targetOffset);
		}

		public long[] getTargetDimensions() {
			return getReorderedLongArray(targetDimensions);
		}

		public long[] getTransformDimensions() {
			return getReorderedLongArray(transformDimensions);
		}

		public double[] getResolution() {
			return getReorderedDoubleArray(resolution);
		}

		public long[] getLabelsSourceMin() {
			if (labelsSourceOffset == null)
				return null;
			else
				return getReorderedLongArray(labelsSourceOffset);
		}

		public long[] getTargetMax() {
			final long[] min = getTargetMin();
			final long[] dimensions = getTargetDimensions();
			final long[] max = new long[ min.length ];
			for (int i = 0; i < min.length; ++i)
				max[i] = min[i] + dimensions[i] - 1;

			return max;
		}
	}

	final static private int[] cellDimensions = new int[] { 64, 64, 8 };

	/**
	 * Creates a transformed version of a 3D volume by mapping each slice with
	 * a mesh transform, mpicbg style.
	 *
	 * @param source extended source
	 * @param sourceInterval the size of the full source before transformation
	 * @param sliceTransforms
	 * @param interpolatorFactory
	 * @param target
	 * @param meshCellSize
	 * @return
	 */
	static public ArrayList<TransformMesh> createMeshes(
			final Interval sourceInterval,
			final List<CoordinateTransform> sliceTransforms,
			final long meshCellSize) {

		final ArrayList<TransformMesh> meshes = new ArrayList<>();

		final long meshRes = Math.max(2, sourceInterval.dimension(0) / meshCellSize);
		System.out.println(String.format("sourceIntervalDimension[0] = %d, meshRes  = %d, meshCellSize = %d", sourceInterval.dimension(0), meshRes, meshCellSize));
		for (int z = 0; z < sliceTransforms.size(); ++z) {

			final TransformMesh mesh =
					new TransformMesh(
							sliceTransforms.get((int)z),
							(int)meshRes,
							sourceInterval.dimension(0),
							sourceInterval.dimension(1));
			mesh.updateAffines();
			meshes.add(mesh);
		}
		return meshes;
	}

	/**
	 * Creates a transformed version of a 3D volume by mapping each slice with
	 * a mesh transform, mpicbg style.
	 *
	 * @param source extended source
	 * @param sourceInterval the size of the full source before transformation
	 * @param sliceTransforms
	 * @param interpolatorFactory
	 * @param target
	 * @param meshCellSize
	 * @return
	 */
	static public <T extends Type<T>> void mapInverseSlices(
			final RandomAccessible<T> source,
			final Interval sourceInterval,
			final List<TransformMesh> meshes,
			final InterpolatorFactory<T, RandomAccessible<T>> interpolatorFactory,
			final RandomAccessibleInterval<T> target,
			final long meshCellSize) {

		for (long z = target.min(2); z < Math.min(meshes.size(), target.max(2) + 1); ++z) {

			System.out.println( z + " " + interpolatorFactory.getClass().getSimpleName() );

			final TransformMeshMapping<T> mapping = new TransformMeshMapping<>(meshes.get((int)z));

			final RealRandomAccessible<T> sourceSlice =
					Views.interpolate(
							Views.hyperSlice(source, 2, z),
							interpolatorFactory);

			final RandomAccessibleInterval<T> targetSlice = Views.hyperSlice(target, 2, z);

			mapping.mapInverse(sourceSlice, targetSlice, Runtime.getRuntime().availableProcessors());
		}
	}

	/**
	 * @param args
	 * @throws Exception
	 */
	public static void main(final String... args) throws Exception {

		final Parameters params = new Parameters();
		try {
			CommandLine.populateCommand(params, args);
		} catch (final RuntimeException e) {
			CommandLine.usage(params, System.err);
			return;
		}

		try (final FileReader transformsReader = new FileReader(new File(params.inTransformations))) {

			final boolean h5Input = params.inFile == null ? false : Files.isRegularFile(Paths.get(params.inFile));
			final boolean h5InputLabels = params.inFileLabels == null ? false : Files.isRegularFile(Paths.get(params.inFileLabels));
			// annotations are always h5, i.e. do not process annotations from N5 FS sources
			final boolean h5InputAnnotations = params.inFileAnnotations == null ? false : Files.isRegularFile(Paths.get(params.inFileAnnotations));

			final boolean h5Output = params.outFile.endsWith(".h5") || params.outFile.endsWith(".hdf5") || params.outFile.endsWith(".hdf");

			System.out.println("Opening " + params.inFile);
			final N5Reader rawReader = params.inFile == null ? null : h5Input ? new N5HDF5Reader(params.inFile, 64) : new N5FSReader(params.inFile);
			final N5Reader labelsReader = params.inFileLabels == null ? null : h5InputLabels ? new N5HDF5Reader(params.inFileLabels, 64) : new N5FSReader(params.inFileLabels);
			if (params.inFileAnnotations == null)
				params.inFileAnnotations = params.inFile;

			System.out.println("Opening " + params.outFile + " for writing");

			final IHDF5Writer hdf5Writer = h5Output ? HDF5Factory.open(params.outFile) : null;
			final N5Writer writer = h5Output ? new N5HDF5Writer(hdf5Writer, 64) : new N5FSWriter(params.outFile);

			final FinalInterval targetInterval =
					new FinalInterval(params.getTargetMin(), params.getTargetMax());

			final FinalInterval sourceInterval =
					new FinalInterval(params.getTransformDimensions());

			final ArrayList<CoordinateTransform> transforms = new ArrayList<>();
			final TrakEM2Export trakem2Export = new Gson().fromJson(transformsReader, TrakEM2Export.class);
			for (final ArrayList<TransformSpec> transformList : trakem2Export.transforms) {
				final CoordinateTransformList<CoordinateTransform> ctl = new CoordinateTransformList<>();
				for (final TransformSpec t : transformList)
					ctl.add(t.createTransform());

				transforms.add(ctl);
			}

			final ArrayList<TransformMesh> meshes = createMeshes(sourceInterval, transforms, params.meshCellSize);


			final double[] resolution = params.getResolution();

			/* raw pixels */
			if (rawReader != null) {

				for (final String raw : params.raws) {

					System.out.println("Loading raw pixels " + raw);
					final RandomAccessibleInterval<UnsignedByteType> rawSource = N5Utils.open(rawReader, raw);

					/* deform */
					final PlanarImg<UnsignedByteType, ByteArray> rawTarget = PlanarImgs.unsignedBytes(params.getTargetDimensions());

					final RandomAccessibleInterval<UnsignedByteType> rawTargetInterval =
							Views.translate(
									rawTarget,
									params.getTargetMin());

					mapInverseSlices(
							Views.extendValue(rawSource, new UnsignedByteType(0)),
							sourceInterval,
							meshes,
							new ClampingNLinearInterpolatorFactory<>(),
							rawTargetInterval,
							params.meshCellSize);

					/* save */
					System.out.println("writing " + params.outFile);
					System.out.println("  " + raw);
					final ExecutorService exec = h5Output ? Executors.newSingleThreadExecutor() : Executors.newFixedThreadPool(Runtime.getRuntime().availableProcessors());
					N5Utils.save(rawTarget, writer, raw, cellDimensions, new GzipCompression(), exec);

					exec.shutdown();

					writer.setAttribute(raw, "resolution", h5Output ? new double[] { resolution[2], resolution[1], resolution[0] } : resolution);
					writer.setAttribute(raw, "offset", new double[] {0, 0, 0});
//					writer.setAttribute(raw, "offset", h5Output ? new double[] {
//									targetInterval.min(2) * resolution[2],
//									targetInterval.min(1) * resolution[1],
//									targetInterval.min(0) * resolution[0] } : new double[] {
//
//									targetInterval.min(0) * resolution[0],
//									targetInterval.min(1) * resolution[1],
//									targetInterval.min(2) * resolution[2] });
				}
			}

			/* labels */
			if (labelsReader != null) {

				for (int i = 0; i < params.labels.size(); ++i) {

					final String labelsPath = params.labels.get(i);
					final String labelsExportPath = i < params.labelExportNames.size() ? params.labelExportNames.get(i) : labelsPath;

					System.out.println("labels " + labelsPath + " > " + labelsExportPath);

					RandomAccessibleInterval<RealType<?>> labelsSource = (RandomAccessibleInterval)N5Utils.open(labelsReader, labelsPath);
					for (int d = 0; d < labelsSource.numDimensions();)
						if (labelsSource.dimension(d) == 1)
							labelsSource = Views.hyperSlice(labelsSource, d, 0);
						else ++d;

					/* optionally override label source offset */
					final long[] labelsSourceMin = params.getLabelsSourceMin();
					if (labelsSourceMin != null)
						labelsSource = Views.translate(
								labelsSource,
								labelsSourceMin[0] - labelsSource.min(0),
								labelsSourceMin[1] - labelsSource.min(1),
								labelsSourceMin[2] - labelsSource.min(2));
					else {
						double[] labelOffset = labelsReader.getAttribute(labelsPath, "offset", double[].class);
						if (labelOffset != null) {
							if (h5InputLabels)
								labelOffset = new double[]{
										labelOffset[2],
										labelOffset[1],
										labelOffset[0]};
								labelsSource = Views.translate(
										labelsSource,
										(long)Math.round(labelOffset[0] / resolution[0]) - labelsSource.min(0),
										(long)Math.round(labelOffset[1] / resolution[1]) - labelsSource.min(1),
										(long)Math.round(labelOffset[2] / resolution[2]) - labelsSource.min(2));
						}
					}

					final RandomAccessibleInterval<UnsignedLongType> longLabelsSource;
					if (params.threshold != null) {
						final double threshold = params.threshold;
						longLabelsSource = Converters.convert(
								labelsSource,
								(a, b) -> {
									final double v = a.getRealDouble();
									b.set(v < params.threshold ? Label.TRANSPARENT : 1);
								},
								new UnsignedLongType());
					} else
						longLabelsSource = (RandomAccessibleInterval)labelsSource;


					writer.createDataset(
							labelsExportPath,
							Intervals.dimensionsAsLongArray(targetInterval),
							cellDimensions,
							DataType.UINT64,
							new GzipCompression());

					final ExecutorService exec = h5Output ? Executors.newSingleThreadExecutor() : Executors.newFixedThreadPool(Runtime.getRuntime().availableProcessors());

					/* process in cellSize[2] thick slices to save some memory */
					for (long zOffset = targetInterval.min(2); zOffset <= targetInterval.max(2); zOffset += cellDimensions[2]) {

						/* deform */
						final RandomAccessibleInterval<UnsignedLongType> labelsTarget =
								Views.translate(
										new PlanarImgFactory<UnsignedLongType>(new UnsignedLongType()).create(
												new long[] {
														targetInterval.dimension(0),
														targetInterval.dimension(1),
														Math.min(targetInterval.max(2) - zOffset + 1, cellDimensions[2])}),
										targetInterval.min(0),
										targetInterval.min(1),
										zOffset);

						/* clear canvas by filling with outside */
						for (final UnsignedLongType t : Views.iterable(labelsTarget))
							t.set(Label.OUTSIDE);

						mapInverseSlices(
								Views.extendValue(longLabelsSource, new UnsignedLongType(Label.OUTSIDE)),
								sourceInterval,
								meshes,
								new NearestNeighborInterpolatorFactory<>(),
								labelsTarget,
								params.meshCellSize);

						/* save */
						System.out.println("writing " + labelsPath);
						N5Utils.saveBlock(
								labelsTarget,
								writer,
								labelsExportPath,
								new long[] {
										0,
										0,
										(zOffset - targetInterval.min(2)) / cellDimensions[2]},
								exec);
					}

					exec.shutdown();

					writer.setAttribute(labelsExportPath, "resolution", h5Output ? new double[]{resolution[2], resolution[1], resolution[0]} : resolution);
					writer.setAttribute(labelsExportPath, "offset", new double[] {0, 0, 0});
//					writer.setAttribute(labelsExportPath, "offset", h5Output ? new double[]{
//							targetInterval.min(2) * resolution[2],
//							targetInterval.min(1) * resolution[1],
//							targetInterval.min(0) * resolution[0]} : new double[]{
//
//							targetInterval.min(0) * resolution[0],
//							targetInterval.min(1) * resolution[1],
//							targetInterval.min(2) * resolution[2]});
				}
			}

			if (hdf5Writer != null)
				hdf5Writer.close();

			/* annotations */
			final long[] annotationsOffset = params.getLabelsSourceMin() == null ? new long[]{0, 0, 0} : params.getLabelsSourceMin();

			final AnnotationsHdf5Store annotationsStore = new AnnotationsHdf5Store(params.inFileAnnotations, new LocalIdService());
			final Annotations annotations = annotationsStore.read();

			final Collection<Annotation> annotationsCollection = annotations.getAnnotations();
			annotationsCollection.forEach(
					a -> {
						final double[] slicePosition = new double[2];
						final RealPoint position = a.getPosition();
						slicePosition[0] = (position.getDoublePosition(0) + annotationsOffset[0]) / resolution[0];
						slicePosition[1] = (position.getDoublePosition(1) + annotationsOffset[1]) / resolution[1];
						final int zIndex = (int)Math.round((position.getDoublePosition(2) + annotationsOffset[2]) / resolution[2]);
						System.out.println(zIndex);
						System.out.print(net.imglib2.util.Util.printCoordinates(slicePosition) + " -> ");
						if (zIndex >= 0 && zIndex < transforms.size()) {
							final InvertibleCoordinateTransform transform = meshes.get(zIndex);
							try {
								transform.applyInverseInPlace(slicePosition);
							} catch (final NoninvertibleModelException e) {
								System.err.println("Unable to transfer synapse annotation " + a.getId() + " at " + net.imglib2.util.Util.printCoordinates(slicePosition));
							}
							System.out.println(net.imglib2.util.Util.printCoordinates(slicePosition));
						}
						position.setPosition((slicePosition[0] - targetInterval.min(0)) * resolution[0], 0);
						position.setPosition((slicePosition[1] - targetInterval.min(1)) * resolution[1], 1);
						position.move(-targetInterval.min(2) * resolution[2], 2);
					});

			final AnnotationsHdf5Store annotationsOutStore = new AnnotationsHdf5Store(params.outFile, new LocalIdService());
			annotationsOutStore.write(annotations);
		}
	}
}
