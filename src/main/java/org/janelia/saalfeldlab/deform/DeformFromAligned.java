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
import net.imglib2.converter.Converter;
import net.imglib2.converter.Converters;
import net.imglib2.interpolation.InterpolatorFactory;
import net.imglib2.interpolation.randomaccess.ClampingNLinearInterpolatorFactory;
import net.imglib2.interpolation.randomaccess.NearestNeighborInterpolatorFactory;
import net.imglib2.realtransform.RealTransform;
import net.imglib2.realtransform.RealTransformRandomAccessible;
import net.imglib2.realtransform.RealViews;
import net.imglib2.realtransform.Scale3D;
import net.imglib2.type.numeric.integer.LongType;
import net.imglib2.type.numeric.integer.UnsignedByteType;
import net.imglib2.view.Views;

/**
 * @author Stephan Saalfeld &lt;saalfelds@janelia.hhmi.org&gt;
 *
 */
public class DeformFromAligned {

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

		@Parameter(names = { "--offset", "-m" }, description = "offset (min coordinate of the output interval, CSV in numpy order)")
		public String offset = "0,0,0";

		@Parameter(names = { "--size", "-s" }, description = "size (dimensions of the output interval, CSV in numpy order)")
		public String size = "200,3072,3072";

		public static long[] getReorderedLongArray(final String csv) {
			final String[] valueStrings = csv.split(",");
			final long[] values = new long[ valueStrings.length ];
			for (int i = 0; i < valueStrings.length; ++i)
				values[values.length - i - 1] = Long.parseLong(valueStrings[i]);

			return values;
		}

		public long[] getMin() {
			return getReorderedLongArray(offset);
		}

		public long[] getDimensions() {
			return getReorderedLongArray(size);
		}

		public long[] getMax() {
			final long[] min = getMin();
			final long[] dimensions = getDimensions();
			final long[] max = new long[ min.length ];
			for (int i = 0; i < min.length; ++i)
				max[i] = min[i] + dimensions[i] - 1;

			return max;
		}
	}

	final static private int[] cellDimensions = new int[] { 64, 64, 8 };

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
							interval));
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
			final FinalInterval targetInterval =
					new FinalInterval(params.getMin(), params.getMax());

			final RandomAccessibleInterval<UnsignedByteType> rawTarget = mapInverseSlices(
					Views.extendValue(rawSource, new UnsignedByteType(0)),
					targetInterval,
					transforms,
					new ClampingNLinearInterpolatorFactory<>());

			/* save */
			System.out.println("writing " + params.outFile);

			System.out.println("  " + rawPath);
			H5Utils.saveUnsignedByte(
					rawTarget,
					outFile,
					rawPath,
					cellDimensions);

			H5Utils.saveAttribute(new double[] { 40.0, 4.0, 4.0 }, writer, rawPath, "resolution");
			H5Utils.saveAttribute(new double[] { targetInterval.min( 2 ), targetInterval.min( 1 ), targetInterval.min( 0 ) }, writer, rawPath, "offset");

			/* labels */
			final ArrayList<RandomAccessibleInterval<LongType>> labelsTargets = new ArrayList<>();
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
				final RandomAccessibleInterval<LongType> labelsTarget = mapInverseSlices(
						Views.extendValue(longLabelsSource, new LongType(Label.OUTSIDE)),
						targetInterval,
						transforms,
						new NearestNeighborInterpolatorFactory<>());

				labelsTargets.add(labelsTarget);

				/* save */
				System.out.println("writing " + labelsPath);
				H5Utils.saveUnsignedLong(
						labelsTarget,
						outFile,
						labelsPath,
						cellDimensions);

				H5Utils.saveAttribute(new double[] { 40.0, 4.0, 4.0 }, writer, labelsPath, "resolution");
				H5Utils.saveAttribute(new double[] { targetInterval.min( 2 ), targetInterval.min( 1 ), targetInterval.min( 0 ) }, writer, labelsPath, "offset");
			}

			writer.close();

			Util.display(
					RealViews.affine(
							Views.interpolate(
									Views.extendZero(rawTarget),
									new NearestNeighborInterpolatorFactory<>()),
							new Scale3D(1, 1, 10)),
					RealViews.affine(
							Views.interpolate(
									Views.extendValue(labelsTargets.get(0), new LongType(Label.OUTSIDE)),
									new NearestNeighborInterpolatorFactory<>()),
							new Scale3D(1, 1, 10)),
					new FinalInterval(
							trakem2Export.width,
							trakem2Export.height,
							rawTarget.dimension(2) * 10));
		}
	}
}
