/**
 *
 */
package org.janelia.saalfeldlab.deform;

import java.io.File;
import java.io.IOException;
import java.util.Arrays;

import com.beust.jcommander.JCommander;
import com.beust.jcommander.Parameter;

import bdv.img.h5.H5Utils;
import bdv.labels.labelset.Label;
import bdv.labels.labelset.LabelMultisetType;
import ch.systemsx.cisd.hdf5.HDF5Factory;
import ch.systemsx.cisd.hdf5.IHDF5Reader;
import ch.systemsx.cisd.hdf5.IHDF5Writer;
import net.imglib2.Cursor;
import net.imglib2.FinalInterval;
import net.imglib2.RandomAccessibleInterval;
import net.imglib2.converter.Converter;
import net.imglib2.converter.Converters;
import net.imglib2.interpolation.randomaccess.NearestNeighborInterpolatorFactory;
import net.imglib2.realtransform.RealViews;
import net.imglib2.realtransform.Scale3D;
import net.imglib2.type.numeric.integer.LongType;
import net.imglib2.type.numeric.integer.UnsignedByteType;
import net.imglib2.view.Views;

/**
 * @author Stephan Saalfeld &lt;saalfelds@janelia.hhmi.org&gt;
 *
 */
public class CropLabelBounds {

	public static class Parameters {

		@Parameter(names = { "--infile", "-i" }, description = "input CREMI-format HDF5 file name")
		public String inFile;

		@Parameter(names = { "--outfile", "-o" }, description = "output CREMI-format HDF5 file name")
		public String outFile;
	}

	final static private int[] cellDimensions = new int[] { 64, 64, 8 };


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

		final String labelsDataset = "neuron_ids";
		final String cleftsDataset = "clefts";
		final String rawDataset = "raw";

		System.out.println("Opening " + params.inFile);
		final IHDF5Reader reader = HDF5Factory.openForReading(params.inFile);
		final IHDF5Writer writer = HDF5Factory.open(params.outFile);

		// support both file_format 0.0 and >=0.1
		final String volumesPath = reader.isGroup("/volumes") ? "/volumes" : "";
		final String labelsPath = reader.isGroup(volumesPath + "/labels") ? volumesPath + "/labels" : "";

		/* raw pixels */
		final String rawPath = volumesPath + "/" + rawDataset;
		final RandomAccessibleInterval<UnsignedByteType> rawSource = Util.loadRaw(reader, rawPath, cellDimensions);

		/* labels */
		final String fragmentsPath = labelsPath + "/" + labelsDataset;
		final RandomAccessibleInterval<LabelMultisetType> labelsSource = Util.loadLabels(reader, fragmentsPath, cellDimensions);

		final RandomAccessibleInterval<LongType> longLabelsSource = Converters.convert(labelsSource,
				new Converter<LabelMultisetType, LongType>() {
					@Override
					public void convert(final LabelMultisetType a, final LongType b) {
						b.set(a.entrySet().iterator().next().getElement().id());
					}
				}, new LongType());

		final long[] min = new long[]{Long.MAX_VALUE, Long.MAX_VALUE, Long.MAX_VALUE};
		final long[] max = new long[]{Long.MIN_VALUE, Long.MIN_VALUE, Long.MIN_VALUE};
		final Cursor<LongType> labelCursor = Views.flatIterable(longLabelsSource).localizingCursor();
		while (labelCursor.hasNext()) {
			if (labelCursor.next().get() != Label.OUTSIDE) {
				for (int d = 0; d < 3; ++d) {
					min[d] = Math.min(min[d], labelCursor.getLongPosition(d));
					max[d] = Math.max(max[d], labelCursor.getLongPosition(d));
				}
			}
		}

		System.out.println(String.format("min = %s; max = %s", Arrays.toString(min), Arrays.toString(max)));

		/* save */
		System.out.println("writing " + params.outFile);

		final File outFile = new File(params.outFile);
		System.out.println("  " + rawPath);
		H5Utils.saveUnsignedByte(
				Views.offsetInterval(rawSource, new FinalInterval(min, max)),
				outFile,
				rawPath,
				cellDimensions);

		writer.float64().setArrayAttr(rawPath, "resolution", new double[]{40.0, 4.0, 4.0});
		writer.float64().setArrayAttr(rawPath, "offset", new double[]{40.0 * min[2], 4.0 * min[1], 4.0 * min[0]});

		System.out.println("  " + fragmentsPath);
		H5Utils.saveUnsignedLong(
				Views.offsetInterval(longLabelsSource, new FinalInterval(min, max)),
				outFile,
				fragmentsPath,
				cellDimensions);

		writer.float64().setArrayAttr(fragmentsPath, "resolution", new double[] { 40.0, 4.0, 4.0 });
		writer.float64().setArrayAttr(fragmentsPath, "offset", new double[]{40.0 * min[2], 4.0 * min[1], 4.0 * min[0]});

		/* clefts */
		final String cleftsPath = labelsPath + "/" + cleftsDataset;
		if (reader.exists(cleftsPath)) {
			final RandomAccessibleInterval<LabelMultisetType> cleftsSource = Util.loadLabels(reader, cleftsPath, cellDimensions);

			final RandomAccessibleInterval<LongType> longCleftsSource = Converters.convert(cleftsSource,
					new Converter<LabelMultisetType, LongType>() {
						@Override
						public void convert(final LabelMultisetType a, final LongType b) {
							b.set(a.entrySet().iterator().next().getElement().id());
						}
					}, new LongType());

			System.out.println("  " + cleftsPath);
			H5Utils.saveUnsignedLong(
					Views.offsetInterval(longCleftsSource, new FinalInterval(min, max)),
					outFile,
					cleftsPath,
					cellDimensions);
			writer.float64().setArrayAttr(cleftsPath, "resolution", new double[] { 40.0, 4.0, 4.0 });
			writer.float64().setArrayAttr(cleftsPath, "offset", new double[]{40.0 * min[2], 4.0 * min[1], 4.0 * min[0]});
		}

		writer.close();

		Util.display(
				RealViews.affine(
						Views.interpolate(
								Views.extendZero(Views.offsetInterval(rawSource, new FinalInterval(min, max))),
								new NearestNeighborInterpolatorFactory<>()),
						new Scale3D(1, 1, 10)),
				RealViews.affine(
						Views.interpolate(
								Views.extendValue(Views.offsetInterval(longLabelsSource, new FinalInterval(min, max)), new LongType(Label.OUTSIDE)),
								new NearestNeighborInterpolatorFactory<>()),
						new Scale3D(1, 1, 10)),
				new FinalInterval(
						max[0] - min[0] + 1,
						max[1] - min[1] + 1,
						(max[2] - min[2] + 1) * 10));
	}
}
