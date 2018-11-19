/**
 *
 */
package org.janelia.saalfeldlab.deform;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

import bdv.img.cache.VolatileGlobalCellCache;
import bdv.img.h5.H5UnsignedByteSetupImageLoader;
import ch.systemsx.cisd.hdf5.HDF5DataTypeInformation;
import ch.systemsx.cisd.hdf5.HDF5Factory;
import ch.systemsx.cisd.hdf5.IHDF5Reader;
import ij.IJ;
import ij.ImagePlus;
import ij.process.ByteProcessor;
import mpicbg.spim.data.generic.sequence.ImgLoaderHints;
import net.imglib2.RandomAccessibleInterval;
import net.imglib2.type.numeric.NumericType;
import net.imglib2.type.numeric.integer.UnsignedByteType;
import net.imglib2.view.Views;
import picocli.CommandLine;
import picocli.CommandLine.Option;

/**
 * @author Stephan Saalfeld &lt;saalfelds@janelia.hhmi.org&gt;
 *
 */
public class ExportTiffSeries {

	public static class Parameters {

		@Option(names={"--infile", "-i"}, description = "input CREMI-format HDF5 file name")
		public String inFile;

		@Option(names={"--outpath", "-o"}, description = "output path")
		public String outPath;

		@Option(names={"--dataset", "-d"}, description = "dataset to be exported")
		public List< String > datasets;
	}

	final static private int[] cellDimensions = new int[] { 64, 64, 8 };

	static public <T extends NumericType<T>> List<RandomAccessibleInterval<T>> slice(final RandomAccessibleInterval<T> source) {

		final ArrayList<RandomAccessibleInterval<T>> slices = new ArrayList<>();
		for (int z = 0; z < source.dimension(2); ++z)
			slices.add(Views.hyperSlice(source, 2, z));

		return slices;
	}

	static public void saveSlices(final RandomAccessibleInterval<UnsignedByteType> source, final String basePath) {

		for (int z = 0; z < source.dimension(2); ++z) {
			final RandomAccessibleInterval<UnsignedByteType> slice = Views.hyperSlice(source, 2, z);
			final ByteProcessor bp = new ByteProcessor((int)slice.dimension(0), (int)slice.dimension(0));
			int i = 0;
			for (final UnsignedByteType t : Views.flatIterable(slice))
				bp.set(i++, t.get());
			final ImagePlus imp = new ImagePlus("", bp);
			IJ.saveAsTiff(imp, basePath + String.format("%05d", z) + ".tif");
		}
	}

	/**
	 * @param args
	 * @throws IOException
	 */
	public static void main(final String[] args) throws IOException {

		final Parameters params = new Parameters();
		try {
			CommandLine.populateCommand(params, args);
		} catch (final RuntimeException e) {
			CommandLine.usage(params, System.err);
			return;
		}

		System.out.println("Opening " + params.inFile);
		final IHDF5Reader reader = HDF5Factory.openForReading(params.inFile);

		for (final String dataset : params.datasets) {
			if (reader.exists(dataset)) {
				final HDF5DataTypeInformation typeInfo = reader.object().getDataSetInformation(dataset).getTypeInformation();
				if (typeInfo.tryGetJavaType().isAssignableFrom( byte.class ))
				{
					final H5UnsignedByteSetupImageLoader raw = new H5UnsignedByteSetupImageLoader(
							reader,
							dataset,
							0,
							cellDimensions,
							new VolatileGlobalCellCache(1, 1));
					saveSlices( raw.getImage(0, ImgLoaderHints.LOAD_COMPLETELY), params.outPath);
				}
			}
		}
	}


}
