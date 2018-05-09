/**
 *
 */
package org.janelia.saalfeldlab.deform;

import org.janelia.saalfeldlab.n5.DatasetAttributes;
import org.janelia.saalfeldlab.n5.GzipCompression;
import org.janelia.saalfeldlab.n5.N5Writer;
import org.janelia.saalfeldlab.n5.hdf5.N5HDF5Writer;
import org.janelia.saalfeldlab.n5.imglib2.N5Utils;

import com.beust.jcommander.JCommander;
import com.beust.jcommander.Parameter;

import net.imglib2.FinalInterval;
import net.imglib2.position.FunctionRandomAccessible;
import net.imglib2.type.numeric.integer.UnsignedLongType;
import net.imglib2.view.Views;

/**
 * @author Stephan Saalfeld &lt;saalfelds@janelia.hhmi.org&gt;
 *
 */
public class TrainingValidationMask {

	public static class Parameters {

		@Parameter(names = { "--file", "-f" }, required = true, description = "input CREMI-format HDF5 file name")
		public String file;

		@Parameter(names = { "--label", "-l" }, description = "label dataset" )
		public String label;

		@Parameter(names = { "--validationThreshold", "-v" }, description = "validation threshold in [0,1]" )
		public double validationThreshold = 0.75;
	}

	/**
	 * @param args
	 * @throws Exception
	 */
	public static void main(final String... args) throws Exception {

		final Parameters params = new Parameters();
		new JCommander(params, args);

		System.out.println("Opening " + params.file);
		final N5Writer writer = new N5HDF5Writer(params.file, 256, 256, 26);

		final double[] resolution = writer.getAttribute(params.label, "resolution", double[].class);
		double[] offset = writer.getAttribute(params.label, "offset", double[].class);
		if (offset == null)
			offset = new double[] {0, 0, 0};

		final DatasetAttributes datasetAttributes = writer.getDatasetAttributes(params.label);
		final FinalInterval interval = new FinalInterval(datasetAttributes.getDimensions());
		final long validationThreshold = Math.round(datasetAttributes.getDimensions()[1] * 0.75);

		final FunctionRandomAccessible<UnsignedLongType> trainFunction = new FunctionRandomAccessible<UnsignedLongType>(3, (x, y) -> y.set(x.getLongPosition(1) < validationThreshold ? 1 : 0), UnsignedLongType::new);
		final FunctionRandomAccessible<UnsignedLongType> validationFunction = new FunctionRandomAccessible<UnsignedLongType>(3, (x, y) -> y.set(x.getLongPosition(1) >= validationThreshold ? 1 : 0), UnsignedLongType::new);

		N5Utils.save(
				Views.interval(trainFunction, interval),
				writer,
				"/volumes/masks/training",
				datasetAttributes.getBlockSize(),
				new GzipCompression());
		writer.setAttribute("/volumes/masks/training", "resolution", resolution);
		writer.setAttribute("/volumes/masks/training", "offset", offset);

		N5Utils.save(
				Views.interval(validationFunction, interval),
				writer,
				"/volumes/masks/validation",
				datasetAttributes.getBlockSize(),
				new GzipCompression());
		writer.setAttribute("/volumes/masks/validation", "resolution", resolution);
		writer.setAttribute("/volumes/masks/validation", "offset", offset);

		System.out.println("Done");
	}
}
