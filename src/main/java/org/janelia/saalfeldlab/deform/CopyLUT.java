/**
 *
 */
package org.janelia.saalfeldlab.deform;

import java.io.IOException;

import com.beust.jcommander.JCommander;
import com.beust.jcommander.Parameter;

import bdv.img.h5.H5Utils;
import ch.systemsx.cisd.hdf5.HDF5Factory;
import gnu.trove.map.hash.TLongLongHashMap;

/**
 * @author Stephan Saalfeld &lt;saalfelds@janelia.hhmi.org&gt;
 *
 */
public class CopyLUT {

	public static class Parameters {

		@Parameter(names = { "--infile", "-i" }, description = "input CREMI-format HDF5 file name")
		public String inFile;

		@Parameter(names = { "--outfile", "-o" }, description = "output CREMI-format HDF5 file name")
		public String outFile;

		@Parameter( names = { "--assignment", "-a" }, description = "fragment segment assignment table" )
		public String assignment = "/fragment_segment_lut";
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

		final TLongLongHashMap lut = H5Utils.loadLongLongLut( HDF5Factory.openForReading(params.inFile), params.assignment, 1024 );

		H5Utils.saveLongLongLut(lut, params.outFile, params.assignment, 1024);
	}
}
