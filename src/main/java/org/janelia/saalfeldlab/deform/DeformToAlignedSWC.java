/**
 *
 */
package org.janelia.saalfeldlab.deform;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.nio.charset.Charset;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.nio.file.StandardOpenOption;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import org.apache.commons.csv.CSVFormat;
import org.apache.commons.csv.CSVParser;
import org.apache.commons.csv.CSVRecord;

import com.beust.jcommander.JCommander;
import com.beust.jcommander.Parameter;
import com.google.gson.Gson;

import mpicbg.trakem2.transform.CoordinateTransform;
import mpicbg.trakem2.transform.CoordinateTransformList;
import net.imglib2.Interval;
import net.imglib2.RandomAccessible;
import net.imglib2.RandomAccessibleInterval;
import net.imglib2.RealRandomAccessible;
import net.imglib2.interpolation.InterpolatorFactory;
import net.imglib2.type.Type;
import net.imglib2.view.Views;

/**
 * @author Stephan Saalfeld &lt;saalfelds@janelia.hhmi.org&gt;
 *
 */
public class DeformToAlignedSWC {

	public static class Parameters {

		@Parameter(names = { "--infile", "-i" }, required = true, description = "input SWC file name")
		public String inFile;

		@Parameter(names = { "--outfile", "-o" }, required = true, description = "output SWC file name")
		public String outFile;

		@Parameter(names = { "--intransformations", "-t" }, required = true, description = "input JSON export of alignment transformations, formatted as a list of lists")
		public String inTransformations;

		@Parameter(names = { "--intransformations_offset" }, description = "optional transformation offset")
		private String transformOffsetString = null;

		@Parameter(names = { "--intransformations_size" }, required = true, description = "transformation field of view")
		private String transformSizeString = null;

		@Parameter(names = { "--resolution" }, description = "resolution")
		private String resolutionString = null;

		public static double[] getDoubleArray(final String csv) {
			final String[] valueStrings = csv.split(",");
			final double[] values = new double[ valueStrings.length ];
			for (int i = 0; i < valueStrings.length; ++i)
				values[i] = Double.parseDouble(valueStrings[i]);

			return values;
		}

		public double[] getTransformOffset() {

			if (transformOffsetString == null)
				return new double[3];
			else
				return getDoubleArray(transformOffsetString);
		}

		public double[] getTransformSize() {

			return getDoubleArray(transformSizeString);
		}

		public double[] getResolution() {

			if (resolutionString == null)
				return new double[] {1, 1, 1};
			else
				return getDoubleArray(resolutionString);
		}
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
			final long meshCellSize,
			final int numProcessors) {

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

			mapping.map(sourceSlice, targetSlice, numProcessors);
		}
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

		mapSlices(
				source,
				sourceInterval,
				sliceTransforms,
				interpolatorFactory,
				target,
				meshCellSize,
				Runtime.getRuntime().availableProcessors());
	}

	/**
	 * @param args
	 * @throws Exception
	 */
	public static void main(final String... args) throws Exception {

		final Parameters params = new Parameters();
		new JCommander(params, args);

		/* transforms */
		TrakEM2Export trakem2Export = null;
		try (final FileReader transformsReader = new FileReader(new File(params.inTransformations))) {
			trakem2Export = new Gson().fromJson(transformsReader, TrakEM2Export.class);
		}
		if (trakem2Export == null)
			throw new FileNotFoundException("Could not find transforms " + params.inTransformations);

		final ArrayList<CoordinateTransform> transforms = new ArrayList<>();
		for (final ArrayList<TransformSpec> transformList : trakem2Export.transforms) {
			final CoordinateTransformList<CoordinateTransform> ctl = new CoordinateTransformList<>();
			for (final TransformSpec t : transformList)
				ctl.add(t.createTransform());

			transforms.add(ctl);
		}

		final double[] offset = params.getTransformOffset();
		final double[] size = params.getTransformSize();
		final double[] resolution = params.getResolution();

		/* csv */
		final Path inPath = Paths.get(params.inFile);
		final Path outPath = Paths.get(params.outFile);
		final CSVParser parser = CSVParser.parse(inPath, Charset.forName("UTF-8"), CSVFormat.DEFAULT);

		try (final BufferedWriter writer = Files.newBufferedWriter(outPath, Charset.forName("UTF-8"), StandardOpenOption.CREATE, StandardOpenOption.WRITE, StandardOpenOption.TRUNCATE_EXISTING)) {

			writer.write("neuron, skeleton_id, treenode_id, parent_treenode_id, x, y, z, r\n");

			for (final CSVRecord csvRecord : parser) {

				try {
					final double[] coordinate = new double[]{
							Double.parseDouble(csvRecord.get(4)) / resolution[0] - offset[0],
							Double.parseDouble(csvRecord.get(5)) / resolution[1] - offset[1],
							Double.parseDouble(csvRecord.get(6)) / resolution[2] - offset[2]};

					if (
							coordinate[0] < 0 || coordinate[0] > size[0] ||
							coordinate[1] < 0 || coordinate[1] > size[1] ||
							coordinate[2] < 0 || coordinate[2] > size[2] ) {

						System.out.println("Skipping out of FOV coordinate " + Arrays.toString(coordinate));
						continue;
					}

					final CoordinateTransform transform = transforms.get((int)Math.round(coordinate[2]));
					final double[] coordinate2D = new double[]{
							coordinate[0],
							coordinate[1]};
					transform.applyInPlace(coordinate2D);

					final String outRecord = String.format(
							"\"%s\",%s,%s,%s,%d,%d,%d,%s\n",
							csvRecord.get(0),
							csvRecord.get(1),
							csvRecord.get(2),
							csvRecord.get(3),
							(int)Math.round(coordinate2D[0] * resolution[0]),
							(int)Math.round(coordinate2D[1] * resolution[1]),
							(int)Math.round(coordinate[2] * resolution[2]),
							csvRecord.get(7));

					writer.write(outRecord);
					System.out.println(outRecord);
				} catch (final Exception e) {
					e.printStackTrace(System.err);
				}
			}
		}
	}
}

