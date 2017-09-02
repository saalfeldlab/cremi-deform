/**
 *
 */
package org.janelia.saalfeldlab.deform;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;
import java.util.List;

import org.junit.experimental.theories.ParameterSignature;

import com.beust.jcommander.JCommander;
import com.beust.jcommander.Parameter;
import com.google.gson.Gson;

import bdv.bigcat.annotation.Annotation;
import bdv.bigcat.annotation.Annotations;
import bdv.bigcat.annotation.AnnotationsHdf5Store;
import bdv.img.h5.H5Utils;
import bdv.labels.labelset.Label;
import bdv.labels.labelset.LabelMultisetType;
import bdv.util.LocalIdService;
import ch.systemsx.cisd.hdf5.HDF5Factory;
import ch.systemsx.cisd.hdf5.IHDF5Reader;
import ch.systemsx.cisd.hdf5.IHDF5Writer;
import io.scif.img.cell.SCIFIOCellImgCleaningProvider;
import mpicbg.trakem2.transform.CoordinateTransform;
import mpicbg.trakem2.transform.CoordinateTransformList;
import net.imglib2.FinalInterval;
import net.imglib2.Interval;
import net.imglib2.RandomAccessible;
import net.imglib2.RandomAccessibleInterval;
import net.imglib2.RealPoint;
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

		@Parameter(names = { "--infile", "-i" }, required = true, description = "input CREMI-format HDF5 file name")
		public String inFile;

		@Parameter(names = { "--infile_labels", "-j" }, description = "input for labels CREMI-format HDF5 file name (default == --infile")
		public String inFileLabels = null;

		@Parameter(names = { "--infile_annotations", "-k" }, description = "input for annotaions CREMI-format HDF5 file name (default == --infile")
		public String inFileAnnotations = null;

		@Parameter(names = { "--raw", "-r" }, description = "raw datasets, multiple entries possible" )
		public List<String> raws = new ArrayList<>();

		@Parameter(names = { "--label", "-l" }, description = "label dataset, multiple entries possible" )
		public List<String> labels = new ArrayList<>();

		@Parameter(names = { "--outfile", "-o" }, required = true, description = "output CREMI-format HDF5 file name")
		public String outFile;

		@Parameter(names = { "--intransformations", "-t" }, required = true, description = "input JSON export of alignment transformations, formatted as a list of lists")
		public String inTransformations;

		@Parameter(names = { "--meshcellsize", "-c" }, description = "mesh cell size for rendering the output")
		public long meshCellSize = 64;
		
		@Parameter(names = { "--numProcessors" }, description = "number of processors for parallel mapping, default all")
		public int numProcessors = Runtime.getRuntime().availableProcessors();
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
		
		System.out.printf("Opening inputs: \"%s\", \"%s\", \"%s\"...", params.inFile, params.inFileLabels, params.inFileAnnotations);
		System.out.println();
		
		final IHDF5Reader rawReader = HDF5Factory.openForReading(params.inFile);
		final IHDF5Reader labelsReader = params.inFileLabels == null ? rawReader : HDF5Factory.openForReading(params.inFileLabels);
		if (params.inFileAnnotations == null)
			params.inFileAnnotations = params.inFile;
		
		System.out.println("Opening " + params.outFile + " for writing");
		final File outFile = new File(params.outFile);
		final IHDF5Writer writer = HDF5Factory.open(params.outFile);

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

		final FinalInterval sourceInterval =
				new FinalInterval(
						trakem2Export.width,
						trakem2Export.height,
						trakem2Export.transforms.size());
		
		double[] transformResolution = null;
		
		/* raw pixels */
		for (final String rawPath : params.raws) {
			
			final RandomAccessibleInterval<UnsignedByteType> rawSource = Util.loadRaw(rawReader, rawPath, cellDimensions);
			
			double[] resolution = H5Utils.loadAttribute(rawReader, rawPath, "resolution");
			if (transformResolution == null)
				transformResolution = resolution; 
			
			H5Utils.createUnsignedByte(writer, rawPath, sourceInterval, cellDimensions);
			
			/* process in mellSize[2] thick slices to save some memory */
			for (int zOffset = 0; zOffset < sourceInterval.dimension(2); zOffset += cellDimensions[2]) {

				/* deform */
				final RandomAccessibleInterval<UnsignedByteType> rawTarget =
						Views.translate(
								PlanarImgs.unsignedBytes(
										sourceInterval.dimension(0),
										sourceInterval.dimension(1),
										Math.min(sourceInterval.dimension(2), cellDimensions[2])),
								0,
								0,
								zOffset);

				mapSlices(
						Views.extendValue(rawSource, new UnsignedByteType(0)),
						sourceInterval,
						transforms,
						new ClampingNLinearInterpolatorFactory<>(),
						rawTarget,
						params.meshCellSize);

				/* save */
				System.out.println("writing " + rawPath);
				H5Utils.saveUnsignedByte(
						rawTarget,
						outFile,
						rawPath,
						cellDimensions);
			}
			
			if (resolution != null)
				H5Utils.saveAttribute(resolution, writer, rawPath, "resolution");
			H5Utils.saveAttribute(new double[]{0, 0, 0}, writer, rawPath, "offset");
		}

		/* labels */
		for (final String labelsPath : params.labels) {

			final RandomAccessibleInterval<LabelMultisetType> labelsSource = Util.loadLabels(labelsReader, labelsPath, cellDimensions);

			double[] resolution = H5Utils.loadAttribute(labelsReader, labelsPath, "resolution");
			if (transformResolution == null)
				transformResolution = resolution;
			
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
				System.out.println("writing " + labelsPath);
				H5Utils.saveUnsignedLong(
						labelsTarget,
						outFile,
						labelsPath,
						cellDimensions);
			}

			if (resolution != null)
				H5Utils.saveAttribute(resolution, writer, labelsPath, "resolution");
			H5Utils.saveAttribute(new double[]{0, 0, 0}, writer, labelsPath, "offset");
		}
		
		rawReader.close();
		labelsReader.close();
		writer.close();
			
		/* annotations */
		if (transformResolution == null)
			transformResolution = new double[]{1, 1, 1};
		
		final double[] annotationsResolution = transformResolution;
		
		final AnnotationsHdf5Store annotationsStore = new AnnotationsHdf5Store(params.inFileAnnotations, new LocalIdService());
		final Annotations annotations = annotationsStore.read();
		
		final Collection<Annotation> annotationsCollection = annotations.getAnnotations();
		annotationsCollection.forEach(
				a -> {
					final double[] slicePosition = new double[2];
					final RealPoint position = a.getPosition();
					slicePosition[0] = position.getDoublePosition(0) / annotationsResolution[2];
					slicePosition[1] = position.getDoublePosition(1) / annotationsResolution[1];
					final int zIndex = (int)Math.round(position.getDoublePosition(2) / annotationsResolution[0]);
					if (zIndex >= 0 && zIndex < transforms.size()) {
						final CoordinateTransform transform = transforms.get(zIndex);
						transform.applyInPlace(slicePosition);	
					}
					position.setPosition(slicePosition[0] * annotationsResolution[2], 0);
					position.setPosition(slicePosition[1] * annotationsResolution[1], 1);
				});
		
		final AnnotationsHdf5Store annotationsOutStore = new AnnotationsHdf5Store(params.outFile, new LocalIdService());
		annotationsOutStore.write(annotations);
		
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
