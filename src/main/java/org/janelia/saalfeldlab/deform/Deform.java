/**
 * 
 */
package org.janelia.saalfeldlab.deform;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Random;

import bdv.bigcat.label.FragmentSegmentAssignment;
import bdv.bigcat.ui.GoldenAngleSaturatedARGBStream;
import bdv.img.h5.H5LabelMultisetSetupImageLoader;
import bdv.img.h5.H5UnsignedByteSetupImageLoader;
import bdv.labels.labelset.Label;
import bdv.labels.labelset.LabelMultisetType;
import bdv.util.Bdv;
import bdv.util.BdvFunctions;
import bdv.util.BdvStackSource;
import bdv.util.LocalIdService;
import ch.systemsx.cisd.hdf5.HDF5Factory;
import ch.systemsx.cisd.hdf5.IHDF5Reader;
import mpicbg.spim.data.generic.sequence.ImgLoaderHints;
import net.imglib2.FinalInterval;
import net.imglib2.Interval;
import net.imglib2.RandomAccessible;
import net.imglib2.RandomAccessibleInterval;
import net.imglib2.RealRandomAccessible;
import net.imglib2.converter.Converter;
import net.imglib2.converter.Converters;
import net.imglib2.interpolation.InterpolatorFactory;
import net.imglib2.interpolation.randomaccess.ClampingNLinearInterpolatorFactory;
import net.imglib2.interpolation.randomaccess.NearestNeighborInterpolatorFactory;
import net.imglib2.realtransform.RealTransform;
import net.imglib2.realtransform.RealTransformRandomAccessible;
import net.imglib2.realtransform.RealViews;
import net.imglib2.realtransform.Scale3D;
import net.imglib2.type.numeric.ARGBType;
import net.imglib2.type.numeric.integer.LongType;
import net.imglib2.type.numeric.integer.UnsignedByteType;
import net.imglib2.view.Views;

/**
 * @author Stephan Saalfeld &lt;saalfelds@janelia.hhmi.org&gt;
 *
 */
public class Deform {
	final static private int[] cellDimensions = new int[] { 64, 64, 8 };

	final static private RandomAccessibleInterval<LabelMultisetType> loadLabels(final IHDF5Reader reader,
			final String dataset) throws IOException {
		final RandomAccessibleInterval<LabelMultisetType> fragmentsPixels;
		if (reader.exists(dataset)) {
			final H5LabelMultisetSetupImageLoader fragments = new H5LabelMultisetSetupImageLoader(reader, null, dataset,
					1, cellDimensions);
			fragmentsPixels = fragments.getImage(0, 0);
		} else {
			System.out.println("no labels found cooresponding to requested dataset '" + dataset + "'");
			fragmentsPixels = null;
		}
		return fragmentsPixels;
	}

	final static public void display(
			final RealRandomAccessible< UnsignedByteType > raw,
			final RealRandomAccessible< LongType > labels,
			final Interval interval) {
		final FragmentSegmentAssignment assignment = new FragmentSegmentAssignment(new LocalIdService());
		final GoldenAngleSaturatedARGBStream argbStream = new GoldenAngleSaturatedARGBStream(assignment);
		BdvStackSource< UnsignedByteType > source = BdvFunctions.show(raw, interval, "raw", Bdv.options());
		BdvFunctions.show(
				Converters.convert(
						labels,
						new Converter<LongType, ARGBType>(){

							@Override
							public void convert(LongType input, ARGBType output) {
								final long id = input.get();
								if (id == Label.TRANSPARENT || id == Label.INVALID)
									output.set(0);
								else {
									final int argb = argbStream.argb(input.get());
									final int r = ((argb >> 16) & 0xff) / 4;
									final int g = ((argb >> 8) & 0xff) / 4;
									final int b = (argb & 0xff) / 4;
									
									output.set(((((r << 8) | g) << 8) | b) | 0xff000000);
								}
							}	
						},
						new ARGBType()),
				interval, "labels", Bdv.options().addTo(source.getBdvHandle()));
	}
	
	/**
	 * Creates the inverse thin plate spline transform for jittered points on a grid.
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
				p.add(new double[]{x, y});
				q.add(new double[]{
						x + jitterRadius * (2 * rnd.nextDouble() - 1),
						y + jitterRadius * (2 * rnd.nextDouble() - 1)});
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
	
	static public ArrayList<ThinplateSplineTransform> make2DSectionJitterTransforms(
			final Interval interval,
			final double controlPointSpacing,
			final double jitterRadius) {
		
		final Random rnd = new Random();

		final ArrayList<ThinplateSplineTransform> sliceTransforms = new ArrayList<>();

		for (int z = 0; z < interval.dimension(2); ++z)
			sliceTransforms.add(
					make2DSectionJitterTransform(
							rnd,
							interval,
							controlPointSpacing,
							jitterRadius));

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
	 * @param args
	 * @throws IOException
	 */
	public static void main(String[] args) throws IOException {
		final String projectFile = args[0];
		final double controlPointSpacing = 512;
		final double jitterRadius = 32;
		//final String outFile = args[1];

		String labelsDataset = "neuron_ids";
		if (args.length > 1)
			labelsDataset = args[1];

		String rawDataset = "raw";
		if (args.length > 2)
			rawDataset = args[2];

		System.out.println("Opening " + projectFile);
		final IHDF5Reader reader = HDF5Factory.open(projectFile);

		// support both file_format 0.0 and >=0.1
		final String volumesPath = reader.isGroup("/volumes") ? "/volumes" : "";
		final String labelsPath = reader.isGroup(volumesPath + "/labels") ? volumesPath + "/labels" : "";

		/* raw pixels */
		final String rawPath = volumesPath + "/" + rawDataset;
		final H5UnsignedByteSetupImageLoader raw = new H5UnsignedByteSetupImageLoader(reader, rawPath, 0,
				cellDimensions);
		final RandomAccessibleInterval<UnsignedByteType> rawPixels = raw.getImage(0, ImgLoaderHints.LOAD_COMPLETELY);

		/* labels */
		final RandomAccessibleInterval<LabelMultisetType> labels = loadLabels(reader, labelsPath + "/" + labelsDataset);

		final RandomAccessibleInterval<LongType> longLabels = Converters.convert(labels,
				new Converter<LabelMultisetType, LongType>() {
					@Override
					public void convert(LabelMultisetType a, LongType b) {
						b.set(a.entrySet().iterator().next().getElement().id());
					}
				}, new LongType());

		/* deform */
		final ArrayList<ThinplateSplineTransform> jitterTransforms = make2DSectionJitterTransforms(
				rawPixels,
				controlPointSpacing,
				jitterRadius);
		
		final RandomAccessibleInterval<UnsignedByteType> deformedRawPixels = jitterSlices(
				Views.extendValue(rawPixels, new UnsignedByteType(0)),
				rawPixels,
				jitterTransforms,
				new ClampingNLinearInterpolatorFactory<UnsignedByteType>());
		
		final RandomAccessibleInterval<LongType> deformedLongLabels = jitterSlices(
				Views.extendValue(longLabels, new LongType(Label.TRANSPARENT)),
				longLabels,
				jitterTransforms,
				new NearestNeighborInterpolatorFactory<>());

		display(
				RealViews.affine(
						Views.interpolate(
								Views.extendZero(deformedRawPixels),
								new NearestNeighborInterpolatorFactory<>()),
						new Scale3D(1, 1, 10)),
				RealViews.affine(
						Views.interpolate(
								Views.extendZero(deformedLongLabels),
								new NearestNeighborInterpolatorFactory<>()),
						new Scale3D(1, 1, 10)),
				new FinalInterval(
						rawPixels.dimension(0),
						rawPixels.dimension(1),
						rawPixels.dimension(2) * 10));

	}
}
