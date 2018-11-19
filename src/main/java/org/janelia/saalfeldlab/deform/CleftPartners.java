/**
 *
 */
package org.janelia.saalfeldlab.deform;

import java.util.Collection;

import org.janelia.saalfeldlab.n5.N5Reader;
import org.janelia.saalfeldlab.n5.hdf5.N5HDF5Reader;
import org.janelia.saalfeldlab.n5.imglib2.N5Utils;

import bdv.bigcat.annotation.Annotation;
import bdv.bigcat.annotation.Annotations;
import bdv.bigcat.annotation.AnnotationsHdf5Store;
import bdv.bigcat.annotation.PostSynapticSite;
import bdv.bigcat.annotation.PreSynapticSite;
import bdv.labels.labelset.Label;
import bdv.util.LocalIdService;
import net.imglib2.RandomAccessibleInterval;
import net.imglib2.RealPoint;
import net.imglib2.interpolation.randomaccess.NearestNeighborInterpolatorFactory;
import net.imglib2.realtransform.AffineGet;
import net.imglib2.realtransform.AffineRealRandomAccessible;
import net.imglib2.realtransform.RealViews;
import net.imglib2.realtransform.Scale3D;
import net.imglib2.type.numeric.integer.UnsignedLongType;
import net.imglib2.util.Util;
import net.imglib2.view.Views;
import picocli.CommandLine;
import picocli.CommandLine.Option;

/**
 * @author Stephan Saalfeld &lt;saalfelds@janelia.hhmi.org&gt;
 *
 */
public class CleftPartners {

	public static class Parameters {

		@Option(names = { "--infile", "-i" }, description = "input CREMI-format HDF5 file name (default == --infile")
		public String inFile = null;
	}

	/**
	 * @param args
	 * @throws Exception
	 */
	public static void main(final String... args) throws Exception {

//		new ImageJ();

		final Parameters params = new Parameters();
		try {
			CommandLine.populateCommand(params, args);
		} catch (final RuntimeException e) {
			CommandLine.usage(params, System.err);
			return;
		}

//		System.out.println("Opening input " + params.inFile);

		final N5Reader labelsReader = new N5HDF5Reader(params.inFile);
		RandomAccessibleInterval<UnsignedLongType> labelsSource = N5Utils.open(labelsReader, "/volumes/labels/neuron_ids");
		RandomAccessibleInterval<UnsignedLongType> cleftsSource = N5Utils.open(labelsReader, "/volumes/labels/clefts");
		final double[] labelsResolution = labelsReader.getAttribute("/volumes/labels/neuron_ids", "resolution", double[].class);
		final Scale3D labelsScale = new Scale3D(labelsResolution[2], labelsResolution[1], labelsResolution[0]);
		final double[] labelsOffset = labelsReader.getAttribute("/volumes/labels/neuron_ids", "offset", double[].class);
		if (labelsOffset != null) {
			labelsSource = Views.translate(
					labelsSource,
					(long)Math.round(labelsOffset[2] / labelsResolution[2]) - labelsSource.min(0),
					(long)Math.round(labelsOffset[1] / labelsResolution[1]) - labelsSource.min(1),
					(long)Math.round(labelsOffset[0] / labelsResolution[0]) - labelsSource.min(2));
		}
		final double[] cleftsResolution = labelsReader.getAttribute("/volumes/labels/clefts", "resolution", double[].class);
		final Scale3D cleftsScale = new Scale3D(cleftsResolution[2], cleftsResolution[1], cleftsResolution[0]);
		final double[] cleftsOffset = labelsReader.getAttribute("/volumes/labels/clefts", "offset", double[].class);
		if (cleftsOffset != null) {
			cleftsSource = Views.translate(
					cleftsSource,
					(long)Math.round(cleftsOffset[2] / cleftsResolution[2]) - cleftsSource.min(0),
					(long)Math.round(cleftsOffset[1] / cleftsResolution[1]) - cleftsSource.min(1),
					(long)Math.round(cleftsOffset[0] / cleftsResolution[0]) - cleftsSource.min(2));
		}

		final AffineRealRandomAccessible<UnsignedLongType, AffineGet> labelsSourceScaled = RealViews.affineReal(Views.interpolate(Views.extendValue(labelsSource, new UnsignedLongType(Label.OUTSIDE)), new NearestNeighborInterpolatorFactory<>()), labelsScale);
		final AffineRealRandomAccessible<UnsignedLongType, AffineGet> cleftsSourceScaled = RealViews.affineReal(Views.interpolate(Views.extendValue(cleftsSource, new UnsignedLongType(Label.OUTSIDE)), new NearestNeighborInterpolatorFactory<>()), cleftsScale);

		/* annotations */
		final AnnotationsHdf5Store annotationsStore = new AnnotationsHdf5Store(params.inFile, new LocalIdService());
		final Annotations annotations = annotationsStore.read();

		final Collection<Annotation> annotationsCollection = annotations.getAnnotations();
		final AffineRealRandomAccessible<UnsignedLongType, AffineGet>.AffineRealRandomAccess labelsAccess = labelsSourceScaled.realRandomAccess();
		final AffineRealRandomAccessible<UnsignedLongType, AffineGet>.AffineRealRandomAccess cleftsAccess = cleftsSourceScaled.realRandomAccess();

		System.out.println("pre_label, pre_id, pre_x, pre_y, pre_z, post_label, post_id, post_x, post_y, post_z, cleft");

		annotationsCollection.forEach(
				a -> {
					if (a instanceof PostSynapticSite) {
						final PreSynapticSite b = ((PostSynapticSite)a).getPartner();
						final RealPoint start = b.getPosition();
						labelsAccess.setPosition(start);
						final long preLabel = labelsAccess.get().get();
						final RealPoint end = a.getPosition();
						labelsAccess.setPosition(end);
						final long postLabel = labelsAccess.get().get();

						final RealPoint d = new RealPoint(
								end.getDoublePosition(0) - start.getDoublePosition(0),
								end.getDoublePosition(1) - start.getDoublePosition(1),
								end.getDoublePosition(2) - start.getDoublePosition(2));
						final double distance = Util.distance(start, end);

						long cleftLabel = Label.TRANSPARENT;
						cleftsAccess.setPosition(start);
						for (double i = 0; i < distance; i += 0.5) {
							final double step = i / distance;
							cleftsAccess.setPosition(start.getDoublePosition(0) + step * d.getDoublePosition(0), 0);
							cleftsAccess.setPosition(start.getDoublePosition(1) + step * d.getDoublePosition(1), 1);
							cleftsAccess.setPosition(start.getDoublePosition(2) + step * d.getDoublePosition(2), 2);
							cleftLabel = cleftsAccess.get().get();
							if (cleftLabel != Label.TRANSPARENT)
								break;
						}
						System.out.println(String.format(
								"%d, %d, %.2f, %.2f, %.2f, %d, %d, %.2f, %.2f, %.2f, %s",
								preLabel,
								b.getId(),
								start.getDoublePosition(0),
								start.getDoublePosition(1),
								start.getDoublePosition(2),
								postLabel,
								a.getId(),
								end.getDoublePosition(0),
								end.getDoublePosition(1),
								end.getDoublePosition(2),
								(cleftLabel == Label.TRANSPARENT ? "-1" : cleftLabel)));
					}
				});
	}
}
