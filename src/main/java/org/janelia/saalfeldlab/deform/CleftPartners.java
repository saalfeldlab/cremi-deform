/**
 *
 */
package org.janelia.saalfeldlab.deform;

import java.util.Collection;

import org.janelia.saalfeldlab.n5.N5Reader;
import org.janelia.saalfeldlab.n5.hdf5.N5HDF5Reader;
import org.janelia.saalfeldlab.n5.imglib2.N5Utils;

import com.beust.jcommander.JCommander;
import com.beust.jcommander.Parameter;

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

/**
 * @author Stephan Saalfeld &lt;saalfelds@janelia.hhmi.org&gt;
 *
 */
public class CleftPartners {

	public static class Parameters {

		@Parameter(names = { "--infile", "-i" }, description = "input CREMI-format HDF5 file name (default == --infile")
		public String inFile = null;
	}

	/**
	 * @param args
	 * @throws Exception
	 */
	public static void main(final String... args) throws Exception {

		final Parameters params = new Parameters();
		new JCommander(params, args);

		System.out.println("Opening input " + params.inFile);

		final N5Reader labelsReader = new N5HDF5Reader(params.inFile);
		final RandomAccessibleInterval<UnsignedLongType> labelsSource = N5Utils.open(labelsReader, "/volumes/labels/neuron_ids");
		final RandomAccessibleInterval<UnsignedLongType> cleftsSource = N5Utils.open(labelsReader, "/volumes/labels/clefts");
		final double[] resolution = labelsReader.getAttribute("/volumes/labels/neuron_ids", "resolution", double[].class);
		final Scale3D scale = new Scale3D(resolution[2], resolution[1], resolution[0]);
		final AffineRealRandomAccessible<UnsignedLongType, AffineGet> labelsSourceScaled = RealViews.affineReal(Views.interpolate(Views.extendValue(labelsSource, new UnsignedLongType(Label.OUTSIDE)), new NearestNeighborInterpolatorFactory<>()), scale);
		final AffineRealRandomAccessible<UnsignedLongType, AffineGet> cleftsSourceScaled = RealViews.affineReal(Views.interpolate(Views.extendValue(cleftsSource, new UnsignedLongType(Label.OUTSIDE)), new NearestNeighborInterpolatorFactory<>()), scale);

		/* annotations */
		final AnnotationsHdf5Store annotationsStore = new AnnotationsHdf5Store(params.inFile, new LocalIdService());
		final Annotations annotations = annotationsStore.read();

		final Collection<Annotation> annotationsCollection = annotations.getAnnotations();
		final AffineRealRandomAccessible<UnsignedLongType, AffineGet>.AffineRealRandomAccess labelsAccess = labelsSourceScaled.realRandomAccess();
		final AffineRealRandomAccessible<UnsignedLongType, AffineGet>.AffineRealRandomAccess cleftsAccess = cleftsSourceScaled.realRandomAccess();
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
						System.out.printf("%6d -> %6d via %6s", preLabel, postLabel, (cleftLabel == Label.TRANSPARENT ? "None" : cleftLabel));
						System.out.println();
					}
				});

		System.out.println("Done");
	}
}
