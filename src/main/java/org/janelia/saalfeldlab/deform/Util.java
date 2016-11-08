/**
 * License: GPL
 *
 * This program is free software; you can redistribute it and/or
 * modify it under the terms of the GNU General Public License 2
 * as published by the Free Software Foundation.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program; if not, write to the Free Software
 * Foundation, Inc., 59 Temple Place - Suite 330, Boston, MA  02111-1307, USA.
 */
package org.janelia.saalfeldlab.deform;

import java.io.IOException;
import java.util.stream.DoubleStream;

import bdv.bigcat.label.FragmentSegmentAssignment;
import bdv.bigcat.ui.GoldenAngleSaturatedARGBStream;
import bdv.img.h5.H5LabelMultisetSetupImageLoader;
import bdv.img.h5.H5UnsignedByteSetupImageLoader;
import bdv.img.h5.H5Utils;
import bdv.labels.labelset.Label;
import bdv.labels.labelset.LabelMultisetType;
import bdv.util.Bdv;
import bdv.util.BdvFunctions;
import bdv.util.BdvStackSource;
import bdv.util.LocalIdService;
import ch.systemsx.cisd.hdf5.IHDF5Reader;
import mpicbg.spim.data.generic.sequence.ImgLoaderHints;
import net.imglib2.Interval;
import net.imglib2.RandomAccessibleInterval;
import net.imglib2.RealRandomAccessible;
import net.imglib2.converter.Converter;
import net.imglib2.converter.Converters;
import net.imglib2.type.numeric.ARGBType;
import net.imglib2.type.numeric.integer.LongType;
import net.imglib2.type.numeric.integer.UnsignedByteType;
import net.imglib2.view.Views;

/**
 *
 *
 * @author Stephan Saalfeld &lt;saalfelds@janelia.hhmi.org&gt;
 */
public class Util {

	private Util(){}

	/**
	 * Load raw pixels with the correct offset.
	 *
	 * @param reader
	 * @param dataset
	 * @return
	 * @throws IOException
	 */
	final static public RandomAccessibleInterval<UnsignedByteType> loadRaw(
			final IHDF5Reader reader,
			final String dataset,
			final int[] cellDimensions) throws IOException {
		final RandomAccessibleInterval<UnsignedByteType> rawPixels;
		if (reader.exists(dataset)) {
			final H5UnsignedByteSetupImageLoader raw =
					new H5UnsignedByteSetupImageLoader(
							reader,
							dataset,
							0,
							cellDimensions);
			final double[] resolution;
			if (reader.object().hasAttribute(dataset, "offset")) {
				final double[] offset = H5Utils.loadAttribute(reader, dataset, "offset");
				if (reader.object().hasAttribute(dataset, "resolution")) {
					resolution = H5Utils.loadAttribute(reader, dataset, "resolution");
					for (int i = 0; i < offset.length; ++i)
						offset[i] /= resolution[i];
				}
				/* in CREMI, all offsets are integers */
				final long[] longOffset = DoubleStream.of(offset).mapToLong(a -> Math.round(a)).toArray();
				rawPixels = Views.translate(
						raw.getImage(0, ImgLoaderHints.LOAD_COMPLETELY),
						new long[]{
								longOffset[2],
								longOffset[1],
								longOffset[0]
						});
			}
			else
				rawPixels = raw.getImage(0, 0);
		} else {
			System.out.println("no raw pixels found cooresponding to requested dataset '" + dataset + "'");
			rawPixels = null;
		}
		return rawPixels;
	}

	/**
	 * Load labels with the correct offset.
	 *
	 * @param reader
	 * @param dataset
	 * @return
	 * @throws IOException
	 */
	final static public RandomAccessibleInterval<LabelMultisetType> loadLabels(
			final IHDF5Reader reader,
			final String dataset,
			final int[] cellDimensions) throws IOException {
		RandomAccessibleInterval<LabelMultisetType> fragmentsPixels;
		if (reader.exists(dataset)) {
			final H5LabelMultisetSetupImageLoader fragments =
					new H5LabelMultisetSetupImageLoader(
							reader,
							null,
							dataset,
							1,
							cellDimensions);
			final double[] resolution;
			if (reader.object().hasAttribute(dataset, "offset")) {
				try {
					final double[] offset = H5Utils.loadAttribute(reader, dataset, "offset");
					if (reader.object().hasAttribute(dataset, "resolution")) {
						resolution = H5Utils.loadAttribute(reader, dataset, "resolution");
						for (int i = 0; i < offset.length; ++i)
							offset[i] /= resolution[i];
					}
					/* in CREMI, all offsets are integers */
					final long[] longOffset = DoubleStream.of(offset).mapToLong(a -> Math.round(a)).toArray();
					fragmentsPixels = Views.translate(
							fragments.getImage(0, 0),
							new long[]{
									longOffset[2],
									longOffset[1],
									longOffset[0]
							});
				}
				catch (final Exception e) {
					fragmentsPixels = fragments.getImage(0, 0);
				}
			}
			else
				fragmentsPixels = fragments.getImage(0, 0);
		} else {
			System.out.println("no labels found cooresponding to requested dataset '" + dataset + "'");
			fragmentsPixels = null;
		}
		return fragmentsPixels;
	}

	final static public void display(
			final RealRandomAccessible<UnsignedByteType> raw,
			final RealRandomAccessible<LongType> labels,
			final Interval interval) {
		final FragmentSegmentAssignment assignment = new FragmentSegmentAssignment(new LocalIdService());
		final GoldenAngleSaturatedARGBStream argbStream = new GoldenAngleSaturatedARGBStream(assignment);
		final BdvStackSource<UnsignedByteType> source = BdvFunctions.show(raw, interval, "raw", Bdv.options());
		BdvFunctions.show(
				Converters.convert(
						labels,
						new Converter<LongType, ARGBType>() {

							@Override
							public void convert(final LongType input, final ARGBType output) {
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
}
