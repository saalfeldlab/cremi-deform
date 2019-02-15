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
package org.saalfeldlab.dsb;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.Iterator;

import org.janelia.saalfeldlab.n5.GzipCompression;
import org.janelia.saalfeldlab.n5.hdf5.N5HDF5Writer;
import org.janelia.saalfeldlab.n5.imglib2.N5Utils;

import ch.systemsx.cisd.hdf5.HDF5Factory;
import ch.systemsx.cisd.hdf5.IHDF5Writer;
import ij.IJ;
import ij.ImagePlus;
import net.imglib2.Cursor;
import net.imglib2.converter.Converters;
import net.imglib2.img.array.ArrayImg;
import net.imglib2.img.array.ArrayImgs;
import net.imglib2.img.basictypeaccess.array.IntArray;
import net.imglib2.img.basictypeaccess.array.LongArray;
import net.imglib2.img.imageplus.ImagePlusImg;
import net.imglib2.img.imageplus.ImagePlusImgs;
import net.imglib2.type.numeric.ARGBType;
import net.imglib2.type.numeric.IntegerType;
import net.imglib2.type.numeric.integer.UnsignedLongType;

/**
 * Convert Kaggle training or test data into hdf5
 *
 * @author Stephan Saalfeld &lt;saalfelds@janelia.hhmi.org&gt;
 */
public class DSBConvert {

	/**
	 * @param args
	 * @throws IOException
	 */
	public static void main(final String[] args) throws IOException {

		final String trainingPath = "/groups/saalfeld/saalfeldlab/kaggle-nuclei/stage1_train";
//		final String trainingPath = "/groups/saalfeld/saalfeldlab/kaggle-nuclei/stage1_test";
//		final String trainingPath = "/groups/saalfeld/saalfeldlab/kaggle-nuclei/test";
//		final String hdf5Path = "/groups/saalfeld/saalfeldlab/kaggle-nuclei/stage1_test.hdf5";
		final String hdf5Path = "/groups/saalfeld/saalfeldlab/kaggle-nuclei/stage1_train.rgb.hdf5";

		final IHDF5Writer hdf5Writer = HDF5Factory.open(hdf5Path);
		final N5HDF5Writer n5 = new N5HDF5Writer(hdf5Writer, -1, -1, -1);

		Files
		.list(Paths.get(trainingPath))
		.filter(path -> path.toFile().isDirectory())
		.forEach(path -> {

			final String groupName = path.getFileName().toString();
			final Iterator<Path> iter;
			final ArrayImg<UnsignedLongType, LongArray> rai;
			try {
				n5.createGroup(groupName);
				final ImagePlus imp = IJ.openImage(Files.list(Paths.get(path.toAbsolutePath().toString(), "images")).iterator().next().toAbsolutePath().toString());
				System.out.println(groupName);
				final ImagePlusImg<ARGBType, IntArray> rgbImg = (ImagePlusImg<ARGBType, IntArray>)(ImagePlusImg)ImagePlusImgs.from(imp);
				N5Utils.save(Converters.argbChannel(rgbImg, 1), n5, groupName + "/image-r", new int[]{2048, 2048}, new GzipCompression());
				N5Utils.save(Converters.argbChannel(rgbImg, 2), n5, groupName + "/image-g", new int[]{2048, 2048}, new GzipCompression());
				N5Utils.save(Converters.argbChannel(rgbImg, 3), n5, groupName + "/image-b", new int[]{2048, 2048}, new GzipCompression());
//				final RandomAccessibleInterval<UnsignedByteType> rgbStack = Views.stack(
//						Converters.argbChannel(rgbImg, 1),
//						Converters.argbChannel(rgbImg, 2),
//						Converters.argbChannel(rgbImg, 3));
//				N5Utils.save(rgbStack, n5, groupName + "/image", new int[]{2048, 2048, 3}, new GzipCompression());
				final long[] targetValues = new long[imp.getWidth() * imp.getHeight()];
//				Arrays.fill(targetValues, Label.TRANSPARENT);
				rai = ArrayImgs.unsignedLongs(targetValues, imp.getWidth(), imp.getHeight());
				iter = Files.list(Paths.get(path.toAbsolutePath().toString(), "masks")).iterator();
			} catch (final IOException e) {
				e.printStackTrace();
				return;
			}
			for (long id = 1; iter.hasNext(); ++id) {
				final ImagePlus imp = IJ.openImage(iter.next().toAbsolutePath().toString());
				final ImagePlusImg img = ImagePlusImgs.from(imp);
				final Cursor<IntegerType> sourceCursor = (Cursor<IntegerType>)(Cursor)img.cursor();
				final Cursor<UnsignedLongType> targetCursor = rai.cursor();
				while (sourceCursor.hasNext())
					if (sourceCursor.next().getInteger() == 0)
						targetCursor.fwd();
					else
						targetCursor.next().set(id);
			}
			try {
				N5Utils.save(rai, n5, groupName + "/mask", new int[]{2048, 2048}, new GzipCompression());
			} catch (final IOException e) {
				e.printStackTrace();
			}
		});
		hdf5Writer.close();

	}

}
