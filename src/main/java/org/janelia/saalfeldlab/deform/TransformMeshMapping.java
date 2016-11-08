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

import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.atomic.AtomicInteger;

import mpicbg.models.AffineModel2D;
import mpicbg.util.Util;
import net.imglib2.FinalInterval;
import net.imglib2.RandomAccess;
import net.imglib2.RandomAccessibleInterval;
import net.imglib2.RealRandomAccess;
import net.imglib2.RealRandomAccessible;
import net.imglib2.realtransform.AffineTransform2D;
import net.imglib2.type.Type;
import net.imglib2.util.Pair;

/**
 * Naive wrapper around mpicbg.
 *
 * TODO replace with ImgLib2 based mesh and rendering
 * mechanism in next version.
 *
 * @param <T>
 *
 * @author Stephan Saalfeld &lt;saalfelds@janelia.hhmi.org&gt;
 */
public class TransformMeshMapping<T extends Type<T>> {

	final protected TransformMesh mesh;

	public TransformMeshMapping(final TransformMesh mesh) {
		this.mesh = mesh;
	}

	abstract private class AbstractMapTriangleThread extends Thread {
		final protected AtomicInteger i;
		final protected List<Pair<AffineModel2D, double[][]>> triangles;
		final protected RealRandomAccessible<T> source;
		final protected RandomAccessibleInterval<T> target;

		AbstractMapTriangleThread(
				final AtomicInteger i,
				final List<Pair<AffineModel2D, double[][]>> triangles,
				final RealRandomAccessible<T> source,
				final RandomAccessibleInterval<T> target) {

			this.i = i;
			this.triangles = triangles;
			this.source = source;
			this.target = target;
		}
	}

	final private class MapTriangleThread extends AbstractMapTriangleThread {

		MapTriangleThread(
				final AtomicInteger i,
				final List<Pair<AffineModel2D, double[][]>> triangles,
				final RealRandomAccessible<T> source,
				final RandomAccessibleInterval<T> target) {

			super(i, triangles, source, target);
		}

		@Override
		final public void run() {
			int k = i.getAndIncrement();
			while (!isInterrupted() && k < triangles.size()) {
				mapTriangle(triangles.get(k), source, target);
				k = i.getAndIncrement();
			}
		}
	}

	final private class MapInverseTriangleThread extends AbstractMapTriangleThread {

		MapInverseTriangleThread(
				final AtomicInteger i,
				final List<Pair<AffineModel2D, double[][]>> triangles,
				final RealRandomAccessible<T> source,
				final RandomAccessibleInterval<T> target) {

			super(i, triangles, source, target);
		}

		@Override
		final public void run() {
			int k = i.getAndIncrement();
			while (!isInterrupted() && k < triangles.size()) {
				mapInverseTriangle(triangles.get(k), source, target);
				k = i.getAndIncrement();
			}
		}
	}

	final static protected <T extends Type<T>> void mapTriangle(
			final Pair<AffineModel2D, double[][]> ai,
			final RealRandomAccessible<T> source,
			final RandomAccessibleInterval<T> target) {

		final long w = target.max(0);
		final long h = target.max(1);

		final AffineModel2D a = ai.getA();
		final AffineTransform2D affine = new AffineTransform2D();
		final double[] coefficients = new double[6];
		a.toArray(coefficients);
		affine.set(
				coefficients[0], coefficients[2], coefficients[4],
				coefficients[1], coefficients[3], coefficients[5]);
		final double[][] pq = ai.getB();

		final double[] min = new double[2];
		final double[] max = new double[2];
		TransformMesh.calculateTargetBoundingBox(pq, min, max);

		final long minX = Math.max(0, Util.roundPos(min[0]));
		final long minY = Math.max(0, Util.roundPos(min[1]));
		final long maxX = Math.min(w, Util.roundPos(max[0]));
		final long maxY = Math.min(h, Util.roundPos(max[1]));

		final RealRandomAccess<T> sourceAccess = source.realRandomAccess();
		final RandomAccess<T> targetAccess = target.randomAccess(new FinalInterval(new long[] { minX, minY }, new long[] { maxX, maxY }));
		for (long y = minY; y <= maxY; ++y) {
			targetAccess.setPosition(y, 1);
			for (long x = minX; x <= maxX; ++x) {
				targetAccess.setPosition(x, 0);
				if (TransformMesh.isInTargetTriangle(pq, x, y)) {
					try {
						affine.applyInverse(sourceAccess, targetAccess);
					} catch (final Exception e) {
						e.printStackTrace(System.err);
						continue;
					}

					targetAccess.get().set(sourceAccess.get());
				}
			}
		}
	}

	final public void map(final RealRandomAccessible<T> source, final RandomAccessibleInterval<T> target, final int numThreads) {
		final ArrayList<Pair<AffineModel2D, double[][]>> av = mesh.getAV();
		if (numThreads > 1) {
			final AtomicInteger i = new AtomicInteger(0);
			final ArrayList<Thread> threads = new ArrayList<Thread>(numThreads);
			for (int k = 0; k < numThreads; ++k) {
				final Thread mtt = new MapTriangleThread(i, av, source, target);
				threads.add(mtt);
				mtt.start();
			}
			for (final Thread mtt : threads) {
				try {
					mtt.join();
				} catch (final InterruptedException e) {}
			}
		} else {
			for (final Pair<AffineModel2D, double[][]> triangle : av) {
				mapTriangle(triangle, source, target);
			}
		}
	}

	final static protected <T extends Type<T>> void mapInverseTriangle(
			final Pair<AffineModel2D, double[][]> ai,
			final RealRandomAccessible<T> source,
			final RandomAccessibleInterval<T> target) {

		final AffineModel2D a = ai.getA();
		final AffineTransform2D affine = new AffineTransform2D();
		final double[] coefficients = new double[6];
		a.toArray(coefficients);
		affine.set(
				coefficients[0], coefficients[2], coefficients[4],
				coefficients[1], coefficients[3], coefficients[5]);
		final double[][] pq = ai.getB();

		final double[] min = new double[2];
		final double[] max = new double[2];
		TransformMesh.calculateSourceBoundingBox(pq, min, max);

		final long minX = Math.max(target.min(0), Util.roundPos(min[0]));
		final long minY = Math.max(target.min(1), Util.roundPos(min[1]));
		final long maxX = Math.min(target.max(0), Util.roundPos(max[0]));
		final long maxY = Math.min(target.max(1), Util.roundPos(max[1]));

		final RealRandomAccess<T> sourceAccess = source.realRandomAccess();
		final RandomAccess<T> targetAccess = target.randomAccess(new FinalInterval(new long[] { minX, minY }, new long[] { maxX, maxY }));
		for (long y = minY; y <= maxY; ++y) {
			targetAccess.setPosition(y, 1);
			for (long x = minX; x <= maxX; ++x) {
				targetAccess.setPosition(x, 0);
				if (TransformMesh.isInSourceTriangle(pq, x, y)) {
					try {
						affine.apply(targetAccess, sourceAccess);
					} catch (final Exception e) {
						e.printStackTrace(System.err);
						continue;
					}

					targetAccess.get().set(sourceAccess.get());
				}
			}
		}
	}

	final public void mapInverse(final RealRandomAccessible<T> source, final RandomAccessibleInterval<T> target, final int numThreads) {
		final ArrayList<Pair<AffineModel2D, double[][]>> av = mesh.getAV();
		if (numThreads > 1) {
			final AtomicInteger i = new AtomicInteger(0);
			final ArrayList<Thread> threads = new ArrayList<Thread>(numThreads);
			for (int k = 0; k < numThreads; ++k) {
				final Thread mtt = new MapInverseTriangleThread(i, av, source, target);
				threads.add(mtt);
				mtt.start();
			}
			for (final Thread mtt : threads) {
				try {
					mtt.join();
				} catch (final InterruptedException e) {}
			}
		} else {
			for (final Pair<AffineModel2D, double[][]> triangle : av) {
				mapTriangle(triangle, source, target);
			}
		}
	}
}
