/**
 *
 */
package org.janelia.saalfeldlab.deform;

import mpicbg.trakem2.transform.CoordinateTransform;
import net.imglib2.RealLocalizable;
import net.imglib2.RealPositionable;
import net.imglib2.realtransform.RealTransform;

/**
 * @author Stephan Saalfeld &lt;saalfelds@janelia.hhmi.org&gt;
 *
 */
/**
 * Wraps an mpicbg CoordinateTransform as an ImgLib2 RealTransform
 *
 * @author Stephan Saalfeld &lt;saalfelds@janelia.hhmi.org&gt;
 */
public class CoordinateTransformRealTransform implements RealTransform {

	final protected CoordinateTransform ct;
	final protected int numDimensions;
	final double[] point;

	public CoordinateTransformRealTransform(final CoordinateTransform ct, final int numDimensions) {
		this.ct = ct;
		this.numDimensions = numDimensions;
		point = new double[numDimensions];
	}

	@Override
	public int numSourceDimensions() {

		return numDimensions;
	}

	@Override
	public int numTargetDimensions() {

		return numDimensions;
	}

	@Override
	public void apply(final double[] source, final double[] target) {

		if (source != target)
			System.arraycopy(source, 0, target, 0, numDimensions);

		ct.applyInPlace(target);
	}

	@Override
	public void apply(final float[] source, final float[] target) {

		for (int d = 0; d < numDimensions; ++d)
			point[d] = source[d];

		ct.applyInPlace(point);

		for (int d = 0; d < numDimensions; ++d)
			target[d] = (float)point[d];
	}

	@Override
	public void apply(final RealLocalizable source, final RealPositionable target) {

		source.localize(point);;
		ct.applyInPlace(point);
		target.setPosition(point);
	}

	@Override
	public CoordinateTransformRealTransform copy() {

		return new CoordinateTransformRealTransform(ct, numDimensions);
	}
}