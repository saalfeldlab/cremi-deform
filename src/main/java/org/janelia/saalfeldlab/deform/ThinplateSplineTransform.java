/**
 * 
 */
package org.janelia.saalfeldlab.deform;

import jitk.spline.ThinPlateR2LogRSplineKernelTransform;
import net.imglib2.RealLocalizable;
import net.imglib2.RealPositionable;
import net.imglib2.realtransform.RealTransform;

/**
 * @author Stephan Saalfeld &lt;saalfelds@janelia.hhmi.org&gt;
 *
 */
public class ThinplateSplineTransform implements RealTransform {

	final protected ThinPlateR2LogRSplineKernelTransform tps;
	final protected double[] a;
	final protected double[] b;

	final static private ThinPlateR2LogRSplineKernelTransform init(
			final double[][] p,
			final double[][] q) {

		assert p.length == q.length;

		final ThinPlateR2LogRSplineKernelTransform tps =
				new ThinPlateR2LogRSplineKernelTransform(p.length, p, q);

		return tps;
	}
	
	public ThinplateSplineTransform(final ThinPlateR2LogRSplineKernelTransform tps) {

		this.tps = tps;
		a = new double[tps.getNumDims()];
		b = new double[a.length];
	}

	public ThinplateSplineTransform(
			final double[][] p,
			final double[][] q) {

		this(init(p, q));
	}
		
	
	@Override
	public void apply(double[] source, double[] target) {

		tps.apply(source, target);
	}

	@Override
	public void apply(float[] source, float[] target) {

		for (int d = 0; d < a.length; ++d)
			a[d] = source[d];

		tps.apply(a, b);

		for (int d = 0; d < target.length; ++d)
			target[d] = (float)b[d];
	}

	@Override
	public void apply(RealLocalizable source, RealPositionable target) {

		source.localize(a);
		tps.apply(a, b);
		target.setPosition(b);
	}

	@Override
	public ThinplateSplineTransform copy() {

		/* tps is stateless and constant and can therefore be reused */
		return new ThinplateSplineTransform(tps);
	}

	@Override
	public int numSourceDimensions() {

		return tps.getNumDims();
	}

	@Override
	public int numTargetDimensions() {

		return tps.getNumDims();
	}
}
