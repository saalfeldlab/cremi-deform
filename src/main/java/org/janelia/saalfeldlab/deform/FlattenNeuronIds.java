package org.janelia.saalfeldlab.deform;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import com.beust.jcommander.JCommander;
import com.beust.jcommander.Parameter;

import bdv.bigcat.label.FragmentSegmentAssignment;
import bdv.img.h5.H5LabelMultisetSetupImageLoader;
import bdv.img.h5.H5Utils;
import bdv.labels.labelset.Label;
import bdv.labels.labelset.LabelMultisetType;
import bdv.util.IdService;
import bdv.util.LocalIdService;
import ch.systemsx.cisd.hdf5.HDF5Factory;
import ch.systemsx.cisd.hdf5.IHDF5Writer;
import net.imglib2.Cursor;
import net.imglib2.RandomAccessibleInterval;
import net.imglib2.converter.Converter;
import net.imglib2.converter.Converters;
import net.imglib2.img.cell.CellImg;
import net.imglib2.img.cell.CellImgFactory;
import net.imglib2.type.numeric.integer.LongType;
import net.imglib2.util.Pair;
import net.imglib2.view.IntervalView;
import net.imglib2.view.RandomAccessiblePair;
import net.imglib2.view.Views;

public class FlattenNeuronIds
{
	static private class Parameters
	{
		@Parameter( names = { "--infile", "-i" }, description = "Input file path" )
		public String inFile = "";

		@Parameter( names = { "--label", "-l" }, description = "label dataset" )
		public String label = "/volumes/labels/neuron_ids";

		@Parameter( names = { "--slice", "-s" }, description = "slices to be cleared from any labels" )
		public List< Long > badSlices = Arrays.asList( new Long[ 0 ] );

		@Parameter( names = { "--canvas", "-c" }, description = "canvas dataset" )
		public String canvas = "/volumes/labels/painted_neuron_ids";

		@Parameter( names = { "--export", "-e" }, description = "export dataset" )
		public String export = "/volumes/labels/merged_neuron_ids";
	}

	final static private int[] cellDimensions = new int[]{ 64, 64, 8 };

	final private ArrayList< H5LabelMultisetSetupImageLoader > labels = new ArrayList<>();

	private FragmentSegmentAssignment assignment;

	private IdService idService = new LocalIdService();

	/**
	 * Writes max(a,b) into a
	 *
	 * @param a
	 * @param b
	 */
	final static private void max( final long[] a, final long[] b )
	{
		for ( int i = 0; i < a.length; ++i )
			if ( b[ i ] > a[ i ] )
				a[ i ] = b[ i ];
	}

	public static void main( final String[] args ) throws Exception
	{
		final Parameters params = new Parameters();
		new JCommander( params, args );

		System.out.println( "Opening " + params.inFile );
		final IHDF5Writer writer = HDF5Factory.open( params.inFile );

		/* dimensions */
		final long[] maxRawDimensions = new long[]{ 0, 0, 0 };

		/* labels */
		final H5LabelMultisetSetupImageLoader labelLoader;
		if ( writer.exists( params.label ) )
		{
			 labelLoader = new H5LabelMultisetSetupImageLoader(
							writer,
							null,
							params.label,
							1,
							cellDimensions );
			labelLoader.getVolatileImage( 0, 0 ).dimensions( maxRawDimensions );
		}
		else
		{
			System.err.println( "no label dataset '" + params.label + "' found" );
			return;
		}

		/* resolution */
		final double[] resolution;
		if ( writer.object().hasAttribute( params.label, "resolution" ) )
			resolution = writer.float64().getArrayAttr( params.label, "resolution" );
		else
			resolution = new double[] { 1, 1, 1 };

		/* canvas (to which the brush paints) */
		/* TODO this has to change into a virtual container with temporary storage */
		CellImg< LongType, ?, ? > canvas = null;
		if ( writer.exists( params.canvas ) )
			canvas = H5Utils.loadUnsignedLong( writer, params.canvas, cellDimensions );
		else
		{
			canvas = new CellImgFactory< LongType >( cellDimensions ).create( maxRawDimensions, new LongType() );
			for ( final LongType t : canvas )
				t.set( Label.TRANSPARENT );
		}

		/* pair labels */
		final RandomAccessiblePair< LabelMultisetType, LongType > labelCanvasPair =
				new RandomAccessiblePair<>(
						labelLoader.getImage( 0 ),
						canvas );
		final RandomAccessibleInterval< Pair< LabelMultisetType, LongType > > pairInterval =
				Views.offsetInterval( labelCanvasPair, canvas );

		final Converter< Pair< LabelMultisetType, LongType >, LongType > converter =
				new Converter< Pair< LabelMultisetType, LongType >, LongType >()
				{
					@Override
					public void convert(
							final Pair< LabelMultisetType, LongType > input,
							final LongType output )
					{
						final long inputB = input.getB().get();
						if ( inputB == Label.TRANSPARENT )
						{
							output.set( input.getA().entrySet().iterator().next().getElement().id() );
						}
						else
						{
							output.set( inputB );
						}
					}
				};

		final RandomAccessibleInterval< LongType > source =
				Converters.convert(
						pairInterval,
						converter,
						new LongType() );

		System.out.println( "Writing merged labels into canvas..." );

		/* copy merge into canvas */
		final Cursor< LongType > sourceCursor = Views.flatIterable( source ).cursor();
		final Cursor< LongType > targetCursor = Views.flatIterable( canvas ).cursor();

		while ( sourceCursor.hasNext() )
		{
			final LongType sourceType = sourceCursor.next();
			final LongType targetType = targetCursor.next();
			final long sourceValue = sourceType.get();
			targetType.set( sourceValue );
		}

		/* clear bad slices */
		for ( final long z : params.badSlices )
		{
			System.out.println( "Clearing bad slice " + z + "..." );
			final IntervalView< LongType > slice = Views.hyperSlice( canvas, 2, z );
			for ( final LongType t : slice )
				t.set( Label.TRANSPARENT );
		}

		System.out.println( "Saving " + params.export + "..." );

		H5Utils.saveUnsignedLong( canvas, writer, params.export, cellDimensions );
		H5Utils.saveAttribute( resolution, writer, params.export, "resolution" );

		writer.close();

		System.out.println( "Done" );
	}
}
