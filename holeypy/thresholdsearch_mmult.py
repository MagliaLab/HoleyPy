# -*- coding: utf-8 -*-
import numpy as np

#TODO: I'm not sure where to put this..
def thresholdsearch_mmult( signal, sampling_period, L0, l1, dwelltime=0, skip=2, trace=0, t0=0, t1=-1 ):
    try:
        signal = signal[ trace ][ t0:t1 ]
        index = []
        level_index = np.array([])
        L1_events = np.array([])
        L1_start_events = np.array([])
        L1_end_events = np.array([])
        for c, l0 in enumerate( L0 ):
            index = []
            # While dwelltime suggests that also spikes can be seen, atleast 2 datapoints are required to be an event
            n_filter = dwelltime / float( sampling_period ) if dwelltime / float( sampling_period ) > 2 else 2
            
            a = np.where( ( abs( np.array( signal ) ) > ( abs( l0 ) + abs( l1 ) ) ) )[0]
            a = a[np.in1d(np.array(a), np.array( index ), invert=True)]
            a_end_idx = np.where( np.diff( a ) > skip )[0]
            a_end_idx = np.append( a_end_idx, len(a)-1 )
            a_start_idx = np.delete( np.insert(a_end_idx+1, 0, 0), -1, 0 )
            
                        
            # Only keep those events that are atleast 2 or n_filter data points long
            idx = np.where( a_end_idx - a_start_idx > n_filter )[0]
            L1_start, L1_end = a[ a_start_idx[ idx ] ], a[ a_end_idx[ idx ] ] 
            L1 = np.array( [ signal[i:j] for i, j in zip( L1_start, L1_end ) ] )
            
            level_index = np.concatenate( ( level_index, np.array( len(L1_start)*[c] ) ) )
            L1_events = np.concatenate( ( L1_events, L1 ) )
            L1_start_events = np.concatenate( ( L1_start_events, L1_start ) )
            L1_end_events = np.concatenate( ( L1_end_events, L1_end ) )
            
            for start, end in zip( L1_start, L1_end ):
                for i in range( int( end-start ) ):
                    index.append( int( start + i ) )
        resort_index = L1_start_events.argsort()
        level_index = level_index[resort_index]
        L1_events = L1_events[resort_index]
        L1_start_events = L1_start_events[resort_index]
        L1_end_events = L1_end_events[resort_index]
        L0_start, L0_end = np.delete( np.insert( L1_end+1, 0, 0 ), -1, 0 ), L1_start                          # Set L0 relative to L1
        L0_events = np.array( [ signal[i:j] for i, j in zip( L0_start, L0_end ) ] )
        return ( L0_events, L1_events, L0_start, L0_end, L1_start, L1_end, level_index )
    except Exception as e:
        raise e