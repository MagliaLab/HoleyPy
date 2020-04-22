# -*- coding: utf-8 -*-
import numpy as np

def join_traces( signal, sampling_period, t0=0, t1=-1 ):
    if (t0>=0) & (t0<=len( signal[0] )):
        t0 = int( t0 / sampling_period )
    else:
        t0 = 0
    if (t1>=0) & (t1<=len( signal[0] )):
        t1 = int( t1 / sampling_period )
    else:
        t1 = -1
    Y = np.array([])
    for trace in signal:
        Y = np.concatenate( ( Y, np.array( trace[t0:t1] ) ) )
    return [Y]