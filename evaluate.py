import numpy as np

from scipy import signal

def decoding_performance_func(e):
    return 1/(2*len(e))*np.inner(e,e)

def decoding_performance(e_log):
    N = np.shape(e_log)[0]
    return 1/(2*N) * np.einsum('i...,i...',e_log, e_log ) #Einsum really is nice :)

def decoding_performance_mean(e_log):
    if len(e_log) == 0:
        return -1
    N = np.shape(e_log)[0]
    M = np.shape(e_log)[1]
    return 1/(2*N*M) * np.einsum('ij,ij',e_log, e_log ) #Einsum really is nice :)

def rolling_average(log, window_length = 1000):

    cumsum = np.cumsum(log)

    return (cumsum[window_length:] - cumsum[:-window_length])/window_length


def rolling_decoding_performance(e_log, window_length = 3001):
    #assert window_length % 2 == 1 #Savgol requires an odd windowlength
    N = np.shape(e_log)[0]
    performance = 1/(2*N) * np.einsum('i...,i...',e_log, e_log ) #Einsum really is nice :)
    return np.convolve(performance, np.ones(window_length)/window_length, 'valid')
    #return signal.savgol_filter(performance, window_length, 1)


