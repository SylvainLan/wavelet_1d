import numpy as np
from numpy.fft import fft, ifft


def analysis_decim(data, h_filter, g_filter, nb_scale=3):
    if np.mod(len(data), np.power(2, nb_scale)) != 0:
        raise ValueError('should have dyadic length')
    n = len(data)
    analysis_data = []
    analysis_data_scale = []
    idx_begin = 0
    for scale in range(nb_scale):
        dataf = fft(data)
        g_scale = g_filter[idx_begin:idx_begin+n]
        analysis_data_scale = ifft(np.conj(g_scale)*dataf)
        analysis_data_scale = analysis_data_scale[::2]
        analysis_data = np.append(analysis_data, analysis_data_scale)

        h_scale = h_filter[idx_begin:idx_begin+n]
        analysis_data_scale = ifft(np.conj(h_scale)*dataf)
        analysis_data_scale = analysis_data_scale[::2]
        data = analysis_data_scale

        idx_begin += n
        n = np.int(n/2)
    analysis_data = np.append(analysis_data, analysis_data_scale)
    return analysis_data


def analysis_undecim(data, h_filter, g_filter, nb_scale=3):
    if np.mod(len(data), np.power(2, nb_scale)) != 0:
        raise ValueError('should have dyadic length')
    n = len(data)
    analysis_data = np.zeros(n*(nb_scale+1), dtype=complex)
    idx_begin = 0
    xf = fft(data)
    nj = n
    for scale in range(nb_scale):
        g_scale = g_filter[idx_begin:idx_begin+nj]
        h_scale = h_filter[idx_begin:idx_begin+nj]
        g_scale = np.tile(g_scale, np.int(np.power(2, scale)))
        h_scale = np.tile(h_scale, np.int(np.power(2, scale)))

        analysis_data[n*scale:n*(scale+1)] = ifft(np.conj(g_scale)*xf)
        xf = np.conj(h_scale)*xf

        idx_begin += nj
        nj = np.int(nj/2)
    analysis_data[-n:] = ifft(xf)
    return analysis_data


def synthesis_decim(analysis_data, hs_filter, gs_filter, nb_scale=3):
    if np.mod(len(analysis_data), np.power(2, nb_scale)) != 0:
        raise ValueError('should have dyadic length')
    n = len(analysis_data)
    idx_coeff = 0
    analysis_data = analysis_data[::-1]
    nj = np.int(n/np.power(2, nb_scale))
    data = analysis_data[idx_coeff:idx_coeff+nj]
    data = data[::-1]
    idx_coeff += nj

    idx_filter = 0
    hs_filter = hs_filter[::-1]
    gs_filter = gs_filter[::-1]
    for scale in range(nb_scale):
        c_ = np.zeros(2*len(data), dtype=complex)
        c_[::2] = data
        data = fft(c_)
        h_scale = hs_filter[idx_filter:idx_filter+2*nj]
        g_scale = gs_filter[idx_filter:idx_filter+2*nj]
        h_scale = h_scale[::-1]
        g_scale = g_scale[::-1]

        w = analysis_data[idx_coeff:idx_coeff+nj]
        w = w[::-1]
        c_[::2] = w
        w = fft(c_)

        data = ifft(data*h_scale + w*g_scale)
        idx_coeff += nj
        nj = 2*nj
        idx_filter += nj
    return data


def synthesis_undecim(analysis_data, hs_filter, gs_filter, nb_scale=3):
    if np.mod(len(analysis_data), np.power(2, nb_scale)) != 0:
        raise ValueError('should have dyadic length')
    n = len(analysis_data)
    analysis_data = analysis_data[::-1]
    idx_begin = 0
    nw = np.int(n/np.power(2, nb_scale - 1))
    data = analysis_data[idx_begin:idx_begin+nw]
    data = fft(data[::-1])
    nj = np.int(nw/np.power(2, nb_scale))
    hs_filter = hs_filter[::-1]
    gs_filter = gs_filter[::-1]
    idx_filter = 0
    idx_begin += nw
    for scale in range(nb_scale):
        h_scale = hs_filter[idx_filter:idx_filter+2*nj]
        g_scale = gs_filter[idx_filter:idx_filter+2*nj]
        h_scale = h_scale[::-1]
        g_scale = g_scale[::-1]
        h_scale = np.tile(h_scale, np.int(nw/(2*nj)))
        g_scale = np.tile(g_scale, np.int(nw/(2*nj)))

        analysis_dataf = analysis_data[idx_begin:idx_begin+nw]
        analysis_dataf = analysis_dataf[::-1]
        analysis_dataf = fft(analysis_dataf)
        data = (data*h_scale + analysis_dataf*g_scale)/2.

        idx_filter += 2*nj
        idx_begin += nw
        nj *= 2
    data = ifft(data)
    return data

