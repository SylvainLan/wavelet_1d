import numpy as np
from numpy.fft import fft, ifft


def transform(x, h_filter, g_filter, decim=1, direct=1, n_scales=3):
    if np.mod(len(x), np.power(2, n_scales)) != 0:
        raise ValueError('should have dyadic length')
    n = len(x)
    if direct:
        if decim:
            res = []
            res_scale = []
            idx_begin = 0
            for scale in range(n_scales):
                xf = fft(x)
                g_scale = g_filter[idx_begin:idx_begin+n]
                res_scale = ifft(np.conj(g_scale)*xf)
                res_scale = res_scale[::2]
                res = np.append(res, res_scale)

                h_scale = h_filter[idx_begin:idx_begin+n]
                res_scale = ifft(np.conj(h_scale)*xf)
                res_scale = res_scale[::2]
                x = res_scale

                idx_begin += n
                n = np.int(n/2)
            res = np.append(res, res_scale)
        else:
            res = np.zeros(len(x)*(n_scales+1), dtype=complex)
            idx_begin = 0
            xf = fft(x)
            nj = n
            for scale in range(n_scales):
                g_scale = g_filter[idx_begin:idx_begin+nj]
                h_scale = h_filter[idx_begin:idx_begin+nj]
                g_scale = np.tile(g_scale, np.int(np.power(2, scale)))
                h_scale = np.tile(h_scale, np.int(np.power(2, scale)))

                res[n*scale:n*(scale+1)] = ifft(np.conj(g_scale)*xf)
                xf = np.conj(h_scale)*xf

                idx_begin += nj
                nj = np.int(nj/2)
            res[-n:] = ifft(xf)
    else:
        if decim:
            idx_coeff = 0
            x = x[::-1]
            nj = np.int(n/np.power(2, n_scales))
            res = x[idx_coeff:idx_coeff+nj]
            res = res[::-1]
            idx_coeff += nj

            idx_filter = 0
            h_filter = h_filter[::-1]
            g_filter = g_filter[::-1]
            for scale in range(n_scales):
                c_ = np.zeros(2*len(res), dtype=complex)
                c_[::2] = res
                res = fft(c_)
                h_scale = h_filter[idx_filter:idx_filter+2*nj]
                g_scale = g_filter[idx_filter:idx_filter+2*nj]
                h_scale = h_scale[::-1]
                g_scale = g_scale[::-1]

                w = x[idx_coeff:idx_coeff+nj]
                w = w[::-1]
                c_[::2] = w
                w = fft(c_)

                res = ifft(res*h_scale + w*g_scale)
                idx_coeff += nj
                nj = 2*nj
                idx_filter += nj
        else:
            x = x[::-1]
            idx_begin = 0
            nw = np.int(n/np.power(2, n_scales - 1))
            res = x[idx_begin:idx_begin+nw]
            res = fft(res[::-1])
            nj = np.int(nw/np.power(2, n_scales))
            h_filter = h_filter[::-1]
            g_filter = g_filter[::-1]
            idx_filter = 0
            idx_begin += nw
            for scale in range(n_scales):
                h_scale = h_filter[idx_filter:idx_filter+2*nj]
                g_scale = g_filter[idx_filter:idx_filter+2*nj]
                h_scale = h_scale[::-1]
                g_scale = g_scale[::-1]
                h_scale = np.tile(h_scale, np.int(nw/(2*nj)))
                g_scale = np.tile(g_scale, np.int(nw/(2*nj)))

                xf = x[idx_begin:idx_begin+nw]
                xf = xf[::-1]
                xf = fft(xf)
                res = (res*h_scale + xf*g_scale)/2.

                idx_filter += 2*nj
                idx_begin += nw
                nj *= 2
            res = ifft(res)

    return res
