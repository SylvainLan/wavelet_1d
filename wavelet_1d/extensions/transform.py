from ..base.transform import OneDWaveletTransformBase
from scipy.io import loadmat
from ..transform.transform import analysis_decim, analysis_undecim, synthesis_undecim, synthesis_decim
import pywt


class FILTEROneDWaveletTransformBase(OneDWaveletTransformBase):
    __use_pywt__ = None

    def _set_transformation_parameters(self):
        self.use_pywt = self.__use_pywt__
        if not self.__use_pywt__:
            self.filter_bank = load_filter(self.nb_scale)

    def _analysis(self, data, **kwargs):
        raise NotImplementedError("Abstract method should not be declared "
                                  "in derivate classes.")

    def _synthesis(self, analysis_data, analysis_header):
        raise NotImplementedError("Abstract method should not be declared "
                                  "in derivate classes.")


class DecimatedActiveletTransform(FILTEROneDWaveletTransformBase):
    __use_pywt__ = 0

    def _analysis(self, data, **kwargs):
        analysis_data = analysis_decim(data, self.filter_bank['h_filter'], self.filter_bank['g_filter'], self.nb_scale)
        analysis_header = None
        return analysis_data, analysis_header

    def _synthesis(self, analysis_data, analysis_header):
        data = synthesis_decim(analysis_data, self.filter_bank['hi_filter'], self.filter_bank['gi_filter'],
                               self.nb_scale)
        return data


class UndecimatedActiveletTransform(FILTEROneDWaveletTransformBase):
    __use_pywt__ = 0

    def _analysis(self, data, **kwargs):
        analysis_data = analysis_undecim(data, self.filter_bank['h_filter'], self.filter_bank['g_filter'],
                                         self.nb_scale)
        analysis_header = None
        return analysis_data, analysis_header

    def _synthesis(self, analysis_data, analysis_header):
        data = synthesis_undecim(analysis_data, self.filter_bank['hi_filter'], self.filter_bank['gi_filter'],
                                 self.nb_scale)
        return data


class PyWTransform(FILTEROneDWaveletTransformBase):
    __use_pywt__ = 1

    def _analysis(self, data, **kwargs):
        coeffs = pywt.wavedec(data, wavelet=self.name, level=self.nb_scale)
        analysis_data, analysis_header = pywt.coeffs_to_array(coeffs)
        return analysis_data, analysis_header

    def _synthesis(self, analysis_data, analysis_header):
        coeffs = pywt.array_to_coeffs(analysis_data, analysis_header, output_format='wavedec')
        data = pywt.waverec(coeffs, wavelet=self.name)
        return data


def load_filter(nb_scale):
    filter_load = loadmat('data/filters.mat')  # TODO change the way filters are loaded
    filter_bank = {'h_filter': filter_load['allH'][0],
                   'g_filter': filter_load['allG'][0],
                   'hi_filter': filter_load['allHd'][0],
                   'gi_filter': filter_load['allGd'][0]}
    return filter_bank

