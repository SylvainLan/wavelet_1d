from .utils import load_transform


class Wavelet1(object):
    """
    The 1D wavelet transform class
    """
    def __init__(self, wavelet_name, nb_scale=3, verbose=0):
        """
        Initialize the 'Wavelet1' class

        :param wavelet_name: str
            the wavelet name to be used in the decomposition
        :param nb_scale: int, default 3
            number of decomposition scales
        :param verbose: int, default 0
            verbosity level
        """
        self.nb_scale = nb_scale
        self.transform = load_transform(wavelet_name, nb_scale)

    def op(self, data):
        self.transform.data = data
        self.transform.analysis()
        analysis_data = self.transform.analysis_data
        return analysis_data

    def adj_op(self, coeffs):
        self.transform.analysis_data = coeffs
        signal = self.transform.synthesis()
        return signal
