class OneDWaveletTransformBase:
    def __init__(self, nb_scale, verbose=0, **kwargs):
        self.nb_scale = nb_scale
        self.name = None
        self.filter_bank = None
        self.use_pywt = None

        self._data = None
        self._analysis_data = None
        self._analysis_header = None

        if 'name' in kwargs.keys():
            self.name = kwargs['name']

    def _set_data(self, data):
        self._data = data
        self._set_transformation_parameters()

    def _get_data(self):
        return self._data

    def _set_analysis_data(self, data):
        self._analysis_data = data

    def _get_analysis_data(self):
        return self._analysis_data

    def _set_analysis_header(self, analysis_header):
        self._analysis_header = analysis_header

    def _get_analysis_header(self):
        return self._analysis_header

    def analysis(self, **kwargs):
        if self._data is None:
            raise ValueError("Please specify first the input data.")
        self._analysis_data, self._analysis_header = self._analysis(
            self._data)

    def _analysis(self, data, **kwargs):
        raise NotImplementedError("Abstract method should not be declared "
                                  "in derivate classes.")

    def synthesis(self, **kwargs):
        if self._analysis_data is None:
            raise ValueError("Please specify first the decomposition "
                             "coefficients array.")
        if self.use_pywt and self._analysis_header is None:
            raise ValueError("Please specify first the decomposition "
                             "coefficients header.")
        return self._synthesis(self._analysis_data, self._analysis_header)

    def _synthesis(self, analysis_data, analysis_header):
        raise NotImplementedError("Abstract method should not be declared "
                                  "in derivate classes.")

    def _set_transformation_parameters(self):
        raise NotImplementedError("Abstract method should not be declared "
                                  "in derivate classes.")

    data = property(_get_data, _set_data)
    analysis_data = property(_get_analysis_data, _set_analysis_data)
    analysis_header = property(_get_analysis_header, _set_analysis_header)
