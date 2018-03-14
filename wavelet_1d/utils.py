from wavelet_1d.extensions.transform import UndecimatedActiveletTransform, DecimatedActiveletTransform, PyWTransform


def load_transform(wavelet_name, nb_scale, **kwargs):
    """
    Load a transform using its name
    :param wavelet_name: str
        Name of the 1D wavelet
    :param nb_scale: int
        Number of decomposition scales
    :return: transform
        WaveletTransform1D instance

    """
    if wavelet_name == 'activeletDecim':
        return DecimatedActiveletTransform(nb_scale)
    elif wavelet_name == 'activeletUndecim':
        return UndecimatedActiveletTransform(nb_scale)
    else:
        kwargs["name"] = wavelet_name
        return PyWTransform(nb_scale, **kwargs)
