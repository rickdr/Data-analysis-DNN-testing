#  Copyright (c) 2021. NavInfo EU
#  All rights reserved.
#
#  You may only use this code under the terms of the NavInfo license.
#
import torch
import numpy as np
import cv2


class Filter():
    """
    Base class for all perturbing filters

    """

    def __init__(self, *args, **kwargs):
        self.kwargs = kwargs
        self._estimator = kwargs.get('estimator')
        self._requires_normalization = kwargs.get('requires_normalization', True)

    def __str__(self):
        return self.__class__.__name__

    def verify_params(self):
        pass

    def perturb(self, x, y=None, **kwargs):
        """
        Perturb the data provided in x an y and return a tuple of the perturbed data
        The method alos applies input transforms if required
        :param x: the data points
        :param y: the labels
        :param kwargs: additional arguments
        """
        if self._requires_normalization:
            x = self._estimator.inverse_transform_input(x)
            x, y = self._perturb(x, y, **kwargs)
            x = self._estimator.forward_transform_input(x)
        else:
            x, y = self._perturb(x, y, **kwargs)
        return x, y

    def _perturb(self, x, y=None, **kwargs):
        """
        Perturb the data provided in x an y and return a tuple of the perturbed data
        :param x: the data points
        :param y: the labels
        :param kwargs: additional arguments
        """
        raise NotImplementedError


class Noise(Filter):
    """
    Marker class for all noises
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

def _rgb2gray(image, axis=2, convert=False, hls2rgb=False, hsv2rgb=False):
    img_dtype = np.float32 if image.dtype == np.float32 or image.dtype == np.float64 else np.uint8
    image = image.astype(img_dtype)
    image = cv2.cvtColor(image, cv2.COLOR_HLS2RGB) if hls2rgb else image
    image = cv2.cvtColor(image, cv2.COLOR_HSV2RGB) if hsv2rgb else image
    image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if convert else image
    if len(image.shape) < 3:
        image = np.expand_dims(image, axis)
    return image.transpose((2, 0, 1))

def _rgb2hls(image, hls=False, hsv=False):
    img_dtype = np.float32 if image.dtype == np.float32 or image.dtype == np.float64 else np.uint8
    x_ = image.transpose(1, 2, 0).copy().astype(img_dtype)
    x_ = cv2.cvtColor(x_, cv2.COLOR_GRAY2RGB) if image.shape[0] == 1 else x_
    x_hls = x_.copy()
    if hls:
        x_hls = cv2.cvtColor(x_.astype(img_dtype), cv2.COLOR_RGB2HLS)
        x_hls = np.array(x_hls, dtype=img_dtype)
    if hsv:
        x_hls = cv2.cvtColor(x_.astype(img_dtype), cv2.COLOR_RGB2HSV)
        x_hls = np.array(x_hls, dtype=img_dtype)
    return np.array(x_, dtype=img_dtype), x_hls


def clip_values(func):
    def wrapper(self, *args, **kwargs):
        def _clip_values(x, y = None):
            # self._estimator = None
            if self._estimator is not None and self._estimator.clip_values is not None:
                x = torch.clip(x, self._estimator.clip_values[0], self._estimator.clip_values[1])
            return x, y
        x, y = func(self, *args, **kwargs)
        return _clip_values(x, y)
    return wrapper


class GaussianNoise(Noise):
    @clip_values
    def _perturb(self, x, y=None, **kwargs):
        x = x.cpu()
        x = x + torch.randn(x.size()) * kwargs.get('sigma', 0.5)
        return x, y


class SaltAndPepperNoise(Noise):
    @clip_values
    def _perturb(self, x, y=None, **kwargs):
        x = x.cpu().detach().numpy()
        noise_indices = np.random.random(x.shape)
        # set a random number of pixels to max value and a random number of pixels to min value as defined by user
        x[noise_indices>=1.0-kwargs.get('threshold',0.005)] = kwargs.get('upper_value',1.0)
        x[noise_indices<=kwargs.get('threshold',0.005)] = kwargs.get('lower_value',0.0)
        return torch.from_numpy(x), y