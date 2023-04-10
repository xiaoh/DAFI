
"""
Collection of data-processing algorithms.
"""

import warnings

import numpy as np


class PreProcess(object):

    def __init__(self, ):
        self.stats = None

    def scale(self, x, stats=None):
        if stats is None:
            stats = self.calc_stats(x)
        return self._scale(x, stats)

    def _scale(self, x, stats):
        pass

    def unscale(self, x, stats):
        pass

    def calc_stats(self, x):
        pass

    def update_stats(self, x):
        if self.stats==None:
            self.stats = self.calc_stats(x)
        else:
            self.stats = self._update_stats(x)

    def _update_stats(self, x):
        pass


class Scale(PreProcess):

    def __init__(self, min=0.0, max=1.0):
        super(self.__class__, self).__init__()
        self.gmin = min
        self.gmax = max

    def _scale(self, x, stats):
        xmin, xmax = stats
        return (x - xmin)/(xmax-xmin) * (self.gmax - self.gmin) + self.gmin

    def unscale(self, x, stats):
        xmin, xmax = stats
        return (x-self.gmin) * (xmax - xmin) / (self.gmax - self.gmin) + xmin

    def calc_stats(self, x):
        xmin = np.min(x, axis=0)
        xmax = np.max(x, axis=0)
        return [xmin, xmax]

    def _update_stats(self, x):
        xmin, xmax = self.stats
        xmin_new, xmax_new = self.calc_stats(x)
        xmin = np.fmin(xmin, xmin_new)
        xmax = np.fmax(xmax, xmax_new)
        self.stats = [xmin, xmax]
        return [xmin, xmax]

class LocScale(PreProcess):

    def __init__(self, ts=1.0):
        super(self.__class__, self).__init__()
        self.ts = ts
        # self.gmax = max

    def _scale(self, x, ts):
        ts = ts.reshape(ts.shape[0], 1)
        x_raw = x / ts / ts
        return x_raw/ (np.abs(x_raw) + 1.0 / ts / ts)

    # def unscale(self, x, stats):
    #     # xmin, xmax = stats
    #     return x #(x-self.gmin) * (xmax - xmin) / (self.gmax - self.gmin) + xmin

    # def calc_stats(self, x):
    #     xmin = np.min(x, axis=0)
    #     xmax = np.max(x, axis=0)
    #     return [xmin, xmax]

    # def _update_stats(self, x):
    #     xmin, xmax = self.stats
    #     xmin_new, xmax_new = self.calc_stats(x)
    #     xmin = np.fmin(xmin, xmin_new)
    #     xmax = np.fmax(xmax, xmax_new)
    #     self.stats = [xmin, xmax]
    #     return [xmin, xmax]

class Normalize(PreProcess):

    def __init__(self, mean=0.0, std=1.0):
        super(self.__class__, self).__init__()
        self.gmean = mean
        self.gstd = std

    def _scale(self, x, stats):
        mean, std, _ = stats
        return ((x - mean)/std)*self.gstd + self.gmean

    def unscale(self, x, stats):
        mean, std, _ = stats
        return x * std + mean

    def calc_stats(self, x):
        mean = np.mean(x, axis=0)
        std = np.std(x, axis=0)
        ndata = x.shape[0]
        return [mean, std, ndata]

    def _update_stats(self, x):
        mean, std, ndata = self.stats
        mean_new, std_new, ndata_new = self.calc_stats(x)
        mean_combined = (ndata*mean + ndata_new*mean_new) / (ndata + ndata_new)
        t1 = ndata * (std**2 + (mean - mean_combined)**2)
        t2 = ndata_new * (std_new**2 + (mean_new - mean_combined)**2)
        std_combined = np.sqrt((t1 + t2) / (ndata+ndata_new))
        ndata_combined = ndata + ndata_new
        self.stats = [mean_combined, std_combined, ndata_combined]
