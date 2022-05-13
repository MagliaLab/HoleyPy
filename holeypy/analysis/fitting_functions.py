# -*- coding: utf-8 -*-
from scipy.special import gamma, gammainc, gammaincc
from scipy.stats import gamma as Gamma
from scipy.stats import gennorm
from scipy.stats import erlang
from scipy.optimize import (curve_fit, minimize)
from scipy import asarray as ar, exp
import numpy as np
import time

from scipy.fft import rfft
from scipy import signal

'''
# rNDF: regular Normal Distribution Function
def rNDF(x, a, x0, sigma):
    return a*exp(-(x-x0)**2/(2*sigma**2))


# gNDF: generalised Normal Distribution Function
def gNDF(x, A, x0, sigma, B, C):
    E = -((x - x0) / (2 * sigma))**B
    return (A*exp(E)) + C
'''

from contextlib import contextmanager
import threading
import _thread


class TimeoutException(Exception):
    def __init__(self, msg=''):
        self.msg = msg


@contextmanager
def time_limit(seconds, msg=''):
    timer = threading.Timer(seconds, lambda: _thread.interrupt_main())
    timer.start()
    try:
        yield
    except KeyboardInterrupt:
        raise TimeoutException("Timed out for operation {}".format(msg))
    finally:
        # if the action ends in specified time, timer is canceled
        timer.cancel()


class Adept:
    def __init__(self, popt=None):
        self.popt = popt

    def get_fit(self, x):
        return self.gNDF(np.array(x), *self.popt)

    def fit(self, x, y, I0=-100):
        try:
            a = np.max(y)
            x0 = np.mean(x)
            sigma = float(max(x)-min(x)) / 3
            c = sigma / 5
            with time_limit(0.2, 'sleep'):
                self.popt, _ = curve_fit(self.adept2state, x, -1 * abs(y), p0=(I0, a, x0-sigma, x0+sigma, c), maxfev=1000)
        except:
            pass
        finally:
            if self.popt is not None:
                return True
            else:
                return False

    def adept2state(self, t, I0, a, mu_1, mu_2, tau):
        rise_capacitance = (np.exp(-(t - mu_2) / tau) - 1)
        fall_capacitance = (1 - np.exp(-(t - mu_1) / tau))
        rise_capacitance[abs(rise_capacitance) == np.inf] = 0
        fall_capacitance[abs(fall_capacitance) == np.inf] = 0
        rise = rise_capacitance * np.heaviside(t - mu_2, 1)
        fall = fall_capacitance * np.heaviside(t - mu_1, 1)
        return I0 + a * (rise + fall)

    def dwell_time(self):
        return abs(self.popt[3] - self.popt[2])


class gNDF:
    def __init__(self, popt=None):
        self.popt = popt
        self.name = "gNDF"
        self.x = None
        self.y = None

    def features(self):
        features = ['Amplitude_block', 'Localisation', 'Sigma', 'Beta',
                    'Open_current', 'Dwell_time', 'Fs_event']
        if self.popt is not None:
            features_dict = dict(zip(features[0:len(features)-1], self.popt))
            p = self.get_prob()
            # print(self.effective_event_frequency(p))
            features_dict['Dwell_time'] = self.dwell_time(p=p)
            features_dict['Fs_event'] = self.effective_event_frequency(p)
            # features_dict['Noise_component'] = ','.join(str(v) for v in self.noise_component())
            return features_dict
        else:
            return features

    def get_fit(self, x):
        return self.gNDF(np.array(x), *self.popt)

    def get_pdf(self, x):
        return self.gNDF_PDF(np.array(x), *self.popt)

    def fit(self, x, y, I0=-100, sigma=None, a=None):
        self.x = x
        self.y = y
        try:
            if a == None:
                a = np.max(y)
            x0 = np.mean(x)
            if sigma == None:
                sigma = float(max(x)-min(x)) / 3
            b = 2.72
            c = I0
            with time_limit(0.2, 'sleep'):
                self.popt, _ = curve_fit(self.gNDF, x, -1 * abs(y), p0=(a, x0, sigma, b, c), maxfev=1000)
        except:
            pass
        finally:
            if self.popt is not None:
                return True
            else:
                return False

    def gNDF(self, x, a, x0, sigma, b, c, cutoff=0.001, tau=0.0001):
        reg = 1 / (2 * sigma * gamma(1 / b))
        # gNDF_fit = a * (gennorm.pdf(x, b, loc=x0, scale=sigma) / reg) + c
        pdf = np.array(gennorm.pdf(x, b, loc=x0, scale=sigma))
        max_pdf = gennorm.pdf(x0, b, loc=x0, scale=sigma)
        gNDF_fit = a * (pdf / max_pdf) + c
        return gNDF_fit

    def noise_component(self):
        return (self.y - self.get_fit(self.x)) * self.get_pdf(self.x)

    def frequency_spectrum(self, x, y):
        # Number of sample points
        N = len(x)
        # sample spacing
        T = np.mean(np.diff(x))
        xf = np.linspace(0.0, 1.0 / (2.0 * T), int(N / 2))

        y_res = (y - self.get_fit(x) * 10 ** -12) * self.get_pdf(x)
        yf = rfft(y_res)
        yf = 2.0 / N * np.abs(yf[:N // 2])
        b, a = signal.bessel(N=4, Wn=(2 * np.pi) / T, btype='lowpass', analog=True)
        _, gain = signal.freqs(b, a, worN=xf * 2 * np.pi)
        gain[xf > 1 / T] = 1
        gain = np.abs(gain)
        yf = (yf / gain) ** 2
        return xf, yf

    def effective_event_frequency(self, p):
        sigma = self.popt[2]
        beta = self.popt[3]
        s = 2*sigma*gamma((1/beta)+1)
        # print(np.log(1-p*s))
        delta_s = sigma*(-np.log(1-p*s))**(1/beta)
        return 1 / (2*delta_s)

    def dwell_time(self, p=0.001):
        sigma = self.popt[2]
        beta = self.popt[3]
        s = 2 * sigma * gamma((1 / beta) + 1)
        return (1+2*p)*s
        '''
        start = self.event_start(cutoff=p)
        end = self.event_end(start)
        return end - start
        '''

    def get_prob(self):
        res = minimize(self._validate_prob, np.array([0.001]), bounds=[(0, 0.01)])
        return res.x[0]

    def _validate_prob(self, p):
        dw = (1+2*p)*2*gamma((1/self.popt[3])+1)
        er = np.sign(p-0.5)*erlang.ppf(2*abs(p-0.5), 1/self.popt[3])**(1/self.popt[3])
        return (dw+er)**2

    def event_start(self, cutoff=0.001):
        return self.gNDF_iCDF(cutoff, *self.popt)

    def event_end(self, event_start):
        return event_start + ((1 + 2 * self.gNDF_CDF(event_start, *self.popt)) / self.gNDF_PDF(self.popt[1], *self.popt))

    def gNDF_dt(self, x, A, x0, sigma, B, C):
        e1 = (B ** 2) * ((1 / sigma) ** (B + 1)) * (x0 - x) * (abs(x0 - x) ** (B - 2)) * np.exp(
            -((1 / sigma) ** B) * abs(x0 - x) ** B)
        return e1 / (2 * gamma(1 / B))

    def gNDF_PDF(self, x, A, x0, sigma, B, C):
        return gennorm.pdf(x, B, loc=x0, scale=sigma)

    def gNDF_CDF(self, x, A, x0, sigma, B, C):
        return gennorm.cdf(x, B, loc=x0, scale=sigma)

    def gNDF_iCDF(self, x, A, x0, sigma, B, C):
        return gennorm.ppf(x, B, loc=x0, scale=sigma)

    def gNDF_var(self, A, x0, sigma, B, C):
        return gennorm.std(B, loc=x0, scale=sigma)