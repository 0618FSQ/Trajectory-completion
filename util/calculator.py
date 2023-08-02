import numpy as np
import math
import scipy.interpolate as spi

DELTA = 1e-8


class Interpolate(object):
    def __init__(self, x, y):
        if np.any(x[1:] == x[:-1]):
            x[np.where(x[1:] == x[:-1])] = x[np.where(x[1:] == x[:-1])]+DELTA
        self.ipo = spi.interp1d(x, y, kind='quadratic')

    def __call__(self, x):
        iy = self.ipo(x)
        return iy

class Calculator:
    def __init__(self, x, y, time):
        self.x = self.trans_data_format(x)
        self.y = self.trans_data_format(y)
        self.time = np.arange(0, time, 0.1)
        self.v_x = self.differential(self.time, self.x)
        self.v_y = self.differential(self.time, self.y)
        self.acc = np.sqrt(self.v_x ** 2 + self.v_y ** 2)
        self.acc_x = self.differential(self.time, self.v_x)
        self.acc_y = self.differential(self.time, self.v_y)
        self.jerk_x = self.differential(self.time, self.acc_x)
        self.jerk_y = self.differential(self.time, self.acc_y)
        self.jerk = self.differential(self.time, self.acc)
        self.heading = np.arctan2(self.v_y, self.v_x)
        self.cur = np.gradient(self.heading) / np.sqrt(np.gradient(x) ** 2 + np.gradient(y)**2)
    
    def __call__(self):
        v_x = self.v_x * np.cos(self.heading) + self.v_y * np.sin(self.heading)
        v_y = self.v_y * np.cos(self.heading) - self.v_x * np.sin(self.heading)
        acc_x = self.acc_x * np.cos(self.heading) + self.acc_y * np.sin(self.heading)
        acc_y = self.acc_y * np.cos(self.heading) - self.acc_x * np.sin(self.heading)
        jerk_x = self.jerk_x * np.cos(self.heading) + self.jerk_y * np.sin(self.heading)
        jerk_y = self.jerk_y * np.cos(self.heading) - self.jerk_x * np.sin(self.heading)
        return v_x, v_y, acc_x, acc_y, jerk_x, jerk_y

    @classmethod
    def trans_data_format(cls, x):
        if isinstance(x, list):
            return np.array(x)
        elif isinstance(x, np.ndarray):
            return x
        else:
            raise RuntimeError("Data format is error")
    
    @classmethod
    def differential(cls, x, y):
        assert len(x) == len(y)
        interpolate = Interpolate(x, y,)
        ans = []
        length = len(x)
        for i in range(length):
            if i != length - 1:
                ans.append(
                    (y[i] - interpolate(x[i] + DELTA)) / DELTA
                )
            else:
                ans.append(
                    -(y[i] - interpolate(x[i] - DELTA)) / DELTA
                )
        return np.array(ans)
