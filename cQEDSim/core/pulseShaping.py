import qutip as qt
import numpy as np
import scipy
from scipy import constants
from scipy.linalg import expm, sinm, cosm
import itertools


class discretized_pulse_with_gauusian_filter():
    def __init__(self, args):
        """
        args = {'N': awg_input, 'dt': dt,
                'p': resolution of discretization,
                'w0': 0.40 * 2 * np.pi [rad*Hz], 'uk': vUk}
        """
        self.N = args['N']
        self.M = self.N * args['p']
        self.dt = args['dt']
        self.ddt = self.dt / (args['p'])
        self.w0 = args['w0']
        self.uk = args['uk']
        self.sl = self.calc_sl()
        self.tljs = np.zeros([self.N, self.M])
        self.cal_tljs()
        self.t_points = np.arange(self.ddt, self.dt*self.N + self.ddt, self.ddt, dtype=np.float64)

    def heaviside_step_ddt(self, t, j):
        return np.heaviside(t-(j)*self.ddt, 1) - np.heaviside(t-(j+1)*self.ddt, 1)

    def heaviside_step_dt(self, t, j):
        return np.heaviside(t-(j)*self.dt, 1) - np.heaviside(t-(j+1)*self.dt, 1)

    def tlj(self, l, j):
        # gauusian_filter
        dt, ddt, w0 = self.dt, self.ddt, self.w0
        erf_L = scipy.special.erf(w0*( l*ddt - dt * j ) / 2)
        erf_R = scipy.special.erf(w0*( l*ddt - dt * (j+1) ) / 2)
        return 0.5 * (erf_L - erf_R)

    def cal_tljs(self):
        for j in range(self.N):
            self.tljs[j] = [ self.tlj(l, j) for l in range(self.M) ]

    def calc_sl(self):
        uk = self.uk
        res = [sum([self.tlj(l, j) * uk[j] for j in range(self.N)]) for l in range(self.M)]
        return np.array(res)

    def recalc_sl(self, uk):
        self.uk = uk
        self.sl = self.calc_sl()
        return self.sl

    def wf_before(self, t):
        t = t
        j = int(t / self.dt)
        if j >= self.N:
            res = 0
        elif j <= 0:
            res = 0
        else:
            res = self.heaviside_step_dt(t, j) * self.uk[j]
        return res

    def wf_after(self, t, ags=None):
        t = t
        j = int(t / self.ddt)
        if j >= self.M:
            res = 0
        elif j < 0:
            res = 0
        else:
            res = self.heaviside_step_ddt(t, j) * self.sl[j]
        return res