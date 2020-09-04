import qutip as qt
import numpy as np
import scipy
from scipy import constants
from scipy.linalg import expm, sinm, cosm
import itertools

import matplotlib.pyplot as plt

pi = np.pi
e = constants.e
h = constants.h
hbar = constants.hbar
ep0 = constants.epsilon_0
mu0 = constants.mu_0

er = 11.9 # Si

# unit[m]
H = 450e-6
H1 = 1200e-6
t = 0.05e-6
Lam0 = 4.5e-8  # 侵入長

TC = 9.29 # [K] # Nb
TB = 20e-3 # [K] # Base temp.


def C_cpw_per_L(s=10e-6, w=6e-6, er=11.9, h1=750e-6, h=450e-6):
    k = np.tanh( pi * s/(4 * h) )/np.tanh( pi * (s + 2 * w)/(4 * h) )
    k1 = np.tanh( pi * s/(4 * h1) )/np.tanh( pi * (s + 2 * w)/(4 * h1) )

    K = scipy.special.ellipk(k)
    K_d = scipy.special.ellipk(np.sqrt(1 - k**2))
    K1 = scipy.special.ellipk(k1)
    K1_d = scipy.special.ellipk(np.sqrt(1 - k1**2))

    Cair = 2 * ep0 * ( K/K_d + K1/K1_d )

    q = (K/K_d) / (K/K_d + K1/K1_d)
    e_eff = 1 + q * (er - 1)
    Ccpw = e_eff * Cair
    return [Ccpw, Cair, e_eff] # unit[F/m]

def L_cpw_per_L(s=10e-6, w=6e-6, er=11.9, h1=750e-6, h=450e-6):
    k = np.tanh( pi * s/(4 * h) )/np.tanh( pi * (s + 2 * w)/(4 * h) )
    k1 = np.tanh( pi * s/(4 * h1) )/np.tanh( pi * (s + 2 * w)/(4 * h1) )

    K = scipy.special.ellipk(k)
    K_d = scipy.special.ellipk(np.sqrt(1 - k**2))
    K1 = scipy.special.ellipk(k1)
    K1_d = scipy.special.ellipk(np.sqrt(1 - k1**2))
    return (mu0/2) / (K/K_d + K1/K1_d) # unit[H/m]

def phase_velocity(C_per_L, L_per_L):
    return 1/np.sqrt(C_per_L*L_per_L)

def L_kin(s=10e-6, w=6e-6, t=.05e-6, er=11.9, Tb=TB, Tc=TC, lam0=Lam0, h=450e-6):
    A = (-t/pi) + 0.5*np.sqrt((2*t/pi)**2 + s**2)
    B = (s**2)/(4*A)
    C = B - t/pi + np.sqrt((t/pi)**2 + w**2)
    D = (2*t)/pi + C
    k = np.tanh( pi * s/(4 * h) )/np.tanh( pi * (s + 2 * w)/(4 * h) )
    K = scipy.special.ellipk(k)
    lam = lam0/(1 - (Tb/Tc)**4)**(0.5)
    Lkin = mu0*lam*(C/(4*A*D*K))*( 1.7/(np.sinh(t/(2*lam))) + 0.4/np.sqrt(((B/A)**2 - 1 )*(1 - (B/D)**2)) )
    return Lkin

def impedance(s=10e-6, w=6e-6, t=.05e-6, er=11.9, Tb=TB, Tc=TC, lam0=Lam0, kin=False, h1=750e-6, h=450e-6):
    cperl = C_cpw_per_L(s, w, er, h1, h)[0] #+ 20e-15
    lperl = L_cpw_per_L(s, w, er, h1, h)
    if kin:
        lperl = lperl + L_kin(s, w, t, er, Tb, Tc, lam0)
    return np.sqrt(lperl/cperl)

def freqency(l, s=10e-6, w=6e-6, t=.05e-6, er=11.9, Tb=TB, Tc=TC, lam0=Lam0, kin=False, h1=750e-6, h=450e-6):
    cperl = C_cpw_per_L(s, w, er, h1, h)[0] #+ 20e-15
    lperl = L_cpw_per_L(s, w, er, h1, h)
    if kin:
        lperl = lperl + L_kin(s, w, t, er, Tb, Tc, lam0, h)
    nu = phase_velocity(cperl, lperl)
    f = nu/(2*l)
    return f

def freqency4(l, s=10e-6, w=6e-6, t=.05e-6, er=11.9, Tb=TB, Tc=TC, lam0=Lam0, kin=False, h1=750e-6, h=450e-6):
    cperl = C_cpw_per_L(s, w, er, h1, h)[0] #+ 20e-15
    lperl = L_cpw_per_L(s, w, er, h1, h)
    if kin:
        lperl = lperl + L_kin(s, w, t, er, Tb, Tc, lam0, h)
    nu = phase_velocity(cperl, lperl)
    f = nu/(4*l)
    return f

def resonatorLength(f, s=10e-6, w=6e-6, t=.05e-6, er=11.9, Tb=TB, Tc=TC, lam0=Lam0, kin=False, h1=750e-6, h=450e-6, lam=True):
    cperl = C_cpw_per_L(s, w, er, h1, h)[0] #+ 20e-15
    lperl = L_cpw_per_L(s, w, er, h1, h)
    if kin:
        lperl = lperl + L_kin(s, w, t, er, Tb, Tc, lam0, h)
    nu = phase_velocity(cperl, lperl)
    if lam == 0:
        l = nu/(2*f)
    else:
        l = nu/(4*f)
    return l

def TD(f, s=10e-6, w=6e-6, t=.05e-6, er=11.9, Tb=TB, Tc=TC, lam0=Lam0, kin=False, h1=750e-6, h=450e-6, lam=True):
    l = resonatorLength(s, w, t, f, er, Tb, Tc, lam0, kin, h1, h, lam)
    eff = C_cpw_per_L(s, w, er, h1, h)[2]
    c = 3*10**8
    print('Len={:.3f}'.format(l*1e6))
    return l*np.sqrt(eff)/c

def Qext_halfLambda(wr, Cin, Cout, Z0=50):
    return (pi/2)/(((wr*Z0)**2)*(Cin**2 + Cout**2))