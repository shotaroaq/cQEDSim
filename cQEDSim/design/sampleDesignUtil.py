import qutip as qt
import numpy as np
import scipy
from scipy import constants
from scipy.linalg import expm, sinm, cosm

import matplotlib.pyplot as plt

pi = np.pi
e = constants.e
h = constants.h
hbar = constants.hbar
ep0 = constants.epsilon_0
mu0 = constants.mu_0
Phi0 = h/(2*e)
kb = constants.Boltzmann


def to_dBm(P):
    return 10*np.log10(P/1e-3)

def to_Watt(dBm):
    return 1e-3*10**(dBm/10)

def Anbe(R):
    # input  : R in ohm
    # output : Ic in A
    return 1.764*pi*kb*1.2/(2*e*R)

def Ic_to_R(Ic):
    # input  : Ic in A
    # output : R in ohm
    return 1.764*pi*kb*1.2/(2*e*Ic)

def EJ_to_Ic(EJ):
    # input  : EJ in Hz
    # output : Ic in A
    return (EJ) * (4*pi*e)

def Ic_to_EJ(Ic):
    # input  : EJ in Hz
    # output : Ic in A
    return (Ic) / (4*pi*e)

def EJ_to_LJ(EJ):
    # input  : EJ in Hz
    # output : LJ in H
    IC = EJ_to_Ic(EJ)
    return Phi0/(2*pi*IC)

def Ic_to_LJ(IC):
    # input  : IC in A
    # output : LJ in H
    return Phi0/(2*pi*IC)

def LJ_to_Ic(LJ):
    # input  : LJ in H
    # output : IC in A
    return Phi0/(2*pi*LJ)

def Ec_to_C(EC):
    # input  : EC in Hz
    # output : C in F
    return (e**2)/(2*h*EC)

def C_to_Ec(C):
    # input  : C in F
    # output : EC in Hz
    return (e**2)/(2*h*C)

def calc_EJ_from_R(R):
    # input  : R in ohm
    # output : EJ in Hz
    ic = Anbe(R)
    ej = ic/(4*pi*e)
    return ej

def calc_c_to_g2(Cr, Cq, Cg, fr, fq):
    # coupling const between resonator and Transmon
    # output : g in Hz
    A = 4 * pi * 1 * np.sqrt( (Cr * Cq) / (fr * fq * 4 * pi ** 2) )
    return Cg/A

def calc_Cg2(Cr, Cq, fr, fq, g_target):
    # g_target in Hz
    # return target coupling capacitance between Q & R
    return 4 * pi * g_target * np.sqrt( (Cr * Cq) / (fr * fq * 4 * pi ** 2) )

def calc_g_direct(c1, c2, cg):
    # coupling const between Transmons
    return (4*e**2)*cg/(c1*c2)/h

def H_Transmon(Ec, Ej, N, ng):
    """
    Return the charge qubit hamiltonian as a Qobj instance.
    Ej : josephson energy in Hz
    Ec : charging energy in Hz
    N : maximum cooper pair deference
    ng : voltage bias for island
    """
    Ec = Ec*1e9
    Ej = Ej*1e9
    m = np.diag(4 * Ec * (np.arange(-N,N+1)-ng)**2) + 0.5 * Ej * (np.diag(-np.ones(2*N), 1) + np.diag(-np.ones(2*N), -1))
    return qt.Qobj(m)

def Transmon_ene_levels(EJ, EC, N, PLOT=1):
    # Ej : josephson energy in Hz
    # Ec : charging energy in Hz
    # N : maximum cooper pair deference
    if PLOT==1:
        ng = 0
        enes = H_Transmon(EC, EJ, N, ng).eigenenergies()
    elif PLOT==0:
        ng_vec = np.linspace(-4, 4, 100)
        energies = np.array([H_Transmon(EC, EJ, N, ng).eigenenergies() for ng in ng_vec])
        enes = energies[49]
    if PLOT==0:
        fig, axes = plt.subplots(1,2, figsize=(16,6))
        for n in range(len(energies[0,:])):
            ene = energies[:,n] - energies[:,0]
            axes[0].plot(ng_vec, ene)

        axes[0].plot(ng_vec, [9.8 for _ in range(len(ng_vec))], linestyle='dashed', color='red')
        axes[0].set_ylim(0.1, 50)
        axes[0].set_xlabel(r'$n_g$', fontsize=18)
        axes[0].set_ylabel(r'$E_n$', fontsize=18)
        axes[0].grid()

        for n in range(len(energies[0,:])):
            axes[1].plot(ng_vec, (energies[:,n]-energies[:,0])/(energies[:,1]-energies[:,0]))

        axes[1].set_ylim(-0.1, 3)
        axes[1].set_xlabel(r'$n_g$', fontsize=18)
        axes[1].set_ylabel(r'$(E_n-E_0)/(E_1-E_0)$', fontsize=18)
        axes[1].grid()
    
    return [enes[i]-enes[0] for i in range(len(enes))]