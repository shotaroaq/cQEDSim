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
Phi0 = h/(2*e)
kb = constants.Boltzmann

def ket(Nq, i):
    return qt.basis(Nq, i)

def ket_2Qsys(i, j, Nq1, Nq2):
    a = ket(Nq1, i)
    b = ket(Nq2, j)
    return qt.tensor(a, b)

def ket_3Qsys(i, j, k, Nq1, Nq2, Nqc):
    a = ket(Nq1, i)
    b = ket(Nq2, j)
    c = ket(Nqc, k)
    return qt.tensor(a, b, c)

def Bose_distributon(f, T):
    f = f * 1e9
    if T==0:
        res = 0
    else:
        res = ( np.exp(h*f/(kb*T)) - 1 )**(-1)
    return res

def Maxwell_Boltzmann_distributon(fs, T, N):
    if N < 1:
        print('N should be >=1')
    # f1 = fs[1] * 1e9
    # a = np.exp(-f1*h/(kb*T))
    Z = 0
    for i in range(len(fs[0:N])):
        fi = fs[i] * 1e9
        Z = Z + np.exp(-fi*h/(kb*T))
    pops = [np.exp(-fs[i]*1e9*h/(kb*T))/Z for i in range(len(fs[0:N]))]
    return pops

######### N-level paulis #########
def pI_N(Nq):
    return ket(Nq, 0) * ket(Nq, 0).dag() + ket(Nq, 1) * ket(Nq, 1).dag()

def pX_N(Nq):
    return ket(Nq, 0) * ket(Nq, 1).dag() + ket(Nq, 1) * ket(Nq, 0).dag()

def pY_N(Nq):
    return 1j*ket(Nq, 0) * ket(Nq, 1).dag() - 1j*ket(Nq, 1) * ket(Nq, 0).dag()

def pZ_N(Nq):
    return ket(Nq, 0) * ket(Nq, 0).dag() - ket(Nq, 1) * ket(Nq, 1).dag()

def Rarb(p:list, Nq):
    SX, SY, SZ = pX_N(Nq), pY_N(Nq), pZ_N(Nq)
    return ( -1j*(p[0]*SX + p[1]*SY + p[2]*SZ)/2 ).expm()

def Rarb2Q(p:list, Nq):
    R1 = Rarb(p[0:3], Nq)
    R2 = Rarb(p[3:6], Nq)
    res = qt.tensor(R1, R2)
    return res

def iniState1Qsys(Nq:int, s:int, mode='ket'):
    q1 = ket(Nq, s)
    psi0 = q1
    if mode == 'rho':
        ini = psi0*psi0.dag()
    else:
        ini = psi0
    return ini

def iniState1Q1Rsys(Nq:int, Nf:int, s:int, t:int, mode='ket'):
    q1 = ket(Nq, s)
    r1 = qt.fock(Nf, t)
    psi0 = qt.tensor(q1, r1)
    if mode == 'rho':
        ini = psi0*psi0.dag()
    else:
        ini = psi0
    return ini

def iniState2Qsys(Nq1:int, Nq2:int, s1:int, s2:int, mode='ket'):
    q1 = ket(Nq1, s1)
    q2 = ket(Nq2, s2)
    psi0 = qt.tensor(q1, q2)
    if mode == 'rho':
        ini = psi0*psi0.dag()
    else:
        ini = psi0
    return ini


class transmon():
    def __init__(self, EC=10, EJ=0.3, f01=None, alpha=None, N=10, Nq=3, Temp=20e-3):
        # Unit in [GHz]
        if f01 is None and alpha is None:
            self.EC = EC
            self.EJ = EJ
            self.N = N
            self.enes = self.calcChargeQubitLevels(self.EC, self.EJ, self.N)
            self.f01 = self.enes[1]
            self.anh = self.enes[2] - 2*self.enes[1]
        else:
            self.f01 = f01
            self.anh = - abs(alpha)
            self.enes = [0] + [i*self.f01 + (i-1)*self.anh for i in range(1, N)]
        self.nth_q = Maxwell_Boltzmann_distributon(self.enes, Temp, N)
        self.P0 = iniState1Qsys(Nq, 0, mode='rho')
        self.P1 = iniState1Qsys(Nq, 1, mode='rho')
        # Thermal_state_ket = 0
        # for i in range(Nq):
        #     Thermal_state_ket += ket(Nq, i)*self.nth_q[i]
        # self.Thermal_state_ket = Thermal_state_ket
        # self.Thermal_state_dm = self.Thermal_state_ket*self.Thermal_state_ket.dag()
        self.Thermal_state_dm = qt.thermal_dm(Nq, self.nth_q[1])
        if Nq != None:
            self.Q_duffingOscillator(Nq)

    def chargeQubitHamilonian(self, Ec, Ej, N, ng):
        """
        Return the charge qubit hamiltonian as a Qobj instance.

        Parameters
        ----------
        Ec : float
            unit is [GHz]

        Ej : float
            unit is [GHz]

        N : int
            Difference in the number of Cooper pairs

        ng : float:
            Voltage bias for Charge qubit.

        Returns
        -------
        qobj of Charge qubit hamiltonian.
        """
        m = np.diag(4 * Ec * (np.arange(-N,N+1)-ng)**2) + 0.5 * Ej * (np.diag(-np.ones(2*N), 1) +
                                                                np.diag(-np.ones(2*N), -1))
        return qt.Qobj(m)

    def Q_duffingOscillator(self, Nq=5):
        self.Nq = Nq

        Iq = qt.qeye(Nq)
        b = qt.destroy(Nq)
        nb = b.dag()*b

        self.X = pX_N(Nq)
        self.Y = pY_N(Nq)
        self.Z = pZ_N(Nq)

        self.Iq = Iq
        self.nb = nb

        self.b = b
        q1_lab = self.f01 * nb + 0.5 * self.anh * nb * (nb - Iq)
        self.Hqlab = q1_lab
        return q1_lab

    def calcChargeQubitLevels(self, EC, EJ, N):
        """
        Return the list of charge qubit eigen energy at flux optimal point.

        Parameters
        ----------
        Ec : float
            unit is [GHz]

        Ej : float
            unit is [GHz]

        N : int
            Difference in the number of Cooper pairs

        Returns
        -------
        list of eigenenergies.
        """
        ng_vec = [0]
        ene = np.array([self.chargeQubitHamilonian(EC, EJ, N, ng).eigenenergies() for ng in ng_vec])
        return [ene[0][i] - ene[0][0] for i in range(len(ene[0])) ]


class resonator():
    def __init__(self, fr, Qc, Nf=5, Temp=20e-3):
        self.fr = fr
        self.Nf = Nf
        self.Qc = Qc
        self.a = qt.destroy(Nf)
        self.ad = self.a.dag()
        self.na = self.ad * self.a
        self.Hr = fr * self.na
        self.Ir = qt.qeye(Nf)
        self.kappa = fr/Qc
        self.nth_a = Bose_distributon(fr, Temp)
        self.Thermal_state_dm = qt.thermal_dm(Nf, self.nth_a)


class QR():
    def __init__(self, Q, fr, g):
        # Unit in [GHz]
        self.fr = fr
        self.g = g
        self.Q = Q
        self.detuning = Q.f01 - fr
        self.thermal_photon = qt.utilities.n_thermal(fr, Q.f01)
        self.f01_dressed = Q.f01 + ( 2 * (g**2) / self.detuning ) * ( self.thermal_photon + 1/2 )
        self.X = ((g**2)/self.detuning)*(Q.anh/(Q.f01+Q.anh-fr))


class QRQ():
    def __init__(self, Q1, Q2, frb, g1, g2):
        # Unit in [GHz]
        self.frb = frb
        self.g1 = g1
        self.g2 = g2
        self.Q1 = Q1
        self.Q1 = Q2
        self.detuning1 = Q1.f01 - frb
        self.thermal_photon1 = qt.utilities.n_thermal(frb, Q1.f01)
        self.f01_dressed1 = Q1.f01 + ( 2 * (g1**2) / self.detuning1 ) * ( self.thermal_photon1 + 1/2 )
        self.X1 = ((g1**2)/self.detuning1)*(Q1.anh/(Q1.f01+Q1.anh-frb))

        self.detuning2 = Q2.f01 - frb
        self.thermal_photon2 = qt.utilities.n_thermal(frb, Q2.f01)
        self.f01_dressed2 = Q2.f01 + ( 2 * (g2**2) / self.detuning2 ) * ( self.thermal_photon2 + 1/2 )
        self.X2 = ((g2**2)/self.detuning2)*(Q2.anh/(Q2.f01+Q2.anh-frb))

        self.D12 = self.f01_dressed1 - self.f01_dressed2
        self.J = g1*g2*( self.detuning1 + self.detuning2 ) / ( 2 * self.detuning1 * self.detuning2 )
        self.f01_coupled1 = self.f01_dressed1 + (self.J**2)/self.D12
        self.f01_coupled2 = self.f01_dressed2 - (self.J**2)/self.D12

class QQ():
    # For direct coupling simulation
    def __init__(self, Q1, Q2, g12):
        # duffing oscillator model
        # Unit in [GHz]
        self.g12 = g12
        self.Q1 = Q1
        self.Q2 = Q2
        self.Nq1, self.Nq2 = Q1.Nq, Q2.Nq

        iq1, iq2 = qt.qeye(self.Nq1), qt.qeye(self.Nq2)
        b1, b2 = qt.destroy(self.Nq1), qt.destroy(self.Nq2)
        self.b1, self.b2 = b1, b2
        self.iq1, self.iq2 = iq1, iq2
        self.nb1, self.nb2 = self.b1.dag()*self.b1, self.b2.dag()*self.b2

        self.B1 = qt.tensor(b1, iq2)
        self.B2 = qt.tensor(iq1, b2)
        self.Iq1 = qt.tensor(iq1, iq2)
        self.Iq2 = qt.tensor(iq1, iq2)
        self.Nb1 = self.B1.dag()*self.B1
        self.Nb2 = self.B2.dag()*self.B2
        # Drive term @rotating frame
        self.Hd1_real = self.B1 + self.B1.dag()
        self.Hd1_imag = (- self.B1 + self.B1.dag())*1j
        self.Hd2_real = (self.B2 + self.B2.dag())
        self.Hd2_imag = (- self.B2 + self.B2.dag())*1j

        self.X1 = qt.tensor(pX_N(self.Nq1), iq2)
        self.Y1 = qt.tensor(pY_N(self.Nq1), iq2)
        self.Z1 = qt.tensor(pZ_N(self.Nq1), iq2)

        self.X2 = qt.tensor(iq1, pX_N(self.Nq2))
        self.Y2 = qt.tensor(iq1, pY_N(self.Nq2))
        self.Z2 = qt.tensor(iq1, pZ_N(self.Nq2))

        bbbb1 = self.B1.dag()*self.B1.dag()*self.B1*self.B1
        bbbb2 = self.B2.dag()*self.B2.dag()*self.B2*self.B2
        self.duff_part1 = 0.5 * self.Q1.anh * self.Nb1 * (self.Nb1 - self.Iq1) # 0.5 * Q1.anh * bbbb1
        self.duff_part2 = 0.5 * self.Q2.anh * self.Nb2 * (self.Nb2 - self.Iq2) # 0.5 * Q2.anh * bbbb2
        self.Hq1 = Q1.f01 * self.Nb1 + self.duff_part1 #  - self.Iq1*0
        self.Hq2 = Q2.f01 * self.Nb2 + self.duff_part2 #  - self.Iq2*0
        self._int12 = self.B1*self.B2.dag() + self.B1.dag()*self.B2

        self.Hint12 = g12*(self._int12)
        self.Hint = self.Hint12
        self.Hlab = self.Hq1 + self.Hq2 + self.Hint
        self.calcStaticZZ(self.Hlab)
        self.fd1 = self.eigenlevels[0][self.keys['10']] - self.eigenlevels[0][self.keys['00']]
        self.fd2 = self.eigenlevels[0][self.keys['01']] - self.eigenlevels[0][self.keys['00']]

        # ref : https://doi.org/10.1103/PhysRevApplied.12.054023
        self.staticZZ = self.eigenlevels[0][self.keys['11']] - self.eigenlevels[0][self.keys['10']] - self.eigenlevels[0][self.keys['01']]

    def dressedEnergyLevels(self, H=None):
        if self.Nq1 == self.Nq2:
            Nq = self.Nq2
        else:
            print('Should be Nq1 = Nq2')
        if H == None:
            eigenlevels = self.Hlab.eigenstates()
        else:
            eigenlevels = H.eigenstates()
        keys = {}
        for i in range(Nq):
            for j in range(Nq):
                k = ket_2Qsys(i, j, Nq, Nq)
                e = np.abs([(k.dag() * eigenlevels[1])[i].tr() for i in range(Nq**2)])
                index = np.argmax(e)
                keys['{}{}'.format(i, j)] = index

        self.keys = keys
        self.eigenlevels = eigenlevels

    def plotDressedEnergyLevels(self, figname=1):
        if self.Nq1 == self.Nq2:
            Nq = self.Nq2
        else:
            print('Should be Nq1 = Nq2')
        d = self.keys
        enes = self.eigenlevels
        plt.figure(figname, dpi=150)
        plt.title(r'$|Q1, Q2\rangle$')
        for i in range(Nq):
            for j in range(Nq):
                key = '{}{}'.format(i,j)
                if key == '22':
                    break
                index = d[key]
                ene = enes[0][index]
                if i < j:#p
                    s = abs(i-j)
                    t = s+1
                elif i > j:#m
                    t = -abs(i-j)+1
                    s = t-1
                elif i == j:
                    s = 0
                    t = 1
                plt.hlines(ene, s, t)
                plt.text(s, ene+0.4, '|'+key+'>'+':{:.4f}GHz'.format(ene))

        plt.ylim([-1.0, ene+3])
        plt.ylabel('Eigen energy [GHz]')
        plt.xticks(color='None')
        plt.tick_params(length=0)
        plt.grid()

    def toRotFrameHamiltonian(self, fd, Scl=0, target=0):
        q1_rot = (self.Q1.f01-fd) * self.nb1 + 0.5 * self.Q1.anh * self.nb1 * (self.nb1 - self.iq1)
        q2_rot = (self.Q2.f01-fd) * self.nb2 + 0.5 * self.Q2.anh * self.nb2 * (self.nb2 - self.iq2)
        self.Hqrot = qt.tensor(q1_rot, self.iq2) + qt.tensor(self.iq1, q2_rot)
        if target == 0:
            Hdrive = self.B1 + self.B1.dag()
        else:
            Hdrive = self.B2 + self.B2.dag()
        return self.Hqrot + self.Hint + Scl * Hdrive

    def calcStaticZZ(self, H):
        self.dressedEnergyLevels(H=H)
        self.staticZZ = self.eigenlevels[0][self.keys['11']] - self.eigenlevels[0][self.keys['10']] - self.eigenlevels[0][self.keys['01']]
        return self.staticZZ


class QQQ():
    # For tunable coupling simulation
    def __init__(self, Q1, Q2, Qc, gc1, gc2, g12):
        # duffing oscillator model
        # Unit in [GHz]
        self.gc1 = gc1
        self.gc2 = gc2
        self.g12 = g12
        self.Q1 = Q1
        self.Q2 = Q2
        self.Qc = Qc
        self.Nq1, self.Nq2, self.Nqc = Q1.Nq, Q2.Nq, Qc.Nq

        iq1, iq2, iqc = qt.qeye(self.Nq1), qt.qeye(self.Nq2), qt.qeye(self.Nqc)
        b1, b2, bc = qt.destroy(self.Nq1), qt.destroy(self.Nq2), qt.destroy(self.Nqc)

        self.B1 = qt.tensor(b1, iq2, iqc)
        self.B2 = qt.tensor(iq1, b2, iqc)
        self.Bc = qt.tensor(iq1, iq2, bc)
        self.Iq1 = qt.tensor(iq1, iq2, iqc)
        self.Iq2 = qt.tensor(iq1, iq2, iqc)
        self.Iqc = qt.tensor(iq1, iq2, iqc)
        self.Nb1 = self.B1.dag()*self.B1
        self.Nb2 = self.B2.dag()*self.B2
        self.Nbc = self.Bc.dag()*self.Bc

        bbbb1 = self.B1.dag()*self.B1.dag()*self.B1*self.B1
        bbbb2 = self.B2.dag()*self.B2.dag()*self.B2*self.B2
        bbbbc = self.Bc.dag()*self.Bc.dag()*self.Bc*self.Bc
        self.duff_part1 = 0.5 * self.Q1.anh * self.Nb1 * (self.Nb1 - self.Iq1) # 0.5 * Q1.anh * bbbb1
        self.duff_part2 = 0.5 * self.Q2.anh * self.Nb2 * (self.Nb2 - self.Iq2) # 0.5 * Q2.anh * bbbb2
        self.duff_partc = 0.5 * self.Qc.anh * self.Nbc * (self.Nbc - self.Iqc) # 0.5 * Qc.anh * bbbbc
        self.Hq1 = Q1.f01 * self.Nb1 + self.duff_part1 #  - self.Iq1*0
        self.Hq2 = Q2.f01 * self.Nb2 + self.duff_part2 #  - self.Iq2*0
        self.Hqc = Qc.f01 * self.Nbc + self.duff_partc #  - self.Iqc*0
        self._intc1 = self.B1*self.Bc.dag() + self.B1.dag()*self.Bc
        self._intc2 = self.B2*self.Bc.dag() + self.B2.dag()*self.Bc
        self._int12 = self.B1*self.B2.dag() + self.B1.dag()*self.B2
        # self._intc1 = (self.B1 + self.B1.dag())*(self.Bc + self.Bc.dag())
        # self._intc2 = (self.B2 + self.B2.dag())*(self.Bc + self.Bc.dag())
        # self._int12 = (self.B1 + self.B1.dag())*(self.B2 + self.B2.dag())
        self.Hintc1 = gc1*self._intc1
        self.Hintc2 = gc2*self._intc2
        self.Hint12 = g12*self._int12
        self.Hint = self.Hintc1 + self.Hintc2 + self.Hint12
        self.Hlab = self.Hq1 + self.Hq2 + self.Hqc + self.Hint
        self.eigenlevels = self.Hlab.eigenstates()
        self.dressedEnergyLevels()
        self.fd1 = self.eigenlevels[0][self.keys['100']] - self.eigenlevels[0][self.keys['000']]
        self.fd2 = self.eigenlevels[0][self.keys['010']] - self.eigenlevels[0][self.keys['000']]

        # ref : https://doi.org/10.1103/PhysRevApplied.12.054023
        self.staticZZ = self.eigenlevels[0][self.keys['110']] - self.eigenlevels[0][self.keys['100']] - self.eigenlevels[0][self.keys['010']]
        self.effectiveCoupling = gc1*gc2*(1/(Q1.f01-Qc.f01)+1/(Q2.f01-Qc.f01))*0.5 + g12

    def dressedEnergyLevels(self):
        if self.Nq1 == self.Nq2:
            Nq = self.Nq2
        else:
            print('Should be Nq1 = Nq2')
        eigenlevels = self.eigenlevels
        keys = {}
        for i in range(Nq):
            for j in range(Nq):
                for k in range(Nq):
                    bra = ket_3Qsys(i, j, k, Nq, Nq, Nq).dag()
                    e = np.abs([(bra * eigenlevels[1])[i].tr() for i in range(Nq**3)])
                    index = np.argmax(e)
                    keys['{}{}{}'.format(i, j, k)] = index

        self.keys = keys

    def plotDressedEnergyLevels(self, coupler_exitation_stop=0):
        # coupler_exitation_stop : coupler exitation number to be plotted.
        ces = coupler_exitation_stop
        if self.Nq1 == self.Nq2:
            Nq = self.Nq2
        else:
            print('Should be Nq1 = Nq2')
        d = self.keys
        enes = self.eigenlevels
        plt.figure(1, dpi=150)
        cmap = plt.get_cmap("tab10")
        plt.title(r'$|Q1, Q2, Qc\rangle$')
        for i in range(Nq):
            for j in range(Nq):
                for k in range(Nq):
                    key = '{}{}{}'.format(i, j, k)
                    if key == '220' or k > ces:
                        break
                    index = d[key]
                    ene = enes[0][index]
                    if i < j:#p
                        s = abs(i-j)
                        t = s+1
                    elif i > j:#m
                        t = -abs(i-j)+1
                        s = t-1
                    elif i == j:
                        s = 0
                        t = 1
                    plt.hlines(ene, s, t, color=cmap(k))
                    plt.text(s, ene+0.4, '|'+key+r'$\rangle$'+':{:.3f}GHz'.format(ene))
        plt.ylim([-1.0, ene+3])
        plt.ylabel('Eigen energy [GHz]')
        plt.xticks(color='None')
        plt.tick_params(length=0)
        plt.grid()

class RQRQR():
    def __init__(self, QR1, QR2, frb, g1, g2):
        # Unit in [GHz]
        self.frb = frb
        self.g1 = g1
        self.g2 = g2
        self.QR1 = QR1
        self.QR2 = QR2
        self.detuning1 = QR1.f01_dressed - frb
        self.thermal_photon1 = qt.utilities.n_thermal(frb, QR1.f01_dressed)
        self.f01_dressed1 = QR1.f01_dressed + ( 2 * (g1**2) / self.detuning1 ) * ( self.thermal_photon1 + 1/2 )
        self.X1 = ((g1**2)/self.detuning1)*(QR1.Q.anh/(QR1.f01_dressed + QR1.Q.anh - frb))

        self.detuning2 = QR2.f01_dressed - frb
        self.thermal_photon2 = qt.utilities.n_thermal(frb, QR2.f01_dressed)
        self.f01_dressed2 = QR2.f01_dressed + ( 2 * (g2**2) / self.detuning2 ) * ( self.thermal_photon2 + 1/2 )
        self.X2 = ((g2**2)/self.detuning2)*(QR2.Q.anh/(QR2.f01_dressed + QR2.Q.anh - frb))

        self.D12 = self.f01_dressed1 - self.f01_dressed2
        self.J = g1*g2*( self.detuning1 + self.detuning2 ) / ( 2 * self.detuning1 * self.detuning2 )
        self.f01_coupled1 = self.f01_dressed1 + (self.J**2)/self.D12
        self.f01_coupled2 = self.f01_dressed2 - (self.J**2)/self.D12


class labFrame2Qhamiltonian_DuffingOscillator():
    def __init__(self, RQRQR, Nq1, Nq2):
        self.Nq1, self.Nq2 = Nq1, Nq2

        Iq1, Iq2 = qt.qeye(Nq1), qt.qeye(Nq2)
        b1, b2 = qt.destroy(Nq1), qt.destroy(Nq2)
        Nb1, Nb2 = b1.dag()*b1, b2.dag()*b2

        self.X1 = qt.tensor(pX_N(Nq1), Iq2)
        self.Y1 = qt.tensor(pY_N(Nq1), Iq2)
        self.Z1 = qt.tensor(pZ_N(Nq1), Iq2)

        self.X2 = qt.tensor(Iq1, pX_N(Nq2))
        self.Y2 = qt.tensor(Iq1, pY_N(Nq2))
        self.Z2 = qt.tensor(Iq1, pZ_N(Nq2))

        self.Iq1, self.Iq2 = Iq1, Iq2
        self.Nb1, self.Nb2 = Nb1, Nb2
        self.QR1 = RQRQR.QR1
        self.QR2 = RQRQR.QR2
        J = RQRQR.J

        self.B1 = qt.tensor(b1, Iq2)
        self.B2 = qt.tensor(Iq1, b2)
        bbbb1 = b1.dag()*b1.dag()*b1*b1
        bbbb2 = b2.dag()*b2.dag()*b2*b2
        # Drive term @rotating frame
        self.Hd1_real = self.B1 + self.B1.dag()
        self.Hd1_imag = (- self.B1 + self.B1.dag())*1j
        self.Hd2_real = (self.B2 + self.B2.dag())
        self.Hd2_imag = (- self.B2 + self.B2.dag())*1j

        q1_lab = self.QR1.f01_dressed * Nb1 + 0.5 * self.QR1.Q.anh * Nb1 * (Nb1 - Iq1)
        q2_lab = self.QR2.f01_dressed * Nb2 + 0.5 * self.QR2.Q.anh * Nb2 * (Nb2 - Iq2)
        self.Hqlab = qt.tensor(q1_lab, Iq2) + qt.tensor(Iq1, q2_lab)
        self.Hint = J * ( qt.tensor(b1, b2.dag()) + qt.tensor(b1.dag(), b2) )
        self.Hlab = self.Hqlab + self.Hint

        self.dressedEnergyLevels()
        self.fd1 = self.eigenlevels[0][self.keys['10']] - self.eigenlevels[0][self.keys['00']]
        self.fd2 = self.eigenlevels[0][self.keys['01']] - self.eigenlevels[0][self.keys['00']]

    def dressedEnergyLevels(self):
        if self.Nq1 == self.Nq2:
            Nq = self.Nq2
        else:
            print('Should be Nq1 = Nq2')
        eigenlevels = self.Hlab.eigenstates()
        keys = {}
        for i in range(Nq):
            for j in range(Nq):
                k = ket_2Qsys(i, j, Nq, Nq)
                e = np.abs([(k.dag() * eigenlevels[1])[i].tr() for i in range(Nq**2)])
                index = np.argmax(e)
                keys['{}{}'.format(i, j)] = index

        self.keys = keys
        self.eigenlevels = eigenlevels

    def plotDressedEnergyLevels(self):
        if self.Nq1 == self.Nq2:
            Nq = self.Nq2
        else:
            print('Should be Nq1 = Nq2')
        d = self.keys
        enes = self.eigenlevels
        plt.figure(1)
        for i in range(Nq):
            for j in range(Nq):
                key = '{}{}'.format(i,j)
                if key == '22':
                    break
                index = d[key]
                ene = enes[0][index]
                if i < j:#p
                    s = abs(i-j)
                    t = s+1
                elif i > j:#m
                    t = -abs(i-j)+1
                    s = t-1
                elif i == j:
                    s = 0
                    t = 1
                plt.hlines(ene, s, t)
                plt.text(s, ene+0.4, '|'+key+'>'+':{:.3f}GHz'.format(ene))

        plt.ylim([-1.0, ene+3])
        plt.ylabel('Eigen energy [GHz]')
        plt.xticks(color='None')
        plt.tick_params(length=0)
        plt.grid()

    def toRotFrameHamiltonian(self, fd:float):
        Nb1, Nb2 = self.Nb1, self.Nb2
        Iq1, Iq2 = self.Iq1, self.Iq2

        q1_rot = (self.QR1.f01_dressed-fd) * Nb1 + 0.5 * self.QR1.Q.anh * Nb1 * (Nb1 - Iq1)
        q2_rot = (self.QR2.f01_dressed-fd) * Nb2 + 0.5 * self.QR2.Q.anh * Nb2 * (Nb2 - Iq2)
        self.Hqrot = qt.tensor(q1_rot, self.Iq2) + qt.tensor(self.Iq1, q2_rot)
        return self.Hqrot + self.Hint

    def toDoublyRotFrameHamiltonian(self, fd1:float, fd2:float):
        Nb1, Nb2 = self.Nb1, self.Nb2
        Iq1, Iq2 = self.Iq1, self.Iq2

        q1_rot = (self.QR1.f01_dressed-fd1) * Nb1 + 0.5 * self.QR1.Q.anh * Nb1 * (Nb1 - Iq1)
        q2_rot = (self.QR2.f01_dressed-fd2) * Nb2 + 0.5 * self.QR2.Q.anh * Nb2 * (Nb2 - Iq2)
        self.Hqrot = qt.tensor(q1_rot, self.Iq2) + qt.tensor(self.Iq1, q2_rot)
        return self.Hqrot + self.Hint

class labFrame1Qhamiltonian_DuffingOscillator():
    def __init__(self, QR, Nq):
        self.Nq = Nq

        Iq = qt.qeye(Nq)
        b = qt.destroy(Nq)
        Nb = b.dag()*b

        self.X = pX_N(Nq)
        self.Y = pY_N(Nq)
        self.Z = pZ_N(Nq)

        self.Iq = Iq
        self.Nb = Nb
        self.QR = QR

        self.B = b
        # Drive term @rotating frame
        self.f01_dressed = QR.f01_dressed
        self.Hd1_real = self.B + self.B.dag()
        self.Hd1_imag = (- self.B + self.B.dag())*1j
        q1_lab = self.QR.f01_dressed * Nb + 0.5 * self.QR.Q.anh * Nb * (Nb - Iq)
        self.Hqlab = q1_lab
        self.Hlab = self.Hqlab

    def calcUrot(self, t_list):
        Urots = []
        for t in t_list:
            u = (1j*self.f01_dressed*t*self.Nb).expm()
            Urots.append(u)
        return Urots

class labFrame1Q_1R_hamiltonian():
    def __init__(self, Q, R, g):
        """
        params
        ---
        Q : class instance
            transmon()
        R : class instance
            resonator()
        g : float in [GHz]
            coupling constant
        """
        self.Nq = Q.Nq
        self.Nf = R.Nf
        self.Ir = Ir = R.Ir
        self.Iq = Iq = Q.Iq
        self.II = qt.tensor(Iq, Ir)

        self.f01 = Q.f01
        self.anh = Q.anh
        self.fr = R.fr
        self.g = g
        self.Q = Q
        self.R = R
        self.detuning = Q.f01 - R.fr
        # self.thermal_photon = qt.utilities.n_thermal(self.fr, Q.f01)
        # self.f01_dressed = Q.f01 + ( 2 * (g**2) / self.detuning ) * ( self.thermal_photon + 1/2 )

        self.X = qt.tensor(Q.X, Ir)
        self.Y = qt.tensor(Q.Y, Ir)
        self.Z = qt.tensor(Q.Z, Ir)
        self.P0 = qt.tensor(Q.P0, Ir)
        self.P1 = qt.tensor(Q.P1, Ir)

        self.Na = qt.tensor(Iq, R.na)
        self.Nb = qt.tensor(Q.nb, Ir)
        self.A = A = qt.tensor(Iq, R.a)
        self.B = B = qt.tensor(Q.b, Ir)

        self.HQ1 = qt.tensor(Q.Hqlab, Ir)
        self.HR1 = qt.tensor(Iq, R.Hr)
        self.Hint = g * ( B*A.dag() + B.dag()*A )
        self.Hd1_real = A + A.dag()
        self.Hd1_imag = (- A + A.dag())*1j

        self.Hlab = self.HQ1 + self.HR1 + self.Hint
        self.dressedEnergyLevels()
        self.fq10 = self.eigenlevels[0][self.keys['10']] - self.eigenlevels[0][self.keys['00']]
        self.fq11 = self.eigenlevels[0][self.keys['11']] - self.eigenlevels[0][self.keys['01']]
        self.fr0 = self.eigenlevels[0][self.keys['01']] - self.eigenlevels[0][self.keys['00']]
        self.fr1 = self.eigenlevels[0][self.keys['11']] - self.eigenlevels[0][self.keys['10']]

    def calcUrot(self, t_list, fd):
        Urots = []
        for t in t_list:
            u = (1j*fd*(self.Na + self.Nb)*t).expm()
            Urots.append(u)
        return Urots

    def addDecoherence(self):
        pass
        return

    def calcDispersiveShift(self):
        eigenlevels = self.Hlab.eigenstates()
        e0 = qt.tensor(qt.basis(self.Nq, 1), qt.fock(self.Nf, 0))
        g1 = qt.tensor(qt.basis(self.Nq, 0), qt.fock(self.Nf, 1))
        e1 = qt.tensor(qt.basis(self.Nq, 1), qt.fock(self.Nf, 1))
        ket_try = [e0, g1, e1]
        ket_keys = ['e0', 'g1', 'e1']
        disp_dic = {}
        for i in range(3):
            e = np.abs([(ket_try[i].dag() * eigenlevels[1])[j].tr() for j in range(self.Nq*self.Nf)])
            index = np.argmax(e)
            disp_dic[ket_keys[i]] = eigenlevels[0][index]

        disp_dic['chi'] = (disp_dic['e1'] - disp_dic['e0'] - disp_dic['g1'])/2
        self.dispersiveshift = disp_dic
        return disp_dic

    def toRotFrameHamiltonian(self, fd:float):
        q1_rot = (self.f01-fd) * self.Nb + 0.5 * self.Q.anh * self.Nb * (self.Nb - self.II)
        r1_rot = (self.fr-fd) * self.Na
        self.Hrot = q1_rot + r1_rot + self.Hint
        return self.Hrot
    
    def dressedEnergyLevels(self, H=None):
        Nq = self.Nq
        Nf = self.Nf
        if H == None:
            eigenlevels = self.Hlab.eigenstates()
        else:
            eigenlevels = H.eigenstates()
        keys = {}
        for i in range(Nq):
            for j in range(2):
                k = ket_2Qsys(i, j, Nq, Nf)
                e = np.abs([(k.dag() * eigenlevels[1])[i].tr() for i in range(Nq*Nf)])
                index = np.argmax(e)
                keys['{}{}'.format(i, j)] = index

        self.keys = keys
        self.eigenlevels = eigenlevels

    def plotDressedEnergyLevels(self, figname=1):
        Nq = self.Nq
        Nf = self.Nf
        d = self.keys
        enes = self.eigenlevels
        plt.figure(figname, dpi=150)
        plt.title(r'$|Transmon, Resonator\rangle$')
        for i in range(Nq):
            for j in range(2):
                key = '{}{}'.format(i,j)
                if key == '22':
                    break
                index = d[key]
                ene = enes[0][index]
                if i < j:#p
                    s = abs(i-j)
                    t = s+1
                elif i > j:#m
                    t = -abs(i-j)+1
                    s = t-1
                elif i == j:
                    s = 0
                    t = 1
                plt.hlines(ene, s, t)
                plt.text(s, ene+0.4, '|'+key+r'$\rangle$'+':{:.4f}GHz'.format(ene))

        plt.ylim([-1.0, ene+3])
        plt.ylabel('Eigen energy [GHz]')
        plt.xticks(color='None')
        plt.tick_params(length=0)
        plt.grid()

class timeEvo():
    def __init__(self):
        return 0
