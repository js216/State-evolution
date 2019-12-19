import numpy as np
from numpy import sqrt

# Units and constants

I_Tl = 1/2    # I1 in Ramsey's notation
I_F = 1/2     # I2 in Ramsey's notation

# TlF constants. Data from D.A. Wilkening, N.F. Ramsey,
# and D.J. Larson, Phys Rev A 29, 425 (1984). Everything in Hz.

Brot = 6689920000 # Hz
c1 = 126030.0     # Hz
c2 = 17890.0      # Hz
c3 = 700.0        # Hz
c4 = -13300.0     # Hz
mu_J = 35         # Hz/gauss
mu_Tl = 1240.5    # Hz/gauss
mu_F = 2003.63    # Hz/gauss
D_TlF = 4.2282 * 0.393430307 *5.291772e-9/4.135667e-15 # [Hz/(V/cm)]

class BasisState:
    # constructor
    def __init__(self, J, mJ, I1, m1, I2, m2):
        self.J, self.mJ  = J, mJ
        self.I1, self.m1 = I1, m1
        self.I2, self.m2 = I2, m2

    # equality testing
    def __eq__(self, other):
        return self.J==other.J and self.mJ==other.mJ \
                    and self.I1==other.I1 and self.I2==other.I2 \
                    and self.m1==other.m1 and self.m2==other.m2

    # inner product
    def __matmul__(self, other):
        if self == other:
            return 1
        else:
            return 0

    # superposition: addition
    def __add__(self, other):
        if self == other:
            return State([ (2,self) ])
        else:
            return State([ (1,self), (1,other) ])

    # superposition: subtraction
    def __sub__(self, other):
        return self + -1*other

    # scalar product (psi * a)
    def __mul__(self, a):
        return State([ (a, self) ])

    # scalar product (a * psi)
    def __rmul__(self, a):
        return self * a

    def print_quantum_numbers(self):
        print( self.J,"%+d"%self.mJ,"%+0.1f"%self.m1,"%+0.1f"%self.m2 )

class State:
    # constructor
    def __init__(self, data=[], remove_zero_amp_cpts=True):
        # check for duplicates
        for i in range(len(data)):
            amp1,cpt1 = data[i][0], data[i][1]
            for amp2,cpt2 in data[i+1:]:
                if cpt1 == cpt2:
                    raise AssertionError("duplicate components!")
        # remove components with zero amplitudes
        if remove_zero_amp_cpts:
            self.data = [(amp,cpt) for amp,cpt in data if amp!=0]
        else:
            self.data = data
        # for iteration over the State
        self.index = len(self.data)

    # superposition: addition
    # (highly inefficient and ugly but should work)
    def __add__(self, other):
        data = []
        # add components that are in self but not in other
        for amp1,cpt1 in self.data:
            only_in_self = True
            for amp2,cpt2 in other.data:
                if cpt2 == cpt1:
                    only_in_self = False
            if only_in_self:
                data.append((amp1,cpt1))
        # add components that are in other but not in self
        for amp1,cpt1 in other.data:
            only_in_other = True
            for amp2,cpt2 in self.data:
                if cpt2 == cpt1:
                    only_in_other = False
            if only_in_other:
                data.append((amp1,cpt1))
        # add components that are both in self and in other
        for amp1,cpt1 in self.data:
            for amp2,cpt2 in other.data:
                if cpt2 == cpt1:
                    data.append((amp1+amp2,cpt1))
        return State(data)

    # superposition: subtraction
    def __sub__(self, other):
        return self + -1*other

    # scalar product (psi * a)
    def __mul__(self, a):
        return State( [(a*amp,psi) for amp,psi in self.data] )

    # scalar product (a * psi)
    def __rmul__(self, a):
        return self * a

    # scalar division (psi / a)
    def __truediv__(self, a):
        return self * (1/a)

    # negation
    def __neg__(self):
        return -1.0 * self

    # inner product
    def __matmul__(self, other):
        result = 0
        for amp1,psi1 in self.data:
            for amp2,psi2 in other.data:
                result += amp1.conjugate()*amp2 * (psi1@psi2)
        return result

    # iterator methods
    def __iter__(self):
        return self

    def __next__(self):
        if self.index == 0:
            raise StopIteration
        self.index -= 1
        return self.data[self.index]

    # direct access to a component
    def __getitem__(self, i):
        return self.data[i]

def J2(psi):
    return State([(psi.J*(psi.J+1),psi)])

def Jz(psi):
    return State([(psi.mJ,psi)])

def I1z(psi):
    return State([(psi.m1,psi)])

def I2z(psi):
    return State([(psi.m2,psi)])

def Jp(psi):
    amp = sqrt((psi.J-psi.mJ)*(psi.J+psi.mJ+1))
    ket = BasisState(psi.J, psi.mJ+1, psi.I1, psi.m1, psi.I2, psi.m2)
    return State([(amp,ket)])

def Jm(psi):
    amp = sqrt((psi.J+psi.mJ)*(psi.J-psi.mJ+1))
    ket = BasisState(psi.J, psi.mJ-1, psi.I1, psi.m1, psi.I2, psi.m2)
    return State([(amp,ket)])

def I1p(psi):
    amp = sqrt((psi.I1-psi.m1)*(psi.I1+psi.m1+1))
    ket = BasisState(psi.J, psi.mJ, psi.I1, psi.m1+1, psi.I2, psi.m2)
    return State([(amp,ket)])

def I1m(psi):
    amp = sqrt((psi.I1+psi.m1)*(psi.I1-psi.m1+1))
    ket = BasisState(psi.J, psi.mJ, psi.I1, psi.m1-1, psi.I2, psi.m2)
    return State([(amp,ket)])

def I2p(psi):
    amp = sqrt((psi.I2-psi.m2)*(psi.I2+psi.m2+1))
    ket = BasisState(psi.J, psi.mJ, psi.I1, psi.m1, psi.I2, psi.m2+1)
    return State([(amp,ket)])

def I2m(psi):
    amp = sqrt((psi.I2+psi.m2)*(psi.I2-psi.m2+1))
    ket = BasisState(psi.J, psi.mJ, psi.I1, psi.m1, psi.I2, psi.m2-1)
    return State([(amp,ket)])

def Jx(psi):
    return .5*( Jp(psi) + Jm(psi) )

def Jy(psi):
    return -.5j*( Jp(psi) - Jm(psi) )

def I1x(psi):
    return .5*( I1p(psi) + I1m(psi) )

def I1y(psi):
    return -.5j*( I1p(psi) - I1m(psi) )

def I2x(psi):
    return .5*( I2p(psi) + I2m(psi) )

def I2y(psi):
    return -.5j*( I2p(psi) - I2m(psi) )

def com(A, B, psi):
    ABpsi = State()
    # operate with A on all components in B|psi>
    for amp,cpt in B(psi):
        ABpsi += amp * A(cpt)
    return ABpsi

def Hrot(psi):
    return Brot * J2(psi)

def Hc1(psi):
    return c1 * ( com(I1z,Jz,psi) + .5*(com(I1p,Jm,psi)+com(I1m,Jp,psi)) )

def Hc2(psi):
    return c2 * ( com(I2z,Jz,psi) + .5*(com(I2p,Jm,psi)+com(I2m,Jp,psi)) )

def Hc4(psi):
    return c4 * ( com(I1z,I2z,psi) + .5*(com(I1p,I2m,psi)+com(I1m,I2p,psi)) )

def Hc3a(psi):
    return 15*c3/c1/c2 * com(Hc1,Hc2,psi) / ((2*psi.J+3)*(2*psi.J-1))

def Hc3b(psi):
    return 15*c3/c1/c2 * com(Hc2,Hc1,psi) / ((2*psi.J+3)*(2*psi.J-1))

def Hc3c(psi):
    return -10*c3/c4/Brot * com(Hc4,Hrot,psi) / ((2*psi.J+3)*(2*psi.J-1))

def Hff(psi):
    return Hrot(psi) + Hc1(psi) + Hc2(psi) + Hc3a(psi) + Hc3b(psi) \
            + Hc3c(psi) + Hc4(psi)

def HZx(psi):
    if psi.J != 0:
        return -mu_J/psi.J*Jx(psi) - mu_Tl/psi.I1*I1x(psi) - mu_F/psi.I2*I2x(psi)
    else:
        return -mu_Tl/psi.I1*I1x(psi) - mu_F/psi.I2*I2x(psi)

def HZy(psi):
    if psi.J != 0:
        return -mu_J/psi.J*Jy(psi) - mu_Tl/psi.I1*I1y(psi) - mu_F/psi.I2*I2y(psi)
    else:
        return -mu_Tl/psi.I1*I1y(psi) - mu_F/psi.I2*I2y(psi)

def HZz(psi):
    if psi.J != 0:
        return -mu_J/psi.J*Jz(psi) - mu_Tl/psi.I1*I1z(psi) - mu_F/psi.I2*I2z(psi)
    else:
        return -mu_Tl/psi.I1*I1z(psi) - mu_F/psi.I2*I2z(psi)

def R10(psi):
    amp1 = sqrt((psi.J-psi.mJ)*(psi.J+psi.mJ)/(8*psi.J**2-2))
    ket1 = BasisState(psi.J-1, psi.mJ, psi.I1, psi.m1, psi.I2, psi.m2)
    amp2 = sqrt((psi.J-psi.mJ+1)*(psi.J+psi.mJ+1)/(6+8*psi.J*(psi.J+2)))
    ket2 = BasisState(psi.J+1, psi.mJ, psi.I1, psi.m1, psi.I2, psi.m2)
    return State([(amp1,ket1),(amp2,ket2)])

def R1m(psi):
    amp1 = -.5*sqrt((psi.J+psi.mJ)*(psi.J+psi.mJ-1)/(4*psi.J**2-1))
    ket1 = BasisState(psi.J-1, psi.mJ-1, psi.I1, psi.m1, psi.I2, psi.m2)
    amp2 = .5*sqrt((psi.J-psi.mJ+1)*(psi.J-psi.mJ+2)/(3+4*psi.J*(psi.J+2)))
    ket2 = BasisState(psi.J+1, psi.mJ-1, psi.I1, psi.m1, psi.I2, psi.m2)
    return State([(amp1,ket1),(amp2,ket2)])

def R1p(psi):
    amp1 = -.5*sqrt((psi.J-psi.mJ)*(psi.J-psi.mJ-1)/(4*psi.J**2-1))
    ket1 = BasisState(psi.J-1, psi.mJ+1, psi.I1, psi.m1, psi.I2, psi.m2)
    amp2 = .5*sqrt((psi.J+psi.mJ+1)*(psi.J+psi.mJ+2)/(3+4*psi.J*(psi.J+2)))
    ket2 = BasisState(psi.J+1, psi.mJ+1, psi.I1, psi.m1, psi.I2, psi.m2)
    return State([(amp1,ket1),(amp2,ket2)])

def HSx(psi):
    return -D_TlF * ( R1m(psi) - R1p(psi) )

def HSy(psi):
    return -D_TlF * 1j * ( R1m(psi) + R1p(psi) )

def HSz(psi):
    return -D_TlF * sqrt(2)*R10(psi)

def QN(Jmax):
    return np.array([BasisState(J,mJ,I_Tl,m1,I_F,m2)
      for J in range(Jmax+1)
      for mJ in range(-J,J+1)
      for m1 in np.arange(-I_Tl,I_Tl+1)
      for m2 in np.arange(-I_F,I_F+1)])

def HMatElems(H, QN):
    result = np.empty((len(QN),len(QN)), dtype=complex)
    for i,a in enumerate(QN):
        for j,b in enumerate(QN):
            result[i,j] = (1*a)@H(b)
    return result

def H_mat_elem(Jmax):
    # generate matrix elements
    Hff_m = HMatElems(Hff, QN(Jmax))
    HSx_m = HMatElems(HSx, QN(Jmax))
    HSy_m = HMatElems(HSy, QN(Jmax))
    HSz_m = HMatElems(HSz, QN(Jmax))
    HZx_m = HMatElems(HZx, QN(Jmax))
    HZy_m = HMatElems(HZy, QN(Jmax))
    HZz_m = HMatElems(HZz, QN(Jmax))

    return Hff_m, HSx_m, HSy_m, HSz_m, HZx_m, HZy_m, HZz_m

def H_eff(H_arr):
    # diagonalize
    e_vals, e_states = np.linalg.eigh(H_arr)

    # truncate and subtract rotational energy
    Delta = e_vals[:,16:36] - 6*Brot
    V = e_states[:, 16:36, 16:36]

    # transform back to non-diagonal basis
    return np.einsum("aij,aj,ajl -> ail", V, Delta, np.linalg.inv(V))

def fit_Heff(field_arr, H_orig):
    # give fields sensible names
    Ex, Ey, Ez = field_arr[:,0], field_arr[:,1], field_arr[:,2]
    Bx, By, Bz = field_arr[:,3], field_arr[:,4], field_arr[:,5]

    # define list of potential parameters
    A = np.array([
        Ex*0+1,
        Ex, Ey, Ez, Bx, By, Bz,
        Ex**2, Ey**2, Ez**2, Bx**2, By**2, Bz**2,
        Ex*Ey, Ex*Ez, Ey*Ez,
        Bx*By, Bx*Bz, By*Bz,
        Ex**3, Ey**3, Ez**3, Bx**3, By**3, Bz**3,
        Ex**2*Ey, Ex**2*Ez, Ey**2*Ez, Ex*Ey**2, Ex*Ez**2, Ey*Ez**2,
        Bx**2*By, Bx**2*Bz, By**2*Bz, Bx*By**2, Bx*Bz**2, By*Bz**2,
    ]).T

    # find H_eff for each field value
    B = H_eff(H_orig(field_arr)).reshape(-1, 400)

    # do least-squares fit
    coeffs, _, _, _ = np.linalg.lstsq(A, B, rcond=None)

    return coeffs.reshape(-1, 20, 20)

def Heff_fit(field_arr, fp):
    # give fields sensible names
    Ex, Ey, Ez = field_arr[:,0], field_arr[:,1], field_arr[:,2]
    Bx, By, Bz = field_arr[:,3], field_arr[:,4], field_arr[:,5]

    # use field parameters to obtain matrices
    constant  = fp[np.newaxis,0,:,:]
    linear    = np.einsum("ai,ijk", np.array(field_arr),                      fp[1:7,:,:])
    quadratic = np.einsum("ai,ijk", np.array(field_arr)**2,                   fp[7:13,:,:])
    cross_E   = np.einsum("ia,ijk", np.array([Ex*Ey, Ex*Ez, Ey*Ez]),          fp[13:16,:,:])
    cross_B   = np.einsum("ia,ijk", np.array([Bx*By, Bx*Bz, By*Bz]),          fp[16:19,:,:])
    cubic     = np.einsum("ai,ijk", np.array(field_arr)**3,                   fp[19:25,:,:])
    cross_E2a = np.einsum("ia,ijk", np.array([Ex**2*Ey, Ex**2*Ez, Ey**2*Ez]), fp[25:28,:,:])
    cross_E2b = np.einsum("ia,ijk", np.array([Ex*Ey**2, Ex*Ez**2, Ey*Ez**2]), fp[28:31,:,:])
    cross_B2a = np.einsum("ia,ijk", np.array([Bx**2*By, Bx**2*Bz, By**2*Bz]), fp[31:34,:,:])
    cross_B2b = np.einsum("ia,ijk", np.array([Bx*By**2, Bx*Ez**2, By*Bz**2]), fp[34:37,:,:])

    return constant + linear + quadratic + cross_E + cross_B \
                    + cubic + cross_E2a + cross_E2b + cross_B2a + cross_B2b

def generate_Hamiltonian(Jmax, fname):
    np.save(fname, H_mat_elem(Jmax))

def load_H_mat(fname):
    Hff_m, HSx_m, HSy_m, HSz_m, HZx_m, HZy_m, HZz_m = np.load(fname)
    H_m = np.stack([Hff_m, HSx_m, HSy_m, HSz_m, HZx_m, HZy_m, HZz_m], axis=2)
    return H_m

def load_Hamiltonian(fname):
    Hff_m, HSx_m, HSy_m, HSz_m, HZx_m, HZy_m, HZz_m = np.load(fname)
    Hfield_m = np.stack([HSx_m, HSy_m, HSz_m, HZx_m, HZy_m, HZz_m], axis=2)
    return lambda fields : Hff_m + np.einsum("ax,ijx->aij", fields, Hfield_m)
