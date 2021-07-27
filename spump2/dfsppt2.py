"""
V13 
    - This version does UHF -> UMP2 -> P.
    - Using simplified projector as default. 
    - CO version using Van Voorhis' equations.
    - Using DF-GMP2.
        - But calculating MO eris without df.
        
    - Optimize with einsum.
"""

from functools import reduce
from itertools import combinations
import ctypes
import numpy
import scipy.special

from pyscf import lib
from pyscf.lib import logger
from pyscf import gto, scf, mp, fci, ao2mo
from pyscf.scf import hf
from pyscf import df
from pyscf.ao2mo import _ao2mo

from . import lib as tools
libnp_helper = tools.load_library("libnp_helper")

SWITCH_SIZE = 800 #need test

def ao2mo_slow(eri_ao, mo_coeffs):
    nao = mo_coeffs[0].shape[0]
    eri_ao_s1 = ao2mo.restore(1, eri_ao, nao)
    return lib.einsum('pqrs,pi,qj,rk,sl->ijkl', eri_ao_s1.reshape([nao]*4),
                      mo_coeffs[0].conj(), mo_coeffs[1],
                      mo_coeffs[2].conj(), mo_coeffs[3])

class SPPT2:
    def __init__(self, mf):
        """
        Attributes
            mol (gto Mole object): Stores the molecule.
            
            mo_eri (ndarray): ERIs in MO basis.
            
            ao_ovlp (ndarray): Overlap matrix S in AO basis.
            
            quad (ndarray): Array of [alphas, betas, ws], storing quadrature points 
                            `alphas, betas` and weights `ws`.
                                  
            nelec (int): Number of electrons.
            
            nmo (int): Total number of spatial orbitals (occupied + unoccupied).
            
            S (float): Spin quantum number S = (n_alpha - n_beta) / 2.
            
            e_tot (float): Total energy.
            
            e_hf (float): HF energy.
            
            e_elec_hf (float): Electronic part of HF energy.
        """
        self._scf = mf
        self.mol = mf.mol
        if getattr(mf, 'with_df', None):
            self.with_df = mf.with_df
        else:
            self.with_df = df.DF(mf.mol)
            self.with_df.auxbasis = df.make_auxbasis(mf.mol, mp2fit=True)
        self.max_memory = mf.max_memory
        self.verbose = mf.verbose
        self.stdout = mf.stdout

        self.rotated_mo_coeffs = None
        self.mo_eri = None
        self.ao_ovlp = None
        self.s1e = None
        self.quad = None
        self.nelec = self.mol.nelectron
        self.nocc = self.nelec
        self.nmo = 2 * self.mol.nao # Number of spatial MOs.
        self.S = self.mol.spin / 2 # mol.spin = n_alpha - n_beta
        self.e_tot = None
        self.e_hf = None
        self.e_elec_hf = None

    def generate_quad_part(self, N_beta=None, verbose=False):
        """
        Generate quadrature points and weights.
        """
        # For beta.
        # Number of singly occupied orbitals.
        Omega_max = 0
        
        if self.nelec <= self.nmo:
            Omega_max = self.nelec
        
        elif self.nelec > self.nmo:
            Omega_max = 2 * self.nmo - self.nelec
            
        # Number of points.
        thresh = int(numpy.ceil((Omega_max/2 + self.S + 1) / 2))
        
        if N_beta is None:
            N_beta = thresh
        
        if N_beta < thresh:
            print(f"Ng less than threshold {thresh}.")
            N_beta = thresh
        
        # Quadrature points and weights.
        # x = cos(beta) in [-1, 1]
        xs, ws = numpy.polynomial.legendre.leggauss(N_beta)
        betas = numpy.arccos(xs)
        sorted_inds = numpy.argsort(betas) # sort in order of increasing beta.
        betas.sort()
        ws = ws[sorted_inds]
        
        if verbose:
            print(f'betas: \n{betas}\n')
            print(f'ws: \n{ws}\n')
                  
        return betas, ws
 
    def generate_quad_full(self, N_alpha=None, N_beta=None, N_gamma=None, verbose=False):
        """
        Generate quadrature points and weights. For integration over alpha and gamma, we use 
        a Trapezoid grid; for integration over beta, we use a Gauss-Legendre grid.
        """
        # beta.
        # Number of singly occupied orbitals.
        Omega_max = 0
        
        if self.nelec <= self.nmo:
            Omega_max = self.nelec
        
        elif self.nelec > self.nmo:
            Omega_max = 2 * self.nmo - self.nelec
            
        # Minimum number of points.
        thresh = int(numpy.ceil((Omega_max/2 + self.S + 1) / 2))
        
        if N_beta is None:
            N_beta = thresh
        
        if N_beta < thresh:
            print(f"Ng less than threshold {thresh}.")
            N_beta = thresh
        
        # Quadrature points and weights.
        # x = cos(beta) in [-1, 1]
        xs, ws = numpy.polynomial.legendre.leggauss(N_beta)
        betas = numpy.arccos(xs)
        sorted_inds = numpy.argsort(betas)
        betas.sort()
        ws = ws[sorted_inds]
        
        # alpha.
        if N_alpha is None:
            N_alpha = 2 * N_beta
            
        alphas = numpy.linspace(2 * numpy.pi / N_alpha, 2*numpy.pi, N_alpha)
        
        # gamma.
        if N_gamma is None:
            N_gamma = 2 * N_gamma
            
        gammas = numpy.linspace(2 * numpy.pi/ N_gamma, 2*numpy.pi, N_gamma)
        
        if verbose:
            print(f'alphas: \n{alphas}\n')
            print(f'gammas: \n{gammas}\n')
            print(f'betas: \n{betas}\n')
            print(f'ws: \n{ws}\n')
                  
        return alphas, betas, gammas, ws
        
    def generate_quad(self, proj='part', N_alpha=None, N_beta=None, N_gamma=None, verbose=False):
        alphas, betas, gammas, ws = [0], [0], [0], [0]
        
        if proj == 'full':
            alphas, betas, gammas, ws = self.generate_quad_full(N_alpha=N_alpha, N_beta=N_beta, N_gamma=N_gamma, verbose=verbose)
        
        elif proj == 'part':
            betas, ws = self.generate_quad_part(N_beta=N_beta, verbose=verbose)
        
        self.quad = numpy.array([alphas, betas, gammas, ws], dtype=object)
        return alphas, betas, gammas, ws
        
    def get_wigner_d(self, s, m, k, beta):
        """
        Wigner small d-matrix.
        """
        nmin = int(max(0, k-m))
        nmax = int(min(s+k, s-m))
        sum_arr = []

        for n in range(nmin, nmax+1):
            num = numpy.sqrt(
                scipy.special.factorial(s+k) * scipy.special.factorial(s-k) *
                scipy.special.factorial(s+m) * scipy.special.factorial(s-m))

            denom = (scipy.special.factorial(s+k-n) * scipy.special.factorial(s-n-m) * 
                     scipy.special.factorial(n-k+m) * scipy.special.factorial(n))
            
            sign = (-1.)**(n - k + m)
            cos = (numpy.cos(beta/2))**(2*s - 2*n + k - m)
            sin = (numpy.sin(beta/2))**(2*n - k + m)
            sum_arr.append(sign * num / denom * cos * sin)
            
        return sum(sum_arr)

    def do_uhf(self, dm_init=0, dm0=None, mixing_parameter=numpy.pi/4):
        """
        Performs a UHF calculation on the molecule. Returns a GHF object.
        
        Args
            dm_init (int): Specifies method to use to generate initial density matrix.
            dm0 (ndarray): Input density matrix.
        """
        def init_dm_0():
            uhf = scf.UHF(self.mol).run()
            dm = uhf.make_rdm1()
            dm[0][0] += 1.
            return dm
        
        def init_dm_1(dm0=dm0, mixing_parameter=mixing_parameter):
            ''' 
            Generate density matrix with broken spatial and spin symmetry by mixing
            HOMO and LUMO orbitals following ansatz in Szabo and Ostlund, Sec 3.8.7.

            psi_1a = numpy.cos(q)*psi_homo + numpy.sin(q)*psi_lumo
            psi_1b = numpy.cos(q)*psi_homo - numpy.sin(q)*psi_lumo

            psi_2a = -numpy.sin(q)*psi_homo + numpy.cos(q)*psi_lumo
            psi_2b =  numpy.sin(q)*psi_homo + numpy.cos(q)*psi_lumo
            
            Returns
                Density matrices, a list of 2D ndarrays for alpha and beta spins
            '''
            # opt: q, mixing parameter 0 < q < 2 pi

            # Based on init_guess_by_1e.
            h1e = scf.hf.get_hcore(self.mol)
            s1e = self.get_s1e()
            rhf = scf.RHF(self.mol).run(dm0)
            mo_energy, mo_coeff = rhf.eig(h1e, s1e)
            mo_occ = rhf.get_occ(mo_energy=mo_energy, mo_coeff=mo_coeff)

            homo_idx = 0
            lumo_idx = 1

            for i in range(len(mo_occ)-1):
                if mo_occ[i] > 0 and mo_occ[i+1] == 0:
                    homo_idx = i
                    lumo_idx = i + 1

            psi_homo = mo_coeff[:, homo_idx]
            psi_lumo = mo_coeff[:, lumo_idx]

            Ca = numpy.zeros_like(mo_coeff)
            Cb = numpy.zeros_like(mo_coeff)


            # Mix homo and lumo of alpha and beta coefficients.
            q = mixing_parameter

            for k in range(mo_coeff.shape[0]):
                if k == homo_idx:
                    Ca[:,k] = numpy.cos(q)*psi_homo + numpy.sin(q)*psi_lumo
                    Cb[:,k] = numpy.cos(q)*psi_homo - numpy.sin(q)*psi_lumo
                    continue

                if k == lumo_idx:
                    Ca[:,k] = -numpy.sin(q)*psi_homo + numpy.cos(q)*psi_lumo
                    Cb[:,k] =  numpy.sin(q)*psi_homo + numpy.cos(q)*psi_lumo
                    continue

                Ca[:,k] = mo_coeff[:,k]
                Cb[:,k] = mo_coeff[:,k]

            dm = scf.UHF(self.mol).make_rdm1( (Ca,Cb), (mo_occ,mo_occ) )
            return dm 
            
            
        dm = None
        
        if dm_init == 0:
            dm = init_dm_0()
            
        elif dm_init == 1:
            dm = init_dm_1(dm0=dm0, mixing_parameter=mixing_parameter)
            
        elif dm_init == 2:
            nao = self.mol.nao
            
            if dm0 is not None:
                dma = dm0[:nao, :nao]
                dmb = dm0[nao:, nao:]
                dm = numpy.array([dma, dmb])
            
        # Do UHF calculation.
        uhf = scf.UHF(self.mol).run(dm)
        self.e_hf = uhf.e_tot
        self.e_elec_hf = self.e_hf - self.mol.energy_nuc() 
        
        # Convert to GHF object.
        ghf = scf.addons.convert_to_ghf(uhf)
        return ghf
    
    def do_mp2(self, uhf):
        """
        Performs an MP2 calculation starting from UHF. We use the GMP2 class
        to store results.
        """
        dfgmp2 = mp.GMP2(uhf).density_fit().run()
        return dfgmp2
    
    def Ry(self, mo_coeff, beta):
        """
        Rotates the spin components of the determinants in the coefficient matrix `mo_coeff` by angle `beta`
        about the y axis.
        
        Args
            mo_coeff (ndarray): MO coefficient matrix from a GHF object.
            beta (float): Rotation angle.
            
        Returns
            The rotated coefficient matrix.
        """
        nao = self.mol.nao
        id_mat = numpy.eye(nao)
        Ry_mat = numpy.block([[numpy.cos(beta/2) * id_mat, -numpy.sin(beta/2) * id_mat],
                              [numpy.sin(beta/2) * id_mat, numpy.cos(beta/2) * id_mat]])
        return numpy.dot(Ry_mat, mo_coeff)
    
    def Rz(self, mo_coeff, theta):
        """
        Rotates the spin components of the determinants in the coefficient matrix `mo_coeff` by angle `theta`
        about the z axis.
        
        Args
            mo_coeff (ndarray): MO coefficient matrix from a GHF object.
            theta (float): Rotation angle.
            
        Returns
            The rotated coefficient matrix.
        """
        nao = self.mol.nao
        id_mat = numpy.eye(nao)
        Rz_mat = numpy.block([[numpy.exp(-1j * theta/2) * id_mat, numpy.zeros((nao, nao))],
                              [numpy.zeros((nao, nao)), numpy.exp(1j * theta/2) * id_mat]])
        return numpy.dot(Rz_mat, mo_coeff)
    
    def get_rotated_mo_coeffs(self, mo_coeff, proj='part', N_alpha=None, N_beta=None, N_gamma=None, verbose=False):
        """
        Generates rotated coefficient matrices from `mo_coeff` at each quadrature point.
        """
        if not isinstance(self.quad, numpy.ndarray):
            self.generate_quad(N_alpha=N_alpha, N_beta=N_beta, N_gamma=N_gamma, verbose=verbose)
        
        #self.generate_quad(proj=proj, N_alpha=N_alpha, N_beta=N_beta, N_gamma=N_gamma, verbose=verbose)
        alphas, betas, gammas, ws = self.quad
        
        # 1st index - alpha
        # 2nd index - beta
        # 3rd index - gamma
        # Angles in increasing order along all axes.
        rotated_mo_coeffs = []
        
        if proj == 'part':
            for b, beta_b in enumerate(betas):
                y = self.Ry(mo_coeff, beta_b)
                rotated_mo_coeffs.append(y)
                
        elif proj == 'full':
            for a, alpha_a in enumerate(alphas):
                mo1 = []

                for b, beta_b in enumerate(betas):
                    mo2 = []

                    for c, gamma_c in enumerate(gammas):
                        z = self.Rz(mo_coeff, gamma_c)
                        yz = self.Ry(z, beta_b)
                        zyz = self.Rz(yz, alpha_a)
                        mo2.append(zyz)

                    mo1.append(mo2)

                rotated_mo_coeffs.append(mo1)
        
        self.rotated_mo_coeffs = numpy.array(rotated_mo_coeffs)
        return self.rotated_mo_coeffs
    
    def quad_coeffs(self, s, m, k, proj='part', N_alpha=None, N_beta=None, N_gamma=None, verbose=False):
        """
        Returns a 3D array of the coefficient at each quad point.
        """
        if not isinstance(self.quad, numpy.ndarray):
            alphas, betas, gammas, ws = self.generate_quad(proj=proj, N_alpha=N_alpha, N_beta=N_beta, 
                                                           N_gamma=N_gamma, verbose=verbose)
            
        alphas, betas, gammas, ws = self.quad
        N_alpha = len(alphas)
        N_beta = len(betas)
        N_gamma = len(gammas)
        
        prefactor = (2*s + 1) / (8 * numpy.pi**2) * (2 * numpy.pi) / N_alpha * (2 * numpy.pi) / N_gamma
        coeffs = None
        
        if proj == 'part':
            coeffs = numpy.zeros(N_beta, dtype=numpy.float)
            
            for b, beta_b in enumerate(betas):
                coeffs[b] = ws[b] * self.get_wigner_d(s, m, k, beta_b)
                
        elif proj == 'full':
            coeffs = numpy.zeros((N_alpha, N_beta, N_gamma), dtype=numpy.complex)
        
            for a, alpha_a in enumerate(alphas):
                for b, beta_b in enumerate(betas):
                    for c, gamma_c in enumerate(gammas):
                        coeffs[a, b, c] = ws[b] * self.get_wigner_d(s, m, k, beta_b) * \
                                            numpy.exp(-1j * m * alpha_a) * numpy.exp(-1j * k * gamma_c)
        
        coeffs = prefactor * coeffs
        return coeffs
    
    def get_matrix_element(self, mo_coeff1, mo_coeff2, mo_occ, verbose=False):
        """
        Computes the matrix element <Psi_{UHF}|H R|Psi_{UHF}>.
        
        Args
            mo_coeff1 (ndarray): MO coefficient matrix for the HF state associated with the left determinant.
            mo_coeff2 (ndarray): ... right determinant.
            mo_occ (ndarray): MO occupation numbers for the left determinant. Used to specify ground state
                              determinants.
        """
        #if not isinstance(self.mo_eri, numpy.ndarray):
        #    if mo_coeff1.dtype != numpy.complex:
        #        self.get_mo_eri_real(mo_coeff1)
        #    else:
        #        self.get_mo_eri_complex(mo_coeff1)
        if self.mo_eri is None:
            self.mo_eri = self.ao2mo(mo_coeff1)

        mo_hcore = self.get_mo_hcore(mo_coeff1)
        A = self.get_A_matrix(mo_coeff1, mo_coeff2, mo_occ)
        M = self.get_M_matrix(A)
        trans_rdm1 = self.get_trans_rdm1(A, M)
        #Minv = numpy.linalg.inv(M)
        ovlp = self.det_ovlp(mo_coeff1, mo_coeff2, mo_occ, mo_occ)
        
        # Be careful with convention of dm1 and dm2
        #   dm1[q,p] = <p^\dagger q>
        #   dm2[p,q,r,s] = < p^\dagger r^\dagger s q >
        #   E = einsum('pq,qp', h1, dm1) + .5 * einsum('pqrs,pqrs', eri, dm2)
        # When adding dm1 contribution, dm1 subscripts need to be flipped
        #
        # sum{ h_ik rho_ki + 1/2 <ij||kl> rho_ki rho_lj}
        eri_opop = self.mo_eri.opop
        h1  = mo_hcore[:self.nocc]
        h1 += .5 * lib.einsum('ikjl,ki->jl', eri_opop, trans_rdm1)
        h1 -= .5 * lib.einsum('iljk,ki->jl', eri_opop, trans_rdm1)
        e1 = ovlp * numpy.einsum('ik,ki', h1, trans_rdm1)
        return e1
    
    def get_A_matrix(self, mo_coeff1, mo_coeff2, mo_occ2):
        """
        Returns the uxn matrix A = C^{\dagger} S C'.
        
        Args
            mo_coeff1 (ndarray): MO coefficient matrix for the HF state associated with the left determinant.
            mo_coeff2 (ndarray): ... right determinant.
            mo_occ2 (ndarray): MO occupation numbers for the right determinant. Used to specify ground or doubly
                               excited determinants.
        
        Returns
            The A matrix.
        """ 
        return self.get_mo_ovlp(mo_coeff1, mo_coeff2[:,mo_occ2>0])
    
    def get_M_matrix(self, A_matrix):
        """
        Computes the nxn matrix M.
        """
        return A_matrix[:self.nelec, :self.nelec] 
 
    def get_trans_rdm1(self, A, M):
        """
        Computes the transition density matrix from the A, M matrices.
        
        Args
            A (ndarray): The A matrix.
            M (ndarray): The M matrix.
        
        Returns
            The transition density matrix.
        """
        # A - uxn 
        # M - nxn
        # M_block - nxu
        # AM_block - uxu
        u = self.nmo
        n = self.nelec
        Minv = numpy.linalg.inv(M)
        #Minv_block = numpy.block([[Minv, numpy.zeros((n, u-n))]])            
        #trans_rdm1 = numpy.dot(A, Minv_block)
        trans_rdm1 = numpy.dot(A, Minv)
        return trans_rdm1
    
    def get_co_ovlp_oo(self, mo_coeff1, mo_coeff2):
        """
        Computes the occupied-occupied (oo) block of the CO overlap matrix by doing an SVD of the oo block 
        of the MO overlap matrix.
        """
        mo_ovlp_oo = self.get_mo_ovlp(mo_coeff1[:, :self.nocc], mo_coeff2[:, :self.nocc])
        u, s, vt = numpy.linalg.svd(mo_ovlp_oo)
        return u, s, vt.T.conj()
    
    def get_co_ovlp_vv(self, mo_coeff1, mo_coeff2):
        """
        Computes the virtual-virtual (vv) block of the CO overlap matrix by doing an SVD of the vv block 
        of the MO overlap matrix.
        """
        mo_ovlp_vv = self.get_mo_ovlp(mo_coeff1[:, self.nocc:], mo_coeff2[:, self.nocc:])
        u, s, vt = numpy.linalg.svd(mo_ovlp_vv)
        return u, s, vt.T.conj()
    
    def get_co_ovlp_vo(self, mo_coeff1, mo_coeff2, v_oo):
        """
        Computes the virtual-occupied (vo) block of the CO overlap matrix by transforming the vo block 
        of the MO overlap matrix as follows:
        
            S^{CO, vo} = S^{MO, vo} V^{oo}.
        
        This is because for the rows we just have C^{MO, vv}, but for the columns we have:
        
            C^{CO, oo} = C^{MO, oo} V^{oo}
        """
        mo_ovlp_vo = self.get_mo_ovlp(mo_coeff1[:,self.nocc:], mo_coeff2[:,:self.nocc])
        co_ovlp_vo = numpy.dot(mo_ovlp_vo, v_oo)
        return co_ovlp_vo
    
    def get_co_ovlp_ov(self, mo_coeff1, mo_coeff2, u_oo):
        """
        Computes the occupied-virtual (ov) block of the CO overlap matrix by transforming the ov block 
        of the MO overlap matrix as follows:
        
            S^{CO, ov} = [U^{oo}]^\dagger S^{MO, ov}
            
        This is because for the rows we have:
        
            C^{CO, oo} = C^{MO, oo} U^{oo}
            
        but for the columns we just have C^{MO, vv}.
        """
        mo_ovlp_ov = self.get_mo_ovlp(mo_coeff1[:,:self.nocc], mo_coeff2[:,self.nocc:])
        co_ovlp_ov = numpy.dot(u_oo.T.conj(), mo_ovlp_ov)
        return co_ovlp_ov
    
    def get_co_ovlp(self, mo_coeff1, mo_coeff2, return_all=False):
        """
        CO transformation of only the oo block of S.
        """
        u_oo, co_ovlp_oo_diag, v_oo = self.get_co_ovlp_oo(mo_coeff1, mo_coeff2)
        co_ovlp_vo = self.get_co_ovlp_vo(mo_coeff1, mo_coeff2, v_oo)
        co_ovlp_ov = self.get_co_ovlp_ov(mo_coeff1, mo_coeff2, u_oo)
        co_ovlp_oo = numpy.diag(co_ovlp_oo_diag)
        co_ovlp_vv = self.get_mo_ovlp(mo_coeff1[:,self.nocc:], mo_coeff2[:,self.nocc:])
        co_ovlp = numpy.block([[co_ovlp_oo, co_ovlp_ov], [co_ovlp_vo, co_ovlp_vv]])
        if return_all:
            return co_ovlp, co_ovlp_oo_diag, u_oo, v_oo
        return co_ovlp
 
    def energy_01_term1(self, mo_coeff1, mo_coeff2, mo_occ, co_t2):
        """
        Computes E_{HF} * sum_{i<j, a<b} [ t_{ijab} * <Psi_{UHF} | R | {Psi_{UHF}}_{ij}^{ab}> ] using Eq. 14
        from J. Chem. Phys. 139, 174104 (2013).
        
        - No CO transformation for vv block.
        """
        nmo = self.nmo
        nocc = self.nelec
        nvir = nmo - nocc
        
        # Get the U, S, V matrices.
        u_oo, co_ovlp_oo_diag, v_oo = self.get_co_ovlp_oo(mo_coeff1, mo_coeff2)
        
        # Get the ov-block of the CO overlap matrix.
        co_ovlp_ov = self.get_co_ovlp_ov(mo_coeff1, mo_coeff2, u_oo)
        
        # Get inverse of the oo-block of the CO overlap matrix.
        co_ovlp_oo_inv = 1. / co_ovlp_oo_diag
        
        prefactor = 0.5 * self.e_elec_hf * numpy.prod(co_ovlp_oo_diag)

        tmp = co_ovlp_ov * co_ovlp_oo_inv[:,None]
        term = lib.einsum('ijab,ia,jb->', co_t2, tmp, tmp)
        return prefactor * term 
        
    def get_S_ijpr_matrix_slow(self, co_ovlp, co_ovlp_oo_diag):
        """
        Constructs the S_ijpr tensor used in the evaluation of <Psi_{UHF}|H R|Psi_{UHF}^{1}>.
        """
        nocc = self.nocc
        nvir = self.nmo - nocc
        S_ijpr_mat = numpy.empty((nocc, nocc, nvir, nvir))
        
        for i in range(nocc):
            for j in range(nocc):
                inds = (i, j)
                _nocc = nocc - len(numpy.unique(inds))
                co_ovlp_ij = numpy.delete(co_ovlp, inds, axis=0) # Delete rows i, j
                co_ovlp_ij = numpy.delete(co_ovlp_ij, inds, axis=1) # Delete cols i, j
                co_ovlp_ov_ij = co_ovlp_ij[:_nocc, _nocc:]
                co_ovlp_vo_ij = co_ovlp_ij[_nocc:, :_nocc]

                co_ovlp_oo_diag_ij = numpy.delete(co_ovlp_oo_diag, inds, axis=0) # Delete elements i, j
                co_ovlp_oo_inv_ij = 1. / co_ovlp_oo_diag_ij
                
                # [ co_ovlp_ij.T ]_{pk} = [ co_ovlp_ij ]_{kp}
                S_ijpr_mat[i, j] = (co_ovlp[nocc:, nocc:] - 
                                    numpy.einsum('pk,k,kr->pr', co_ovlp_vo_ij, co_ovlp_oo_inv_ij, co_ovlp_ov_ij))
        
        return S_ijpr_mat

    def get_S_ijpr_matrix(self, co_ovlp, co_ovlp_oo_diag):
        """
        Constructs the S_ijpr tensor used in the evaluation of <Psi_{UHF}|H R|Psi_{UHF}^{1}>.
        """
        nocc = self.nocc
        nvir = self.nmo - nocc

        co_ovlp_vv = co_ovlp[nocc:,nocc:]
        co_ovlp_ov = co_ovlp[:nocc,nocc:]
        co_ovlp_vo = co_ovlp[nocc:,:nocc]
        co_ovlp_oo_inv = 1. / co_ovlp_oo_diag
        #S_ipr = numpy.einsum("pk,kr,k->kpr", co_ovlp_vo, co_ovlp_ov, co_ovlp_oo_inv)
        S_ipr = co_ovlp_vo.T[:,:,None] * co_ovlp_ov[:,None,:] * co_ovlp_oo_inv[:,None,None]
        S_pr = numpy.sum(S_ipr, axis=0)

        tmp = co_ovlp_vv - S_pr
        tmp1 = S_ipr[:,None,:,:] + S_ipr[None,:,:,:]
        idx = numpy.arange(nocc, dtype=numpy.intp)
        tmp1[idx,idx,:,:] *= .5
        S_ijpr_mat = tmp[None,None,:,:] + tmp1
        return S_ijpr_mat

    def get_S_ipr_matrix_slow(self, co_ovlp, co_ovlp_oo_diag):
        """
        Constructs the S_ipr tensor used in the evaluation of <Psi_{UHF}|H R|Psi_{UHF}^{1}>.
        """
        nocc = self.nocc
        _nocc = nocc - 1
        nvir = self.nmo - nocc
        S_ipr_mat = numpy.empty((nocc, nvir, nvir))
        
        for i in range(nocc):
            co_ovlp_i = numpy.delete(co_ovlp, i, axis=0) # Delete row i
            co_ovlp_i = numpy.delete(co_ovlp_i, i, axis=1) # Delete col i
            co_ovlp_ov_i = co_ovlp_i[:_nocc, _nocc:]
            co_ovlp_vo_i = co_ovlp_i[_nocc:, :_nocc]

            co_ovlp_oo_diag_i = numpy.delete(co_ovlp_oo_diag, i, axis=0) # Delete element i
            co_ovlp_oo_inv_i = 1. / co_ovlp_oo_diag_i
            
            # [ co_ovlp_i.T ]_{pk} = [ co_ovlp_i ]_{kp}
            S_ipr_mat[i] = (co_ovlp[nocc:, nocc:] - 
                           numpy.einsum('pk,k,kr->pr', co_ovlp_vo_i, co_ovlp_oo_inv_i, co_ovlp_ov_i))
        
        return S_ipr_mat

    def get_S_ipr_matrix(self, co_ovlp, co_ovlp_oo_diag):
        """
        Constructs the S_ipr tensor used in the evaluation of <Psi_{UHF}|H R|Psi_{UHF}^{1}>.
        """
        nocc = self.nocc
        nvir = self.nmo - nocc

        co_ovlp_vv = co_ovlp[nocc:,nocc:]
        co_ovlp_ov = co_ovlp[:nocc,nocc:]
        co_ovlp_vo = co_ovlp[nocc:,:nocc]
        co_ovlp_oo_inv = 1. / co_ovlp_oo_diag
        #S_ipr = numpy.einsum("pk,kr,k->kpr", co_ovlp_vo, co_ovlp_ov, co_ovlp_oo_inv)
        S_ipr = co_ovlp_vo.T[:,:,None] * co_ovlp_ov[:,None,:] * co_ovlp_oo_inv[:,None,None]
        S_pr = numpy.sum(S_ipr, axis=0)

        tmp = co_ovlp_vv - S_pr
        S_ipr_mat = tmp[None,:,:] + S_ipr
        return S_ipr_mat

    def energy_01_term2_case1(self, co_t2, co_eri, co_ovlp, co_ovlp_oo_diag):
        """
        Case 1: i = k, j = l
        """
        nocc = self.nocc
        nmo = self.nmo
        nvir = nmo - nocc
        co_eri_oovv = co_eri.transpose(0,2,1,3) - co_eri.transpose(0,2,3,1)
        co_ovlp_oo_inv = 1. / co_ovlp_oo_diag

        S_ijpr_matrix = self.get_S_ijpr_matrix(co_ovlp, co_ovlp_oo_diag)

        S_ijpr_matrix = numpy.asarray(S_ijpr_matrix, order='C')
        co_t2 = numpy.asarray(co_t2, order='C')
        sumc_ijad = numpy.empty((nocc,nocc,nvir,nvir), order='C', dtype=co_ovlp.dtype)
        sumb_ijad = numpy.empty((nocc,nocc,nvir,nvir), order='C', dtype=co_ovlp.dtype)
        #for i in range(nocc):
        #    sumc_ijad[i] = numpy.einsum('jca,jcd->jad', S_ijpr_matrix[i], co_eri_oovv[i])
        #    sumb_ijad[i] = numpy.einsum('jab,jdb->jad', co_t2[i], S_ijpr_matrix[i])

        if co_ovlp.dtype == numpy.double:
            fn = getattr(libnp_helper, "contract_o2v3", None)
        elif co_ovlp.dtype == numpy.complex:
            fn = getattr(libnp_helper, "contract_o2v3_cmplx", None)
        try:
            case0 = 0
            case1 = 1
            fn(sumc_ijad.ctypes.data_as(ctypes.c_void_p), 
               S_ijpr_matrix.ctypes.data_as(ctypes.c_void_p), 
               co_eri_oovv.ctypes.data_as(ctypes.c_void_p), 
               ctypes.c_int(nocc), ctypes.c_int(nvir), ctypes.c_int(case0))
            fn(sumb_ijad.ctypes.data_as(ctypes.c_void_p), 
               co_t2.ctypes.data_as(ctypes.c_void_p),
               S_ijpr_matrix.ctypes.data_as(ctypes.c_void_p),
               ctypes.c_int(nocc), ctypes.c_int(nvir), ctypes.c_int(case1))
        except:
            raise RuntimeError

        Sij_inv = co_ovlp_oo_inv[:,None]*co_ovlp_oo_inv[None,:]
        tmp = sumb_ijad * Sij_inv[:,:,None,None]
        sumadij = lib.einsum('ijad,ijad->', sumc_ijad, tmp)

        return sumadij * 0.25 * numpy.prod(co_ovlp_oo_diag)
    
    def energy_01_term2_case2(self, co_t2, co_eri, co_ovlp, co_ovlp_oo_diag):
        """
        Case 2: i = k, j != l
        """
        nocc = self.nocc
        nmo = self.nmo
        nvir = nmo - nocc
        co_eri_oovv = co_eri.transpose(0,2,1,3) - co_eri.transpose(0,2,3,1)
        co_ovlp_oo_inv = 1. / co_ovlp_oo_diag
        co_ovlp_vo = co_ovlp[nocc:, :nocc]
        co_ovlp_ov = co_ovlp[:nocc, nocc:]

        S_ipr_matrix = self.get_S_ipr_matrix(co_ovlp, co_ovlp_oo_diag)
        
        # 1st term.
        tmp = co_ovlp_vo * co_ovlp_oo_inv[None,:]
        sumld_ic = lib.einsum('ilcd,dl->ic', co_eri_oovv, tmp)
        tmp = co_ovlp_ov * co_ovlp_oo_inv[:,None]
        sumjb_ia = lib.einsum('ijab,jb->ia', co_t2, tmp)
        tmp = sumld_ic[:,:,None] * sumjb_ia[:,None,:] * co_ovlp_oo_inv[:,None,None]
        term1 = numpy.einsum('ica,ica->', tmp, S_ipr_matrix)

        # 2nd term.
        sumd_ijc = numpy.einsum('ijcd,dj->ijc', co_eri_oovv, co_ovlp_vo)
        sumb_ija = numpy.einsum('ijab,jb->ija', co_t2, co_ovlp_ov)

        tmp = sumd_ijc[:,:,:,None] * sumb_ija[:,:,None,:]
        tmp1 = (co_ovlp_oo_inv * co_ovlp_oo_inv)[None,:] * co_ovlp_oo_inv[:,None]
        tmp1 = tmp1[:,:,None,None] * S_ipr_matrix[:,None,:,:]
        term2 = -numpy.einsum('ijca,ijca->', tmp, tmp1)

        return (term1 + term2) * numpy.prod(co_ovlp_oo_diag)
    
    def energy_01_term2_case3(self, co_t2, co_eri, co_ovlp, co_ovlp_oo_diag):
        """
        Case 3: i != k, j != l
        """
        nocc = self.nocc
        nmo = self.nmo
        nvir = nmo - nocc
        co_eri_oovv = co_eri.transpose(0,2,1,3) - co_eri.transpose(0,2,3,1)
        co_ovlp_oo_inv = 1. / co_ovlp_oo_diag
        co_ovlp_vo = co_ovlp[nocc:, :nocc]
        co_ovlp_ov = co_ovlp[:nocc, nocc:]

        co_ovlp_oo_inv2 = co_ovlp_oo_inv * co_ovlp_oo_inv
        co_ovlp_oo_inv_ij = co_ovlp_oo_inv[:,None] * co_ovlp_oo_inv[None,:]

        sumcd_ij = numpy.einsum('ijcd,ci,dj->ij', co_eri_oovv, co_ovlp_vo, co_ovlp_vo)
        sumab_ij = numpy.einsum('ijab,ia,jb->ij', co_t2, co_ovlp_ov, co_ovlp_ov)

        # 1st term.
        sumklcd = numpy.sum(sumcd_ij * co_ovlp_oo_inv_ij)
        sumijab = numpy.sum(sumab_ij * co_ovlp_oo_inv_ij)
        term1 = 0.25 * sumklcd * sumijab
        
        # 2nd term.
        sumlcd_i = numpy.dot(sumcd_ij, co_ovlp_oo_inv)
        sumjab_i = numpy.dot(sumab_ij, co_ovlp_oo_inv)
        term2 = -numpy.sum(sumlcd_i * sumjab_i * co_ovlp_oo_inv2)
        
        # 3rd term.
        term3 = 0.5 * numpy.sum(sumcd_ij * sumab_ij * 
                                co_ovlp_oo_inv2[:,None] * co_ovlp_oo_inv2[None,:])
        
        return (term1 + term2 + term3) * numpy.prod(co_ovlp_oo_diag)
    
    def energy_01_term2(self, co_t2, co_eri, co_ovlp, co_ovlp_oo_diag, verbose=False):
        case1 = self.energy_01_term2_case1(co_t2, co_eri, co_ovlp, co_ovlp_oo_diag)
        case2 = self.energy_01_term2_case2(co_t2, co_eri, co_ovlp, co_ovlp_oo_diag)
        case3 = self.energy_01_term2_case3(co_t2, co_eri, co_ovlp, co_ovlp_oo_diag)
        term = case1 + case2 + case3
                
        if verbose:
            print(f'case1 = {case1}')
            print(f'case2 = {case2}')
            print(f'case3 = {case3}')
            
        return term
    
    def energy_01(self, ump2, mo_coeff2):
        """
        - No CO transformation for vv block.
        - Transform t_ijab with V, V^\dagger.
        - Compute mo_eri once, use `get_co_from_mo_eri`.
        """
        nocc = self.nelec
        nvir = self.nmo - nocc
        
        mo_coeff1 = ump2.mo_coeff
        mo_occ = ump2.mo_occ
        mo_t2 = ump2.t2
        
        co_ovlp, co_ovlp_oo_diag, u_oo, v_oo = self.get_co_ovlp(mo_coeff1, mo_coeff2, return_all=True)
        u = numpy.block([[u_oo, numpy.zeros((nocc, nvir))], [numpy.zeros((nvir, nocc)), numpy.eye(nvir)]])
        co_coeff1 = self.get_co_coeff(mo_coeff1, u)
        co_t2 = self.get_co_t2(mo_t2, v_oo)
        co_eri = self.get_co_eri_from_mo_eri(mo_coeff1, u_oo)
        
        term1 = self.energy_01_term1(mo_coeff1, mo_coeff2, mo_occ, co_t2)
        term2 = self.energy_01_term2(co_t2, co_eri, co_ovlp, co_ovlp_oo_diag)
        norm_01 = term1 / self.e_elec_hf
        
        return term1 + term2, norm_01
    
    def energy(self, s, m, k, ump2, proj='part', N_alpha=None, N_beta=None, N_gamma=None, 
               just_hf=False, verbose=False):
        """
        Compute the total energy <Psi_{UHF}|H P|Psi_{UMP2}> / <Psi_{UHF}|P|Psi_{UMP2}>.
        
        - No CO transformation for vv block.
        - Transform t_ijab with V, V^\dagger
        
        Args
            s (int): Total spin S eigenvalue.
            m (int): Final Sz eigenvalue.
            k (int): Initial Sz eigenvalue.
            
        Returns
            <Psi_{UHF}|H P|Psi_{UMP2}> / <Psi_{UHF}|P|Psi_{UMP2}> or <Psi_{UHF}|H P|Psi_{UMP2}>
        """ 
        mo_coeff = ump2.mo_coeff
        mo_occ = ump2.mo_occ
        
        if not isinstance(self.rotated_mo_coeffs, numpy.ndarray):
            self.get_rotated_mo_coeffs(mo_coeff, proj=proj, N_alpha=N_alpha, N_beta=N_beta, N_gamma=N_gamma, verbose=verbose)
        
        rot_mo_coeffs = self.rotated_mo_coeffs
        coeffs = self.quad_coeffs(s, m, k, proj=proj, N_alpha=N_alpha, N_beta=N_beta, N_gamma=N_gamma)
        energy_00 = 0.
        energy_01 = 0.
        norm_00 = 0.
        norm_01 = 0.
        
        if proj == 'part':
            for b in range(N_beta):
                rot_mo_coeff = rot_mo_coeffs[b]
                coeff = coeffs[b]
                _energy_00 = self.get_matrix_element(mo_coeff, rot_mo_coeff, mo_occ, verbose=verbose)
                _norm_00 = self.det_ovlp(mo_coeff, rot_mo_coeff, mo_occ, mo_occ)
                energy_00 += coeff * _energy_00
                norm_00 += coeff * _norm_00

                if just_hf:
                    continue

                _energy_01, _norm_01 = self.energy_01(ump2, rot_mo_coeff)
                energy_01 += coeff * _energy_01
                norm_01 += coeff * _norm_01
            
        elif proj == 'full':
            for a in range(N_alpha):
                for b in range(N_beta):
                    for c in range(N_gamma):
                        rot_mo_coeff = rot_mo_coeffs[a, b, c]
                        coeff = coeffs[a, b, c]
                        _energy_00 = self.get_matrix_element(mo_coeff, rot_mo_coeff, mo_occ, verbose=verbose)
                        _norm_00 = self.det_ovlp(mo_coeff, rot_mo_coeff, mo_occ, mo_occ)
                        energy_00 += coeff * _energy_00
                        norm_00 += coeff * _norm_00

                        if just_hf:
                            continue

                        _energy_01, _norm_01 = self.energy_01(ump2, rot_mo_coeff)
                        energy_01 += coeff * _energy_01
                        norm_01 += coeff * _norm_01
                    
        energy = (energy_00 + energy_01) / (norm_00 + norm_01) + self.mol.energy_nuc()
        return energy
    
    def e_corr(self, s, m, k, ump2, proj='part', N_alpha=None, N_beta=None, N_gamma=None, verbose=False):
        """
        Compute the correlation energy <Psi_{UHF}|H P|Psi_{UHF}^{1}> / <Psi_{UHF}|P|Psi_{UMP2}>.
        
        - No CO transformation for vv block.
        - Transform t_ijab with V, V^\dagger
        
        Args
            s (int): Total spin S eigenvalue.
            m (int): Final Sz eigenvalue.
            k (int): Initial Sz eigenvalue.
            
        Returns
            <Psi_{UHF}|H P|Psi_{UHF}^{1}> / <Psi_{UHF}|P|Psi_{UMP2}>
        """ 
        mo_coeff = ump2.mo_coeff
        mo_occ = ump2.mo_occ
        
        if not isinstance(self.rotated_mo_coeffs, numpy.ndarray):
            self.get_rotated_mo_coeffs(mo_coeff, proj=proj, N_alpha=N_alpha, N_beta=N_beta, N_gamma=N_gamma, verbose=verbose)
        
        rot_mo_coeffs = self.rotated_mo_coeffs
        coeffs = self.quad_coeffs(s, m, k, proj=proj, N_alpha=N_alpha, N_beta=N_beta, N_gamma=N_gamma)
        energy_01 = 0.
        norm_01 = 0.
        norm_00 = 0.
        
        if proj == 'part':
            for b in range(N_beta):
                rot_mo_coeff = rot_mo_coeffs[b]
                coeff = coeffs[b]
                _norm_00 = self.det_ovlp(mo_coeff, rot_mo_coeff, mo_occ, mo_occ)
                _energy_01, _norm_01 = self.energy_01(ump2, rot_mo_coeff)
                energy_01 += coeff * _energy_01
                norm_01 += coeff * _norm_01
                norm_00 += coeff * _norm_00

        elif proj == 'full':
            for a in range(N_alpha):
                for b in range(N_beta):
                    for c in range(N_gamma):
                        rot_mo_coeff = rot_mo_coeffs[a, b, c]
                        coeff = coeffs[a, b, c]
                        _norm_00 = self.det_ovlp(mo_coeff, rot_mo_coeff, mo_occ, mo_occ)
                        _energy_01, _norm_01 = self.energy_01(ump2, rot_mo_coeff)
                        energy_01 += coeff * _energy_01
                        norm_01 += coeff * _norm_01
                        norm_00 += coeff * _norm_00
                    
        ecorr = energy_01 / (norm_00 + norm_01)
        return ecorr
            
    def det_ovlp(self, mo_coeff1, mo_coeff2, mo_occ1, mo_occ2):
        """
        Calculates the overlap between two different determinants. It is the product
        of single values of molecular orbital overlap matrix.
        
        WARNING: Has some sign errors when computing overlaps between double excitations
                 < Psi_{ij}^{ab} | Psi_{kl}^{cd} >.
        """
        if numpy.sum(mo_occ1) != numpy.sum(mo_occ2):
            raise RuntimeError('Electron numbers are not equal. Electronic coupling does not exist.')

        s = self.get_mo_ovlp(mo_coeff1[:,mo_occ1>0], mo_coeff2[:,mo_occ2>0])
        #u, s, vt = numpy.linalg.svd(s)
        #return numpy.prod(s)
        return numpy.linalg.det(s)

    def get_s1e(self, mol=None):
        if self.s1e is None:
            if mol is None:
                mol = self.mol
            self.s1e = hf.get_ovlp(mol)
        return self.s1e

    def get_ao_ovlp(self):
        """
        Compute the AO overlap matrix S.
        """
        if self.ao_ovlp is None:
            self.ao_ovlp = self._scf.get_ovlp()
        return self.ao_ovlp
    
    def get_mo_ovlp(self, mo_coeff1, mo_coeff2):
        """
        Calculates the MO overlap matrix.
        """
        nao = mo_coeff1.shape[0] // 2
        if nao > SWITCH_SIZE:
            moa1 = mo_coeff1[:nao]
            mob1 = mo_coeff1[nao:]
            moa2 = mo_coeff2[:nao]
            mob2 = mo_coeff2[nao:]
            s1e = self.get_s1e()
            s = (reduce(numpy.dot, (moa1.T.conj(), s1e, moa2))
                 +reduce(numpy.dot, (mob1.T.conj(), s1e, mob2)))
        else:
            ao_ovlp = self.get_ao_ovlp()
            s = reduce(numpy.dot, (mo_coeff1.T.conj(), ao_ovlp, mo_coeff2))
        return s
    
    def get_ao_hcore(self):
        """
        Computes the AO `hcore` matrix.
        """
        return self._scf.get_hcore()
    
    def get_mo_hcore(self, mo_coeff):
        """
        Transforms the AO `hcore` matrix to the MO basis.
        """
        ao_hcore = self.get_ao_hcore()
        mo_hcore = reduce(numpy.dot, (mo_coeff.conj().T, ao_hcore, mo_coeff))
        return mo_hcore
    
    def get_ao_eri(self):
        """
        Computes the AO eris.
        """
        return self.mol.intor('int2e', aosym='s8')
        
    def get_mo_eri_complex(self, mo_coeff):
        """
        Transforms the AO eris to MO eris. For complex-valued mo_coeffs.
        
        Args
            mo_coeff (ndarray): GHF mo_coeff
            
        Returns
            mo_eri with shape (nmo, nmo, nmo, nmo)
        """
        if not isinstance(self.mo_eri, numpy.ndarray):
            ao_eri = self.get_ao_eri()
            moa = mo_coeff[:self.mol.nao]
            mob = mo_coeff[self.mol.nao:]
            self.mo_eri = ao2mo_slow(ao_eri, (moa,moa,moa,moa))
            self.mo_eri += ao2mo_slow(ao_eri, (mob,mob,mob,mob))
            self.mo_eri += ao2mo_slow(ao_eri, (moa,moa,mob,mob))
            self.mo_eri += ao2mo_slow(ao_eri, (mob,mob,moa,moa))
        
        return self.mo_eri
    
    def get_mo_eri_real(self, mo_coeff):
        """
        Transforms the AO eris to MO eris. For real-valued mo_coeffs.
        
        Args
            mo_coeff (ndarray): GHF mo_coeff
            
        Returns
            mo_eri with shape (nmo, nmo, nmo, nmo)
        """
        if not isinstance(self.mo_eri, numpy.ndarray):
            ao_eri = self.get_ao_eri()
            nmo = self.nmo
            moa = mo_coeff[:self.mol.nao]
            mob = mo_coeff[self.mol.nao:]
            self.mo_eri = ao2mo.kernel(ao_eri, (moa,moa,moa,moa), compact=False)
            self.mo_eri += ao2mo.kernel(ao_eri, (mob,mob,mob,mob), compact=False)
            self.mo_eri += ao2mo.kernel(ao_eri, (moa,moa,mob,mob), compact=False)
            self.mo_eri += ao2mo.kernel(ao_eri, (mob,mob,moa,moa), compact=False)
            self.mo_eri = self.mo_eri.reshape(nmo, nmo, nmo, nmo)
            
        return self.mo_eri

    def ao2mo(self, mo_coeff=None, nocc=None):
        return _make_eris_incore(self, mo_coeff, nocc)

    def get_co_coeff(self, mo_coeff, u):
        return numpy.dot(mo_coeff, u)
    
    def get_co_eri_from_mo_eri(self, mo_coeff, u_oo):
        """
        Computes the CO eris from the MO eris. The orbitals are from |Psi_{UHF}>.
        """
        if self.mo_eri is None:
            self.mo_eri = self.ao2mo(mo_coeff)

        mo_eri_oovv = self.mo_eri.ovov.transpose(0,2,1,3)
        co_eri_oovv = lib.einsum('pi,qj,pqab->ijab', u_oo.conj(), u_oo.conj(), mo_eri_oovv)
        co_eri = co_eri_oovv.transpose(0,2,1,3)
        return co_eri

    def get_co_t2(self, mo_t2, v_oo):
        """
        CO t2 amplitudes derived from MO t2 amplitudes elucidated with Garnet's method.
        """
        co_t2 = lib.einsum('pi,qj,pqab->ijab', v_oo, v_oo, mo_t2)
        return co_t2

# Other functions.
    
def build_chain(n, bond_length, spin=None, basis='sto-6g'):
    """
    Builds a hydrogen chain of n H atoms.

    Args
        n (int): number of H atoms.
        bond_length (float): bond length between adjacent H atoms.
        spin (int): 2S, num. alpha electrons - num. beta electrons 
        basis (str): type of basis to use.

    Returns
        Hchain
    """
    print('Building molecule...')
    print('Using basis set {}'.format(basis))
    Hchain = gto.Mole()
    Hchain.atom = [['H', (bond_length * i, 0., 0.)] for i in range(n)]
    Hchain.basis = basis
    Hchain.spin = spin # 2S, num. alpha electrons - num. beta electrons 
    
    if spin is None:
        Hchain.spin = 0
    
        if n % 2 == 1:
            Hchain.spin = 1
        
    Hchain.build()
    return Hchain

def get_double(mo_occ, i, j, a, b):
    """
    Reorders the `mo_occ` array so that all the occupied MOs in the doubly excited 
    Slater determinant ijab have occupation number 1.

    Args
        mo_occ (ndarray): Array of HF MO occupations.
        i, j (int): Occupied MOs.
        a, b (int): Virtual MOs.

    Returns
        Doubly excited `mo_occ` array.
    """
    mo_occ_ijab = numpy.copy(mo_occ[:])
    mo_occ_ijab[i], mo_occ_ijab[a] = mo_occ_ijab[a], mo_occ_ijab[i]
    mo_occ_ijab[j], mo_occ_ijab[b] = mo_occ_ijab[b], mo_occ_ijab[j]
    return mo_occ_ijab

def get_all_doubles(mo_occ, nocc, nmo):
    """
    Gets all possible doubles from the `mo_occ` at quad point (alpha, beta).

    Args
        mo_occ (ndarray): Array of HF MO occupations.

    Returns
        Dict {'ijab': mo_occ_ijab}
    """
    all_doubles = {}

    #occ_pairs = list(combinations([self.nelec-2, self.nelec-1], 2))
    occ_pairs = list(combinations(range(nocc), 2))
    vir_pairs = list(combinations(range(nocc, nmo), 2))

    for i, j in occ_pairs:
        for a, b in vir_pairs:
            key = str(i) + ',' + str(j) + ',' + str(a) + ',' + str(b)
            mo_occ_ijab = get_double(mo_occ, i, j, a, b)
            all_doubles[key] = mo_occ_ijab

    return all_doubles

def _ao2mo_loop(mp, eris):
    mo_coeff = eris.mo_coeff
    nao = mo_coeff.shape[0]
    complex_orb = mo_coeff.dtype == numpy.complex
    nocc, nmo = eris.nocc, eris.nmo

    moa = mo_coeff[:nao//2]
    mob = mo_coeff[nao//2:]
    if complex_orb:
        moa_RR = moa.real
        moa_II = moa.imag
        moa_RI = numpy.concatenate((moa.real[:,:nocc], moa.imag), axis=1)
        moa_IR = numpy.concatenate((moa.imag[:,:nocc], moa.real), axis=1)

        mob_RR = mob.real
        mob_II = mob.imag
        mob_RI = numpy.concatenate((mob.real[:,:nocc], mob.imag), axis=1)
        mob_IR = numpy.concatenate((mob.imag[:,:nocc], mob.real), axis=1)

    ijslice = (0, nocc, 0, nmo)
    ijslice1 = (0, nocc, nocc, nmo+nocc)
    Lova = Lovb = buf = None

    with_df = mp.with_df
    naux = with_df.get_naoaux()
    mem_now = lib.current_memory()[0]
    max_memory = max(2000, mp.max_memory*.95-mem_now)
    fac = 1 if not complex_orb else 2
    buf_size = max_memory*1e6/8 - (nocc*nmo)**2*fac
    if buf_size < 0:
        raise RuntimeError(f"At least {(nocc*nmo)**2*fac*8/1e9} Gb of memory is needed.")
    blksize = int(min(naux, max(with_df.blockdim, buf_size / (4*nocc*nmo))))

    for eri1 in with_df.loop(blksize=blksize):
        if complex_orb:
            Lova  = _ao2mo.nr_e2(eri1, moa_RR, ijslice, aosym='s2', out=Lova)
            Lova += _ao2mo.nr_e2(eri1, moa_II, ijslice, aosym='s2', out=buf)
            Lova += 1j * _ao2mo.nr_e2(eri1, moa_RI, ijslice1, aosym='s2', out=buf)
            Lova -= 1j * _ao2mo.nr_e2(eri1, moa_IR, ijslice1, aosym='s2', out=buf)

            Lovb  = _ao2mo.nr_e2(eri1, mob_RR, ijslice, aosym='s2', out=Lovb)
            Lovb += _ao2mo.nr_e2(eri1, mob_II, ijslice, aosym='s2', out=buf)
            Lovb += 1j * _ao2mo.nr_e2(eri1, mob_RI, ijslice1, aosym='s2', out=buf)
            Lovb -= 1j * _ao2mo.nr_e2(eri1, mob_IR, ijslice1, aosym='s2', out=buf)
        else:
            Lova = _ao2mo.nr_e2(eri1, moa, ijslice, aosym='s2', out=Lova)
            Lovb = _ao2mo.nr_e2(eri1, mob, ijslice, aosym='s2', out=Lovb)
        yield Lova, Lovb

def _make_eris_incore(mp, mo_coeff=None, nocc=None):
    cput0 = (logger.process_clock(), logger.perf_counter())
    eris = _ChemistsERIs()
    eris._common_init_(mp, mo_coeff, nocc)

    mo_coeff = eris.mo_coeff
    naux = mp.with_df.get_naoaux()
    nocc, nmo = eris.nocc, eris.nmo

    Lova = numpy.empty((naux, nocc*nmo), dtype=mo_coeff.dtype)
    Lovb = numpy.empty((naux, nocc*nmo), dtype=mo_coeff.dtype)

    p1 = 0
    for qova, qovb in _ao2mo_loop(mp, eris):
        p0, p1 = p1, p1 + qova.shape[0]
        Lova[p0:p1] = qova
        Lovb[p0:p1] = qovb

    buf  = lib.dot(Lova.T, Lova)
    buf += lib.dot(Lovb.T, Lovb)
    buf += lib.dot(Lova.T, Lovb)
    buf += lib.dot(Lovb.T, Lova)
    eris.opop = buf.reshape(nocc, nmo, nocc, nmo)
    eris.ovov = eris.opop[:,nocc:,:,nocc:]
    cput1 = logger.timer(mp, '_make_eris_incore', *cput0)
    return eris

class _ChemistsERIs:
    def __init__(self, mol=None):
        self.mol = mol
        self.mo_coeff = None
        self.nocc = None
        self.nmo = None
        #self.orbspin = None
        self.ovov = None
        self.opop = None

    def _common_init_(self, mp, mo_coeff=None, nocc=None):
        if mo_coeff is None:
            mo_coeff = mp.mo_coeff
        if mo_coeff is None:
            raise RuntimeError('mo_coeff, mo_energy are not initialized.\n'
                               'You may need to call mf.kernel() to generate them.')
        self.mo_coeff = mo_coeff
        if self.mol is None:
            self.mol = mp.mol
        self.nmo = mo_coeff.shape[-1]
        if nocc is None:
            nocc = mp.nocc
        self.nocc = nocc
        return self


if __name__ == '__main__':
    from pyscf import fci
    
    mol = gto.Mole()
    mol.atom = [['H', (i*2,0.,0.)] for i in range(2)]
    mol.basis = 'sto-3g'
    mol.build() 
    
    s, m, k = 0, 0, 0
    N_alpha = N_gamma = 1
    N_beta = 10

    test = SPPT2(mol)
    uhf = test.do_uhf()
    ump2 = test.do_mp2(uhf)
    e = test.energy(s, m, k, ump2, N_alpha=N_alpha, N_beta=N_beta, N_gamma=N_gamma)
