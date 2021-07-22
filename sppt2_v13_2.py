"""
V13 
    - This version does UHF -> UMP2 -> P.
    - Using simplified projector as default. 
    - CO version using Van Voorhis' equations.
    - Using DF-GMP2.
        - But calculating MO eris without df.
        
    - Optimize with einsum.
"""


from pyscf import gto, scf, mp, fci, lib, ao2mo
from pyscf.mp import gmp2_slow
import numpy
import scipy.special
from functools import reduce
from itertools import combinations
from prof import profile
import math

class SPPT2:
    def __init__(self, mol):
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
        self.mol = mol
        self.rotated_mo_coeffs = None
        self.mo_eri = None
        self.ao_ovlp = None
        self.quad = None
        self.nelec = mol.nelectron
        self.nmo = 2 * mol.nao # Number of spatial MOs.
        self.S = mol.spin / 2 # mol.spin = n_alpha - n_beta
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
            s1e = scf.hf.get_ovlp(self.mol)
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
        if not isinstance(self.mo_eri, numpy.ndarray):
            if mo_coeff1.dtype != numpy.complex:
                self.get_mo_eri_real(mo_coeff1)
                
            else:
                self.get_mo_eri_complex(mo_coeff1)
            
        mo_hcore = self.get_mo_hcore(mo_coeff1)
        A = self.get_A_matrix(mo_coeff1, mo_coeff2, mo_occ)
        M = self.get_M_matrix(A)
        trans_rdm1 = self.get_trans_rdm1(A, M)
        Minv = numpy.linalg.inv(M)
        ovlp = self.det_ovlp(mo_coeff1, mo_coeff2, mo_occ, mo_occ)
        
        # Be careful with convention of dm1 and dm2
        #   dm1[q,p] = <p^\dagger q>
        #   dm2[p,q,r,s] = < p^\dagger r^\dagger s q >
        #   E = einsum('pq,qp', h1, dm1) + .5 * einsum('pqrs,pqrs', eri, dm2)
        # When adding dm1 contribution, dm1 subscripts need to be flipped
        #
        # sum{ h_ik rho_ki + 1/2 <ij||kl> rho_ki rho_lj}
        part1 = numpy.einsum('ikjl, ki->jl', self.mo_eri, trans_rdm1)
        part2 = numpy.einsum('iljk, ki->jl', self.mo_eri, trans_rdm1)
        term = numpy.einsum('ik, ki', mo_hcore, trans_rdm1) + 0.5 * (
               numpy.einsum('jl, lj', part1, trans_rdm1) - numpy.einsum('jl, lj', part2, trans_rdm1))
        
        return ovlp * term

    
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
        if not isinstance(self.ao_ovlp, numpy.ndarray):
            self.get_ao_ovlp()
            
        return reduce(numpy.dot, (mo_coeff1.T.conj(), self.ao_ovlp, mo_coeff2[:, mo_occ2>0]))

    
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
        Minv_block = numpy.block([[Minv, numpy.zeros((n, u-n))]])            
        trans_rdm1 = numpy.dot(A, Minv_block)
        return trans_rdm1
    
    
    def get_co_ovlp_oo(self, mo_coeff1, mo_coeff2):
        """
        Computes the occupied-occupied (oo) block of the CO overlap matrix by doing an SVD of the oo block 
        of the MO overlap matrix.
        """
        mo_ovlp_oo = self.get_mo_ovlp(mo_coeff1[:, :self.nelec], mo_coeff2[:, :self.nelec])
        u, s, vt = numpy.linalg.svd(mo_ovlp_oo)
        return u, s, vt.T.conj()
    
    
    def get_co_ovlp_vv(self, mo_coeff1, mo_coeff2):
        """
        Computes the virtual-virtual (vv) block of the CO overlap matrix by doing an SVD of the vv block 
        of the MO overlap matrix.
        """
        mo_ovlp_vv = self.get_mo_ovlp(mo_coeff1[:, self.nelec:], mo_coeff2[:, self.nelec:])
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
        if not isinstance(self.ao_ovlp, numpy.ndarray):
            self.get_ao_ovlp()
        
        mo_ovlp_vo = reduce(numpy.dot, (mo_coeff1[:, self.nelec:].T.conj(), self.ao_ovlp, mo_coeff2[:, :self.nelec]))
        co_ovlp_vo = numpy.dot(mo_ovlp_vo, v_oo)
        
        #co_coeff1_vir = numpy.dot(mo_coeff1[:, self.nelec:], u_vv)
        #co_coeff2_occ = numpy.dot(mo_coeff2[:, :self.nelec], v_oo)
        #co_ovlp_vo = reduce(numpy.dot, (co_coeff1_vir.T.conj(), self.ao_ovlp, co_coeff2_occ))
       
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
        if not isinstance(self.ao_ovlp, numpy.ndarray):
            self.get_ao_ovlp()

        mo_ovlp_ov = reduce(numpy.dot, (mo_coeff1[:, :self.nelec].T.conj(), self.ao_ovlp, mo_coeff2[:, self.nelec:]))
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
        co_ovlp_vv = reduce(numpy.dot, (mo_coeff1[:, self.nelec:].T.conj(), self.ao_ovlp, mo_coeff2[:, self.nelec:]))
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
        term = numpy.einsum('ijab,ia,jb,i,j->', co_t2, co_ovlp_ov, co_ovlp_ov, co_ovlp_oo_inv, co_ovlp_oo_inv)
        return prefactor * term 
    
        
    def get_S_ijpr_matrix(self, co_ovlp, co_ovlp_oo_diag):
        """
        Constructs the S_ijpr tensor used in the evaluation of <Psi_{UHF}|H R|Psi_{UHF}^{1}>.
        """
        nocc = self.nelec
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
        
    
    def get_S_ipr_matrix(self, co_ovlp, co_ovlp_oo_diag):
        """
        Constructs the S_ipr tensor used in the evaluation of <Psi_{UHF}|H R|Psi_{UHF}^{1}>.
        """
        nocc = self.nelec
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
    
    
    def energy_01_term2_case1(self, co_t2, co_eri, co_ovlp, co_ovlp_oo_diag):
        """
        Case 1: i = k, j = l
        """
        nocc = self.nelec
        nmo = self.nmo
        nvir = nmo - nocc
        co_eri_oovv = co_eri.transpose(0,2,1,3) - co_eri.transpose(0,2,3,1)
        co_ovlp_oo_inv = 1. / co_ovlp_oo_diag

        S_ijpr_matrix = self.get_S_ijpr_matrix(co_ovlp, co_ovlp_oo_diag)
        sumc_ijad = numpy.einsum('ijcd,ijca->ijad', co_eri_oovv, S_ijpr_matrix)
        sumb_ijad = numpy.einsum('ijab,ijdb->ijad', co_t2, S_ijpr_matrix)
        sumadij = numpy.einsum('ijad,ijad,i,j->', sumc_ijad, sumb_ijad, co_ovlp_oo_inv, co_ovlp_oo_inv)
                
        return sumadij * 0.25 * numpy.prod(co_ovlp_oo_diag)
    
    
    def energy_01_term2_case2(self, co_t2, co_eri, co_ovlp, co_ovlp_oo_diag):
        """
        Case 2: i = k, j != l
        """
        nocc = self.nelec
        nmo = self.nmo
        nvir = nmo - nocc
        co_eri_oovv = co_eri.transpose(0,2,1,3) - co_eri.transpose(0,2,3,1)
        co_ovlp_oo_inv = 1. / co_ovlp_oo_diag
        co_ovlp_vo = co_ovlp[nocc:, :nocc]
        co_ovlp_ov = co_ovlp[:nocc, nocc:]

        S_ipr_matrix = self.get_S_ipr_matrix(co_ovlp, co_ovlp_oo_diag)
        
        # 1st term.
        sumld_ic = numpy.einsum('ilcd,dl,l->ic', co_eri_oovv, co_ovlp_vo, co_ovlp_oo_inv)
        sumjb_ia = numpy.einsum('ijab,jb,j->ia', co_t2, co_ovlp_ov, co_ovlp_oo_inv)
        term1 = numpy.einsum('ic,ia,i,ica->', sumld_ic, sumjb_ia, co_ovlp_oo_inv, S_ipr_matrix)
        
        # 2nd term.
        sumd_ijc = numpy.einsum('ijcd,dj->ijc', co_eri_oovv, co_ovlp_vo)
        sumb_ija = numpy.einsum('ijab,jb->ija', co_t2, co_ovlp_ov)
        term2 = -numpy.einsum('ijc,ija,j,i,ica->', sumd_ijc, sumb_ija, 
                              numpy.power(co_ovlp_oo_inv, 2), co_ovlp_oo_inv, S_ipr_matrix)

        return (term1 + term2) * numpy.prod(co_ovlp_oo_diag)

    
    def energy_01_term2_case3(self, co_t2, co_eri, co_ovlp, co_ovlp_oo_diag):
        """
        Case 3: i != k, j != l
        """
        nocc = self.nelec
        nmo = self.nmo
        nvir = nmo - nocc
        co_eri_oovv = co_eri.transpose(0,2,1,3) - co_eri.transpose(0,2,3,1)
        co_ovlp_oo_inv = 1. / co_ovlp_oo_diag
        co_ovlp_vo = co_ovlp[nocc:, :nocc]
        co_ovlp_ov = co_ovlp[:nocc, nocc:]
        
        # 1st term.
        sumklcd = numpy.einsum('klcd,ck,dl,k,l->', co_eri_oovv, co_ovlp_vo, co_ovlp_vo, 
                               co_ovlp_oo_inv, co_ovlp_oo_inv)
        sumijab = numpy.einsum('ijab,ia,jb,i,j->', co_t2, co_ovlp_ov, co_ovlp_ov, co_ovlp_oo_inv, co_ovlp_oo_inv)
        term1 = 0.25 * sumklcd * sumijab
        
        # 2nd term.
        sumlcd_i = numpy.einsum('ilcd,ci,dl,l->i', co_eri_oovv, co_ovlp_vo, co_ovlp_vo, co_ovlp_oo_inv)
        sumjab_i = numpy.einsum('ijab,ia,jb,j->i', co_t2, co_ovlp_ov, co_ovlp_ov, co_ovlp_oo_inv)
        term2 = -numpy.einsum('i,i,i->', sumlcd_i, sumjab_i, numpy.power(co_ovlp_oo_inv, 2))
        
        # 3rd term.
        sumcd_ij = numpy.einsum('ijcd,ci,dj->ij', co_eri_oovv, co_ovlp_vo, co_ovlp_vo)
        sumab_ij = numpy.einsum('ijab,ia,jb->ij', co_t2, co_ovlp_ov, co_ovlp_ov)
        term3 = 0.5 * numpy.einsum('ij,ij,i,j->', sumcd_ij, sumab_ij, 
                                   numpy.power(co_ovlp_oo_inv, 2), numpy.power(co_ovlp_oo_inv, 2))
        
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
    
    
    #@profile(sort_by='cumulative', lines_to_print=30, strip_dirs=True) 
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

        if not isinstance(self.ao_ovlp, numpy.ndarray):
            self.get_ao_ovlp()

        s = reduce(numpy.dot, (mo_coeff1[:, mo_occ1>0].T.conj(), self.ao_ovlp, mo_coeff2[:, mo_occ2>0]))
        u, s, vt = numpy.linalg.svd(s)
        return numpy.prod(s)
    
    
    def get_ao_ovlp(self):
        """
        Compute the AO overlap matrix S.
        """
        s = self.mol.intor_symmetric('int1e_ovlp')
        self.ao_ovlp = scipy.linalg.block_diag(s, s)
        return self.ao_ovlp
    
    
    def get_mo_ovlp(self, mo_coeff1, mo_coeff2):
        """
        Calculates the MO overlap matrix.
        """
        if not isinstance(self.ao_ovlp, numpy.ndarray):
            self.get_ao_ovlp()
        
        return reduce(numpy.dot, (mo_coeff1.T.conj(), self.ao_ovlp, mo_coeff2))
    
    
    def get_ao_hcore(self):
        """
        Computes the AO `hcore` matrix.
        """
        hcore = self.mol.intor_symmetric('int1e_kin') + self.mol.intor_symmetric('int1e_nuc')
        return scipy.linalg.block_diag(hcore, hcore)
    
    
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
            self.mo_eri = gmp2_slow.ao2mo_slow(ao_eri, (moa,moa,moa,moa))
            self.mo_eri += gmp2_slow.ao2mo_slow(ao_eri, (mob,mob,mob,mob))
            self.mo_eri += gmp2_slow.ao2mo_slow(ao_eri, (moa,moa,mob,mob))
            self.mo_eri += gmp2_slow.ao2mo_slow(ao_eri, (mob,mob,moa,moa))
        
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
    
    
    def get_co_coeff(self, mo_coeff, u):
        return numpy.dot(mo_coeff, u)
    
    
    def get_co_eri_from_mo_eri(self, mo_coeff, u_oo):
        """
        Computes the CO eris from the MO eris. The orbitals are from |Psi_{UHF}>.
        
        Only works with mo_eri with shape (nmo, nmo, nmo, nmo).
        """
        if not isinstance(self.mo_eri, numpy.ndarray):
            if mo_coeff.dtype != numpy.complex:
                self.get_mo_eri_real(mo_coeff)
                
            else:
                self.get_mo_eri_complex(mo_coeff)
            
        # (nocc, nvir, nocc, nvir)
        mo_eri_ovov = self.mo_eri[:self.nelec, self.nelec:, :self.nelec, self.nelec:]
        return numpy.einsum('pi,qj,paqb->iajb', u_oo.conj(), u_oo.conj(), mo_eri_ovov)
    
    
    def get_co_t2(self, mo_t2, v_oo):
        """
        CO t2 amplitudes derived from MO t2 amplitudes elucidated with Garnet's method.
        """
        return numpy.einsum('ik,jl,ijab->klab', v_oo, v_oo, mo_t2)
    

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