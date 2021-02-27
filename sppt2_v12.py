"""
V12
    - This version does UHF -> UMP2 -> P.
    - Using simplified projector as default. 
    - CO version using Van Voorhis' equations.
    - Using DF-GMP2.
"""


from pyscf import gto, scf, mp, fci, lib
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
        For all attributes storing quantities over the quadrature points, 
        1st index is over rotation angle alpha and 2nd index is over angle beta.
        
        Attributes
            mol (gto Mole object): Stores the molecule.
            
            ao_eri (ndarray): ERIs in AO basis.
            
            ao_ovlp (ndarray): Overlap matrix S in AO basis.
            
            quad (ndarray): Array of [alphas, betas, ws], storing quadrature points 
                            `alphas, betas` and weights `ws`.
                                  
            nelec (int): Number of electrons.
            
            nmo (int): Total number of spatial orbitals (occupied + unoccupied).
            
            e_tot (complex): Total energy.
        """
        self.mol = mol
        self.rotated_mo_coeffs = None
        self.ao_eri = None
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
        
        self.quad = numpy.array([alphas, betas, gammas, ws])
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
            ''' Generate density matrix with broken spatial and spin symmetry by mixing
            HOMO and LUMO orbitals following ansatz in Szabo and Ostlund, Sec 3.8.7.

            psi_1a = np.cos(q)*psi_homo + np.sin(q)*psi_lumo
            psi_1b = np.cos(q)*psi_homo - np.sin(q)*psi_lumo

            psi_2a = -np.sin(q)*psi_homo + np.cos(q)*psi_lumo
            psi_2b =  np.sin(q)*psi_homo + np.cos(q)*psi_lumo
            Returns: 
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
    
    
    def get_double(self, mo_occ, i, j, a, b):
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
    
    
    def get_all_doubles(self, mo_occ):
        """
        Gets all possible doubles from the `mo_occ` at quad point (alpha, beta).
        
        Args
            mo_occ (ndarray): Array of HF MO occupations.
            
        Returns
            Dict {'ijab': mo_occ_ijab}
        """
        all_doubles = {}
        
        #occ_pairs = list(combinations([self.nelec-2, self.nelec-1], 2))
        occ_pairs = list(combinations(range(self.nelec), 2))
        vir_pairs = list(combinations(range(self.nelec, self.nmo), 2))
        
        for i, j in occ_pairs:
            for a, b in vir_pairs:
                key = str(i) + ',' + str(j) + ',' + str(a) + ',' + str(b)
                mo_occ_ijab = self.get_double(mo_occ, i, j, a, b)
                all_doubles[key] = mo_occ_ijab
        
        return all_doubles
        

    def get_matrix_element(self, mo_coeff1, mo_coeff2, mo_occ1, mo_occ2, verbose=False):
        """
        Computes the matrix elements 
            1. <(0)| H R |(0)>
            2. <(0)| H R |(0)^{ab}_{ij}>
        
        as specified by the input `mo_occ2` parameter.
        
        Args
            mo_coeff1 (ndarray): MO coefficient matrix for the HF state associated with the left determinant.
            mo_coeff2 (ndarray): ... right determinant.
            mo_occ1 (ndarray): MO occupation numbers for the left determinant. Used to specify ground state
                               determinants.
            mo_occ2 (ndarray): ... right determinant.
            divide (bool): Whether to divide by the overlap between the left and right determinants.
        """
        mo_hcore = self.get_mo_hcore(mo_coeff1)
        mo_eri = self.get_mo_eri(mo_coeff1)
        A = self.get_A_matrix(mo_coeff1, mo_coeff2, mo_occ2)
        M = self.get_M_matrix(A)
        trans_rdm1 = self.get_trans_rdm1(A, M)
        Minv = numpy.linalg.inv(M)
        ovlp = self.det_ovlp_v2(mo_coeff1, mo_coeff2, mo_occ1, mo_occ2)
        
        # Be careful with convention of dm1 and dm2
        #   dm1[q,p] = <p^\dagger q>
        #   dm2[p,q,r,s] = < p^\dagger r^\dagger s q >
        #   E = einsum('pq,qp', h1, dm1) + .5 * einsum('pqrs,pqrs', eri, dm2)
        # When adding dm1 contribution, dm1 subscripts need to be flipped
        #
        # sum{ h_ik rho_ki + 1/2 <ij||kl> rho_ki rho_lj}
        part1 = numpy.einsum('ikjl, ki->jl', mo_eri, trans_rdm1)
        part2 = numpy.einsum('iljk, ki->jl', mo_eri, trans_rdm1)
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
        
        """
        co_coeff1_vir = numpy.dot(mo_coeff1[:, self.nelec:], u_vv)
        co_coeff2_occ = numpy.dot(mo_coeff2[:, :self.nelec], v_oo)
        co_ovlp_vo = reduce(numpy.dot, (co_coeff1_vir.T.conj(), self.ao_ovlp, co_coeff2_occ))
        """
        return co_ovlp_vo
    
    
    def get_co_ovlp_vo_v2(self, mo_coeff1, mo_coeff2, u_vv, v_oo):
        """
        Computes the virtual-occupied (vo) block of the CO overlap matrix by transforming the vo block 
        of the MO overlap matrix as follows:
        
            S^{CO, vo} = [U^{vv}]^\dagger S^{MO, vo} V^{oo}.
        
        This is because for the rows we have:
        
            C^{CO, vv} = C^{MO, vv} U^{vv}
            
        for the columns we have:
            
            C^{CO, oo} = C^{MO, oo} V^{oo}
        """
        if not isinstance(self.ao_ovlp, numpy.ndarray):
            self.get_ao_ovlp()
        
        mo_ovlp_vo = reduce(numpy.dot, (mo_coeff1[:, self.nelec:].T.conj(), self.ao_ovlp, mo_coeff2[:, :self.nelec]))
        co_ovlp_vo = reduce(numpy.dot, (u_vv.T.conj(), mo_ovlp_vo, v_oo))
        
        """
        co_coeff1_vir = numpy.dot(mo_coeff1[:, self.nelec:], u_vv)
        co_coeff2_occ = numpy.dot(mo_coeff2[:, :self.nelec], v_oo)
        co_ovlp_vo = reduce(numpy.dot, (co_coeff1_vir.T.conj(), self.ao_ovlp, co_coeff2_occ))
        """
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
    
    
    def get_co_ovlp_ov_v2(self, mo_coeff1, mo_coeff2, u_oo, v_vv):
        """
        Computes the occupied-virtual (ov) block of the CO overlap matrix by transforming the ov block 
        of the MO overlap matrix as follows:
        
            S^{CO, ov} = [U^{oo}]^\dagger S^{MO, ov} V^{vv}
            
        This is because for the rows we have:
        
            C^{CO, oo} = C^{MO, oo} U^{oo}
            
        for the columns we have:
            
            C^{CO, vv} = C^{MO, vv} V^{vv}
        """
        if not isinstance(self.ao_ovlp, numpy.ndarray):
            self.get_ao_ovlp()

        mo_ovlp_ov = reduce(numpy.dot, (mo_coeff1[:, :self.nelec].T.conj(), self.ao_ovlp, mo_coeff2[:, self.nelec:]))
        co_ovlp_ov = reduce(numpy.dot, (u_oo.T.conj(), mo_ovlp_ov, v_vv))
        
        """
        co_coeff1_occ = numpy.dot(mo_coeff1[:, :self.nelec], u_oo)
        co_coeff2_vir = numpy.dot(mo_coeff2[:, self.nelec:], v_vv)
        co_ovlp_ov = reduce(numpy.dot, (co_coeff1_occ.T.conj(), self.ao_ovlp, co_coeff2_vir))
        """
        return co_ovlp_ov
    
    
    def get_co_ovlp(self, mo_coeff1, mo_coeff2, return_all=False):
        """
        CO transformation of only oo-block of S.
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

    
    def get_co_ovlp_v2(self, mo_coeff1, mo_coeff2, return_all=False):
        """
        CO transformation of both oo and vv-block of S.
        """
        u_oo, co_ovlp_oo_diag, v_oo = self.get_co_ovlp_oo(mo_coeff1, mo_coeff2)
        u_vv, co_ovlp_vv_diag, v_vv = self.get_co_ovlp_vv(mo_coeff1, mo_coeff2)
        co_ovlp_vo = self.get_co_ovlp_vo_v2(mo_coeff1, mo_coeff2, u_vv, v_oo)
        co_ovlp_ov = self.get_co_ovlp_ov_v2(mo_coeff1, mo_coeff2, u_oo, v_vv)
        co_ovlp_oo = numpy.diag(co_ovlp_oo_diag)
        co_ovlp_vv = numpy.diag(co_ovlp_vv_diag)
        co_ovlp = numpy.block([[co_ovlp_oo, co_ovlp_ov], [co_ovlp_vo, co_ovlp_vv]])
        
        if return_all:
            return co_ovlp, co_ovlp_oo_diag, co_ovlp_vv_diag, u_oo, u_vv, v_oo, v_vv

        return co_ovlp

    
    def energy_01_term1(self, mo_coeff1, mo_coeff2, mo_occ, co_t2):
        """
        Computes E_{HF} * sum_{i<j, a<b} [ t_{ijab} * <Psi_{UHF} | R | {Psi_{UHF}}_{ij}^{ab}> ] using Eq. 14
        from J. Chem. Phys. 139, 174104 (2013).
        
        - No CO transformation for vv block.
        - Iterate over i, j, a, b.
        """
        nmo = self.nmo
        nocc = self.nelec
        nvir = nmo - nocc
        
        # Get the U, S, V matrices.
        u_oo, co_ovlp_oo_diag, v_oo = self.get_co_ovlp_oo(mo_coeff1, mo_coeff2)
        
        # Get the ov-block of the CO overlap matrix.
        co_ovlp_ov = self.get_co_ovlp_ov(mo_coeff1, mo_coeff2, u_oo)
        
        # Unsimplified equation.
        prefactor = 0.5 * self.e_elec_hf * numpy.prod(co_ovlp_oo_diag)
        term = 0.

        for i in range(nocc):
            for j in range(nocc):
                for a in range(nvir):
                    for b in range(nvir):
                        term += co_t2[i, j, a, b] * co_ovlp_ov[i, a] * co_ovlp_ov[j, b] / (co_ovlp_oo_diag[i] * co_ovlp_oo_diag[j])

        term *= prefactor
        return term
    
    
    def energy_01_term1_v2(self, mo_coeff1, mo_coeff2, mo_occ, co_t2):
        """
        Computes E_{HF} * sum_{i<j, a<b} [ t_{ijab} * <Psi_{UHF} | R | {Psi_{UHF}}_{ij}^{ab}> ] using Eq. 14
        from J. Chem. Phys. 139, 174104 (2013).
        
        - CO transformation for vv block.
        - Iterate over i, j, a, b.
        """
        nmo = self.nmo
        nocc = self.nelec
        nvir = nmo - nocc
        
        # Get the U, S, V matrices.
        u_oo, co_ovlp_oo_diag, v_oo = self.get_co_ovlp_oo(mo_coeff1, mo_coeff2)
        u_vv, co_ovlp_vv_diag, v_vv = self.get_co_ovlp_vv(mo_coeff1, mo_coeff2)
        
        # Get the ov-block of the CO overlap matrix.
        co_ovlp_ov = self.get_co_ovlp_ov_v2(mo_coeff1, mo_coeff2, u_oo, v_vv)
        
        # Unsimplified equation.
        prefactor = 0.5 * self.e_elec_hf * numpy.prod(co_ovlp_oo_diag)
        term = 0.

        for i in range(nocc):
            for j in range(nocc):
                for a in range(nvir):
                    for b in range(nvir):
                        term += co_t2[i, j, a, b] * co_ovlp_ov[i, a] * co_ovlp_ov[j, b] / (co_ovlp_oo_diag[i] * co_ovlp_oo_diag[j])

        term *= prefactor
        return term
    
        
    def get_S_ijpr(self, co_ovlp, co_ovlp_oo_diag, inds):
        """
        Args
            inds (tuple): (i, j, p, r)
        """
        i, j, p, r = inds
        co_ovlp_oo_inv = numpy.diag(1./co_ovlp_oo_diag)
        co_ovlp_oo_inv[i, i] = co_ovlp_oo_inv[j, j] = 0.
        _prod = numpy.dot(co_ovlp_oo_inv, co_ovlp[:self.nelec, r])
        prod = numpy.dot(co_ovlp[p, :self.nelec], _prod)
        s_ijpr = co_ovlp[p, r] - prod
        return s_ijpr
    
    
    def get_S_ipr(self, co_ovlp, co_ovlp_oo_diag, inds):
        """
        Args
            inds (tuple): (i, p, r)
        """
        i, p, r = inds
        co_ovlp_oo_inv = numpy.diag(1./co_ovlp_oo_diag)
        co_ovlp_oo_inv[i, i]  = 0.
        _prod = numpy.dot(co_ovlp_oo_inv, co_ovlp[:self.nelec, r])
        prod = numpy.dot(co_ovlp[p, :self.nelec], _prod)
        s_ipr = co_ovlp[p, r] - prod
        return s_ipr
    
    
#     def get_S_ijpr_matrix(self, co_ovlp, co_ovlp_oo_diag):
#         def get_S_ij_matrix(co_ovlp, co_ovlp_oo_inv, inds):
#             """
#             Args
#                 inds (tuple): (i, j)
#             """
#             i, j = inds
#             co_ovlp_oo_inv_i = co_ovlp_oo_inv[i, i]
#             co_ovlp_oo_inv_j = co_ovlp_oo_inv[j, j]
#             co_ovlp_oo_inv[i, i] = co_ovlp_oo_inv[j, j] = 0.
#             _prod = numpy.dot(co_ovlp_oo_inv, co_ovlp[:self.nelec, :])
#             prod = numpy.dot(co_ovlp[:, :self.nelec], _prod)
#             s_ij = co_ovlp - prod
            
#             # Reset values.
#             co_ovlp_oo_inv[i, i] = co_ovlp_oo_inv_i
#             co_ovlp_oo_inv[j, j] = co_ovlp_oo_inv_j
#             return s_ij
        
#         nocc = self.nelec
#         nmo = self.nmo
#         co_ovlp_oo_inv = numpy.diag(1./co_ovlp_oo_diag)
#         s_ijpr = numpy.array([[get_S_ij_matrix(co_ovlp, co_ovlp_oo_inv, (i, j)) for j in range(nocc)] for i in range(nocc)])
#         return s_ijpr
        
    
#     def get_S_ipr_matrix(self, co_ovlp, co_ovlp_oo_diag):
#         def get_S_i_matrix(co_ovlp, co_ovlp_oo_inv, i):
#             co_ovlp_oo_inv_i = co_ovlp_oo_inv[i, i]
#             co_ovlp_oo_inv[i, i]  = 0.
#             _prod = numpy.dot(co_ovlp_oo_inv, co_ovlp[:self.nelec, :])
#             prod = numpy.dot(co_ovlp[:, :self.nelec], _prod)
#             s_i = co_ovlp - prod
            
#             # Reset value.
#             co_ovlp_oo_inv[i, i] = co_ovlp_oo_inv_i
#             return s_i
        
#         nocc = self.nelec
#         nmo = self.nmo
#         co_ovlp_oo_inv = numpy.diag(1./co_ovlp_oo_diag)
#         s_ipr = numpy.array([get_S_i_matrix(co_ovlp, co_ovlp_oo_inv, i) for i in range(nocc)])
#         return s_ipr
    
    
    def energy_01_term2_case1(self, co_t2, co_eri, co_ovlp, co_ovlp_oo_diag):
        """
        Case 1: i = k, j = l
        """
        nocc = self.nelec
        nmo = self.nmo
        nvir = nmo - nocc
        tot = 0.
        
        for i in range(nocc): # 10
            for j in range(nocc): # 10
                
                sumad = 0.
                for a in range(nvir): # 2
                    for d in range(nvir): # 2
                        _a = a + nocc
                        _d = d + nocc
                        
                        # Sum over c.
                        sumc = 0.
                        for c in range(nvir): # 2
                            _c = c + nocc
                            inds = (i, j, _c, _a)
                            sumc += (co_eri[i, c, j, d] - co_eri[i, d, j, c]) * self.get_S_ijpr(co_ovlp, co_ovlp_oo_diag, inds)
                           
                        # Sum over b.
                        sumb = 0.
                        for b in range(nvir): # 2
                            _b = b + nocc
                            inds = (i, j, _d, _b)
                            sumb += co_t2[i, j, a, b] * self.get_S_ijpr(co_ovlp, co_ovlp_oo_diag, inds)
                            
                        sumad += sumb * sumc
                
                #print(tot)
                tot += sumad / (co_ovlp_oo_diag[i] * co_ovlp_oo_diag[j])
                
        return tot * 0.25 * numpy.prod(co_ovlp_oo_diag)
    
    
    def energy_01_term2_case2(self, co_t2, co_eri, co_ovlp, co_ovlp_oo_diag, verbose=False):
        """
        Case 2: i = k, j != l
        """
        nocc = self.nelec
        nmo = self.nmo
        nvir = nmo - nocc
        tot = 0.
        tot_v2 = 0.
        
        for i in range(nocc):
            for a in range(nvir):
                for c in range(nvir):
                    _a = a + nocc
                    _c = c + nocc
                    
                    # First term.
                    # Sum over l, d.
                    sumld = 0.
                    for l in range(nocc):
                        for d in range(nvir):
                            _d = d + nocc
                            sumld += (co_eri[i, c, l, d] - co_eri[i, d, l, c]) * co_ovlp[_d, l] / co_ovlp_oo_diag[l]

                    # Sum over j, b.
                    sumjb = 0.
                    for j in range(nocc):
                        for b in range(nvir):
                            _b = b + nocc
                            sumjb += co_t2[i, j, a, b] * co_ovlp[j, _b] / co_ovlp_oo_diag[j]
                    
                    # Second term.
                    # Sum over j.
                    sumj = 0.
                    for j in range(nocc):
                        
                        # Sum over b.
                        sumb = 0.
                        for b in range(nvir):
                            _b = b + nocc
                            sumb += co_t2[i, j, a, b] * co_ovlp[j, _b] / co_ovlp_oo_diag[j]
                        
                        # Sum over d.
                        sumd = 0.
                        for d in range(nvir):
                            _d = d + nocc
                            sumd += (co_eri[i, c, j, d] - co_eri[i, d, j, c]) * co_ovlp[_d, j] / co_ovlp_oo_diag[j]
                        
                        sumj += sumb * sumd
                        
                    inds = (i, _c, _a)
                    s_ipr = self.get_S_ipr(co_ovlp, co_ovlp_oo_diag, inds)
                    tot += (sumld * sumjb - sumj) * s_ipr / co_ovlp_oo_diag[i]

        return tot * numpy.prod(co_ovlp_oo_diag)

        
    def energy_01_term2_case3(self, co_t2, co_eri, co_ovlp, co_ovlp_oo_diag):
        """
        Case 3: i != k, j != l
        """
        nocc = self.nelec
        nmo = self.nmo
        nvir = nmo - nocc
        term1 = 0.
        term2 = 0.
        term3 = 0.
        
        # First term.
        sumklcd = 0.
        for k in range(nocc):
            for l in range(nocc):
                for c in range(nvir):
                    for d in range(nvir):
                        _c = c + nocc
                        _d = d + nocc
                        sumklcd += (co_eri[k, c, l, d] - co_eri[k, d, l, c]) * co_ovlp[_c, k] * co_ovlp[_d, l] / (co_ovlp_oo_diag[k] * co_ovlp_oo_diag[l])
                        
        sumijab = 0.
        for i in range(nocc):
            for j in range(nocc):
                for a in range(nvir):
                    for b in range(nvir):
                        _a = a + nocc
                        _b = b + nocc
                        sumijab += co_t2[i, j, a, b] * co_ovlp[i, _a] * co_ovlp[j, _b] / (co_ovlp_oo_diag[i] * co_ovlp_oo_diag[j])
        
        term1 = 0.25 * sumklcd * sumijab
        
        # Second term.
        for i in range(nocc):

            sumlcd = 0.
            for l in range(nocc):
                for c in range(nvir):
                    for d in range(nvir):
                        _c = c + nocc
                        _d = d + nocc
                        sumlcd += (co_eri[i, c, l, d] - co_eri[i, d, l, c]) * co_ovlp[_c, i] * co_ovlp[_d, l] / (co_ovlp_oo_diag[i] * co_ovlp_oo_diag[l])
                        
            sumjab = 0.
            for j in range(nocc):
                for a in range(nvir):
                    for b in range(nvir):
                        _a = a + nocc
                        _b = b + nocc
                        sumjab += co_t2[i, j, a, b] * co_ovlp[i, _a] * co_ovlp[j, _b] / (co_ovlp_oo_diag[i] * co_ovlp_oo_diag[j])
                        
            term2 += sumlcd * sumjab
        
        # Third term.
        for i in range(nocc):
            for j in range(nocc):
                
                sumcd = 0.
                for c in range(nvir):
                    for d in range(nvir):
                        _c = c + nocc
                        _d = d + nocc
                        sumcd += (co_eri[i, c, j, d] - co_eri[i, d, j, c]) * co_ovlp[_c, i] * co_ovlp[_d, j] / (co_ovlp_oo_diag[i] * co_ovlp_oo_diag[j])
                        
                sumab = 0.
                for a in range(nvir):
                    for b in range(nvir):
                        _a = a + nocc
                        _b = b + nocc
                        sumab += co_t2[i, j, a, b] * co_ovlp[i, _a] * co_ovlp[j, _b] / (co_ovlp_oo_diag[i] * co_ovlp_oo_diag[j])
                        
                term3 += sumcd * sumab
        
        term3 *= 0.5
        
        tot = (term1 - term2 + term3) * numpy.prod(co_ovlp_oo_diag)
        return tot

    
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
        - Transform t_ijab with V, V^\dagger
        """
        nocc = self.nelec
        nvir = self.nmo - nocc
        
        mo_coeff1 = ump2.mo_coeff
        mo_occ = ump2.mo_occ
        mo_t2 = ump2.t2
        
        co_ovlp, co_ovlp_oo_diag, u_oo, v_oo = self.get_co_ovlp(mo_coeff1, mo_coeff2, return_all=True)
        u = numpy.block([[u_oo, numpy.zeros((nocc, nvir))], [numpy.zeros((nvir, nocc)), numpy.eye(nvir)]])
        co_coeff1 = self.get_co_coeff(mo_coeff1, u)
        v_vv = numpy.eye(nvir)
        co_t2 = self.get_co_t2(mo_t2, v_oo, v_vv)
        co_eri = self.get_co_eri(co_coeff1)
        
        term1 = self.energy_01_term1(mo_coeff1, mo_coeff2, mo_occ, co_t2)
        term2 = self.energy_01_term2(co_t2, co_eri, co_ovlp, co_ovlp_oo_diag)
        norm_01 = term1 / self.e_elec_hf
        
        return term1 + term2, norm_01
    
    
    def energy_01_v2(self, ump2, mo_coeff2):
        """
        - CO transformation for vv block.
        - Transform t_ijab with V, V^\dagger
        """
        nocc = self.nelec
        nvir = self.nmo - nocc
        
        mo_coeff1 = ump2.mo_coeff
        mo_occ = ump2.mo_occ
        mo_t2 = ump2.t2
        
        co_ovlp, co_ovlp_oo_diag, co_ovlp_vv_diag, u_oo, u_vv, v_oo, v_vv = self.get_co_ovlp_v2(mo_coeff1, mo_coeff2, return_all=True)
        u = numpy.block([[u_oo, numpy.zeros((nocc, nvir))], [numpy.zeros((nvir, nocc)), u_vv]])
        co_coeff1 = self.get_co_coeff(mo_coeff1, u)
        co_t2 = self.get_co_t2(mo_t2, v_oo, v_vv)
        co_eri = self.get_co_eri(co_coeff1)
        
        term1 = self.energy_01_term1_v2(mo_coeff1, mo_coeff2, mo_occ, co_t2)
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
                _energy_00 = self.get_matrix_element(mo_coeff, rot_mo_coeff, mo_occ, mo_occ, verbose=verbose)
                _norm_00 = self.det_ovlp_v2(mo_coeff, rot_mo_coeff, mo_occ, mo_occ)
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
                        _energy_00 = self.get_matrix_element(mo_coeff, rot_mo_coeff, mo_occ, mo_occ, verbose=verbose)
                        _norm_00 = self.det_ovlp_v2(mo_coeff, rot_mo_coeff, mo_occ, mo_occ)
                        energy_00 += coeff * _energy_00
                        norm_00 += coeff * _norm_00

                        if just_hf:
                            continue

                        _energy_01, _norm_01 = self.energy_01(ump2, rot_mo_coeff)
                        energy_01 += coeff * _energy_01
                        norm_01 += coeff * _norm_01
                    
        energy = (energy_00 + energy_01) / (norm_00 + norm_01) + self.mol.energy_nuc()
        return energy
    
    
    def energy_v2(self, s, m, k, ump2, proj='part', N_alpha=None, N_beta=None, N_gamma=None, 
                  just_hf=False, verbose=False):
        """
        Compute the total energy <Psi_{UHF}|H P|Psi_{UMP2}> / <Psi_{UHF}|P|Psi_{UMP2}>.
        
        - CO transformation for vv block.
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
                _energy_00 = self.get_matrix_element(mo_coeff, rot_mo_coeff, mo_occ, mo_occ, verbose=verbose)
                _norm_00 = self.det_ovlp_v2(mo_coeff, rot_mo_coeff, mo_occ, mo_occ)
                energy_00 += coeff * _energy_00
                norm_00 += coeff * _norm_00

                if just_hf:
                    continue

                _energy_01, _norm_01 = self.energy_01_v2(ump2, rot_mo_coeff)
                energy_01 += coeff * _energy_01
                norm_01 += coeff * _norm_01
            
        elif proj == 'full':
            for a in range(N_alpha):
                for b in range(N_beta):
                    for c in range(N_gamma):
                        rot_mo_coeff = rot_mo_coeffs[a, b, c]
                        coeff = coeffs[a, b, c]
                        _energy_00 = self.get_matrix_element(mo_coeff, rot_mo_coeff, mo_occ, mo_occ, verbose=verbose)
                        _norm_00 = self.det_ovlp_v2(mo_coeff, rot_mo_coeff, mo_occ, mo_occ)
                        energy_00 += coeff * _energy_00
                        norm_00 += coeff * _norm_00

                        if just_hf:
                            continue

                        _energy_01, _norm_01 = self.energy_01_v2(ump2, rot_mo_coeff)
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
                _norm_00 = self.det_ovlp_v2(mo_coeff, rot_mo_coeff, mo_occ, mo_occ)
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
                        _norm_00 = self.det_ovlp_v2(mo_coeff, rot_mo_coeff, mo_occ, mo_occ)
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
        
        NOTE: Has some sign errors when computing overlaps between double excitations
              < Psi_{ij}^{ab} | Psi_{kl}^{cd} >.

        Return:
            A list:
                the product of single values: float
                x_a: :math:`\mathbf{U} \mathbf{\Lambda}^{-1} \mathbf{V}^\dagger`
                They are used to calculate asymmetric density matrix
        """
        if numpy.sum(mo_occ1) != numpy.sum(mo_occ2):
            raise RuntimeError('Electron numbers are not equal. Electronic coupling does not exist.')

        if not isinstance(self.ao_ovlp, numpy.ndarray):
            self.get_ao_ovlp()

        s = reduce(numpy.dot, (mo_coeff1[:, mo_occ1>0].T.conj(), self.ao_ovlp, mo_coeff2[:, mo_occ2>0]))
        u, s, vt = numpy.linalg.svd(s)
        return numpy.prod(s)
    
    
    def det_ovlp_v2(self, mo_coeff1, mo_coeff2, mo_occ1, mo_occ2):
        """
        Calculates the overlap between two different determinants by computing a matrix determinant.
        """
        if numpy.sum(mo_occ1) != numpy.sum(mo_occ2):
            raise RuntimeError('Electron numbers are not equal. Electronic coupling does not exist.')

        if not isinstance(self.ao_ovlp, numpy.ndarray):
            self.get_ao_ovlp()
            
        M = reduce(numpy.dot, (mo_coeff1[:, mo_occ1>0].T.conj(), self.ao_ovlp, mo_coeff2[:, mo_occ2>0]))
        return numpy.linalg.det(M)
    
    
    def det_ovlp_v3(self, mo_coeff1, mo_coeff2, _mo_occ1=None, _mo_occ2=None):
        """
        Calculates the overlap between two different determinants by computing a matrix determinant.
        The matrix is constructed by replacing occupied orbitals with virtual orbitals in situ. 
        
        Args
            _mo_occ1 (list): List of orbital indices to indicate occupied-virtual excitations, 
                             e.g. [i, j, a, b]
        """
        if not isinstance(self.ao_ovlp, numpy.ndarray):
            self.get_ao_ovlp()
            
        nocc = self.nelec
        i, j, a, b = 0, 1, 0, 1
        k, l, c, d = 0, 1, 0, 1

        if _mo_occ1 is not None:
            k, l, c, d = _mo_occ1
            
        if _mo_occ2 is not None:
            i, j, a, b = _mo_occ2
            
        mo_occ1 = list(range(nocc)) # [0, 1,..., nocc]
        mo_occ2 = list(range(nocc))
        mo_occ1[k], mo_occ1[l] = c, d # k->c, l->d
        mo_occ2[i], mo_occ2[j] = a, b # i->a, j->b

        M = reduce(numpy.dot, (mo_coeff1[:, mo_occ1].T.conj(), self.ao_ovlp, mo_coeff2[:, mo_occ2]))
        return numpy.linalg.det(M)
                          
    
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
        self.ao_eri = self.mol.intor('int2e', aosym='s8')
        
        
    def get_mo_eri(self, mo_coeff):
        """
        Transforms the AO eris to MO eris.
        """
        if not isinstance(self.ao_eri, numpy.ndarray):
            self.get_ao_eri()
            
        moa = mo_coeff[:self.mol.nao]
        mob = mo_coeff[self.mol.nao:]
        mo_eri = gmp2_slow.ao2mo_slow(self.ao_eri, (moa,moa,moa,moa))
        mo_eri += gmp2_slow.ao2mo_slow(self.ao_eri, (mob,mob,mob,mob))
        mo_eri += gmp2_slow.ao2mo_slow(self.ao_eri, (moa,moa,mob,mob))
        mo_eri += gmp2_slow.ao2mo_slow(self.ao_eri, (mob,mob,moa,moa))
        return mo_eri
    
    
    def get_mo_eri_v2(self, mo_coeff):
        """
        Transforms the AO eris to MO eris.
        """
        if not isinstance(self.ao_eri, numpy.ndarray):
            self.get_ao_eri()
            
        nocc = self.nelec
        nao = self.mol.nao
        nvir = self.nmo - nocc
        orboa = mo_coeff[:nao, :nocc]
        orbob = mo_coeff[nao:, :nocc]
        orbva = mo_coeff[:nao, nocc:]
        orbvb = mo_coeff[nao:, nocc:]
        mo_eri = gmp2_slow.ao2mo_slow(self.ao_eri, (orboa,orbva,orboa,orbva))
        mo_eri += gmp2_slow.ao2mo_slow(self.ao_eri, (orbob,orbvb,orbob,orbvb)) 
        mo_eri1 = gmp2_slow.ao2mo_slow(self.ao_eri, (orboa,orbva,orbob,orbvb))
        mo_eri += mo_eri1
        mo_eri += mo_eri1.transpose(2,3,0,1) 
        return mo_eri
    
    
    def get_co_coeff(self, mo_coeff, u):
        return numpy.dot(mo_coeff, u)
    
    
    def get_co_eri(self, co_coeff):
        """
        Transforms AO eris to CO eris.
        CO[i, j, a, b] where 0 <= i, j < nocc
                             0 <= a, b < nvir.
        """
        if not isinstance(self.ao_eri, numpy.ndarray):
            self.get_ao_eri()
            
        nocc = self.nelec
        nao = self.mol.nao
        nvir = self.nmo - nocc
        orboa = co_coeff[:nao, :nocc]
        orbob = co_coeff[nao:, :nocc]
        orbva = co_coeff[:nao, nocc:]
        orbvb = co_coeff[nao:, nocc:]
        co_eri = gmp2_slow.ao2mo_slow(self.ao_eri, (orboa,orbva,orboa,orbva))
        co_eri += gmp2_slow.ao2mo_slow(self.ao_eri, (orbob,orbvb,orbob,orbvb)) 
        co_eri1 = gmp2_slow.ao2mo_slow(self.ao_eri, (orboa,orbva,orbob,orbvb))
        co_eri += co_eri1
        co_eri += co_eri1.transpose(2,3,0,1) 
        return co_eri
        
    
    def get_co_eri_from_mo_eri(self, mo_eri, u_occ, u_vir):
        """
        Computes the CO eris from the MO eris. The orbitals are from |Psi_{UHF}>.
        """
        mo_eri_ovov = mo_eri[:self.nelec, self.nelec:, :self.nelec, self.nelec:]
        return numpy.einsum('pi, qj, ra, sb, prqs->iajb', u_occ.conj(), u_occ.conj(), u_vir, u_vir, mo_eri_ovov)
    
    
    def get_co_eri_from_mo_eri_v2(self, mo_eri, u_occ):
        """
        Computes the CO eris from the MO eris. The orbitals are from |Psi_{UHF}>.
        """
        mo_eri_ovov = mo_eri[:self.nelec, self.nelec:, :self.nelec, self.nelec:]
        return numpy.einsum('pi, qj, paqb->iajb', u_occ.conj(), u_occ.conj(), mo_eri_ovov)
    
    
    def get_co_t2(self, mo_t2, v_oo, v_vv):
        """
        CO t2 amplitudes derived from MO t2 amplitudes elucidated with Garnet's method.
        """
        return numpy.einsum('ik, jl, ca, db, ijab->klcd', v_oo, v_oo, v_vv, v_vv, mo_t2)
    

# Other functions.
    
def build_chain(n, bond_length, basis='sto-3g'):
    """
    Builds a hydrogen chain of n H atoms.

    Args
        n (int): number of H atoms.
        bond_length (float): bond length between adjacent H atoms.
        basis (str): type of basis to use.

    Returns
        Hchain
    """
    print('Building molecule...')
    print('Using basis set {}'.format(basis))
    Hchain = gto.Mole()
    Hchain.atom = [['H', (bond_length * i, 0., 0.)] for i in range(n)]
    Hchain.basis = basis
    Hchain.spin = 0
    
    if n % 2 == 1:
        Hchain.spin = 1
        
    Hchain.build()
    return Hchain
    
    
def bond_curve(n, s, m, k, N_alpha=None, N_beta=None, N_gamma=None, proj='part', just_hf=False,
               dm_init=0, verbose=False, start=0.5, stop=5., points=50, basis='sto-3g'):
    """
    Computes the bond curve using SPPT2, GMP2, and FCI.
    
    Args
        s (int): Total spin S eigenvalue.
        m (int): Final Sz eigenvalue.
        k (int): Initial Sz eigenvalue.
        divide_norm (bool): Specifies whether to divide by <Psi|Psi>.
    """
    bond_lengths = numpy.linspace(start, stop, points)
    dm0 = None
    gdms = None
    e_sppt2s = []
    e_ump2s = []
    e_fcis = []
    
    for bond_length in bond_lengths:
        print('\nBond Length: {}'.format(bond_length))
        mol = build_chain(n, bond_length, basis)
        nao = mol.nao
        test = SPPT2(mol)
        uhf = test.do_uhf(dm_init=dm_init, dm0=dm0)
        
        if dm_init != 0:
            dm0 = uhf.make_rdm1()
            dm0[0][0] += 1.
            
        ump2 = test.do_mp2(uhf)

        # Test energy.
        e_sppt2 = test.energy(s, m, k, uhf, ump2, N_alpha=N_alpha, N_beta=N_beta, N_gamma=N_gamma, proj=proj,
                              just_hf=just_hf, verbose=verbose)
        
        print('\n=====================================')
        print(f'SPPT2 E = {e_sppt2}\n')

        # FCI energy.
        rhf = scf.RHF(mol).run(verbose=0)
        fci_out = fci.FCI(rhf).run(verbose=0)
        print(f'FCI E = {fci_out.e_tot}')
        print('=====================================\n')

        # Store energies.
        e_sppt2s.append(e_sppt2)
        e_ump2s.append(ump2.e_tot)
        e_fcis.append(fci_out.e_tot)
        
    return bond_lengths, e_sppt2s, e_ump2s, e_fcis
    
                         
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