"""
Optimization
- math.acos much faster than np.arccos
- use more array operations [made changes]
"""

from pyscf import gto, scf, mp, fci
from pyscf.mp import gmp2_slow
import numpy
import scipy.special
from functools import reduce
from itertools import combinations

class SPPT2:
    def __init__(self, mol, fock=None):
        """
        For all attributes storing quantities over the quadrature points, 
        1st index is over rotation angle alpha and 2nd index is over angle beta.
        
        Attributes
            mol (gto Mole object): Stores the molecule.
            
            fock (ndarray): UHF Fock. List of 2D Fock matrices for alpha, beta spin, i.e.
                            fock[0] contains the alpha spin Fock matrix,
                            fock[1] contains the beta spin Fock matrix.

            rotated_focks (ndarray): Stores the rotated Fock matrices at the quadrature 
                                     points.
            
            ao_eri (ndarray): ERIs in AO basis.
            
            ao_ovlp (ndarray): Overlap matrix S in AO basis.
            
            quad (ndarray): Array of [alphas, betas, ws], storing quadrature points 
                            `alphas, betas` and weights `ws`.
            
            ghfs (ndarray): Stores GHF objects at each quad point.
            
            t2s (ndarray): Stores t2 amplitude matrices at each quad point.
                                  
            nelec (int): Number of electrons.
            
            nmo (int): Total number of spatial orbitals (occupied + unoccupied).
            
            e_tot (complex): Total energy.
        """
        self.mol = mol
        self.fock = fock 
        self.rotated_focks = None
        self.ao_eri = None
        self.ao_ovlp = None
        self.quad = None
        self.ghfs = None
        self.t2s = None
        self.quad_rdm1s = None
        self.quad_rdm2s = None
        self.rdm1_sym = None
        self.rdm2_sym = None
        
        self.nelec = mol.nelectron
        self.na = self.fock[0].shape[1] # Number of spatial MOs with spin alpha.
        self.nb = self.fock[1].shape[1] # Number of spatial MOs with spin beta.
        self.nmo = mol.nao
        
        if self.fock is not None:
            self.nmo = 2 * self.fock[0].shape[1] # Number of spatial MOs.
            
        self.S = (self.na - self.nb)/2
        self.e_tot = None
        
        
    def generate_quad(self):
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
        Ng = int(numpy.ceil((Omega_max/2 + self.S + 1) / 2))
        
        # Quadrature points and weights.
        # x = cos(beta) in [-1, 1]
        xs, ws = numpy.polynomial.legendre.leggauss(Ng)
        betas = numpy.arccos(xs)
        sorted_inds = numpy.argsort(betas)
        betas.sort()
        ws = ws[sorted_inds]
        
        # For alpha.
        # Number of points.
        Nt = 2 * Ng
        
        # Quadrature points.
        alphas = numpy.linspace(numpy.pi/Ng, 2*numpy.pi, Nt)
        self.quad = numpy.array([alphas, betas, ws])
        
        return alphas, betas, ws
        
        
    def rotate_fock(self, alpha, beta):
        """
        Get rotated Fock operator.
        """
        fock_a = self.fock[0]
        fock_b = self.fock[1]
        nao = self.mol.nao
        fock = numpy.zeros([2*nao, 2*nao],dtype = numpy.complex128)

        fplus = 0.5 * (fock_a + fock_b)
        fminus = 0.5 * (fock_a - fock_b)
        
        fock[:nao, :nao] += fplus
        fock[nao:, nao:] += fplus

        fock[:nao, :nao] += fminus * numpy.cos(beta)
        fock[:nao, nao:] += fminus * numpy.sin(beta) * numpy.exp(-1j*alpha)
        fock[nao:, :nao] += fminus * numpy.sin(beta) * numpy.exp(1j*alpha)
        fock[nao:, nao:] += fminus * (-numpy.cos(beta))

        return fock
    
    
    def get_quad_rotated_focks(self):
        """
        Get the rotated Fock operators at the quadrature points.
        """
        alphas, betas, ws = None, None, None
        self.rotated_focks = []
        
        if self.quad is None:
            alphas, betas, ws = self.generate_quad()
            
        else: 
            alphas, betas, ws = self.quad
        
        for alpha in alphas:
            focks_beta = []
            
            for beta in betas:
                focks_beta.append(self.rotate_fock(alpha, beta))
                
            self.rotated_focks.append(focks_beta)
                
        self.rotated_focks = numpy.array(self.rotated_focks)
        
        
    def get_wigner_d(self, s, m, k, beta):
        """
        Wigner small d-matrix.
        """
        nmin = max(0, k-m)
        nmax = min(s+k, s-m)
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
    
    
    def do_mp2(self, rotated_fock):
        """
        Performs MP2 on a rotated Fock matrix and returns an GMP2 object.
        """
        # Construct GHF object to feed to MP2.
        ghf = scf.GHF(self.mol)
        ghf.get_fock = lambda *args, **kwargs : rotated_fock
        
        # We don't want to run SCF procedure.
        ghf.max_cycle = 0
        ghf.kernel()
        
        # Construct MP2 object.
        mp2 = mp.GMP2(ghf)
        mp2.kernel()
        
        return mp2
    
    
    def get_quad_ghfs_t2s(self):
        """
        Get the GHF objects and MP2 t2 amplitudes at each quadrature point.
        """
        if self.rotated_focks is None:
            self.get_quad_rotated_focks()
            
        focks = self.rotated_focks
        ghfs = []
        t2s = []
        
        # Iterate over alpha.
        for row_a in focks:
            ghfs_beta = []
            t2s_beta = []
            
            # Iterate over beta.
            for fock_ab in row_a:
                
                # Construct GHF object to feed to MP2.
                ghf = scf.GHF(self.mol)
                ghf.get_fock = lambda *args, **kwargs : fock_ab

                # We don't want to run SCF procedure.
                ghf.max_cycle = 0
                ghf.kernel()

                # Construct MP2 object.
                mp2 = mp.GMP2(ghf)
                mp2.kernel()
                
                ghfs_beta.append(ghf)
                t2s_beta.append(mp2.t2) # t2[i,j,a,b]  (i,j in occ, a,b in virt)
            
            ghfs.append(ghfs_beta)
            t2s.append(t2s_beta)
        
        ghfs = numpy.array(ghfs)
        t2s = numpy.array(t2s)
        self.ghfs = ghfs
        self.t2s = t2s
        return ghfs, t2s
    
    
    def quad_coeffs(self, s, m, k):
        """
        Returns a 2D array of the coefficient at each quad point.
        """
        if k != 0:
            print(k)
            print('k != 0. Quadrature coefficients are all zero.')
            return 0
        
        coeffs = []
        
        if not isinstance(self.quad, numpy.ndarray):
            alphas, betas, ws = self.generate_quad()
            
        alphas, betas, ws = self.quad
        Ng = len(betas)
        prefactor = (2*s + 1) / (4*Ng)
        
        for t, alpha_t in enumerate(alphas):
            coeffs_beta = []
            
            for g, beta_g in enumerate(betas):
                coeffs_beta.append(ws[g] * self.get_wigner_d(s, m, k, beta_g) * numpy.exp(1j * m * alpha_t))
                
            coeffs.append(coeffs_beta)
        
        coeffs = prefactor * numpy.array(coeffs)
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
        
        occ_pairs = list(combinations(range(self.nelec), 2))
        vir_pairs = list(combinations(range(self.nelec, self.nmo), 2))
        
        for i, j in occ_pairs:
            for a, b in vir_pairs:
                key = str(i) + ',' + str(j) + ',' + str(a) + ',' + str(b)
                mo_occ_ijab = self.get_double(mo_occ, i, j, a, b)
                all_doubles[key] = mo_occ_ijab
        
        return all_doubles
        

    def energy_xterms(self, s, m, k, order=3, separate=False):
        """
        Computes the energy cross terms sum{ w_g* w_g' <\tilde{psi}_g | H | \tilde{psi}_g'> } 
        due to the spin-projection.
        
        Args
            s (int): Total spin S eigenvalue.
            m (int): Final Sz eigenvalue.
            k (int): Initial Sz eigenvalue.
            order (int): Specifies the order of the SPPT2 correction.
                        
                         0: sum{ |w_g|^2 E_mp2 <\tilde{psi}_g | \tilde{psi}_g> }
                         1: `0` + sum{ w_g* w_g' <(0) | H | (0)> }
                         2: `1` + sum{ w_g* w_g' [<(0) | H | (1)> + <(1) | H | (0)>] }
                         3: `2` + sum{ w_g* w_g' <(1) | H | (1)> }
        
            separate (bool): Whether to compute the corrections separately, i.e. not add them up.
            
        Returns
            The energy cross terms.
        
        TODO: Optimize loops.
        """
        if not isinstance(self.ghfs, numpy.ndarray):
            self.get_quad_ghfs_t2s()
        
        ghfs, t2s = self.ghfs, self.t2s
        
        # 1st index corresponds to alpha, 2nd index corresponds to beta.
        n_alpha = len(ghfs)
        n_beta = len(ghfs[0])
        
        # Store for each quad point.
        # < psi(0)_g | H | psi(0)_g' >
        matrix_elems_00 = numpy.zeros((n_alpha, n_beta, n_alpha, n_beta), dtype=numpy.complex)
        
        # < psi(0)_g | H | psi(1)_g' >
        matrix_elems_01 = numpy.zeros((n_alpha, n_beta, n_alpha, n_beta), dtype=numpy.complex)
        
        # < psi(1)_g | H | psi(1)_g' >
        matrix_elems_11 = numpy.zeros((n_alpha, n_beta, n_alpha, n_beta), dtype=numpy.complex)
        
        nterm = 0
        
        # Loop over GHFs.
        for row1, ghf_row1 in enumerate(ghfs):
            for row2, ghf_row2 in enumerate(ghfs):
                for col1, ghf1 in enumerate(ghf_row1):
                    for col2, ghf2 in enumerate(ghf_row2):
                        if (row1, col1) == (row2, col2): # We only want diff quad points.
                            continue
                        
                        nterm += 1
                        mo_coeff1 = ghf1.mo_coeff
                        mo_coeff2 = ghf2.mo_coeff
                        mo_occ1 = ghf1.mo_occ
                        mo_occ2 = ghf2.mo_occ
                        
                        # Array of doubly excited mo_occs.
                        mo_occ1_ds = self.get_all_doubles(mo_occ1)
                        mo_occ2_ds = self.get_all_doubles(mo_occ2)
                        
                        # < psi(0)_g | H | psi(0)_g' >
                        matrix_elems_00[row1, col1, row2, col2] = self.get_matrix_element(
                            mo_coeff1, mo_coeff2, mo_occ1, mo_occ2)
                        
                        if order == 1:
                            continue
                        
                        # < psi(0)_g | H | psi(1)_g' >
                        matrix_elem_01 = 0
                        
                        # < psi(1)_g | H | psi(1)_g' >
                        matrix_elem_11 = 0
                        
                        # Loop over doubly excited determinants.
                        for mo_occ2_d in mo_occ2_ds:
                            
                            # Skip <(0)|H|(1)> matrix elements where <(0)|(1)> = 0.
                            ovlp1 = self.det_ovlp(mo_coeff1, mo_coeff2, mo_occ1, mo_occ2_ds[mo_occ2_d])[0]
                            if ovlp1 < 1e-7:
                                continue
                            
                            i, j, a, b = [int(s) for s in mo_occ2_d.split(',')]
                            matrix_elem_sd = self.get_matrix_element(
                                mo_coeff1, mo_coeff2, mo_occ1, mo_occ2_ds[mo_occ2_d])
                            
                            # We don't need to divide by 4 because the array of doubly excited mo_occs only
                            # includes combinations of orbitals.
                            matrix_elem_01 += t2s[row2][col2][i, j, a % self.nelec, b % self.nelec] * matrix_elem_sd 
                            
                            if order == 2:
                                continue
                                
                            for mo_occ1_d in mo_occ1_ds:
                                
                                # Skip <(1)|H|(1)> matrix elements where <(1)|(1)> = 0.
                                ovlp2 = self.det_ovlp(mo_coeff1, mo_coeff2, mo_occ1_ds[mo_occ1_d], mo_occ2_ds[mo_occ2_d])[0]
                                if ovlp2 < 1e-7:
                                    continue
                                
                                h, l, c, d = [int(s) for s in mo_occ1_d.split(',')]
                                matrix_elem_dd = self.get_matrix_element(
                                    mo_coeff1, mo_coeff2, mo_occ1_ds[mo_occ1_d], mo_occ2_ds[mo_occ2_d], doubles=mo_occ1_d)
                                
                                # We don't need to divide by 16 because the arrays of doubly excited mo_occs only
                                # include combinations of orbitals.
                                matrix_elem_11 += numpy.conj(t2s[row1][col1][h, l, c % self.nelec, d % self.nelec]) * \
                                    t2s[row2][col2][i, j, a % self.nelec, b % self.nelec] * matrix_elem_dd
                                
                        matrix_elems_01[row1, col1, row2, col2] = matrix_elem_01
                        matrix_elems_11[row1, col1, row2, col2] = matrix_elem_11
                        
        coeffs = numpy.sqrt(self.quad_coeffs(s, m, k))
        x_energy = 0
        term = 0
        
        if separate:
            # Loop over all pairs of quad points.
            for alpha_g in range(n_alpha):
                for beta_g in range(n_beta):
                    for alpha_gp in range(n_alpha):
                        for beta_gp in range(n_beta):
                            if (alpha_g, beta_g) == (alpha_gp, beta_gp):
                                continue

                            if order == 0:
                                continue

                            elif order == 1:
                                term += numpy.conj(coeffs[alpha_g, beta_g]) * coeffs[alpha_gp, beta_gp] * \
                                    (matrix_elems_00[alpha_g, beta_g, alpha_gp, beta_gp])

                            elif order == 2:
                                term += numpy.conj(coeffs[alpha_g, beta_g]) * coeffs[alpha_gp, beta_gp] * \
                                    (matrix_elems_01[alpha_g, beta_g, alpha_gp, beta_gp] + \
                                     numpy.conj(matrix_elems_01[alpha_gp, beta_gp, alpha_g, beta_g]))

                            elif order == 3:
                                term += numpy.conj(coeffs[alpha_g, beta_g]) * coeffs[alpha_gp, beta_gp] * \
                                    (matrix_elems_11[alpha_g, beta_g, alpha_gp, beta_gp])            
            
            return term, nterm
            
        else:
            for alpha_g in range(n_alpha):
                for beta_g in range(n_beta):
                    for alpha_gp in range(n_alpha):
                        for beta_gp in range(n_beta):
                            if (alpha_g, beta_g) == (alpha_gp, beta_gp):
                                continue

                            if order == 0:
                                continue

                            elif order == 1:
                                x_energy += numpy.conj(coeffs[alpha_g, beta_g]) * coeffs[alpha_gp, beta_gp] * \
                                    (matrix_elems_00[alpha_g, beta_g, alpha_gp, beta_gp])

                            elif order == 2:
                                x_energy += numpy.conj(coeffs[alpha_g, beta_g]) * coeffs[alpha_gp, beta_gp] * \
                                    (matrix_elems_00[alpha_g, beta_g, alpha_gp, beta_gp] + \
                                     matrix_elems_01[alpha_g, beta_g, alpha_gp, beta_gp] + \
                                     numpy.conj(matrix_elems_01[alpha_gp, beta_gp, alpha_g, beta_g]))

                            elif order == 3:
                                x_energy += numpy.conj(coeffs[alpha_g, beta_g]) * coeffs[alpha_gp, beta_gp] * \
                                    (matrix_elems_00[alpha_g, beta_g, alpha_gp, beta_gp] + \
                                     matrix_elems_01[alpha_g, beta_g, alpha_gp, beta_gp] + \
                                     numpy.conj(matrix_elems_01[alpha_gp, beta_gp, alpha_g, beta_g]) + \
                                     matrix_elems_11[alpha_g, beta_g, alpha_gp, beta_gp])            
                            
            return x_energy
    
    
    def energy(self, s, m, k, order=3, divide_norm=True, just_mp2=False):
        """
        Compute the total energy <Psi|H|Psi> / <Psi|Psi>.
        
        Args
            s (int): Total spin S eigenvalue.
            m (int): Final Sz eigenvalue.
            k (int): Initial Sz eigenvalue.
            order (int): Specifies the order of the SPPT2 correction.
            divide_norm (bool): Specifies whether to divide by <Psi|Psi>.
            just_mp2 (bool): Specifies whether to just use E_mp2 without multiplying by the normalization.
                             Used for checking.
            
        Returns
            <Psi|H|Psi> or <Psi|H|Psi> / <Psi|Psi>.
        """
        print('\n*** Check energy ***\n')
        if self.rotated_focks is None:
            self.get_quad_rotated_focks()

        alphas, betas, ws = self.quad
        Ng = len(betas)
        E = 0
        
        print(f'alphas = {alphas}')
        print(f'betas = {betas}')
        print(f'ws = {ws}')
        print(f'Ng = {Ng}\n')
        
        coeffs = self.quad_coeffs(s, m, k)
        
        for t, alpha_t in enumerate(alphas):
            for g, beta_g in enumerate(betas):
                rotated_fock = self.rotated_focks[t, g]
                mp2 = self.do_mp2(rotated_fock)
                t2 = mp2.t2
                mo_coeff = mp2.mo_coeff
                mo_occ = mp2.mo_occ
                mp2_norm = 1 + self.norm_11(mo_coeff, mo_coeff, mo_occ, mo_occ, t2, t2)
                
                # Don't want to repeatedly add the nuclear energy.
                E_mp2 = mp2.e_tot - self.mol.energy_nuc()
                
                if just_mp2:
                    # If s = m = k = 0, then d^s_mk = 1 and numpy.exp(1j * m * alpha_t) = 1.
                    # So we won't see a difference from the MP2 energy.
                    E += coeffs[t, g] * E_mp2
                    
                else:
                    E += coeffs[t, g] * E_mp2 * mp2_norm
                
                print(f'E_mp2 = {E_mp2}')
                print(f'alpha_t = {alpha_t}')
                print(f'beta_g = {beta_g}')
                print(f'ws_g = {ws[g]}')
                print(f'd^s_mk = {self.get_wigner_d(s, m, k, beta_g)}\n')
        
        E += self.mol.energy_nuc()
        E += self.energy_xterms(s, m, k, order=order)
        
        # <Psi | H | Psi> / <Psi | Psi>
        self.e_tot = E / self.norm(s, m, k)
        
        if divide_norm:
            return self.e_tot
        
        return E
                            
                            
    def get_matrix_element(self, mo_coeff1, mo_coeff2, mo_occ1, mo_occ2, doubles=None, divide=False, verbose=False):
        """
        Computes the matrix elements 
            1. <psi(0)_g | H | psi(0)_g'>,
            2. <psi(0)_g | H | {psi(0)_g'}_ijab>, or
            3. <{psi(0)_g}_klcd | H | {psi(0)_g'}_ijab>,
        
        as specified by the input `mo_occ`s and `doubles` parameter.
        
        Args
            mo_coeff1 (ndarray): MO coefficient matrix for the HF state associated with the left determinant.
            mo_coeff2 (ndarray): ... right determinant.
            mo_occ1 (ndarray): MO occupation numbers for the left determinant. Used to specify ground or doubly
                               excited determinants.
            mo_occ2 (ndarray): ... right determinant.
            doubles (str): String encoding the doubly excited left determinant.
            divide (bool): Whether to divide by the overlap between the left and right determinants.
        """
        mo_hcore = self.get_mo_hcore(mo_coeff1)
        mo_eri = self.get_mo_eri(mo_coeff1)
        A = self.get_A_matrix(mo_coeff1, mo_coeff2, mo_occ2)
        M = self.get_M_matrix(A)
        
        if doubles is not None:
            M = self.get_M_dtilde_matrix(mo_coeff1, mo_coeff2, mo_occ1, mo_occ2)
        
        trans_rdm1 = self.get_trans_rdm1(A, M, doubles=doubles)
        Minv = numpy.linalg.inv(M)
        ovlp = self.det_ovlp(mo_coeff1, mo_coeff2, mo_occ1, mo_occ2)[0]
        
        # For debugging.
        if verbose:
            print(f'M {M.shape}: \n{M}\n')
            print(f'Minv {Minv.shape}: \n{Minv}\n')
            print(f'A {A.shape}: \n{A}\n')
            print(f'trans rdm1 {trans_rdm1.shape}: \n{trans_rdm1}\n')
        
        part1 = numpy.einsum('ijkl, ki->jl', mo_eri, trans_rdm1)
        part2 = numpy.einsum('ijlk, ki->jl', mo_eri, trans_rdm1)
        term = numpy.einsum('ik, ki', mo_hcore, trans_rdm1) + 0.5 * (
                numpy.einsum('jl, lj', part1, trans_rdm1) - numpy.einsum('jl, lj', part2, trans_rdm1))

        if divide:
            return term
        
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
    
    
    def get_M_dtilde_matrix(self, mo_coeff1, mo_coeff2, mo_occ1, mo_occ2, verbose=False):
        """
        Returns \mathcal{C}^{\dagger} S \mathcal{C}'.
        \dtilde{M} is a (n x n) matrix.
        
        Args
            mo_coeff1 (ndarray): MO coefficient matrix for the HF state associated with the left determinant.
            mo_coeff2 (ndarray): ... right determinant.
            mo_occ1 (ndarray): MO occupation numbers for the left determinant. Used to specify ground or doubly
                               excited determinants.
            mo_occ2 (ndarray): ... right determinant.
        """
        if not isinstance(self.ao_ovlp, numpy.ndarray):
            self.get_ao_ovlp()
        
        if verbose:
            sc = numpy.dot(self.ao_ovlp, mo_coeff2[:, mo_occ2>0])
            csc = numpy.dot(mo_coeff1[:, mo_occ1>0].T.conj(), sc)

            print(f'mo_coeff1[:, mo_occ1>0] {mo_coeff1[:, mo_occ1>0].shape}: \n{mo_coeff1[:, mo_occ1>0]}\n')
            print(f'mo_coeff2[:, mo_occ2>0] {mo_coeff1[:, mo_occ1>0].shape}: \n{mo_coeff2[:, mo_occ2>0]}\n')
            print(f'self.ao_ovlp: \n{self.ao_ovlp}\n')
            print(f'SC {sc.shape}: \n{sc}\n')
            print(f'CSC {csc.shape}: \n{csc}\n')
        
        return reduce(numpy.dot, (mo_coeff1[:, mo_occ1>0].T.conj(), self.ao_ovlp, mo_coeff2[:, mo_occ2>0]))
    

    def get_trans_rdm1(self, A, M, doubles=None):
        """
        Computes the transition density matrix from the A, M matrices.
        
        Args
            A (ndarray): The A matrix.
            M (ndarray): The M matrix.
            doubles (str): String encoding the doubly excited left determinant.
        
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
        
        if doubles is not None:
            i, j, a, b = [int(s) for s in doubles.split(',')]
            Minv_block[:, [i, a]] = Minv_block[:, [a, i]]
            Minv_block[:, [j, b]] = Minv_block[:, [b, j]]
            
        trans_rdm1 = numpy.dot(A, Minv_block)
        return trans_rdm1
    
    
    def det_ovlp(self, mo_coeff1, mo_coeff2, mo_occ1, mo_occ2):
        """
        Calculate the overlap between two different determinants. It is the product
        of single values of molecular orbital overlap matrix.

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
        x = numpy.dot(u/s, vt)
        return numpy.prod(s), x
    
    
    def norm_11(self, mo_coeff1, mo_coeff2, mo_occ1, mo_occ2, t21, t22):
        """
        Computes the normalization <psi(1)_g | psi(1)_g'>.
        
        Note that if `mo_coeff1` == `mo_coeff2`, `mo_occ1` == `mo_occ2`, and 
        `t21` == `t22`, then we compute <psi(1)_g | psi(1)_g>.
        
        Args
            mo_coeff1 (ndarray): MO coefficient matrix for the HF state at quad point g.
            mo_coeff2 (ndarray): ... g'.
            mo_occ1 (ndarray): MO occupation numbers for the HF state at quad point g.
            mo_occ2 (ndarray): ... g'.
            t21 (ndarray): t2 amplitude matrix at quad point g.
            t22 (ndarray): ... g'.
        
        Returns
            <psi(1)_g | psi(1)_g'>
        """
        mo_occ1_ds = self.get_all_doubles(mo_occ1)
        mo_occ2_ds = self.get_all_doubles(mo_occ2)
        ovlp = 0
        
        # Loop over all combinations of doubles.
        for mo_occ1_d in mo_occ1_ds:
            for mo_occ2_d in mo_occ2_ds:
                h, l, c, d = [int(s) for s in mo_occ1_d.split(',')]
                i, j, a, b = [int(s) for s in mo_occ2_d.split(',')]
                ovlp += numpy.conj(t21[h, l, c % self.nelec, d % self.nelec]) * t22[i, j, a % self.nelec, b % self.nelec] * \
                        self.det_ovlp(mo_coeff1, mo_coeff2, mo_occ1_ds[mo_occ1_d], mo_occ2_ds[mo_occ2_d])[0]
            
        return ovlp
    
    
    def norm_01(self, mo_coeff1, mo_coeff2, mo_occ1, mo_occ2, t22):
        """
        Computes the normalization <psi(0)_g | psi(1)_g'>.
        
        Args
            mo_coeff1 (ndarray): MO coefficient matrix for the HF state at quad point g.
            mo_coeff2 (ndarray): ... g'.
            mo_occ1 (ndarray): MO occupation numbers for the HF state at quad point g.
            mo_occ2 (ndarray): ... g'.
            t22 (ndarray): t2 amplitude matrix at quad point g'.
        
        Returns
            <psi(0)_g | psi(1)_g'>
        """
        mo_occ2_ds = self.get_all_doubles(mo_occ2)
        ovlp = 0
        
        # Loop over all combinations of doubles.
        for mo_occ2_d in mo_occ2_ds:
            i, j, a, b = [int(s) for s in mo_occ2_d.split(',')]
            ovlp += numpy.conj(t22[i, j, a % self.nelec, b % self.nelec]) * \
                    self.det_ovlp(mo_coeff1, mo_coeff2, mo_occ1, mo_occ2_ds[mo_occ2_d])[0]
            
        return ovlp
    

    def norm(self, s, m, k):
        """
        Computes the total normalization <Psi | Psi>.
        
        Args
            s (int): Total spin S eigenvalue.
            m (int): Final Sz eigenvalue.
            k (int): Initial Sz eigenvalue.
            
        Returns
            <Psi | Psi>.
        """
        if not isinstance(self.ghfs, numpy.ndarray):
            self.get_quad_ghfs_t2s()
        
        ghfs, t2s = self.ghfs, self.t2s
        coeffs = numpy.sqrt(self.quad_coeffs(s, m, k))
        ovlp = 0
        
        # Loop over ghfs at each quad point.
        for row1, ghf_row1 in enumerate(ghfs):
            for row2, ghf_row2 in enumerate(ghfs):
                for col1, ghf1 in enumerate(ghf_row1):
                    for col2, ghf2 in enumerate(ghf_row2):
                        mo_coeff1 = ghf1.mo_coeff
                        mo_occ1 = ghf1.mo_occ
                        t21 = t2s[row1, col1]
                        
                        if (row1, col1) == (row2, col2):    
                            ovlp += coeffs[row1, col1]**2 * (1 + self.norm_11(mo_coeff1, mo_coeff1, mo_occ1, mo_occ1, t21, t21))
                        
                        else:
                            mo_coeff2 = ghf2.mo_coeff
                            mo_occ2 = ghf2.mo_occ
                            t22 = t2s[row2, col2]
                            ovlp += numpy.conj(coeffs[row1, col1]) * coeffs[row2, col2] * (
                                    self.det_ovlp(mo_coeff1, mo_coeff2, mo_occ1, mo_occ2)[0] + 
                                    self.norm_01(mo_coeff1, mo_coeff2, mo_occ1, mo_occ2, t22) +
                                    numpy.conj(self.norm_01(mo_coeff2, mo_coeff1, mo_occ2, mo_occ1, t21)) + 
                                    self.norm_11(mo_coeff1, mo_coeff2, mo_occ1, mo_occ2, t21, t22))
        
        return ovlp
    
    
    def get_ao_ovlp(self):
        """
        Compute the AO overlap matrix S.
        """
        s = self.mol.intor_symmetric('int1e_ovlp')
        self.ao_ovlp = scipy.linalg.block_diag(s, s)
    
    
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
        mo_hcore = reduce(numpy.dot, (mo_coeff.T, ao_hcore, mo_coeff))
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
        
    
    def check_fock_energy(self):
        print('*** Check rotated Fock energies ***')
        if self.rotated_focks is None:
            self.get_quad_rotated_focks()
        
        alphas, betas, ws = self.quad
        Ng = len(betas)
        
        # TODO: Check factors.
        prefactor = (2*s + 1) / (4*Ng)
        E_euhf = 0.
        
        for g, beta_g in enumerate(betas):
            for t, alpha_t in enumerate(alphas):
                # Construct GHF object.
                ghf = scf.GHF(self.mol)
                rotated_fock = self.rotated_focks[t, g]
                ghf.get_fock = lambda *args, **kwargs : rotated_fock

                # We don't want to run SCF procedure.
                ghf.max_cycle = 0
                ghf.kernel()

                # If s = m = k = 0, then d^s_mk = 1 and numpy.exp(1j * m * alpha_t) = 1.
                E_sum = ws[g] * self.get_wigner_d(s, m, k, beta_g) * numpy.exp(1j * m * alpha_t) * ghf.e_tot
                E_euhf += E_sum
            
        
        E_euhf *= prefactor
        return E_euhf
    

        def get_quad_rdms(self):
        """
        Get the 1-particle and 2-particle MP2 density matrices at the quadrature points.
        """
        if self.rotated_focks is None:
            self.get_quad_rotated_focks()
        
        alphas, betas, w = self.quad
        self.quad_rdm1s = []
        self.quad_rdm2s = []
        
        for alpha in range(len(alphas)):
            rdm1s_beta = []
            rdm2s_beta = []
            
            for beta in range(len(betas)):
                rotated_fock = self.rotated_focks[alpha, beta]
                mp2 = self.do_mp2(rotated_fock)
                
                # Note that rdms are in MO basis!
                rdm1 = mp2.make_rdm1()
                rdm2 = mp2.make_rdm2()
                rdm1s_beta.append(rdm1)
                rdm2s_beta.append(rdm2)
                
            self.quad_rdm1s.append(rdm1s_beta)
            self.quad_rdm2s.append(rdm2s_beta)
            
        self.quad_rdm1s = numpy.array(self.quad_rdm1s)
        self.quad_rdm2s = numpy.array(self.quad_rdm2s)
                

    def get_rdm_sym(self, s, m, k, opt):
        """
        Computes the spin-symmetrized density matrix.
        
        Args
            s (int): Total spin S eigenvalue.
            m (int): Final Sz eigenvalue.
            k (int): Initial Sz eigenvalue.
            opt (str): 'rdm1' - get spin-symmetrized rmd1. 
                       'rdm2' - get spin-symmetrized rmd2. 

        Returns
            The spin-symmetrized density matrix.
        """
        if k != 0:
            print('k != 0. No symmetrized density matrix generated.')
            return 0
        
        quad_rdms = None
        
        if not isinstance(self.quad_rdm1s, numpy.ndarray):
            self.get_quad_rdms()
            
        if opt == 'rdm1':
            quad_rdms = self.quad_rdm1s
            
        elif opt == 'rdm2':
            quad_rdms = self.quad_rdm2s
        
        else:
            raise ValueError('Option not available.')

        alphas, betas, ws = self.quad
        Ng = len(betas)
        rdm_sym = numpy.zeros(quad_rdms[0].shape)
        
        prefactor = (2*s + 1) / (4*Ng)
        
        for t, alpha_t in enumerate(alphas):
            for g, beta_g in enumerate(betas):
                rdm_sym += (
                    ws[g] * self.get_wigner_d(s, m, k, beta_g) * numpy.exp(1j * m * alpha_t) * 
                    quad_rdms[t, g])
        
        rdm_sym *= prefactor
        
        if opt == 'rdm1':
            print('Generated spin-symmetrized rdm1')
            self.rdm1_sym = rdm_sym
            
        elif opt == 'rdm2':
            print('Generated spin-symmetrized rdm2')
            self.rdm2_sym = rdm_sym
            
        return rdm_sym
                

# Other functions.
    
def build_chain(n, bond_length, basis='6-31g'):
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
    Hchain.build()
    return Hchain
    
    
def bond_curve(n, s, m, k, order=3, divide_norm=True, just_mp2=False, start=0.5, stop=5., points=50, basis='sto-3g'):
    """
    Computes the bond curve using SPPT2, GMP2, and FCI.
    
    Args
        s (int): Total spin S eigenvalue.
        m (int): Final Sz eigenvalue.
        k (int): Initial Sz eigenvalue.
        order (int): Specifies the order of the SPPT2 correction.
        divide_norm (bool): Specifies whether to divide by <Psi|Psi>.
    """
    bond_lengths = numpy.linspace(start, stop, points)
    dms = None
    gdms = None
    e_sppt2s = []
    e_gmp2s = []
    e_fcis = []
    

    for bond_length in bond_lengths:
        print('\nBond Length: {}'.format(bond_length))
        mol = build_chain(n, bond_length, basis)
        nao = mol.nao
        
        if dms is None:
            uhf = scf.UHF(mol).run(dms, verbose=0)
            dms = uhf.make_rdm1()

        # Need to do this to get UHF solution.    
        dms[0][0, 0] += 1.0    
        uhf = scf.UHF(mol).run(dms, verbose=0)
        dms = uhf.make_rdm1()
        
        fock = uhf.get_fock()
        test = SPPT2(mol, fock=fock)

        # Test energy.
        e_sppt2 = test.energy(s, m, k, order=order, divide_norm=divide_norm, just_mp2=just_mp2)
        
        print('\n=====================================')
        print(f'SPPT2 E = {e_sppt2}\n')

        # FCI energy.
        rhf = scf.RHF(mol).run(verbose=0)
        fci_out = fci.FCI(rhf).run(verbose=0)
        print(f'FCI E = {fci_out.e_tot}')
        print('=====================================\n')

        # GMP2 energy.
        if gdms is not None:
            gdms[0, 0]+=1.0
            
        ghf = scf.GHF(mol).run(gdms, verbose=0)
        gdms = ghf.make_rdm1()
        gmp2 = mp.GMP2(ghf).run()

        # Store energies.
        e_sppt2s.append(e_sppt2)
        e_gmp2s.append(gmp2.e_tot)
        e_fcis.append(fci_out.e_tot)
        
    return bond_lengths, e_sppt2s, e_gmp2s, e_fcis
    
                         
if __name__ == '__main__':
    from pyscf import fci
    
    mol = gto.Mole()
    # Want to see at larger radii.
    mol.atom = [['H', (i*2,0.,0.)] for i in range(10)]
    mol.basis = 'sto-3g'
    mol.build()
    
    # Need to do this to get UHF solution.
    uhf = scf.UHF(mol).run(verbose=0)
    dms = uhf.make_rdm1()
    dms[0][0, 0] += 1.0    
    uhf = scf.UHF(mol).run(dms, verbose=0)    
    
    fock = uhf.get_fock()
    test = SPPT2(mol, fock=fock)
    s, m, k = 0, 0, 0
    e = test.energy(s, m, k)