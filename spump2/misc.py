from functools import reduce
import numpy as np
from scipy.special import factorial

def get_det_ovlp(mol, mo1, mo2, s1e=None):
    if s1e is None:
        s1e = mol.intor('int1e_ovlp', hermi=1)
    nao = mol.nao
    assert nao == mo1.shape[0] // 2
    assert mo1.shape[0] == mo2.shape[0]

    moa1 = mo1[:nao]
    mob1 = mo1[nao:]
    moa2 = mo2[:nao]
    mob2 = mo2[nao:]
    s = (reduce(np.dot, (moa1.T.conj(), s1e, moa2)) +
         reduce(np.dot, (mob1.T.conj(), s1e, mob2)))
    return np.linalg.det(s)

def apply_Ry(mo_coeff, theta):
    r'''Apply :math:`e^{-i \theta \hat{S}_{y}}` to MOs.

    Args:
        mo_coeff : (2*N, M) ndarray
            GHF MO coefficients.
            N is the number of AOs, and M is the number of MOs.
        theta : float
            Rotation angle.

    Returns:
        mo1 : (2*N, M) ndarray
            MO coefficients of the rotated MOs.
    '''
    nao = mo_coeff.shape[0] // 2
    mo1 = np.empty_like(mo_coeff)
    mo1[:nao] = np.cos(theta/2) * mo_coeff[:nao] - np.sin(theta/2) * mo_coeff[nao:]
    mo1[nao:] = np.sin(theta/2) * mo_coeff[:nao] + np.cos(theta/2) * mo_coeff[nao:]
    return mo1

def apply_Rz(mo_coeff, theta):
    r'''Apply e^{-i \theta \hat{S}_{z}} to MOs.
    '''
    nao = mo_coeff.shape[0] // 2
    mo1 = np.empty_like(mo_coeff, dtype=np.complex128)
    mo1[:nao] = np.exp(-1j * theta/2) * mo_coeff[:nao]
    mo1[nao:] = np.exp( 1j * theta/2) * mo_coeff[nao:]
    return mo1

def quadrature_Ry(n):
    x, w = np.polynomial.legendre.leggauss(n)
    beta = np.arccos(x)
    idx = np.argsort(beta)
    return beta[idx], w[idx]

def quadrature_Rz(n):
    gamma = np.linspace(0, 2*np.pi, n, endpoint=False)
    return gamma, np.repeat(2*np.pi/n, n)

def dmatrix_element(s, m, k, beta):
    r'''Wigner small d-matrix element:
    :math:`d^{s}_{m k} (\beta)`.
    '''
    out = 0.
    cos = np.cos(beta/2)
    sin = np.sin(beta/2)

    nmin = int(max(0, k-m))
    nmax = int(min(s+k, s-m))
    n = np.arange(nmin, nmax+1)
    denom = (factorial(s+k-n) * factorial(n) *
             factorial(m-k+n) * factorial(s-m-n))
    tmp = (-1.)**(m-k+n) * cos**(2*s+k-m-2*n) * sin**(m-k+2*n) / denom

    fac = np.sqrt(factorial(s+k) * factorial(s-k) *
                  factorial(s+m) * factorial(s-m))
    out = tmp.sum() * fac
    return out

def get_norm00_collinear(mol, mo_coeff, s, m, k, n_beta):
    r'''Compute :math:`\int d\Omega w(\Omega) <0|\hat{R}(\Omega)|0>`,
    where :math:`|0>` is an eigenstate of :math:`\hat{S}_z`.
    In this case, :math:`m = k = S_{z}`,
    and the integrations over :math:`\alpha` and :math:`\gamma`
    give constants :math:`2\pi`.
    '''
    assert m == k
    bs, ws = quadrature_Ry(n_beta)
    fac = (2.*s + 1.) / 2.

    out = 0
    for beta, w in zip(bs, ws):
        mo1 = apply_Ry(mo_coeff, beta)
        ovlp = get_det_ovlp(mol, mo_coeff, mo1)
        out += fac * dmatrix_element(s, m, k, beta) * ovlp * w
    return out

def get_norm00(mol, mo_coeff, s, m, k, n_beta, n_gamma):
    r'''Compute :math:`\int d\Omega w(\Omega) <0|\hat{R}(\Omega)|0>`.
    '''
    bs, wbs = quadrature_Ry(n_beta)
    gs, wgs = quadrature_Rz(n_gamma)
    fac = (2.*s + 1.) / (8 * np.pi**2)

    dmat = np.empty((n_beta,))
    for i in range(n_beta):
        dmat[i] = dmatrix_element(s, m, k, bs[i])

    out = 0
    for gamma, wg in zip(gs, wgs):
        mo1 = apply_Rz(mo_coeff, gamma)
        fac1 = fac * np.exp(-1j*k*gamma)
        for i, (beta, wb) in enumerate(zip(bs, wbs)):
            mo2 = apply_Ry(mo1, beta)
            for alpha, wa in zip(gs, wgs):
                mo3 = apply_Rz(mo2, alpha)
                ovlp = get_det_ovlp(mol, mo_coeff, mo3)
                out += fac1 * dmat[i] * ovlp * wa * wb * wg * np.exp(-1j*m*alpha)
    return out


if __name__ == '__main__':
    from pyscf import gto, scf

    mol = gto.Mole()
    mol.atom = '''
        H 0 0 0
        H 0 0 .74
    '''
    mol.basis = 'ccpvdz'
    mol.verbose = 5
    mol.build()

    mf = scf.UHF(mol)
    mf.kernel()
    mf = mf.to_ghf()
    mo0 = mf.mo_coeff[:,mf.mo_occ>0]

    s = 1
    m = 0
    print(get_norm00_collinear(mol, mo0, s, m, m, 10))

    m = 1
    k = -1
    print(get_norm00(mol, mo0, s, m, k, 10, 10))
