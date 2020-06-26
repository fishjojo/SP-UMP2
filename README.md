# SPPT2
To initialize an `SPPT2` object:

```
import sppt2_v2
from pyscf import gto, scf

mol = gto.Mole()
mol.atom = [['H', (2*i, 0., 0.)] for i in range(4)]
mol.basis = 'sto-3g'
mol.build()

# Need to do this to get UHF solution.
uhf = scf.UHF(mol).run()
rdm1 = uhf.make_rdm1()
rdm1[0][0] += 1
uhf = scf.UHF(mol).run(rdm1)

fock = uhf.get_fock()
test = sppt2_v2.SPPT2(mol, fock=fock)
```

To compute the energy up to 0th-order corrections 
<img src="https://latex.codecogs.com/gif.latex?\%20\sum_{g\neq%20g'}%20w_g^*%20w_{g'}%20\langle%20\psi^{(0)}_g%20|%20\hat{H}%20|%20\psi^{(0)}_{g'}%20\rangle"/>:
```
s = m = k = 0

# `order` is not 0, but 1 (sorry for the inconsistency).
# 0 gives just the sum over terms with the E_mp2 energy.
e = test.energy(s, m, k, order=1) 
```

To compute the 0th-order corrections only:
```
term00 = test.energy_xterms(s, m, k, order=1, separate=True)
```

To compute the matrix element <img src="https://latex.codecogs.com/gif.latex?\%20\langle%20\psi^{(0)}_g%20|%20\hat{H}%20|%20\psi^{(0)}_{g'}%20\rangle"/>:
```
# Get ghf at each quadrature point.
ghfs, t2s = test.get_quad_ghfs_t2s()
ghf1 = ghfs[0,0]
ghf2 = ghfs[1,0]
mo_coeff1 = ghf1.mo_coeff
mo_coeff2 = ghf2.mo_coeff
mo_occ1 = ghf1.mo_occ
mo_occ2 = ghf2.mo_occ
matrix_elem = test.get_matrix_element(mo_coeff1, mo_coeff2, mo_occ1, mo_occ2)
```

To compute the norm <img src="https://latex.codecogs.com/gif.latex?\%20\langle\Psi|\Psi\rangle"/>:
```
norm = test.norm(s, m, k)
```

The relevant functions for the energy calculation up to 0th-order corrections are:
```
energy
energy_xterms
get_matrix_element
norm
```

## Index
`Energy.ipynb`: Jupyter notebook with energy plots.  
`normalization.ipynb`: Jupyter notebook with normalization plots.  
`sppt2_v2.py`: Code for SPPT2 implementation.
