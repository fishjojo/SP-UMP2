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

The relevant functions for the 1st-order energy calculation are:
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
