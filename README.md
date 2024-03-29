# SP-UMP2
SP-UMP2 and SPPT2 refer to the same method and are used interchangeably.

## Dependencies
PySCF: `dfgmp2` branch available at https://github.com/shufay/pyscf/tree/dfgmp2.

## Run
To initialize an `SPPT2` object:

```
import sppt2_v13_2
from pyscf import gto

mol = gto.Mole()
mol.atom = [['H', (2*i, 0., 0.)] for i in range(2)]
mol.basis = 'sto-6g'
mol.build()

test = sppt2_v13_2.SPPT2(mol)
```

To compute the energy:
```
# Spin quantum numbers.
s, m, k = 0, 0, 0

# Number of quadrature points.
# The integration is only done over variable beta, so we set N_alpha = N_gamma = 1.
N_alpha = N_gamma = 1
N_beta = 10

test = sppt2_v13_2.SPPT2(mol)
uhf = test.do_uhf()
ump2 = test.do_mp2(uhf)
e = test.energy(s, m, k, ump2, N_alpha=N_alpha, N_beta=N_beta, N_gamma=N_gamma)
```
