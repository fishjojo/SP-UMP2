from pyscf import gto, scf, mp

mol = gto.Mole()
mol.atom = '''
Fe        -0.20236        0.06324       -0.00000
Cl        -1.66556       -0.24136       -1.72410
Cl         1.26074        0.36774        1.72410
Cl         0.46174        2.04564       -0.91380
Cl        -0.86656       -1.91916        0.91380
Cl        -1.82226        1.15104        1.18260
Cl         1.41754       -1.02456       -1.18260
'''
mol.basis = 'ccpvdz'
mol.charge = -4
mol.max_memory=50000
mol.verbose = 5
mol.build()

mf = scf.UHF(mol).density_fit()
mf.kernel(dm0=None)
mf = mf.to_ghf()

mymp = mp.GMP2(mf)
mymp.kernel()

from spump2 import dfsppt2

N_alpha = N_gamma = 1
N_beta = 5
s, m, k = 0, 0, 0

pt2 = dfsppt2.SPPT2(mf)

pt2.e_hf = mf.e_tot
pt2.e_elec_hf = pt2.e_hf - pt2.mol.energy_nuc()

import pstats, cProfile
profiler = cProfile.Profile()
profiler.enable()

e = pt2.energy(s, m, k, mymp, N_alpha=N_alpha, N_beta=N_beta, N_gamma=N_gamma)
print(e)

profiler.disable()
stats = pstats.Stats(profiler).sort_stats('cumtime')
stats.print_stats()

