from pyscf import gto, scf, mp

mol = gto.Mole()
mol.atom = '''
    O 0 0 0
    O 0 0 1.16
'''
mol.basis = 'ccpvtz'
mol.max_memory=2000
mol.verbose = 5
mol.build()

mf = scf.UHF(mol).density_fit()
dm0 = mf.get_init_guess()
dm0[0][0] += 1.
dm0[0] += dm0[0].T
mf.max_cycle = 100
mf.kernel(dm0=dm0)
mf = mf.to_ghf()

mymp = mp.GMP2(mf)
mymp.frozen = 2
mymp.kernel()

from spump2 import dfsppt2

N_alpha = N_gamma = 1
N_beta = 10
s, m, k = 0, 0, 0

pt2 = dfsppt2.SPPT2(mf)

pt2.e_hf = mf.e_tot
pt2.e_elec_hf = pt2.e_hf - pt2.mol.energy_nuc()

e = pt2.energy(s, m, k, mymp, N_alpha=N_alpha, N_beta=N_beta, N_gamma=N_gamma)
print(e)
