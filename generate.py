import sys
import numpy as np
dim = int(sys.argv[1])
N = int(sys.argv[2])
maf = np.loadtxt('../../code/BNN/src/inputs_hmc_test/maf')
maf = maf[::,:dim]
maf = maf[:N]
np.savetxt('maf_%d_%d'%(dim,N),maf)
import os.path
if not os.path.isfile('hc_%d'%N):
    hc = np.loadtxt('../../code/BNN/src/inputs_hmc_test/hc')
    np.savetxt('hc_%d'%N,hc[:N])
