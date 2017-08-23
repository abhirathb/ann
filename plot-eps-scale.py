import numpy as np
import matplotlib.pyplot  as plt

prefix = 'result_22-8/eps-scale/'
suffix = '_1000_100'


h=np.loadtxt(prefix+'ham'+suffix)
plt.plot(h)
plt.title('Hamiltonian')
plt.ylabel('Energy')
plt.xlabel('i where dimension = i*100 + 100')
plt.savefig(prefix + 'ham' + suffix +'.png')

pot=np.loadtxt(prefix+'pot'+suffix)
plt.plot(pot)
plt.title('Potential')
plt.ylabel('Energy')
plt.xlabel('i where dimension = i*100 + 100')
plt.savefig(prefix + 'pot' + suffix +'.png')

kin=np.loadtxt(prefix+'kin'+suffix)
plt.plot(kin)
plt.title('Kinetic Energy')
plt.ylabel('Energy')
plt.xlabel('i where dimension = i*100 + 100')
plt.savefig(prefix + 'kin' + suffix +'.png')

diff=np.loadtxt(prefix+'diff'+suffix)
plt.plot(diff)
plt.title('Difference in Hamiltonian')
plt.ylabel('Energy')
plt.xlabel('i where dimension = i*100 + 100')
plt.savefig(prefix + 'diff' + suffix +'.png')


diff-k=np.loadtxt(prefix+'diff-k'+suffix)
plt.plot(diff-k)
plt.title('Difference in Kinetic Energy')
plt.ylabel('Energy')
plt.xlabel('i where dimension = i*100 + 100')
plt.savefig(prefix + 'diff-k' + suffix +'.png')


ratio=np.loadtxt(prefix+'ratio'+suffix)
plt.plot(ratio)
plt.title('DelH/H')
plt.ylabel('Energy')
plt.xlabel('i where dimension = i*100 + 100')
plt.savefig(prefix + 'ratio' + suffix +'.png')

ratio-k=np.loadtxt(prefix+'ratio-k'+suffix)
plt.plot(ratio-k)
plt.title('DelK/K')
plt.ylabel('Energy')
plt.xlabel('i where dimension = i*100 + 100')
plt.savefig(prefix + 'ratio-k' + suffix +'.png')
