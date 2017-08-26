import numpy as np
import matplotlib.pyplot  as plt

prefix = 'result_22-8/prior-cont/eps-scale/'
suffix = '_1000_100'

x = np.array([0,1,2,3,4])
xticks = ['0.1','0.01','0.001','0.0001','0.00001']

h=np.loadtxt(prefix+'ham'+suffix)
plt.figure()
plt.xticks(x,xticks)
plt.plot(x,h,"o")
plt.title('Hamiltonian')
plt.ylabel('Energy')
plt.xlabel('value of epsilon')
plt.savefig(prefix + 'ham' + suffix +'.png')

pot=np.loadtxt(prefix+'pot'+suffix)
plt.figure()
plt.xticks(x,xticks)
plt.plot(x,pot,"o")
plt.title('Potential')
plt.ylabel('Energy')
plt.xlabel('value of epsilon')
plt.savefig(prefix + 'pot' + suffix +'.png')

kin=np.loadtxt(prefix+'kin'+suffix)
plt.figure()
plt.xticks(x,xticks)
plt.plot(x,kin,"o")
plt.title('Kinetic Energy')
plt.ylabel('Energy')
plt.xlabel('value of epsilon')
plt.savefig(prefix + 'kin' + suffix +'.png')

diff=np.loadtxt(prefix+'diff'+suffix)
plt.figure()
plt.xticks(x,xticks)
plt.plot(x,diff,"o")
plt.title('Difference in Hamiltonian')
plt.ylabel('Energy')
plt.xlabel('value of epsilon')
plt.savefig(prefix + 'diff' + suffix +'.png')


diff_k=np.loadtxt(prefix+'diff-k'+suffix)
plt.figure()
plt.xticks(x,xticks)
plt.plot(x,diff_k,"o")
plt.title('Difference in Kinetic Energy')
plt.ylabel('Energy')
plt.xlabel('value of epsilon')
plt.savefig(prefix + 'diff-k' + suffix +'.png')

diff_u=np.loadtxt(prefix+'diff-u'+suffix)
plt.figure()
plt.xticks(x,xticks)
plt.plot(x,diff_u,"o")
plt.title('Difference in Potential Energy')
plt.ylabel('Energy')
plt.xlabel('value of epsilon')
plt.savefig(prefix + 'diff-u' + suffix +'.png')



ratio=np.loadtxt(prefix+'ratio'+suffix)
plt.figure()
plt.xticks(x,xticks)
plt.plot(x,ratio,"o")
plt.title('DelH/H')
plt.ylabel('Energy')
plt.xlabel('value of epsilon')
plt.savefig(prefix + 'ratio' + suffix +'.png')

ratio_k=np.loadtxt(prefix+'ratio-k'+suffix)
plt.figure()
plt.xticks(x,xticks)
plt.plot(x,ratio_k,"o")
plt.title('DelK/K')
plt.ylabel('Energy')
plt.xlabel('value of epsilon')
plt.savefig(prefix + 'ratio-k' + suffix +'.png')


ratio_u=np.loadtxt(prefix+'ratio-u'+suffix)
plt.figure()
plt.xticks(x,xticks)
plt.plot(x,ratio_u,"o")
plt.title('DelU/U')
plt.ylabel('Energy')
plt.xlabel('value of epsilon')
plt.savefig(prefix + 'ratio-u' + suffix +'.png')
