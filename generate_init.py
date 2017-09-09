import numpy as np
import sys
var_h = float(sys.argv[1])
var_o = float(sys.argv[2])
nd = int(sys.argv[3])
nh = int(sys.argv[4])

w = np.random.normal(0,var_h,(nd,nh)).astype(np.float128)
b = np.random.normal(0,var_h,(nh)).astype(np.float128)
W = np.random.normal(0,var_o,(nh,2)).astype(np.float128)
B = np.random.normal(0,var_o,(2)).astype(np.float128)

np.savetxt('hw',w)
np.savetxt('hb',b)
np.savetxt('ow',W)
np.savetxt('ob',B)

