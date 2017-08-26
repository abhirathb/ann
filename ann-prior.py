import numpy as np
import sys
from scipy.stats import invgamma
#basic constants for run
#this is assuming one hidden layer i.e. a sigmoid layer and a softmax-output layer
num_hidden = 5  #number of units in hidden layer
num_inputs = int(sys.argv[1]) #number of inputs
N = int(sys.argv[2])
eps = float(sys.argv[3])
var = float(sys.argv[4])
#.....

#layer variables:
hidden_weights = np.zeros((num_inputs,num_hidden),np.float128)   #shape = N x H
hidden_biases = np.zeros((num_hidden),np.float128) 
output_weights = np.zeros((num_hidden,2),np.float128)
output_biases = np.zeros((2),np.float128)


init = 100.0
# hidden layer prior settings; shape, scale, sW, sB, mean
hpw_shape = 5.0 #hidden layer prior weights shape
hpw_scale = 2.0 #hidden layer prior weights scale
hidden_sW = (np.tile(init,reps=hidden_weights.shape[0]).reshape(1,hidden_weights.shape[0])).astype(np.float128) # each input unit has one sigma. so you repeat the same initial sigma D times to get a vector
hidden_sB = invgamma.rvs(1.0,scale=1.0, size = (1,1)).astype(np.float128) #the biases have just one common prior
hpw_mean = (np.tile(hpw_scale/(hpw_shape-1),reps=hidden_weights.shape[0]).reshape(1,hidden_weights.shape[0])).astype(np.float128) #maintain means of all the variances of prior of each input weight 
#end of hidden layer prior

#output layer prior: 
opw_shape = 0.1
opw_scale = 0.1
output_sW = np.array([100.],dtype=np.float128) #outputs have exactly one prior for all weights/biases
output_sB = invgamma.rvs(1.0,scale=1.0,size=(1,1)).astype(np.float128)#end output layer prior contribution
#output layer prior end

hidden_weights_grad = np.zeros((num_inputs,num_hidden),np.float128)
hidden_biases_grad = np.zeros((num_hidden),np.float128)
output_weights_grad = np.zeros((num_hidden,2),np.float128)
output_biases_grad = np.zeros((2),np.float128)

hidden_outputs = np.zeros((num_hidden),np.float128)
output_outputs = np.zeros((2),np.float128)
#....

#Sampling Variables:
init_sd_output = 1.0
init_sd_hidden = 1.0
pW = np.random.normal(0,init_sd_output,output_weights.shape)
pB = np.random.normal(0,init_sd_output,output_biases.shape)
pw = np.random.normal(0,init_sd_hidden,hidden_weights.shape)
pb = np.random.normal(0,init_sd_hidden,hidden_biases.shape)
#....


def compute_outputs(hidden_weights, hidden_biases, output_weights, output_biases,inputs):
    h_z = np.dot(inputs,hidden_weights)
    for i in range(len(h_z)):
        h_z[i]+=hidden_biases
        for j in range(len(h_z[i])):
            h_z[i][j] = 1/(1+np.exp(-h_z[i][j]))

    
    output_outputs = np.dot(h_z,output_weights)


    for i in range(len(output_outputs)):
        output_outputs[i]+=output_biases
        o1 = np.exp(output_outputs[i][0])
        o2 = np.exp(output_outputs[i][1])
        output_outputs[i][0] = o1/(o1+o2)
        output_outputs[i][1] = o2/(o1+o2)


    return h_z,output_outputs

def compute_grads(hidden_weights, hidden_outputs,hidden_biases,hsW,hsB, output_weights,output_outputs,output_biases,osW,osB,inputs,outputs):
    diff = outputs - output_outputs    #the main difference term

    dB = np.dot(np.ones((1,np.shape(diff)[0])),diff).reshape((output_weights.shape[1],))
    dB -= output_biases/osB[0]
    dW = np.dot(hidden_outputs.T,diff) 
    dW -= output_weights/osW[0]
    bp = np.dot(diff,output_weights.T)
    prod = hidden_outputs*(1-hidden_outputs)
    back = bp*prod
    db = np.dot(np.ones((1,np.shape(back)[0])),back).reshape((hidden_weights.shape[1],))
    db -= hidden_biases/hsB[0]
    dw = np.dot(inputs.T,back)
    for i in range(len(hidden_weights)):
        dw[i] -= (hidden_weights[i])/(hsW[0][i])

    return dB,dW,db,dw


def prior_contrib(hidden_weights, hidden_biases, hsW, hsB, output_weights, output_biases, osW, osB):
    val = 0 
    for i,j in zip(hidden_weights,hsW[0]):
        val -= (i**2).sum()/(2*j)
#    print osW
    for i in output_weights:
        val -= (i**2).sum()/(osW[0])
    val -= (hidden_biases**2).sum()/(2*hsB[0])
    val -= (output_biases**2).sum()/(2*osB[0])
    print "prior",val
    return val


def Hamiltonian(outputs, output_outputs,pw,pb,pB,pW,hidden_weights,hidden_biases,hidden_sW,hidden_sB,output_weights,output_biases,output_sW,output_sB):
    log = outputs*np.log(output_outputs)
    log = log.sum()
    k = (pw**2).sum() + (pb**2).sum() + (pW**2).sum() + (pB**2).sum()
    
    p = prior_contrib(hidden_weights,hidden_biases,hidden_sW,hidden_sB,output_weights, output_biases, output_sW, output_sB)
    return log,k,log+p,(log+p-k)



def gibbs_update(hidden_weights, hidden_biases, hsW, hsB,hpw_mean,hpw_shape,hpw_scale, output_weights, output_biases, osW, osB,opw_shape,opw_scale):
    #update for ARD
    new_hsW = np.zeros(hsW.shape)
    new_mean = np.zeros(hpw_mean.shape)
    n_w = np.float128(hidden_weights.shape[1])
    hpw_shape_new = hpw_shape+ n_w/2.0
    for i in range(len(hidden_weights)):
        hpw_scale_new=hpw_scale + (hidden_weights[i]**2).sum()/2.0
        new_val = invgamma.rvs(hpw_shape_new,scale=hpw_scale_new,size=1)
        new_hsW[0,i]=np.float128(new_val)
        new_mean[0,i] = np.float128(hpw_scale_new/(hpw_shape_new-1.0))
    
    hsW = new_hsW.astype(np.float128)
    hpw_mean = new_mean.astype(np.float128)
    n_b = np.float128(hidden_biases.shape[0])
    hpb_shape_new = hpw_shape + n_b/2.0
    hpb_scale_new = hpw_scale + (hidden_biases**2).sum()/2.0
    new_val = invgamma.rvs(hpb_shape_new, scale=hpb_scale_new,size=1)
    hsB = np.float128(new_val)
    
    #update for GLP
    n_w = np.float128(output_weights.shape[0]*output_weights.shape[1])
    shape_new = opw_shape + n_w/2.0
    scale_new = opw_scale + (output_weights**2).sum()/2.0
    new_val = invgamma.rvs(shape_new, scale=scale_new, size=1)
    
    osW = np.float128(new_val)

    n_b = np.float128(output_biases.shape[0])
    shape_new = opw_shape+ n_b/2.0
    scale_new = opw_scale + (output_biases**2).sum()/2.0
    new_val = invgamma.rvs(shape_new,scale=scale_new,size=1)
    osB = np.float128(new_val)
    
    return hsW, hsB, hpw_mean, osW, osB 
    
def leap_frog(hw, hb,hsW,hsB, pw,pb, dw,db, ow,ob,osW,osB,pW,pB,dW,dB,eps,inputs,outputs):
    pw += (eps/2.0)*dw
    pb += (eps/2.0)*db
    pW += (eps/2.0)*dW
    pB += (eps/2.0)*dB

    hw += (eps)*pw
    hb += (eps)*pb
    ow += (eps)*pW
    ob += (eps)*pB
    
    hz,oo = compute_outputs(hw,hb,ow,ob,inputs)
    dB,dW,db,dw = compute_grads(hw,hz,hb,hsW,hsB,ow,oo,ob,osW,osB,inputs,outputs)

    pw += (eps/2.0)*dw
    pb += (eps/2.0)*db
    pW += (eps/2.0)*dW
    pB += (eps/2.0)*dB

    return hw,hb,pw,pb,ow,ob,pW,pB


if __name__ == "__main__":
    #:    global hidden_weights, hidden_biases, output_weights, output_biases
    inputs = np.loadtxt('input_files/maf_%d_%d'%(num_inputs,N),dtype=np.float128)
    outputs = np.loadtxt('input_files/hc_%d'%(N),dtype=np.float128)
    hidden_weights = np.random.normal(0,var,(num_inputs,num_hidden)).astype(np.float128)
    #hidden_weights = np.loadtxt('input_files/init_hw')
    hidden_biases +=  np.random.normal(0,var,(num_hidden)).astype(np.float128)
    output_weights += np.random.normal(0,var,(num_hidden,2)).astype(np.float128)
    #output_weights += np.loadtxt('input_files/init_ow')
    output_biases += np.random.normal(0,var,(2)).astype(np.float128)

#    eps = 0.00001
    hidden_outputs,output_outputs = compute_outputs(hidden_weights,hidden_biases, output_weights, output_biases, inputs)
    dB,dW,db,dW = compute_grads(hidden_weights,hidden_outputs, hidden_biases,hidden_sW,hidden_sB, output_weights, output_outputs,output_biases,output_sW,output_sB, inputs,outputs)
    steps = 2
#    print "outputsW:",output_sW
    for i in range(steps):
        print "Step:",(i+1)

        hidden_outputs,output_outputs = compute_outputs(hidden_weights,hidden_biases, output_weights, output_biases, inputs)
        output_biases_grad,output_weights_grad,hidden_biases_grad,hidden_weights_grad = compute_grads(hidden_weights,hidden_outputs, hidden_biases,hidden_sW,hidden_sB, output_weights, output_outputs,output_biases,output_sW,output_sB, inputs,outputs)
        l,k,U,H = Hamiltonian(outputs,output_outputs,pw,pb,pW,pB,hidden_weights,hidden_biases,hidden_sW,hidden_sB,output_weights, output_biases, output_sW, output_sB)
        if i==0:
            H = H[0]
        hidden_sW, hidden_sB, hpw_mean, output_sW, output_sB = gibbs_update(hidden_weights, hidden_biases, hidden_sW, hidden_sB, hpw_mean, hpw_shape, hpw_scale, output_weights, output_biases, output_sW, output_sB, opw_shape, opw_scale)
        hidden_weights,hidden_biases,pw,pb,output_weights,output_biases,pW,pB = leap_frog(hidden_weights, hidden_biases,hidden_sW,hidden_sB,pw,pb,hidden_weights_grad,hidden_biases_grad,output_weights,output_biases,output_sW,output_sB,pW,pB,output_weights_grad,output_biases_grad,eps,inputs,outputs)

        output_biases_grad,output_weights_grad,hidden_biases_grad,hidden_weights_grad = compute_grads(hidden_weights,hidden_outputs,hidden_biases,hidden_sW,hidden_sB, output_weights, output_outputs,output_biases,output_sW,output_sB, inputs,outputs)
        l_new,k_new,U_new,H_new = Hamiltonian(outputs,output_outputs,pw,pb,pW,pB,hidden_weights,hidden_biases,hidden_sW,hidden_sB,output_weights, output_biases, output_sW, output_sB)
        
        print 'current U:',U
        print 'current L:',l
        print 'current K:',k
        print 'current H:',H
        print 'proposed U:',U_new
        print 'proposed L:',L_new
        print 'proposed K:',k_new
        print 'proposed H:',H_new
        print 'diff-h:',H_new-H
        print 'diff-k:',k_new-k
        print 'diff-u:',U_new-U
        print 'diff-l:',l_new-l
        print 'ratio-u:',(U_new-U)/U
        print 'ratio-l:',(l_new-l)/l
        print 'ratio-h:',(H_new-H)/H
        print 'ratio-k:',(k_new-k)/k

print "MEAN:",hpw_mean
