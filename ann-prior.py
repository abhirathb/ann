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
hidden_sW = (np.tile(init,reps=hidden_weights.shape[0]).reshape(1,hidden_weights.shape[0])).astype(np.float128)
hidden_sB = invgamma.rvs(1.0,scale=1.0, size = (1,1)).astype(np.float128)
hpw_mean = (np.tile(hpw_scale/(hpw_shape-1),reps=hidden_weights.shape[0]).reshape(1,hidden_weights.shape[0])).astype(np.float128)
#end of hidden layer prior

#output layer prior: 
opw_shape = 0.1
pqw_scale = 0.1
output_sW = np.array([100.],dtype=np.float128)
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

def compute_grads(hidden_weights, hidden_outputs, output_weights,output_outputs,inputs,outputs):
    diff = outputs - output_outputs    #the main difference term

    dB = np.dot(np.ones((1,np.shape(diff)[0])),diff).reshape((output_weights.shape[1],))
    dW = np.dot(hidden_outputs.T,diff) 
    bp = np.dot(diff,output_weights.T)
    prod = hidden_outputs*(1-hidden_outputs)
    back = bp*prod
    db = np.dot(np.ones((1,np.shape(back)[0])),back).reshape((hidden_weights.shape[1],))
    dw = np.dot(inputs.T,back)
    return dB,dW,db,dw


def prior_contrib(hidden_weights, hidden_biases, hsW, hsB, output_weights, output_biases, osW, osB):
    val = 0 
    for i,j in zip(hidden_weights,hsW[0]):
        val -= (i**2).sum()/(2*j)
    for i in output_weights:
        val -= (i**2).sum()/(osW[0])
    val -= (hidden_biases**2).sum()/(2*hsB[0][0])
    val -= (output_biases**2).sum()/(2*osB[0][0])
    return val
def Hamiltonian(outputs, output_outputs,pw,pb,pB,pW,hidden_weights,hidden_biases,hidden_sW,hidden_sB,output_weights,output_biases,output_sW,output_sB):
    log = outputs*np.log(output_outputs)
    log = log.sum()
    k = (pw**2).sum() + (pb**2).sum() + (pW**2).sum() + (pB**2).sum()
    
    p = prior_contrib(hidden_weights,hidden_biases,hidden_sW,hidden_sB,output_weights, output_biases, output_sW, output_sB)
    return log,k,log+k+p

def leap_frog(hw, hb, pw,pb, dw,db, ow,ob,pW,pB,dW,dB,eps,inputs,outputs):
    pw += (eps/2.0)*dw
    pb += (eps/2.0)*db
    pW += (eps/2.0)*dW
    pB += (eps/2.0)*dB

    hw += (eps)*pw
    hb += (eps)*pb
    ow += (eps)*pW
    ob += (eps)*pB
    
    hz,oo = compute_outputs(hw,hb,ow,ob,inputs)
    dB,dW,db,dw = compute_grads(hw,hz,ow,oo,inputs,outputs)

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
    dB,dW,db,dW = compute_grads(hidden_weights,hidden_outputs, output_weights, output_outputs, inputs,outputs)
    steps = 2

    sys.exit()
    for i in range(2):
        print "Step:",(i+1)

        hidden_outputs,output_outputs = compute_outputs(hidden_weights,hidden_biases, output_weights, output_biases, inputs)
        output_biases_grad,output_weights_grad,hidden_biases_grad,hidden_weights_grad = compute_grads(hidden_weights,hidden_outputs, output_weights, output_outputs, inputs,outputs)
        l,k,H = Hamiltonian(outputs,output_outputs,pw,pb,pW,pB,hidden_weights,hidden_biases,hidden_sW,hidden_sB,output_weights, output_biases, output_sW, output_sB)
        hidden_weights,hidden_biases,pw,pb,output_weights,output_biases,pW,pB = leap_frog(hidden_weights, hidden_biases,pw,pb,hidden_weights_grad,hidden_biases_grad,output_weights,output_biases,pW,pB,output_weights_grad,output_biases_grad,eps,inputs,outputs)

        output_biases_grad,output_weights_grad,hidden_biases_grad,hidden_weights_grad = compute_grads(hidden_weights,hidden_outputs, output_weights, output_outputs, inputs,outputs)
        l_new,k_new,H_new = Hamiltonian(outputs,output_outputs,pw,pb,pW,pB,hidden_weights,hidden_biases,hidden_sW,hidden_sB,output_weights, output_biases, output_sW, output_sB)
        
        print 'current U:',l
        print 'current K:',k
        print 'current H:',H
        print 'proposed U:',l_new
        print 'proposed K:',k_new
        print 'proposed H:',H_new
        print 'diff-h:',H_new-H
        print 'diff-k:',k_new-k
        print 'ratio-h:',(H_new-H)/H
        print 'ratio-k:',(k_new-k)/k


