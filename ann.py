import numpy as np
import sys
from scipy.stats import invgamma


def compute_outputs():
    global hidden_weights,hidden_biases, output_weights, output_biases, inputs,hidden_outputs,output_outputs
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

    hidden_outputs = h_z

def compute_grads():

    global hidden_weights, hidden_outputs,hidden_biases,hidden_sW,hidden_sB, output_weights,output_outputs,output_biases,output_sW,output_sB,inputs,outputs,hidden_weights_grad,hidden_biases_grad, output_weights_grad, output_biases_grad 

    diff = outputs - output_outputs    #the main difference term

    dB = np.dot(np.ones((1,np.shape(diff)[0])),diff).reshape((output_weights.shape[1],))
    dB -= output_biases/output_sB[0]
    dW = np.dot(hidden_outputs.T,diff) 
    dW -= output_weights/output_sW[0]
    bp = np.dot(diff,output_weights.T)
    prod = hidden_outputs*(1-hidden_outputs)
    back = bp*prod
    db = np.dot(np.ones((1,np.shape(back)[0])),back).reshape((hidden_weights.shape[1],))
    db -= hidden_biases/hidden_sB[0]
    dw = np.dot(inputs.T,back)
    for i in range(len(hidden_weights)):
        dw[i] -= (hidden_weights[i])/(hidden_sW[0][i])
    

    hidden_weights_grad = dw
    hidden_biases_grad = db
    output_weights_grad = dW
    output_biases_grad = dB


def prior_contrib():
    global hidden_weights, hidden_biases, hidden_sW, hidden_sB, output_weights, output_biases, output_sW, output_sB
    val = 0 
    for i,j in zip(hidden_weights,hidden_sW[0]):
        val -= (i**2).sum()/(2*j)
    for i in output_weights:
        val -= (i**2).sum()/(output_sW[0])
    val -= (hidden_biases**2).sum()/(2*hidden_sB[0])
    val -= (output_biases**2).sum()/(2*output_sB[0])
    return val


def Hamiltonian():
    global outputs, output_outputs,pw,pb,pB,pW,hidden_weights,hidden_biases,hidden_sW,hidden_sB,output_weights,output_biases,output_sW,output_sB
    log = outputs*np.log(output_outputs)
    log = log.sum()
    k = (pw**2).sum() + (pb**2).sum() + (pW**2).sum() + (pB**2).sum()
    
    p = prior_contrib()
    return log,k,log+p,(log+p-k)



def gibbs_update():
    global hidden_weights, hidden_biases, hidden_sW, hidden_sB,hpw_mean,hpw_shape,hpw_scale, output_weights, output_biases, output_sW, output_sB,opw_shape,opw_scale
#update for ARD
    new_hsW = np.zeros(hidden_sW.shape)
    new_mean = np.zeros(hpw_mean.shape)
    n_w = precision(hidden_weights.shape[1])
    hpw_shape_new = hpw_shape+ n_w/2.0
    for i in range(len(hidden_weights)):
        hpw_scale_new=hpw_scale + (hidden_weights[i]**2).sum()/2.0
        new_val = invgamma.rvs(hpw_shape_new,scale=hpw_scale_new,size=1)
        new_hsW[0,i]=precision(new_val)
        new_mean[0,i] = precision(hpw_scale_new/(hpw_shape_new-1.0))
    
    hidden_sW = new_hsW.astype(precision)
    hpw_mean = new_mean.astype(precision)
    n_b = precision(hidden_biases.shape[0])
    hpb_shape_new = hpw_shape + n_b/2.0
    hpb_scale_new = hpw_scale + (hidden_biases**2).sum()/2.0
    new_val = invgamma.rvs(hpb_shape_new, scale=hpb_scale_new,size=1)
    hidden_sB = precision(new_val)
    #update for GLP
    n_w = precision(output_weights.shape[0]*output_weights.shape[1])
    shape_new = opw_shape + n_w/2.0
    scale_new = opw_scale + (output_weights**2).sum()/2.0
    new_val = invgamma.rvs(shape_new, scale=scale_new, size=1)
    
    output_sW = precision(new_val)

    n_b = precision(output_biases.shape[0])
    shape_new = opw_shape+ n_b/2.0
    scale_new = opw_scale + (output_biases**2).sum()/2.0
    new_val = invgamma.rvs(shape_new,scale=scale_new,size=1)
    output_sB = precision(new_val)
    
    
def leap_frog():
    global hidden_weights, hidden_biases,hidden_sW,hidden_sB, pw,pb, hidden_weights_grad, hidden_biases_grad, output_weights,output_biases,output_sW,output_sB,pW,pB,output_weights_grad,output_biases_grad,eps,inputs,outputs,eps
    pw += (eps/2.0)*hidden_weights_grad
    pb += (eps/2.0)*hidden_biases_grad
    pW += (eps/2.0)*output_weights_grad
    pB += (eps/2.0)*output_biases_grad

    hidden_weights += (eps)*pw
    hidden_biases += (eps)*pb
    output_weights += (eps)*pW
    output_biases += (eps)*pB
    
    compute_outputs()
    compute_grads()

    pw += (eps/2.0)*hidden_weights_grad
    pb += (eps/2.0)*hidden_biases_grad
    pW += (eps/2.0)*output_weights_grad
    pB += (eps/2.0)*output_biases_grad
    


def read_input(fname):
    f = open(fname)
    params = {}
    for i in f.readlines():
        i = i.split(":")
        key = i[0]
        value = i[1][:-1]
        if key[0]!="#":
            params[key]=value
    return params
        

def isfloat(val):
    try:
        a = float(val)
        return True
    except:
        return False
 

if __name__ == "__main__":

    #input file is given as cmd-line arg
    params = read_input(sys.argv[1])
    #precision argument is supposed to say double or single. The default value is double
    precision = np.float32 if params['precision']=="single" else np.float128

    #:    global hidden_weights, hidden_biases, output_weights, output_biases
    #input vector and output vector files are specified
    inputs = np.loadtxt(params['input_vector'],dtype=precision)
    outputs = np.loadtxt(params['output_vector'],dtype=precision)
    num_inputs = len(inputs[0]) #number of inputs is directly inferred from the file
    num_hidden = int(params['num_hidden_units']) #number of hidden units for the NN is specified in input
    
    # the "hidden_weights" parameter allows for you to go two ways: specify a float to specify a variance from which to draw the hidden_weights (centered on 0). The other is to specify a file from which you can directly load values of hidden_weights. Same will be applied for hidden_biases, output_weights, output_biases 

    if isfloat(params['hidden_weights']): 
        var = float(params['hidden_weights'])
        hidden_weights = np.random.normal(0,var,(num_inputs,num_hidden)).astype(precision)
    else:
        hidden_weights = np.loadtxt(params['hidden_weights'],dtype=precision)

    if isfloat(params['hidden_biases']): 
        var = float(params['hidden_biases'])
        hidden_biases = np.random.normal(0,var,(num_hidden)).astype(precision)
    else:
        hidden_biases = np.loadtxt(params['hidden_biases'],dtype=precision)

    if isfloat(params['output_weights']): 
        var = float(params['output_weights'])
        output_weights = np.random.normal(0,var,(num_hidden,2)).astype(precision)
    else:
        output_weights = np.loadtxt(params['output_weights'],dtype=precision)

    if isfloat(params['output_biases']): 
        var = float(params['output_biases'])
        output_biases = np.random.normal(0,var,(2)).astype(precision)
    else:
        output_biases = np.loadtxt(params['output_biases'],dtype=precision)
    
        
    init = float(params['ard_init'])
    # hidden layer prior settings; shape, scale, sW, sB, mean
    hpw_shape = float(params['ard_prior_shape']) #hidden layer prior weights shape
    hpw_scale =  float(params['ard_prior_scale']) #hidden layer prior weights scale
    hidden_sW = (np.tile(init,reps=hidden_weights.shape[0]).reshape(1,hidden_weights.shape[0])).astype(precision) # each input unit has one sigma. so you repeat the same initial sigma D times to get a vector
    hidden_sB = invgamma.rvs(1.0,scale=1.0, size = (1,1)).astype(precision) #the biases have just one common prior
    hpw_mean = (np.tile(hpw_scale/(hpw_shape-1),reps=hidden_weights.shape[0]).reshape(1,hidden_weights.shape[0])).astype(precision) #maintain means of all the variances of prior of each input weight 
    #end of hidden layer prior
    
    opw_shape = 0.1
    opw_scale = 0.1
    output_sW = np.array([100.],dtype=precision) #outputs have exactly one prior for all weights/biases
    output_sB = invgamma.rvs(1.0,scale=1.0,size=(1,1)).astype(precision)#end output layer prior contribution

    hidden_weights_grad = np.zeros((num_inputs,num_hidden),precision)
    hidden_biases_grad = np.zeros((num_hidden),precision)
    output_weights_grad = np.zeros((num_hidden,2),precision)
    output_biases_grad = np.zeros((2),precision)
    
    hidden_outputs = np.zeros((num_hidden),precision)
    output_outputs = np.zeros((2),precision)
    #....
    
    #Sampling Variables:
    init_sd_output = 1.0    #not provided in input parameters because it doesn't appear to be something that is changed so far
    init_sd_hidden = 1.0
    pW = np.random.normal(0,init_sd_output,output_weights.shape).astype(precision)
    pB = np.random.normal(0,init_sd_output,output_biases.shape).astype(precision)
    pw = np.random.normal(0,init_sd_hidden,hidden_weights.shape).astype(precision)
    pb = np.random.normal(0,init_sd_hidden,hidden_biases.shape).astype(precision)
    #....


    eps = float(params['hmc_eps'])
    compute_outputs()
    gibbs_update()
    compute_grads()
    hmc_steps = int(params['hmc_steps'])
    for i in range(hmc_steps):
        print "Step:",(i+1)

        compute_outputs()
        compute_grads()
        l,k,U,H = Hamiltonian()
        gibbs_update()
        leap_frog()
        compute_outputs()
        l_new,k_new, U_new, H_new = Hamiltonian()
        
        print 'current U:',U
        print 'current L:',l
        print 'current K:',k
        print 'current H:',H
        print 'proposed U:',U_new
        print 'proposed L:',l_new
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
