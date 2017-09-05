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

    global hidden_weights, hidden_outputs,hidden_biases,hidden_sW,hidden_sB, output_weights,output_outputs,output_biases,output_sW,output_sB,inputs,outputs,hidden_weights_grad,hidden_biases_grad, output_weights_grad, output_biases_grad,log_on, prior_on, for_init

    diff = outputs - output_outputs    #the main difference term
    dw = np.zeros((num_inputs,num_hidden),precision)
    db = np.zeros((num_hidden),precision)
    dW = np.zeros((num_hidden,2),precision)
    dB = np.zeros((2),precision)
    if log_on:
        dB = np.dot(np.ones((1,np.shape(diff)[0])),diff).reshape((output_weights.shape[1],))
        dW = np.dot(hidden_outputs.T,diff) 
        bp = np.dot(diff,output_weights.T)
        prod = hidden_outputs*(1-hidden_outputs)
        back = bp*prod
        db = np.dot(np.ones((1,np.shape(back)[0])),back).reshape((hidden_weights.shape[1],))
        dw = np.dot(inputs.T,back)
    if prior_on and for_init==False:
        dB -= output_biases/output_sB[0]
        dW -= output_weights/output_sW[0]
        db -= hidden_biases/hidden_sB[0]
        for i in range(len(hidden_weights)):
            dw[i] -= (hidden_weights[i])/(hidden_sW[0][i])
    

    hidden_weights_grad = dw
    hidden_biases_grad = db
    output_weights_grad = dW
    output_biases_grad = dB

def compute_Egrads():

    global hidden_weights, hidden_outputs,hidden_biases,hidden_sW,hidden_sB, output_weights,output_outputs,output_biases,output_sW,output_sB,inputs,outputs,hidden_weights_Egrad,hidden_biases_Egrad, output_weights_Egrad, output_biases_Egrad 

    diff = outputs - output_outputs    #the main difference term
    p = output_outputs[::,0]*output_outputs[::,1]
    p = np.array([p,p]).reshape(len(diff),2)
    diff = diff*p
    dB = np.dot(np.ones((1,np.shape(diff)[0])),diff).reshape((output_weights.shape[1],))
    dW = np.dot(hidden_outputs.T,diff) 
    bp = np.dot(diff,output_weights.T)
    prod = hidden_outputs*(1-hidden_outputs)
    back = bp*prod
    db = np.dot(np.ones((1,np.shape(back)[0])),back).reshape((hidden_weights.shape[1],))
    dw = np.dot(inputs.T,back)

    hidden_weights_Egrad = dw
    hidden_biases_Egrad = db
    output_weights_Egrad = dW
    output_biases_Egrad = dB



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
    global outputs, output_outputs,pw,pb,pB,pW,hidden_weights,hidden_biases,hidden_sW,hidden_sB,output_weights,output_biases,output_sW,output_sB, prior_on, log_on
    if log_on:
        log = outputs*np.log(output_outputs)
        log = log.sum()
    else:
        log = 0
    k = (pw**2).sum() + (pb**2).sum() + (pW**2).sum() + (pB**2).sum()
    if prior_on:
        p = prior_contrib()
    else:
        p=0
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
    

def error():
    global outputs, output_outputs
    rmsd = 0
    for i in range(len(outputs)):
        rmsd += (outputs[i][0] - output_outputs[i][0])**2
    return rmsd/len(outputs)




def initialise():
    global hidden_weights, initialise_steps, hidden_biases, output_weights, output_biases, initialis_eps, hidden_weights_grad, hidden_biases_grad, output_weights_grad, output_biases_grad,for_init, track_prior

    print 'Beginning Initialisation Stes:'
    for_init = True
    for i in range(initialise_steps):
        compute_outputs()

        l,k,U,H = Hamiltonian()
        if params['descent'] == "log":
            compute_grads()
            hidden_weights += initialise_eps*hidden_weights_grad
            hidden_biases += initialise_eps*hidden_biases_grad
            output_weights += initialise_eps*output_weights_grad
            output_biases += initialise_eps*output_biases_grad

        elif params['descent'] == "error":
            compute_Egrads()
            hidden_weights += initialise_eps*hidden_weights_Egrad
            hidden_biases += initialise_eps*hidden_biases_Egrad
            output_weights += initialise_eps*output_weights_Egrad
            output_biases += initialise_eps*output_biases_Egrad

        compute_outputs()
        l_new,k_new, U_new, H_new = Hamiltonian()

        print 'Iteration:',i
        print 'RMSD Accuracy:',error()
        print 'current U:',U
        print 'current L:',l
        print 'current K:',k
        print 'current H:',H
        print 'proposed U:',U_new
        if log_on:
            print 'proposed L:',l_new
        print 'proposed K:',k_new
        print 'proposed H:',H_new
        print 'diff-h:',H_new-H
        print 'diff-k:',k_new-k
        print 'diff-u:',U_new-U
        if log_on:
            print 'diff-l:',l_new-l
        print 'ratio-u:',(U_new-U)/U
        if log_on:
            print 'ratio-l:',(l_new-l)/l
        print 'ratio-h:',(H_new-H)/H
        print 'ratio-k:',(k_new-k)/k
        if track_theta:
            print_theta()
        if track_prior:
            print_variances()


def print_theta():
    global hidden_weights, hidden_biases, output_weights, output_biases
    for i,j in enumerate(hidden_weights):
        for k,l in enumerate(j):
            print "w_%d,%d=%f"%(i,k,l)
    for i,j in enumerate(hidden_biases):
        print "b_%d=%f"%(i,j)

    for i,j in enumerate(output_weights):
        for k,l in enumerate(j):
            print "W_%d,%d=%f"%(i,k,l)
    for i,j in enumerate(output_biases):
        print "B_%d=%f"%(i,j)

def print_momenta():
    global pw, pb, pW, pB
    for i,j in enumerate(pw):
        for k,l in enumerate(j):
            print "pw_%d,%d=%f"%(i,k,l)
    for i,j in enumerate(pb):
        print "pb_%d=%f"%(i,j)

    for i,j in enumerate(pW):
        for k,l in enumerate(j):
            print "pW_%d,%d=%f"%(i,k,l)
    for i,j in enumerate(pB):
        print "pB_%d=%f"%(i,j)



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

def hmc():
    global prior_on
    for i in range(hmc_steps):
        print "Step:",(i+1)

        compute_outputs()
        compute_grads()
        l,k,U,H = Hamiltonian()
        if gibbs_on:
            gibbs_update()
        leap_frog()
        compute_outputs()
        l_new,k_new, U_new, H_new = Hamiltonian()
        
        print 'current U:',U
        print 'current L:',l
        print 'current K:',k
        print 'current H:',H
        print 'proposed U:',U_new
        if log_on:
            print 'proposed L:',l_new
        print 'proposed K:',k_new
        print 'proposed H:',H_new
        print 'diff-h:',H_new-H
        print 'diff-k:',k_new-k
        print 'diff-u:',U_new-U
        if log_on:
            print 'diff-l:',l_new-l
        print 'ratio-u:',(U_new-U)/U
        if log_on:
            print 'ratio-l:',(l_new-l)/l
        print 'ratio-h:',(H_new-H)/H
        print 'ratio-k:',(k_new-k)/k
        if track_theta:
            print_theta()
    
def print_variances():
    global hidden_sW, hidden_sB, output_sW, output_sB
    print hidden_sW
    print hidden_sW.shape
    for i,j in enumerate(hidden_sW[0]):
        print "sw_%d="%(i),j
    print "sb=",(hidden_sB[0])
    print "ow=",(output_sB[0])
    print "ob=",(output_sW[0])


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
        print "I loaded a file"
        hidden_weights = np.loadtxt(params['hidden_weights'],dtype=precision)
        print hidden_weights

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
    
        
    # hidden layer prior settings; shape, scale, sW, sB, mean
    
    hpw_shape = float(params['ard_prior_shape']) #hidden layer prior weights shape
    hpw_scale =  float(params['ard_prior_scale']) #hidden layer prior weights scale
    if isfloat(params['hidden_sw']): 
        init = float(params['hidden_sw'])
        hidden_sW = (np.tile(init,reps=hidden_weights.shape[0]).reshape(1,hidden_weights.shape[0])).astype(precision) # each input unit has one sigma. so you repeat the same initial sigma D times to get a vector
    else:
        hidden_sW = np.loadtxt(params['hidden_sw']).reshape(1,hidden_weights.shape[0]).astype(precision)
    hpw_mean = (np.tile(hpw_scale/(hpw_shape-1),reps=hidden_weights.shape[0]).reshape(1,hidden_weights.shape[0])).astype(precision) #maintain means of all the variances of prior of each input weight 
    if params['hidden_sb']=="auto":
        hidden_sB = invgamma.rvs(1.0,scale=1.0, size = (1,1)).astype(precision) #the biases have just one common prior
    else:
        hidden_sB = np.loadtxt(params['hidden_sb']).reshape((1,1)).astype(precision)
    
    if params['output_sw']=="auto":
        output_sW = np.array([100.],dtype=precision) #outputs have exactly one prior for all weights/biases
    else:
        output_sW = np.loadtxt(params['output_sw']).reshape((1,1)).astype(precision)
    
    if params['output_sb']=="auto":
        output_sB = np.array([100.],dtype=precision) #outputs have exactly one prior for all weights/biases
    else:
        output_sB = np.loadtxt(params['output_sb']).reshape((1,1)).astype(precision)
    #end of hidden layer prior
    gibbs_on = True if params['gibbs_on']=='true' else False

    opw_shape = 0.1
    opw_scale = 0.1

    hidden_weights_grad = np.zeros((num_inputs,num_hidden),precision)
    hidden_biases_grad = np.zeros((num_hidden),precision)
    output_weights_grad = np.zeros((num_hidden,2),precision)
    output_biases_grad = np.zeros((2),precision)
    
    hidden_weights_Egrad = np.zeros((num_inputs,num_hidden),precision)
    hidden_biases_Egrad = np.zeros((num_hidden),precision)
    output_weights_Egrad = np.zeros((num_hidden,2),precision)
    output_biases_Egrad = np.zeros((2),precision)

    hidden_outputs = np.zeros((num_hidden),precision)
    output_outputs = np.zeros((2),precision)
    #....
    for_init = False
    #Sampling Variables:
    init_sd_output = 1.0    #not provided in input parameters because it doesn't appear to be something that is changed so far
    init_sd_hidden = 1.0
    if params['pw']=="auto":
        pw = np.random.normal(0,init_sd_hidden,hidden_weights.shape).astype(precision)
    else:
        pw = np.loadtxt(params['pw'])
    
    if params['pb']=="auto":
        pb = np.random.normal(0,init_sd_hidden,hidden_biases.shape).astype(precision)
    else:
        pb = np.loadtxt(params['pb'])
    if params['pW']=="auto":
        pW = np.random.normal(0,init_sd_output,output_weights.shape).astype(precision)
    else:
        pW = np.loadtxt(params['pW'])
    if params['pB']=="auto":
        pB = np.random.normal(0,init_sd_output,output_biases.shape).astype(precision)
    else:
        pB = np.loadtxt(params['pB'])
    #....


    eps = float(params['hmc_eps'])
    initialise_eps = float(params['initialise_eps'])
    initialise_steps = int(params['initialise_steps'])
    hmc_steps = int(params['hmc_steps'])
    
    log_on = True if not  params['log_on'] else ( False if params['log_on']=="false" else True )
    prior_on = True if not  params['prior_on'] else ( False if params['prior_on']=="false" else True )
    track_theta = True if params['track_theta']=='true' else False
    track_prior = True if params['track_prior']=='true' else False
    
    if params['static_prior']=="false":
        gibbs_update()
    print_variances()
    print_momenta()
    if track_theta:
        print_theta()
    if not params['initialise']=='false':
        initialise()
    if not params['hmc']=='false':
        hmc()

