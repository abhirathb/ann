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
    diff = diff/len(diff)  # so that derivative is of normalised error
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
    global hidden_weights, hidden_biases, hidden_sw, hidden_sb, output_weights, output_biases, output_sw, output_sb
    val = 0 
    for i,j in zip(hidden_weights,hidden_sW[0]):
        ws = (i**2).sum()
        val -= (i**2).sum()/(2*j)
    val -= (output_weights**2).sum()/(2*output_sW[0])
    val -= (hidden_biases**2).sum()/(2*hidden_sB[0])
    val -= (output_biases**2).sum()/(2*output_sB[0])
    return val


def analytical_diff_prior():
    global eps,hidden_weights, hidden_biases, hidden_sw, hidden_sb, output_weights, output_biases, output_sw, output_sb,pw,pW,pb,pB
    err = 0
    for i in range(len(hidden_weights)):
        s = hidden_sW[0][i]
        for j in range(len(hidden_weights[i])):
            x = hidden_weights[i][j]
            p = pw[i][j]
            err += (eps**3) * ((eps**3)*(x**2) - 4*(eps**2)*(p)*(s)*x + 4*eps*(p**2)*(s**2) -4*eps*s*(x**2) + 8*p*x*(s**2))/(32*(s**4))
    for i in range(len(hidden_biases)):
        s = hidden_sB[0]
        x = hidden_biases[i]
        p = pb[i] 
        err += (eps**3) * ((eps**3)*(x**2) - 4*(eps**2)*(p)*(s)*x + 4*eps*(p**2)*(s**2) -4*eps*s*(x**2) + 8*p*x*(s**2))/(32*(s**4))
    s = output_sW[0]
    for i in range(len(output_weights)):
        for j in range(len(output_weights[i])):
            x = output_weights[i][j]
            p = pW[i][j]
            err += (eps**3) * ((eps**3)*(x**2) - 4*(eps**2)*(p)*(s)*x + 4*eps*(p**2)*(s**2) -4*eps*s*(x**2) + 8*p*x*(s**2))/(32*(s**4))

    for i in range(len(output_biases)):
        s = output_sB[0]
        x = output_biases[i]
        p = pB[i] 
        err += (eps**3) * ((eps**3)*(x**2) - 4*(eps**2)*(p)*(s)*x + 4*eps*(p**2)*(s**2) -4*eps*s*(x**2) + 8*p*x*(s**2))/(32*(s**4))

    return err
    

def Hamiltonian():
    global outputs, output_outputs,pw,pb,pB,pW,hidden_weights,hidden_biases,hidden_sW,hidden_sB,output_weights,output_biases,output_sW,output_sB, prior_on, log_on
    if log_on:
        log = outputs*np.log(output_outputs)
        log = log.sum()
    else:
        log = 0
    k = (pw**2).sum() + (pb**2).sum() + (pW**2).sum() + (pB**2).sum()
    k = k/2
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
    global hidden_weights, hidden_biases,hidden_sW,hidden_sB, pw,pb, hidden_weights_grad, hidden_biases_grad, output_weights,output_biases,output_sW,output_sB,pW,pB,output_weights_grad,output_biases_grad,eps,inputs,outputs,eps,for_init
    for_init=False
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
    return 1.0- (rmsd/len(outputs))**0.5
def maj_error():
    global outputs, output_outputs
    err = 0
    for i in range(len(outputs)):
        err += np.round(abs(outputs[i][0] - output_outputs[i][0]))
    return 1.0 - (err/len(outputs))

def save_params():
    global hidden_weights,hidden_biases, output_weights, output_biases, pW,pB,pw,pb
    bkp = {}
    w = hidden_weights.copy()
    bkp['w']=w
    b = hidden_biases.copy()
    bkp['b']=b
    W = output_weights.copy()
    bkp['W']=W
    B = output_biases.copy()
    bkp['B']=B
    bkp['pw'] = pw
    bkp['pb'] = pb
    bkp['pW'] = pW
    bkp['pB'] = pB
    return bkp
def revert_params(bkp):
    global hidden_weights,hidden_biases, output_weights, output_biases, pW,pB,pw,pb
    hidden_weights = bkp['w']
    hidden_biases = bkp['b']
    output_weights = bkp['W']
    output_biases = bkp['B']
    pw = bkp['pw']
    pb = bkp['pb']
    pW = bkp['pW']
    pB = bkp['pB']

def initialise():
    global hidden_weights, initialise_steps, hidden_biases, output_weights, output_biases, initialis_eps, hidden_weights_grad, hidden_biases_grad, output_weights_grad, output_biases_grad,for_init, track_prior, evolve_w, evolve_b,evolve_W,evolve_B

    print 'Beginning Initialisation Steps:'
    for_init = True if params['descent']!="kernel" else False
    for i in range(initialise_steps):
        compute_outputs()

        l,k,U,H = Hamiltonian()
        if params['descent'] == "log" or params['descent']=="kernel":
            compute_grads()
            if evolve_w==True:
                hidden_weights += initialise_eps*hidden_weights_grad
            if evolve_b==True:
                hidden_biases += initialise_eps*hidden_biases_grad
            if evolve_W==True:
                output_weights += initialise_eps*output_weights_grad
            if evolve_B==True:
               output_biases += initialise_eps*output_biases_grad


        elif params['descent'] == "error":
            compute_Egrads()

            if evolve_w==True:
                hidden_weights += initialise_eps*hidden_weights_Egrad
            if evolve_b==True:
                hidden_biases += initialise_eps*hidden_biases_Egrad
            if evolve_W==True:
                output_weights += initialise_eps*output_weights_Egrad
            if evolve_B==True:
                output_biases += initialise_eps*output_biases_Egrad

        compute_outputs()
        l_new,k_new, U_new, H_new = Hamiltonian()

        print 'Iteration:',i
        print 'RMSD Accuracy:',error()
        print 'Majority Accuracy:',maj_error()
        print 'current U:',U
        print 'current L:',l
        print 'current K:',k
        print 'current H:',H
        print 'proposed U:',U_new
        if log_on:
            print 'proposed L:',l_new
        print 'proposed K:',k_new
        print 'proposed H:',H_new
        print 'diff-h:',U_new - k_new -U + k
        print 'diff-k:',k_new-k
        print 'diff-u:',U_new-U
        if log_on:
            print 'diff-l:',l_new-l
        if U!=0:
            print 'ratio-u:',(U_new-U)/U
        if log_on:
            print 'ratio-l:',(l_new-l)/l
        print 'ratio-h:',(H_new-H)/H
        print 'ratio-k:',(k_new-k)/k
        if track_theta:
            print_theta()
        if track_prior:
            print_variances()
        for i,j in enumerate(output_outputs):
            print "f-%d:"%i,j[0]

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
    accept = 0.0
    for i in range(hmc_steps):
        print "Step:",(i+1)
        bkp = copy_params()
        compute_outputs()
        compute_grads()
        l,k,U,H = Hamiltonian()
        if gibbs_on:
            gibbs_update()
        leap_frog()
        compute_outputs()
        l_new,k_new, U_new, H_new = Hamiltonian()
        diff = H_new - H
        alpha = np.min([0,diff])
        u = np.log(np.random.random(1)[0])
        if u<alpha:
            msg="Accept"
            accept+=1.0
            revert_params(bkp)
        else:
            msg="Reject!"
        print 'current U:',U
        print 'current L:',l
        print 'current K:',k
        print 'current H:',H
        print 'proposed U:',U_new
        if log_on:
            print 'proposed L:',l_new
        print 'proposed K:',k_new
        print 'proposed H:',H_new
        print 'diff-h:%e'%(H_new-H)
        print 'diff-k:%e'%(k_new-k)
        print 'diff-u:%e'%(U_new-U)
        if log_on:
            print 'diff-l:%e'%(l_new-l)
        print 'ratio-u:%e'%((U_new-U)/U)
        if log_on:
            print 'ratio-l:%e'%((l_new-l)/l)
        print 'ratio-h:%e'%((H_new-H)/H)
        print 'ratio-k:%e'%((k_new-k)/k)
        print 'RMSD Accuracy:',error()
        print 'Majority Accuracy:',maj_error()
        print 'Acceptance Rate:',(accept/(i+1))
        if track_theta:
            print_theta()
            print_momenta() 
def print_variances():
    global hidden_sW, hidden_sB, output_sW, output_sB
    print hidden_sW
    print hidden_sW.shape
    for i,j in enumerate(hidden_sW[0]):
        print "sw_%d="%(i),j
    print "sb=",(hidden_sB[0])
    print "ow=",(output_sW[0])
    print "ob=",(output_sB[0])


if __name__ == "__main__":

    #input file is given as cmd-line arg
    params = read_input(sys.argv[1])
    #precision argument is supposed to say double or single. The default value is double
    precision = np.float32 if params['precision']=="single" else np.float128
    #:    global hidden_weights, hidden_biases, output_weights, output_biases
    #input vector and output vector files are specified
    outputs = np.loadtxt(params['output_vector'],dtype=precision)
    num_hidden = int(params['num_hidden_units']) #number of hidden units for the NN is specified in input
    
    num_inputs = int(params['num_snp']) #number of input units
    inputs = np.loadtxt(params['input_vector'],dtype=precision)
    num_subjects = int(len(inputs))
    print num_subjects,num_inputs,inputs.shape
    print params['input_vector']
    inputs = inputs.reshape(num_subjects,num_inputs)

    # the "hidden_weights" parameter allows for you to go two ways: specify a float to specify a variance from which to draw the hidden_weights (centered on 0). The other is to specify a file from which you can directly load values of hidden_weights. Same will be applied for hidden_biases, output_weights, output_biases 

    if isfloat(params['hidden_weights']): 
        var = float(params['hidden_weights'])
        hidden_weights = np.random.normal(0,var,(num_inputs,num_hidden)).astype(precision)
    else:
   #     print "I loaded a file"
        hidden_weights = np.loadtxt(params['hidden_weights'],dtype=precision).reshape(num_inputs,num_hidden)
    #    print hidden_weights

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
        hidden_sB = invgamma.rvs(1.0,scale=1.0, size = (1,)).astype(precision) #the biases have just one common prior
    else:
        hidden_sB = np.loadtxt(params['hidden_sb']).reshape((1,)).astype(precision)
    
    if params['output_sw']=="auto":
        output_sW = np.array([100.],dtype=precision).reshape((1,))#outputs have exactly one prior for all weights/biases
    else:
        output_sW = np.loadtxt(params['output_sw']).reshape((1,)).astype(precision)
    
    if params['output_sb']=="auto":
        output_sB = np.array([100.],dtype=precision).reshape((1,)) #outputs have exactly one prior for all weights/biases
    else:
        output_sB = np.loadtxt(params['output_sb']).reshape((1,)).astype(precision)
    #end of hidden layer prior
    gibbs_on = True if params['gibbs_on']=='true' else False
    print "GIBBS:",gibbs_on
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
        print hidden_weights.shape
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
    print "HMC_eps:",eps
    initialise_eps = float(params['initialise_eps'])
    print "Initialize_eps:",initialise_eps
    initialise_steps = int(params['initialise_steps'])
    hmc_steps = int(params['hmc_steps'])
    
    log_on = True if not  params['log_on'] else ( False if params['log_on']=="false" else True )
    prior_on = True if not  params['prior_on'] else ( False if params['prior_on']=="false" else True )
    track_theta = True if params['track_theta']=='true' else False
    track_prior = True if params['track_prior']=='true' else False
    evolve_w = True if params['evolve_w']=='true' else False
    evolve_b = True if params['evolve_b']=='true' else False
    evolve_W = True if params['evolve_W']=='true' else False
    evolve_B = True if params['evolve_B']=='true' else False
    
    if params['static_prior']=="false":
        gibbs_update()
    #print_variances()
    #print_momenta()
    if track_theta:
        print_theta()
    if not params['initialise']=='false':
        initialise()
    if not params['hmc']=='false':
        hmc()
    for i,j in enumerate(hpw_mean):
        print i,j
