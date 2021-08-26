# -*- coding: utf-8 -*-
"""
Created on Fri Apr 26 01:23:00 2019

@author: Trader
"""


#Import tensorflow and numpy
import tensorflow as tf
import numpy as np
import math




l_0 = 3
l_1 = 4
l_2 = 2
lb_0 = l_1
lb_1 = l_2


#batch_size = 10
batch_size = 10



def build_toy_dataset1(n_samples = 100, N = 10 , noise_std = 0.1):
    #Pour obtenir des matrices bi-dimensionelles pour X et tester l'algorithme   
    
    #X2 sera de taille (n_samples,N)

    C2= np.random.randn(N, N) #np.array([[0., -0.1], [1.7, .4]])

    #on pourrait juste utiliser X1
    #X1 = np.dot(np.random.randn(n_samples, 2), C)
    #X2 = np.dot(np.random.randn(n_samples, 2), C)

    X2 = np.random.multivariate_normal(np.zeros(N), C2.T.dot(C2), n_samples)

    #générer W2, une matricee de poids, à utiliser
    C1= np.random.randn(2, 2)
    W2= np.dot(np.random.randn(N, 2), C1)  #on doit avoir la taille (N,2)

    #Y sera une transformation légère de X2, Y = X2*W2 où W2 va être construit à partir du code suivant
    
    the_noise =  np.random.normal(0, noise_std, size=n_samples)
    y = np.cos(np.dot(X2,W2)) + np.array([the_noise,the_noise]).T
    
    return X2,y



#https://stackoverflow.com/questions/10939213/how-can-i-calculate-the-nearest-positive-semi-definite-matrix?noredirect=1&lq=1
#How can I calculate the nearest positive semi-definite matrix?
#positive semi-definite matrices and numerical stability?: 
#https://stackoverflow.com/questions/2115880/positive-semi-definite-matrices-and-numerical-stability?noredirect=1&lq=1
#https://stackoverflow.com/questions/619335/a-simple-algorithm-for-generating-positive-semidefinite-matrices

def near_psd(x, epsilon=0):
    '''
    Calculates the nearest postive semi-definite matrix for a correlation/covariance matrix

    Parameters
    ----------
    x : array_like
      Covariance/correlation matrix
    epsilon : float
      Eigenvalue limit (usually set to zero to ensure positive definiteness)

    Returns
    -------
    near_cov : array_like
      closest positive definite covariance/correlation matrix

    Notes
    -----
    Document source
    http://www.quarchome.org/correlationmatrix.pdf

    '''

    if min(np.linalg.eigvals(x)) > epsilon:
        return x

    # Removing scaling factor of covariance matrix
    n = x.shape[0]
    var_list = np.array([np.sqrt(np.abs(x[i,i])) for i in range(n)])
    y = np.array([[x[i, j]/(var_list[i]*var_list[j]) for i in range(n)] for j in range(n)])

    # getting the nearest correlation matrix
    eigval, eigvec = np.linalg.eig(y)
    val = np.matrix(np.maximum(eigval, epsilon))
    vec = np.matrix(eigvec)
    T = 1/(np.multiply(vec, vec) * val.T)
    T = np.matrix(np.sqrt(np.diag(np.array(T).reshape((n)) )))
    B = T * vec * np.diag(np.array(np.sqrt(val)).reshape((n)))
    near_corr = B*B.T    

    # returning the scaling factors
    near_cov = np.array([[near_corr[i, j]*(var_list[i]*var_list[j]) for i in range(n)] for j in range(n)])

    print("new near_cov, done#")
    return near_cov




#Here is an algorithm to simple a GMRF X, given its mean U and its Precision matrix Q

def sample_GMRF(U,Q): 
    
    """ This function helps to sample Gaussian Markov Random Fields  values """
    
    """ Inputs are : mean (U) and Precision matrix (Q) """
    Q = near_psd(Q, epsilon=0.1)
    x_dim = Q.shape[1]        #la dimension de x
    
    """ Q must be semi-positive definite , or you'll get errors """
    
    
    L = np.linalg.cholesky(Q) #Cholesky de Q pour avoir L
    
    #il faut générer z, une normal standard
    z = np.random.multivariate_normal(np.zeros(x_dim), np.identity(x_dim)) #mean = np.zeros(5); cov[, size] = np.identity(3)
    x =  U + np.linalg.solve(L.T, z) 
    
    return x  # x = U + z*(L.T)^{-1}



# Let'S WRITE HERE THE KULLBAKC_LEIBLER DIVERGENCE

#https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.invwishart.html
from scipy.stats import invwishart

#to sample an inverse wishart distribution

def sample_Sigma(mu_bar,psi_bar):
    
    psi_bar = near_psd(psi_bar, epsilon=0.1)
    
    return  invwishart.rvs(df=mu_bar, scale=psi_bar, size=1, random_state=None)




#markov random fields kbld

#KBLD_MRFG0_MRFG1 = 0.5*( tf.log(tf.matrix_determinant(J_bar)/tf.matrix_determinant(J)) - dim + tf.trace(tf.matmul(J,tf.matrix_inverse(J_bar))) )


def KBLD_MRFG0_MRFG1(J,J_bar) :
        
    dim = J.shape[0]
    
    return 0.5*( tf.log(tf.matrix_determinant(J_bar)/tf.matrix_determinant(J)) - dim + tf.trace(tf.matmul(J,tf.matrix_inverse(J_bar))) )
   
    


def KBLD_IW0_IW1 (mu,psi,mu_bar,psi_bar):
    
    #d = tf.shape(psi_test)[0]
    #d = tf.constant(d,dtype=tf.float64)
    #d = tf.cast(d, tf.float64)
    
    #mu1 = tf.constant(mu - d - 1, dtype = tf.float64)
    #mu_bar1 = tf.constant(mu_bar - d - 1, dtype = tf.float64)
    
    di = int(d.eval(session=sess))
    part1 = ((mu_bar - d - 1 )/2)*tf.log(tf.matrix_determinant(psi_bar)) + ((mu - d - 1 )/2)*tf.log(tf.matrix_determinant(psi)) 
    part2 = tf.reduce_sum([tf.lgamma((mu - d - j )/2)-tf.lgamma((mu_bar - d - j )/2) for j in range(1,di)])
    part3 = ((mu-mu_bar)/2)*( tf.log(tf.matrix_determinant(psi_bar)) + tf.reduce_sum(tf.convert_to_tensor([tf.digamma((mu_bar - d - j )/2) for j in range(1,di)])) )
    part4 = tf.trace(  ((mu_bar)/2)*tf.matmul( tf.matrix_inverse(psi_bar),(psi_bar-psi)) )
    
    return part1 + part2 + part3 + part4


##writing the Three data model, use this "h = tf.nn.relu(tf.matmul(x, W) + b)" for more in-depth changes

def NN1(x, W_0, W_1, b_0, b_1):  
    h1 = tf.tanh(tf.matmul(x, W_0) + b_0)
    h2 = tf.matmul(h1, W_1) + b_1
    return h2    # tf.reshape(h2, [-1])


def NN2(x, W_0, W_1, b_0, b_1):  
    h1 = tf.nn.relu(tf.matmul(x, W_0) + b_0)
    h2 = tf.matmul(h1, W_1) + b_1
    return h2    # tf.reshape(h2, [-1])


def NN3(x, W_0, W_1, W_2, W_3, W_4, W_5, b_0, b_1, b_2, b_3, b_4, b_5):  
    h1 = tf.tanh(tf.matmul(x, W_0) + b_0)
    h2 = tf.matmul(h1, W_1) + b_1
    h3 = tf.matmul(h2, W_2) + b_2
    h4 = tf.matmul(h3, W_3) + b_3
    h5 = tf.matmul(h4, W_4) + b_4 
    h6 = tf.matmul(h5, W_5) + b_5
    return h6


#TO BE  TESTED

def lop_y_proba_NN2(x,y,W,b,Sigma):
    
    #remember : SIZE_gMRF_W = l_0*l_1 + l_1*l_2, look in distribution setting
    
    #use this because in case of batch, restric_size is not l0
    #normalement on prend, rsx = restric_size = x.shape[0], mais avec le batch, on a :

    l_0 = x.shape[1]
    l_1 = 3
    l_2 = y.shape[1]
    lb_0 = l_1
    lb_1 = l_2 
    
    rsx = x.shape[0]
    
    W_0 = W[0:l_0*l_1]
    W_0  = tf.reshape(W_0, [l_0,l_1])
    
    W_1 = W[l_0*l_1:(l_0*l_1 + l_1*l_2 )]
    W_1  = tf.reshape(W_1, [l_1,l_2])
    
    #remember: SIZE_gMRF_b = lb_0 + lb_1
    
    b_0 = b[0:lb_0]
    b_0 = tf.reshape(b_0,[-1])
    
    b_1 = b[lb_0: (lb_0+lb_1)]
    b_1 = tf.reshape(b_1,[-1])
    
    print("KIKI")
    
    def corpus (i): #définie pour un seul y_i = y[i,]
        
        yi = tf.reshape(y[i,], [1,y.shape[1]])
        xi = tf.reshape(x[i,], [1,x.shape[1]])
        
        C = (-l_2/2)*tf.log(2* tf.constant(math.pi,dtype=tf.float64)) - (l_2/2)*tf.log(tf.matrix_determinant(Sigma))
        
        #return C - (1/2)*( tf.matmul((y[i,]- NN2(x[i,], W_0, W_1, b_0, b_1)).T , tf.matmul(tf.matrix_inverse(Sigma),(y[i,]- NN2(x[i,], W_0, W_1, b_0, b_1))) ) )
        return C - (1/2)*( tf.matmul((yi- NN2(xi, W_0, W_1, b_0, b_1)) , tf.matmul(tf.matrix_inverse(Sigma),tf.transpose(yi- NN2(xi, W_0, W_1, b_0, b_1))) ) )

    
    return  tf.add_n([corpus(i) for i in range(rsx)]) 
    
    

# DEFINE ALL NECESSARY DISTRIBUTIONS

#iN spnnr, W IS G-MRF

def Distribution_w (W,mean_W,Q):
    
    SIZE_gMRF_W = l_0*l_1 + l_1*l_2
    SIZE_gMRF_W = tf.cast(SIZE_gMRF_W, tf.float64)
    #Here mean_W == 0
    
    W  = tf.reshape(W, [1,W.size])
    
    part1 = (-SIZE_gMRF_W/2)*tf.log(2*tf.constant(math.pi,dtype=tf.float64)) + (SIZE_gMRF_W/2)*tf.log(tf.matrix_determinant(Q)) 
    part2 = - (1/2)*( tf.matmul(W , tf.matmul(Q,tf.transpose(W) ) ) )
                     
    return tf.exp(part1+part2) #prendre l'exponentielle car la vrai distribution on souhaite utiliser
    

#lb_0 = b_0.shape[0],lb_1 = b_0.shape[1], 
#lb_1 = W_1.shape[0], lb_2 = b_1.shape[1]
#b = [VEC(b_0), VEC(b_1)]


#iN spnnr, b IS G-MRF
             
def Distribution_b (b,mean_b,xi):
             
    SIZE_gMRF_b = lb_0 + lb_1 

    SIZE_gMRF_b = tf.cast(SIZE_gMRF_b, tf.float64)
    #Here mean_W == 0
    
    b  = tf.reshape(b, [1,b.size])
    #here mean_b ==0
    
    part1 = (-SIZE_gMRF_b/2)*tf.log(2*tf.constant(math.pi,dtype=tf.float64)) + (SIZE_gMRF_b/2)*tf.log(tf.matrix_determinant(xi))
    part2 = - (1/2)*( tf.matmul(b , tf.matmul(xi, tf.transpose(b) ) ) )             
    return  tf.exp(part1+part2) ##prendre l'exponentielle car la vrai distribution on souhaite utiliser
             

#in spnnr, Sigma is inverse Wishart, 
#need multivariate gamma function to work
             
def log_mv_gamma(p, a):
    C = p * (p - 1) / 4 * tf.log(tf.constant(math.pi, dtype=tf.float64))
    return C + tf.reduce_sum( tf.lgamma(a - 0.5 * tf.range(1,p+1)) ) 

#mu_bar and psi_bar             
             
#depends on psi_bar and mu_bar             
def Distribution_Sigma (Sigma,mu_bar,psi_bar):
       
    #p_sig = dimension_sigma = Sigma.shape[0]
    p_sig = d  
        
    part1 = ((tf.matrix_determinant(psi_bar)**(mu_bar/2))/ ((2**((mu_bar*p_sig)/2))*tf.exp(log_mv_gamma (p_sig,(mu_bar/2)))))
    part2 = (tf.matrix_determinant(Sigma)**((mu_bar +p_sig-1)/2))*tf.exp(-0.5*tf.trace(tf.matmul(psi_bar,tf.matrix_inverse(Sigma))))                 
             
    return part1*part2



#######################################################################################

#Use shuffling for stochastic gradient: 
#https://stackoverflow.com/questions/4601373/better-way-to-shuffle-two-numpy-arrays-in-unison
def shuffle_in_unison(a, b):
    assert len(a) == len(b)
    shuffled_a = np.empty(a.shape, dtype=a.dtype)
    shuffled_b = np.empty(b.shape, dtype=b.dtype)
    permutation = np.random.permutation(len(a))
    for old_index, new_index in enumerate(permutation):
        shuffled_a[new_index] = a[old_index]
        shuffled_b[new_index] = b[old_index]
    return shuffled_a, shuffled_b    


def Training_Parameters_Estimation(my_qw_evalb,my_xi_evalb, my_psi_bar_evalb, my_mu_bar_evalb, x,y):

    
    l_0 = x.shape[1]
    l_1 = 3
    l_2 = y.shape[1]

    lb_0 = l_1 #(must be equal to l_1)
    lb_1 = l_2 #(must be equal to l_2)


    # INTIALISATIONS

    #init = tf.initialize_all_variables()
    sess = tf.Session()
    #sess.run(init)

    #number of samples per parameters
    exp_time1 = 2

    mean_b = U_b = tf.constant(np.zeros((lb_0 + lb_1)),dtype=tf.float64)
    mean_W = U_w = tf.constant(np.zeros((l_0*l_1 + l_1*l_2)),dtype=tf.float64)


    weights_list = []

    bias_list = []

    Sigma_list = []


    for s in range(exp_time1) : #prendre la moyenne sur le nombre de fois pour le expected

        #Work with :

        ##Generate parameters

        W = sample_GMRF(mean_W.eval(session=sess),my_qw_evalb) # set mean_W here == zeros matrix, the mean is null,sample_GMRF(U_w.eval(session=sess),Q_w.eval(session=sess))

        weights_list.append(W)

        b = sample_GMRF(mean_b.eval(session=sess),my_xi_evalb) # set mean_b here == zeros matrix, the mean is null

        bias_list.append(b)

        Sigma = sample_Sigma(my_mu_bar_evalb,my_psi_bar_evalb) #mu_bar must be superior than IW dimension

        Sigma_list.append(Sigma) 

    return np.mean(weights_list, axis = 0), np.mean(bias_list, axis = 0), np.mean(Sigma_list, axis = 0)    

    
