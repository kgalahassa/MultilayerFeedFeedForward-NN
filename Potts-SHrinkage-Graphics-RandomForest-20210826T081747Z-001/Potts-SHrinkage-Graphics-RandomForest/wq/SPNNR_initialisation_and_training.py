


#Import tensorflow and numpy


# -*- coding: utf-8 -*-
"""
Created on Fri Apr 26 01:23:00 2019

@author: Trader
"""


#Import tensorflow and numpy
import tensorflow as tf
import numpy as np
import math
#https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.invwishart.html
from scipy.stats import invwishart
from numpy.linalg import LinAlgError
import sklearn.datasets as SD
class SPNNR(object):
    
    def __init__(self,xtrain, ytrain, l_0,l_1,l_2,lb_0,lb_1):
        #self.l1 = name
        #self.age = age
        
        """ hidden layer weights matrix is size l_0 *l_1, 
        output layer weights matrix is size l_1*l_2,
        
        first biais/offset vector is size lb_0 (must be equal to l_1),
        second biais/offset vector is size lb_1 (must be equal to l_2)"""
        
        self.l_0 = l_0 #=3
        self.l_1 = l_1 #= 2
        self.l_2 = l_2#= 2
        self.lb_0 = lb_0 #=2
        self.lb_1 = lb_1 #=2.
        
        
        #defines data
        self.x = xtrain
        self.y = ytrain
        
        
        
        #dimension of Sigma, InvWishart (mu, psi), or dimension of psi
        self.d = tf.constant(ytrain.shape[1], dtype=tf.float64)
    
        #batch_size = 10
        self.batch_size = 0.2*xtrain.shape[0]
        
    
    
    #https://stackoverflow.com/questions/10939213/how-can-i-calculate-the-nearest-positive-semi-definite-matrix?noredirect=1&lq=1
    #How can I calculate the nearest positive semi-definite matrix?
    #positive semi-definite matrices and numerical stability?: 
    #https://stackoverflow.com/questions/2115880/positive-semi-definite-matrices-and-numerical-stability?noredirect=1&lq=1
    #https://stackoverflow.com/questions/619335/a-simple-algorithm-for-generating-positive-semidefinite-matrices
    
    global near_psd
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
    
    global  sample_GMRF
    def sample_GMRF(U,Q): 
        
        """ This function helps to sample Gaussian Markov Random Fields  values """
        
        """ Inputs are : mean (U) and Precision matrix (Q) """
        Q = near_psd(Q, epsilon=0.1)
        x_dim = Q.shape[1]        #la dimension de x
        
        """ Q must be semi-positive definite , or you'll get errors """
        
        
        L = np.linalg.cholesky(Q) #Cholesky de Q pour avoir L
        
        #il faut générer z, une normal standard
        z = np.random.multivariate_normal(np.zeros(x_dim), np.identity(x_dim)) #mean = np.zeros(5); cov[, size] = np.identity(3)
        x= U + np.linalg.solve(L.T, z)
        
        return x  # x = U + z*(L.T)^{-1}
    
    
    
    # Let'S WRITE HERE THE KULLBAKC_LEIBLER DIVERGENCE
    
  
    #to sample an inverse wishart distribution
    
    global sample_Sigma
    def sample_Sigma(mu_bar,psi_bar):
        
        psi_bar = near_psd(psi_bar, epsilon=0.1)
        

        my_sigma = invwishart.rvs(df=mu_bar, scale=psi_bar, size=1, random_state=None)
        #print("Third to ++++")
        return  my_sigma  #invwishart.rvs(df=mu_bar, scale=psi_bar, size=1, random_state=None)
    
    
    
    
    #markov random fields kbld
    
    #KBLD_MRFG0_MRFG1 = 0.5*( tf.log(tf.matrix_determinant(J_bar)/tf.matrix_determinant(J)) - dim + tf.trace(tf.matmul(J,tf.matrix_inverse(J_bar))) )
    
    global KBLD_MRFG0_MRFG1
    def KBLD_MRFG0_MRFG1(J,J_bar) :
            
        dim = J.shape[0]
        
        return 0.5*( tf.log(tf.matrix_determinant(J_bar)/tf.matrix_determinant(J)) - dim + tf.trace(tf.matmul(J,tf.matrix_inverse(J_bar))) )
       
        
    
    #global KBLD_IW0_IW1
    def KBLD_IW0_IW1 (self,mu,psi,mu_bar,psi_bar): #**********to be used with self
        
        #d = tf.shape(psi_test)[0]
        #d = tf.constant(d,dtype=tf.float64)
        #d = tf.cast(d, tf.float64)
        
        #mu1 = tf.constant(mu - d - 1, dtype = tf.float64)
        #mu_bar1 = tf.constant(mu_bar - d - 1, dtype = tf.float64)
        
        
              
        sess2 = tf.Session()
        d= self.d
        di = int(d.eval(session=sess2))
        
        part1 = ((mu_bar - d - 1 )/2)*tf.log(tf.matrix_determinant(psi_bar)) + ((mu - d - 1 )/2)*tf.log(tf.matrix_determinant(psi)) 
        part2 = tf.reduce_sum([tf.lgamma((mu - d - j )/2)-tf.lgamma((mu_bar - d - j )/2) for j in range(1,di)])
        part3 = ((mu-mu_bar)/2)*( tf.log(tf.matrix_determinant(psi_bar)) + tf.reduce_sum(tf.convert_to_tensor([tf.digamma((mu_bar - d - j )/2) for j in range(1,di)])) )
        part4 = tf.trace(  ((mu_bar)/2)*tf.matmul( tf.matrix_inverse(psi_bar),(psi_bar-psi)) )
        
        #sess2.close()
        
        return part1 + part2 + part3 + part4
    
    
    ##writing the Three data model, use this "h = tf.nn.relu(tf.matmul(x, W) + b)" for more in-depth changes
    global NN1
    def NN1(x, W_0, W_1, b_0, b_1):  
        h1 = tf.tanh(tf.matmul(x, W_0) + b_0)
        h2 = tf.matmul(h1, W_1) + b_1
        return h2    # tf.reshape(h2, [-1])
    
    global NN2
    def NN2(x, W_0, W_1, b_0, b_1):  
        h1 = tf.tanh(tf.matmul(x, W_0) + b_0) #tf.nn.relu
        h2 = tf.matmul(h1, W_1) + b_1
        return h2    # tf.reshape(h2, [-1])
    
    global NN3
    def NN3(x, W_0, W_1, W_2, W_3, W_4, W_5, b_0, b_1, b_2, b_3, b_4, b_5):  
        h1 = tf.tanh(tf.matmul(x, W_0) + b_0)
        h2 = tf.matmul(h1, W_1) + b_1
        h3 = tf.matmul(h2, W_2) + b_2
        h4 = tf.matmul(h3, W_3) + b_3
        h5 = tf.matmul(h4, W_4) + b_4 
        h6 = tf.matmul(h5, W_5) + b_5
        return h6
    
    
    #TO BE  TESTED
    #global lop_y_proba_NN2
    def lop_y_proba_NN2(self,x,y,W,b,Sigma):#********
        
        
        l_0 = self.l_0
        l_1 = self.l_1
        l_2 = self.l_2
        lb_0 = self.lb_0
        lb_1 = self.lb_1 
        batch_size = self.batch_size
        
        x= self.x
        y= self.y
        
        #remember : SIZE_gMRF_W = l_0*l_1 + l_1*l_2, look in distribution setting
        
        #use this because in case of batch, restric_size is not l0
        #normalement on prend, rsx = restric_size = x.shape[0], mais avec le batch, on a :
        rsx = int(batch_size)
        
        W_0 = W[0:l_0*l_1]
        W_0  = tf.reshape(W_0, [l_0,l_1])
        
        W_1 = W[l_0*l_1:(l_0*l_1 + l_1*l_2 )]
        W_1  = tf.reshape(W_1, [l_1,l_2])
        
        #remember: SIZE_gMRF_b = lb_0 + lb_1
        
        b_0 = b[0:lb_0]
        b_0 = tf.reshape(b_0,[-1])
        
        b_1 = b[lb_0: (lb_0+lb_1)]
        b_1 = tf.reshape(b_1,[-1])
        
        
        
        def corpus (i): #définie pour un seul y_i = y[i,]
            
            yi = tf.reshape(y[i,], [1,y.shape[1]])
            xi = tf.reshape(x[i,], [1,x.shape[1]])
            
            C = (-l_2/2)*tf.log(2* tf.constant(math.pi,dtype=tf.float64)) - (l_2/2)*tf.log(tf.matrix_determinant(Sigma))
            
            #return C - (1/2)*( tf.matmul((y[i,]- NN2(x[i,], W_0, W_1, b_0, b_1)).T , tf.matmul(tf.matrix_inverse(Sigma),(y[i,]- NN2(x[i,], W_0, W_1, b_0, b_1))) ) )
            return C - (1/2)*( tf.matmul((yi- NN2(xi, W_0, W_1, b_0, b_1)) , tf.matmul(tf.matrix_inverse(Sigma),tf.transpose(yi- NN2(xi, W_0, W_1, b_0, b_1))) ) )
    
        
        return  tf.add_n([corpus(i) for i in range(rsx)]) 
        
        
    
    # DEFINE ALL NECESSARY DISTRIBUTIONS
    
    #iN spnnr, W IS G-MRF
    
    def Distribution_w (self,W,mean_W,Q):#*******
        
        l_0 = self.l_0
        l_1 = self.l_1
        l_2 = self.l_2
        
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
                 
    def Distribution_b (self,b,mean_b,xi):
        
        lb_0 = self.lb_0
        lb_1 = self.lb_1

         
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
    global log_mv_gamma         
    def log_mv_gamma(p, a):
        C = p * (p - 1) / 4 * tf.log(tf.constant(math.pi, dtype=tf.float64))
        return C + tf.reduce_sum( tf.lgamma(a - 0.5 * tf.range(1,p+1)) ) 
    
    #mu_bar and psi_bar             
                 
    #depends on psi_bar and mu_bar             
    def Distribution_Sigma (self,Sigma,mu_bar,psi_bar):
           
        #p_sig = dimension_sigma = Sigma.shape[0]
        p_sig = self.d  
            
        part1 = ((tf.matrix_determinant(psi_bar)**(mu_bar/2))/ ((2**((mu_bar*p_sig)/2))*tf.exp(log_mv_gamma (p_sig,(mu_bar/2)))))
        part2 = (tf.matrix_determinant(Sigma)**((mu_bar +p_sig-1)/2))*tf.exp(-0.5*tf.trace(tf.matmul(psi_bar,tf.matrix_inverse(Sigma))))                 
                 
        return part1*part2
    
    
    
    #######################################################################################
    
    #Use shuffling for stochastic gradient: 
    #https://stackoverflow.com/questions/4601373/better-way-to-shuffle-two-numpy-arrays-in-unison
    global shuffle_in_unison
    def shuffle_in_unison(a, b):
        assert len(a) == len(b)
        shuffled_a = np.empty(a.shape, dtype=a.dtype)
        shuffled_b = np.empty(b.shape, dtype=b.dtype)
        permutation = np.random.permutation(len(a))
        for old_index, new_index in enumerate(permutation):
            shuffled_a[new_index] = a[old_index]
            shuffled_b[new_index] = b[old_index]
        return shuffled_a, shuffled_b    
    
    
    
    #######################################################################################INTIALISATION
    
    def initialisation_training(self,x,y):
    
        
        l_0 = self.l_0
        l_1 = self.l_1
        l_2 = self.l_2
        lb_0 = self.lb_0
        lb_1 = self.lb_1 
        batch_size = self.batch_size
        
        x= self.x
        y= self.y
        
        
        
        # INTIALISATIONS
        #tf.reset_default_graph()    
        init = tf.initialize_all_variables()
        sess = tf.Session()
        sess.run(init)
    
        #J = tf.constant(J, dtype=tf.float32)
        #J_bar = tf.constant(J_bar, dtype=tf.float32)
        #dim = tf.constant(dim, dtype=tf.float32)
    
        #l_0 = W_0.shape[0],l_1 = W_0.shape[1], 
        #l_1 = W_1.shape[0], l_2 = W_1.shape[1]
        #W = [VEC(w_0), VEC(W_1)]
        #b = [VEC(b_0), VEC(b_1)]
    
        #q= self.x.shape[0]
    
        #create a class to integrate this
        #l_0 = 3
        #l_1 = 2
        #l_2 = 2
        
        #lb_0 = 2 #(must be equal to l_1)
        #lb_1 = 2 #(must be equal to l_2)
    
        #dimension of Sigma, InvWishart (mu, psi), or dimension of psi
        #d = tf.constant(y.shape[1], dtype=tf.float64)
    
        #batch_size = 10
        #batch_size = 0.2*xtrain.shape[0]
    
        #l_0*l_1 + l_1*l_2 = ...3*2 + 2*2
    
    
        #####################################################
        #####################################################
    
        ######################################### CONDITION INITIALE POUR QW
        
        
        def send_Q (l_0,l_1,l_2,lb_0,lb_1):
            
            """ Return une valeur pour Q_w et xi """
        
            #A = np.random.randint(100, size=(l_0*l_1, l_0*l_1))
            #B = np.random.randint(100, size=(l_1*l_2, l_1*l_2))
        
        
            #https://docs.scipy.org/doc/numpy-1.14.1/reference/generated/numpy.block.html
            #Q must be a "random" positive semi-definite matix
            #How can I calculate the nearest positive semi-definite matrix?
            #https://stackoverflow.com/questions/10939213/how-can-i-calculate-the-nearest-positive-semi-definite-matrix?noredirect=1&lq=1
        
        
            #Q_w = np.block([ [np.matmul(A,A.T),               np.ones((l_0*l_1, l_1*l_2))],   [np.ones((l_1*l_2, l_0*l_1)), np.matmul(B,B.T)               ]
            #])
        
            #Q_w = tf.constant(np.random.randint(100, size=(10, 10)), dtype=tf.float64)
        

            Q_w = SD.make_spd_matrix(l_0*l_1 + l_1*l_2, 42)

            Q_w = tf.constant(Q_w,dtype=tf.float64)
    
            #C = np.random.randint(5, size=(y.shape[1], y.shape[1]))

            psi_bar = SD.make_spd_matrix(l_2, 123)

            psi_bar = tf.constant(psi_bar,dtype=tf.float64)
        
            ##Initialiser pour b
            #Ab = np.random.randint(7, size=(lb_0, lb_0))
            #Bb = np.random.randint(7, size=(lb_1, lb_1))
            #xi = np.block([ [np.matmul(Ab,Ab.T),               np.ones((lb_0, lb_1))],   [np.ones((lb_1, lb_0)), np.matmul(Bb,Bb.T)               ]
            #])
    
            xi = SD.make_spd_matrix(lb_0+lb_1, 123)
            xi = tf.constant(xi,dtype=tf.float64)
    
            return Q_w,xi, psi_bar
        
        def validation(l_0,l_1,l_2,lb_0,lb_1):

                       default_True = 0
                       while default_True == 0:
                           
                             set_Q_w,set_xi, set_psi_bar = send_Q (l_0,l_1,l_2,lb_0,lb_1)
                             
                             try:
                                 near_psd(set_Q_w.eval(session=sess), epsilon=0.1)
                                 near_psd(set_xi.eval(session=sess), epsilon=0.1)
                                 near_psd(set_psi_bar.eval(session=sess), epsilon=0.1)

                                 print("Everything is GOOD!")
                                 default_True == 1
                                 return set_Q_w,set_xi, set_psi_bar

                                 

                             except: 

                                 raise LinAlgError 
                                 default_True == 0   
                    
            
        
        
        Q_w,xi, psi_bar = validation(l_0,l_1,l_2,lb_0,lb_1)
    
        print("I have set up Q_w,xi, psi_bar FOR REAL!")

        mean_W = U_w = tf.constant(np.zeros((l_0*l_1 + l_1*l_2)),dtype=tf.float64)
        
        
        #####################################################
        #####################################################
    
        ######################################### CONDITION INITIALE POUR Q0
        Ao = np.random.randint(8, size=(l_0*l_1, l_0*l_1))
        Bo = np.random.randint(8, size=(l_1*l_2, l_1*l_2))
    
    
        #https://docs.scipy.org/doc/numpy-1.14.1/reference/generated/numpy.block.html
        #Q must be a "random" positive semi-definite matix
        #How can I calculate the nearest positive semi-definite matrix?
        #https://stackoverflow.com/questions/10939213/how-can-i-calculate-the-nearest-positive-semi-definite-matrix?noredirect=1&lq=1
    
    
        Q0 = np.block([ [np.matmul(Ao,Ao.T),               np.ones((l_0*l_1, l_1*l_2))],   [np.ones((l_1*l_2, l_0*l_1)), np.matmul(Bo,Bo.T)               ]
        ])
    
        Q0 =  tf.constant(Q0,dtype=tf.float64)
    
        #################################CONDITION INITIALE POUR xi0
    
        ##Initialiser pour b
        Abo = np.random.randint(10, size=(lb_0, lb_0))
        Bbo = np.random.randint(10, size=(lb_1, lb_1))
        xi0 = np.block([ [np.matmul(Abo,Abo.T),               np.ones((lb_0, lb_1))],   [np.ones((lb_1, lb_0)), np.matmul(Bbo,Bbo.T)               ]
        ])
        xi0 = tf.constant(xi0,dtype=tf.float64)
    
        #################################CONDITION INITIALE POUR psi0
    
        Co = np.random.randint(15, size=(y.shape[1], y.shape[1]))
        psi0 = tf.constant(np.matmul(Co,Co.T),dtype=tf.float64)
    
    
        #################################################
        #################################################
        ##Initialiser les paramètres pour Sigma
        mu0 = mu_bar = tf.constant(y.shape[1]+20,dtype=tf.float64) #y.shape[1] + some_constante

    
        #xi0 = xi
    
        mean_b = U_b = tf.constant(np.zeros((lb_0 + lb_1)),dtype=tf.float64)
    
    
    
    
        #######################################################################################TRAINING
    
    
        #The method to compute the ELBO follows
    
        #weights_vars = [U_0, V_0, U_1, V_1, U_01, V_01] # Can be replaced directly with Q_w
        #bias_vars = [u_0, v_0, u_1, v_1, u_01, v_01]  # Can be replaced directly with Q_b
        #covariance_Sigma_vars = [mu_bar,psi_bar]   #We conservate this 
    
        #Exemple for U_0
        
        n_epochs = 1
    
        for j in range (n_epochs):
        ##select data
            
            batch = np.random.choice(x.shape[0], int(batch_size)) #batch_size = 10
            
            xn,yn = shuffle_in_unison(x,y)
            
            xn,yn = xn [batch,],yn  [batch,]
    
            """ #COMPUTE EXPECTED LOG-LIKELIHOOD for each epoch """
    
            out_expected_Q_w = [] #out_expected_U_0 = []
            out_expected_xi = []
            out_expected_mu_bar = []
            out_expected_psi_bar = []
    
            mean_time = 5

            print("Previous to all: <<<<<<<<<<<<<<<<<<<----------    [[I start now the learning process]] -------------->>>>>>>>>>>: %i"%j)
            for i in range(mean_time): #prendre la moyenne sur le nombre de fois pour le expected
               print("For god sake")
               #Work with :

               my_qw_eval = near_psd(Q_w.eval(session=sess), epsilon = 0.1) 
               print("KALI")
               my_xi_eval = near_psd(xi.eval(session=sess), epsilon = 0.1)
               my_psi_bar_eval = near_psd(psi_bar.eval(session=sess), epsilon = 0)
               my_mu_bar_eval = mu_bar.eval(session=sess)

               print("Second to all: <<<<<<<<<<<<<<<<<<<----------: %i"%j)

            ##Generate parameters

               W = sample_GMRF(mean_W.eval(session=sess),my_qw_eval) # set mean_W here == zeros matrix, the mean is null,sample_GMRF(U_w.eval(session=sess),Q_w.eval(session=sess))
               

               b = sample_GMRF(mean_b.eval(session=sess),my_xi_eval) # set mean_b here == zeros matrix, the mean is null
               
               Sigma = sample_Sigma(int(mu_bar.eval(session=sess)),my_psi_bar_eval) #mu_bar must be superior than IW dimension
               
               ###COMPUTE ALL NECESSARY GRADIENTS


               print("Third to all: <<<<<<<<<<<<<<<<<<<---------: %i"%j)
               #1- Q_w


               Dist_grad_Q = tf.gradients(self.Distribution_w(W,mean_W,Q_w), [Q_w])

               Dist_grad_Q_val = sess.run(Dist_grad_Q,feed_dict={Q_w : my_qw_eval }) 

               print(Dist_grad_Q_val)
              
               my_Q_factor = (self.lop_y_proba_NN2(xn,yn,W,b,Sigma)*self.Distribution_b (b,mean_b,my_xi_eval)*self.Distribution_Sigma(Sigma,my_mu_bar_eval,my_psi_bar_eval)).eval(session=sess)        

               out_expected_Q_w.append(my_Q_factor*Dist_grad_Q_val[0])

               #2- xi

               Dist_grad_xi = tf.gradients(self.Distribution_b (b,mean_b,xi), [xi])

               Dist_grad_xi_val = sess.run(Dist_grad_xi,feed_dict={xi : my_xi_eval }) 

               

               my_xi_factor = (self.lop_y_proba_NN2(xn,yn,W,b,Sigma)*self.Distribution_w(W,mean_W,my_qw_eval)*self.Distribution_Sigma(Sigma,my_mu_bar_eval,my_psi_bar_eval)).eval(session=sess)        

               out_expected_xi.append(my_xi_factor*Dist_grad_xi_val[0])

               #3- psi_bar


               Dist_grad_psi_bar = tf.gradients(self.Distribution_Sigma(Sigma,my_mu_bar_eval,psi_bar), [psi_bar])

               Dist_grad_psi_bar_val = sess.run(Dist_grad_psi_bar,feed_dict={psi_bar : my_psi_bar_eval }) 


               my_psi_bar_factor = (self.lop_y_proba_NN2(xn,yn,W,b,Sigma)*self.Distribution_w(W,mean_W,my_qw_eval)*self.Distribution_b (b,mean_b,my_xi_eval)).eval(session=sess)        

               out_expected_psi_bar.append(my_psi_bar_factor*Dist_grad_psi_bar_val[0])

               #4-mu_bar

               Dist_grad_mu_bar = tf.gradients(self.Distribution_Sigma(Sigma,mu_bar,my_psi_bar_eval), [mu_bar])

               Dist_grad_mu_bar_val = sess.run(Dist_grad_mu_bar,feed_dict={mu_bar : my_mu_bar_eval }) 


               my_mu_bar_factor = (self.lop_y_proba_NN2(xn,yn,W,b,Sigma)*self.Distribution_w(W,mean_W,my_qw_eval)*self.Distribution_b (b,mean_b,my_xi_eval)).eval(session=sess)        

               out_expected_mu_bar.append(my_mu_bar_factor*Dist_grad_mu_bar_val[0])


        #out_expected_Q_w.append(lop_y_proba_NN2(x,y,W,b,Sigma)*tf.gradients(Distribution_w(W,mean_W,Q_w), Q_w)*Distribution_b (b,mean_b,xi)*Distribution_Sigma(Sigma,mu_bar,psi_bar_test))    
        #out_expected_xi.append(lop_y_proba_NN2(xn,yn,W,b,Sigma)*Distribution_w(W,mean_W,Q_w)*tf.gradients(Distribution_b (b,mean_b,xi),xi)*Distribution_Sigma(Sigma,mu_bar,psi_bar))    
        #out_expected_mu_bar.append(lop_y_proba_NN2(xn,yn,W,b,Sigma)*Distribution_w(W,mean_W,Q_w)*Distribution_b (b,mean_b,xi)*tf.gradients(Distribution_Sigma(Sigma,mu_bar,psi_bar), mu_bar))    
        #out_expected_psi_bar.append(lop_y_proba_NN2(xn,yn,W,b,Sigma)*Distribution_w(W,mean_W,Q_w)*Distribution_b (b,mean_b,xi)*tf.gradients(Distribution_Sigma(Sigma,mu_bar,psi_bar), psi_bar))    

            print("<<<<<<<<<<<<<<<<<<<----------    [[I start now the learning process]]%i -------------->>>>>>>>>>>"%j)
            #Then you compute the expected-loglikelihood for U_0    
            out_expected_Q_w = tf.add_n(out_expected_Q_w) #-----CORRIGER car ensemble de tensors
            out_expected_xi = tf.add_n(out_expected_xi) 
            out_expected_mu_bar = tf.add_n(out_expected_mu_bar)
            out_expected_psi_bar = tf.add_n(out_expected_psi_bar)
    
            print(""" #And finally the ELBO_gradient_in_U_0 """)
    
            ###Work with
            my_q0_eval = near_psd(Q0.eval(session=sess), epsilon = 0.1)
            my_xi0_eval = near_psd(xi0.eval(session=sess), epsilon = 0.1)
            my_psi0_eval = near_psd(psi0.eval(session=sess), epsilon = 0.1)
            my_mu0_eval = mu0.eval(session=sess)
            
            ### COMPUTE KBLD GRADIENTS
            
            #1- Q_w
            KBLD_grad_Q = tf.gradients(KBLD_MRFG0_MRFG1(my_q0_eval,Q_w),Q_w)
    
            KBLD_grad_Q_val = sess.run(KBLD_grad_Q,feed_dict={Q_w : my_qw_eval })
            ELBO_grad_Q_w = tf.scalar_mul((1/mean_time),out_expected_Q_w) - KBLD_grad_Q_val[0] #*
            
            #2- xi
            
            KBLD_grad_xi = tf.gradients(KBLD_MRFG0_MRFG1(my_xi0_eval,xi),xi)
    
            KBLD_grad_xi_val = sess.run(KBLD_grad_xi,feed_dict={xi : my_xi_eval })
            ELBO_grad_xi = tf.scalar_mul((1/mean_time),out_expected_xi) - KBLD_grad_xi_val[0]
            
            #3- psi_bar
            
            KBLD_grad_psi_bar = tf.gradients(self.KBLD_IW0_IW1 (my_mu0_eval,my_psi0_eval,my_mu_bar_eval,psi_bar),psi_bar)
    
            KBLD_grad_psi_bar_val = sess.run(KBLD_grad_psi_bar,feed_dict={psi_bar : my_psi_bar_eval })
            ELBO_grad_psi_bar = tf.scalar_mul((1/mean_time),out_expected_psi_bar) - KBLD_grad_psi_bar_val[0]
            
            #4- mu_bar
            
            KBLD_grad_mu_bar = tf.gradients(self.KBLD_IW0_IW1 (my_mu0_eval,my_psi0_eval,mu_bar,my_psi_bar_eval),mu_bar)
    
            KBLD_grad_mu_bar_val = sess.run(KBLD_grad_mu_bar,feed_dict={mu_bar : my_mu_bar_eval })
            #print("EVALUATE SESSION, DONE!",KBLD_grad_mu_bar_val[0])#.eval(session=sess)
            ELBO_grad_mu_bar = tf.scalar_mul((1/mean_time),out_expected_mu_bar) - KBLD_grad_mu_bar_val[0]
            
            
             
            #fix the learning rate
            
            qrate = 10e-3
            
            #find new values 
            
            #1-
            NQ_w = my_qw_eval + tf.scalar_mul(qrate,ELBO_grad_Q_w)
            Q_w = tf.cast(NQ_w,dtype=tf.float64)
         
            #bernouilli_Q_w = np.random.binomial(size=(l_0*l_1 + l_1*l_2,l_0*l_1 + l_1*l_2), n=1, p= 1)
            #bernouilli_Q_w = tf.cast(bernouilli_Q_w, dtype=tf.float64)
            #Q_w = tf.multiply(Q_w, bernouilli_Q_w)
            
            
            #2-
            Nxi = my_xi_eval + tf.scalar_mul(qrate,ELBO_grad_xi)
            xi = tf.cast(Nxi,dtype=tf.float64)
            
            #bernouilli_xi = np.random.binomial(size=(lb_0+lb_1,lb_0+lb_1), n=1, p=1)
            #bernouilli_xi = tf.cast(bernouilli_xi, dtype=tf.float64)
            #xi = tf.multiply(xi, bernouilli_xi)
            
            
            #3-
            Nmu_bar = my_mu_bar_eval + tf.scalar_mul(qrate,ELBO_grad_mu_bar)
            
            mu_bar = tf.cast(Nmu_bar,dtype=tf.float64)
            
            #4-
            Npsi_bar = my_psi_bar_eval + tf.scalar_mul(qrate,ELBO_grad_psi_bar)
            psi_bar = tf.cast(Npsi_bar,dtype=tf.float64)
            
            #bernouilli_psi_bar = np.random.binomial(size=(l_2,l_2), n=1, p= 1)
            #bernouilli_psi_bar = tf.cast(bernouilli_psi_bar, dtype=tf.float64)
            #psi_bar = tf.multiply(psi_bar, bernouilli_psi_bar)
            print(""" ---*UPDATES USING THE ELBO*----""")
        
        #a,b,c,d = Q_w.eval(session=sess), xi.eval(session=sess), psi_bar.eval(session=sess), int(mu_bar.eval(session=sess))
        a = Q_w.eval(session=sess)
        
        b = xi.eval(session=sess)
        
        c = psi_bar.eval(session=sess)
        
        d = int(mu_bar.eval(session=sess))

        #print("EVALUATE SESSION, DONE!",d)
        #sess.close()
        print("EVALUATE SESSION, DONE!")
        print(a,b,c,d)
        return a,b,c,d
    
    
    #################################################################################Save in memory
    
    



