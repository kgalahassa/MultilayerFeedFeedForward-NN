import pandas as pd
import random
import numpy as np
from numpy import linalg as LA


#PottsData = pd.read_csv("DataForPottsClustering.csv", index_col=0, parse_dates=['DATE'])

from scipy.io import arff
data = arff.loadarff('scpf.arff')
df = pd.DataFrame(data[0])
df.dropna(inplace=True)
covariables = df.iloc[:,0:23].as_matrix()
response = df.iloc[:,23:26].as_matrix()
positions = np.arange(143)

from sklearn.cross_validation import train_test_split

covariables_train, covariables_test, response_train, response_test,positions_train,positions_test = train_test_split(covariables, response,positions, test_size=0.33, random_state=42)

#####################################################################

#Pour le training, build training data
xtrain,ytrain =  covariables_train,response_train

#Pour tester . build test data
xtest,ytest = covariables_test,response_test

Train_PottsData = xtrain
Test_PottsData = xtest

q = 10
#####################################################################
sigma = 1
T = 1

sigma = 1

Initial_Spin_Configuration = []

for i in range(len(Train_PottsData)):
    
    Initial_Spin_Configuration.append(random.randint(1,q))


from collections import defaultdict
# function for adding edge to graph 
graph = defaultdict(list) 


# Python program to print connected  
# components in an undirected graph
#https://www.geeksforgeeks.org/connected-components-in-an-undirected-graph/
class Graph: 
      
    # init function to declare class variables 
    def __init__(self,V): 
        self.V = V 
        self.adj = [[] for i in range(V)] 
  
    def DFSUtil(self, temp, v, visited): 
  
        # Mark the current vertex as visited 
        visited[v] = True
  
        # Store the vertex to list 
        temp.append(v) 
  
        # Repeat for all vertices adjacent 
        # to this vertex v 
        for i in self.adj[v]: 
            if visited[i] == False: 
                  
                # Update the list 
                temp = self.DFSUtil(temp, i, visited) 
        return temp 
  
    # method to add an undirected edge 
    def addEdge(self, v, w): 
        self.adj[v].append(w) 
        self.adj[w].append(v) 
  
    # Method to retrieve connected components 
    # in an undirected graph 
    def connectedComponents(self): 
        visited = [] 
        cc = [] 
        for i in range(self.V): 
            visited.append(False) 
        for v in range(self.V): 
            if visited[v] == False: 
                temp = [] 
                cc.append(self.DFSUtil(temp, v, visited)) 
        return cc
    
# This code is contributed by Abhishek Valsan    


My_Potts_Graph = Graph(len(Train_PottsData))



from collections import OrderedDict

def findneighbors(i, Train_PottsData, Initial_Spin_Configuration, k_voisins = 10):
    
    Compute_Norms  = {}
    
    for j in range(len(Train_PottsData)):
        
        if (i != j and Initial_Spin_Configuration[i] == Initial_Spin_Configuration[j] ):
            
            Compute_Norms[j] = LA.norm(Train_PottsData[i,:] - Train_PottsData[j,:])
                                       

    OrderedCompute_Norms = OrderedDict(sorted(Compute_Norms.items(), key=lambda x: x[1]))

    OCN_size  = len(OrderedCompute_Norms)
    
    SelectedOrderedCompute_Norms = list(OrderedCompute_Norms)[(OCN_size -k_voisins):OCN_size ]
                                       
    return SelectedOrderedCompute_Norms      


for i in range(len(Train_PottsData)):
    
    #let's get the top neighbors of observation i
    
    Selected_Neighbors = findneighbors(i, Train_PottsData, Initial_Spin_Configuration, k_voisins = 5)
    
    for j in Selected_Neighbors:
        
        #addEdge(graph,i,j)
        My_Potts_Graph.addEdge(i,j)


Potts_Clusters = My_Potts_Graph.connectedComponents() 


def Compute_Partition (Train_PottsData, _Spin_Configuration, T=1, sigma=1):
    
    
    """ 
    
    Given the Data and Spin Configuration, this function compute the Partition
    
    Parameters : 
    ----------
    
    PottsData: the features data, X
    
    Initial_Spin_Configuration : Initial Spin configuration for all observations
    
    T : The temperature 
    
    sigma : The bandwitch
    
    """
    
    _My_Potts_Graph = Graph(len(Train_PottsData))
    
    for i in range(len(Train_PottsData)):
        #let's get the top neighbors of observation i

        Selected_Neighbors = findneighbors(i, Train_PottsData, _Spin_Configuration, k_voisins = 5)

        for j in Selected_Neighbors:

            #addEdge(graph,i,j)
            _My_Potts_Graph.addEdge(i,j)

                
    _Potts_Clusters = _My_Potts_Graph.connectedComponents() 
    
    return _Potts_Clusters

def Potts_Random_Partition (Train_PottsData, Initial_Partition = Potts_Clusters, Number_of_Random_Partitions = 1000) : 
    
    
    """ 
    
    This function generates _Random_Partitions for a given initial Potts_Clusters
    
    Parameters
    ----------
    
    Initial_Partition : A given initial (random partition) in defaultdict(list) format
    
    Number_of_Random_Partitions: Number of expected random partitions, must be greater than 0 preferably
    
    
    Return    
    ------
    
    Full_Observations_Spin_Configuration : A full list of spin configuration for each generated partition 
    
    Full_Partition_Sets : A full list of all generated partitions
    
    
    """
    
    Full_Observations_Spin_Configuration = defaultdict(list) 
    
    Full_Partition_Sets = defaultdict(list) 
    
    Actual_Partition = Initial_Partition
    
    k = 0
    
    while k < (Number_of_Random_Partitions + 1):
        
        
            #Create the Clustter Component spin configuration 

            _Cluster_Spin_Configuration = []

            for h in range(len(Actual_Partition)):

                _Cluster_Spin_Configuration.append(random.randint(1,q))

            #Find observation spin configuration

            Observations_Spin_Configuration = []

            for observation in range(len(Train_PottsData)):

                Observation_Cluster_index = [ int(observation in Cluster) for Cluster in  Actual_Partition ].index(1)

                Observations_Spin_Configuration.append(_Cluster_Spin_Configuration[Observation_Cluster_index])
            
            
            Full_Observations_Spin_Configuration[k] = Observations_Spin_Configuration
            
            
            New_Partition = Compute_Partition (Train_PottsData, Observations_Spin_Configuration, T=1, sigma=1)

            #print(New_Partition)
            
            Full_Partition_Sets[k] = New_Partition

            Actual_Partition = New_Partition
            
            k = k + 1
            
    return Full_Partition_Sets, Full_Observations_Spin_Configuration


#import time
#start_time = time.time()

#Partitions_Sets,Spin_Configuration_Sets = Potts_Random_Partition (Train_PottsData, Initial_Partition= Potts_Clusters, Number_of_Random_Partitions = 1000)

#print("-3 Partitions generated-- %s seconds ---DONE!" % (time.time() - start_time))

#import pickle
#output = open('Partitions_Sets.pkl', 'wb')
#pickle.dump(Partitions_Sets, output)
#output.close()

import pickle

with open(r"scpf_Partitions_Sets.pkl", "rb") as input_file:
     Partitions_Sets = pickle.load(input_file)

def Cluster_mean(PottsData, Partition, Cluster_index):
    
    """
    
    Compute a Cluster Mean given a partition and cluster index
    
    Parameters
    ---------
    
    PottsData : The Data
    
    Partition : The set of Partition Clusters
    
    Cluster_index : The index of the cluster
    
    
    Return
    ------
    
    Cluster_Mean : A numpy array, size equal number of columns in PottsData.
    
    
    """
    
    
    Cluster_Mean = PottsData[Partition[Cluster_index],:].mean(0)
    
    
    return Cluster_Mean

from scipy import spatial


def New_Observation_Cluster(New_observation,Train_PottsData, Partitions_Sets, partition_position):
    
    New_observation_bonds = []

    
    New_Spin_Configuration = []

    for i in range(len(Train_PottsData)):

        New_Spin_Configuration.append(random.randint(1,q))

    
    New_observation_spin =np.random.choice(New_Spin_Configuration, 1)  #random.randint(1,q)      #random.choice(np.unique(Spin_Configuration_Sets[partition_position]))

    Indexed_Cluster_list_for_A_Given_Partition = []

    for observation in range(len(Train_PottsData)):



         if (New_observation_spin == New_Spin_Configuration[observation] ): #pin_Configuration_Sets[partition_position][observation] ) : 


                 connection_probability = 1 - np.exp(-LA.norm(New_observation - Train_PottsData[observation,:])/(sigma*T) )

                 #print(connection_probability)

                 if   connection_probability > 0.1 :
                      print("HOC")  

                      Observation_Cluster_index = [ int(observation in Cluster) for Cluster in  Partitions_Sets[partition_position] ].index(1) 

                      Indexed_Cluster_list_for_A_Given_Partition.append(Observation_Cluster_index)




    Indexed_Cluster_list_for_A_Given_Partition =np.unique(Indexed_Cluster_list_for_A_Given_Partition)
    
    Choosen_Cluster_ = 1000# just a default vaule that will cause errors if not updated
    
    if len(Indexed_Cluster_list_for_A_Given_Partition)>1:
    
        Indexed_Clusters_Mean_list = []


        for h in range(len(Indexed_Cluster_list_for_A_Given_Partition)):


            Indexed_Clusters_Mean_list.append(Cluster_mean(Train_PottsData,Partitions_Sets[partition_position], Indexed_Cluster_list_for_A_Given_Partition[h]))

        #print(Indexed_Clusters_Mean_list)   

        tree = spatial.KDTree(Indexed_Clusters_Mean_list)

        Choosen_Cluster_ = tree.query(New_observation)[1]
        
    else: 
        
        Choosen_Cluster_ = Indexed_Cluster_list_for_A_Given_Partition[0]
         
        
    return Choosen_Cluster_
    


#Import tensorflow and numpy
import numpy as np

# Load scikit's random forest classifier library for comparison
from sklearn.ensemble import RandomForestRegressor
from numpy import linalg as LA


from SPNNR_initialisation_and_training import SPNNR
#from Training_prediction_RMSE import SPNNR_training_prediction_rmse
#from Test_prediction_RMSE import SPNNR_test_prediction_rmse
from SPNNR_methods import build_toy_dataset1
import tensorflow as tf

#from Train_Test_predictions_RMSE import SPNNR_training_test_prediction_rmse


from SPNNR_methods import sample_GMRF,sample_Sigma, NN2


def SPNNR_training_test_prediction_rmse(my_qw_evalb,my_xi_evalb, my_psi_bar_evalb, my_mu_bar_evalb, x,y,xtest,ytest):

    
    l_0 = x.shape[1]#3
    l_1 = 3

    l_2 = y.shape[1]#2
    
    lb_0 = l_1 #2, (must be equal to l_1)
    lb_1 = l_2 #2, (must be equal to l_2)
    
    print("GRANDEUR### Etape00000")
    y_test_prediction = []
    SPNNR_test_rmse = []

    # INTIALISATIONS
        
    #init = tf.initialize_all_variables()
    sess = tf.Session()
    #sess.run(init)

    #number of samples per parameters
    exp_time1 = 2

    #number of samples per yi
    exp_time2 = 2

    stack1 = np.zeros((y.shape[0],exp_time1, 1,y.shape[1]))
    

    mean_b = U_b = tf.constant(np.zeros((lb_0 + lb_1)),dtype=tf.float64)
    mean_W = U_w = tf.constant(np.zeros((l_0*l_1 + l_1*l_2)),dtype=tf.float64)

    print("GRANDEUR### Etape0")

    for s in range(exp_time1) : #prendre la moyenne sur le nombre de fois pour le expected

        #Work with :

        ##Generate parameters



        W = sample_GMRF(mean_W.eval(session=sess),my_qw_evalb) # set mean_W here == zeros matrix, the mean is null,sample_GMRF(U_w.eval(session=sess),Q_w.eval(session=sess))

        b = sample_GMRF(mean_b.eval(session=sess),my_xi_evalb) # set mean_b here == zeros matrix, the mean is null

        Sigma = sample_Sigma(my_mu_bar_evalb,my_psi_bar_evalb) #mu_bar must be superior than IW dimension

        W_0 = W[0:l_0*l_1]
        W_0  = np.reshape(W_0, [l_0,l_1])
        
        W_1 = W[l_0*l_1:(l_0*l_1 + l_1*l_2 )]
        W_1  = np.reshape(W_1, [l_1,l_2])
        
        #remember: SIZE_gMRF_b = lb_0 + lb_1
        
        b_0 = b[0:lb_0]
        b_0 = np.reshape(b_0,[-1])
        
        b_1 = b[lb_0: (lb_0+lb_1)]
        b_1 = np.reshape(b_1,[-1])
        
        for i in range(x.shape[0]): # remember x.shape[0] = q
        
        
	           xi = tf.reshape(x[i,], [1,x.shape[1]])

	           yi_mean = np.mean(np.random.multivariate_normal(np.reshape(NN2(xi, W_0, W_1, b_0, b_1).eval(session=sess),-1), Sigma,exp_time2), axis=0)
               
	           stack1[i,s,] = yi_mean
        if len(xtest) > 0 :        

            stack1_test = np.zeros((ytest.shape[0],exp_time1, 1,ytest.shape[1]))
            for i in range(xtest.shape[0]): # remember x.shape[0] = q


                   xi_test = tf.reshape(xtest[i,], [1,xtest.shape[1]])

                   yi_mean_test = np.mean(np.random.multivariate_normal(np.reshape(NN2(xi_test, W_0, W_1, b_0, b_1).eval(session=sess),-1), Sigma,exp_time2), axis=0)

                   stack1_test[i,s,] = yi_mean_test 

            #print(y[i,],"And", yi_mean)
            #y_array[i] = np.stack((y_array[i],yi_mean))

    ################################################### TRAINING STATS #####################################################################
    
    errors = np.zeros((y.shape[0],y.shape[1]))



    stack2 = np.zeros((y.shape[0], 1,y.shape[1]))

    for i in range(x.shape[0]):
        
        stack2[i,] = np.mean(stack1[i,],axis=0 )
        
        errors[i,] = y[i,] - np.reshape(stack2[i,],-1)

    print("GRANDEUR### Etape2")
    SPNNR_train_rmse = np.sqrt((LA.norm(errors)**2)/x.shape[0] ) 

    print("GRANDEUR### Etape3")
    #print("The Root Mean Square on train data for the SPNNR model is %f"%SPNNR_train_rmse)
    ################################################### TEST STATS #####################################################################
    
    if len(xtest) > 0 :
    
      print("test GRANDEUR### Etape1")
      
      errors_test = np.zeros((ytest.shape[0],ytest.shape[1]))

      stack2_test = np.zeros((ytest.shape[0], 1,ytest.shape[1]))

      for i in range(xtest.shape[0]):
          
          stack2_test[i,] = np.mean(stack1_test[i,],axis=0 )
         
          errors_test[i,] = ytest[i,] - np.reshape(stack2_test[i,],-1)


      print("test GRANDEUR### Etape2")
      SPNNR_test_rmse = np.sqrt((LA.norm(errors_test)**2)/xtest.shape[0] )  

      print("test GRANDEUR### Etape3")
      #print("The Root Mean Square on test data for the SPNNR model is %f"%SPNNR_test_rmse)

      y_test_prediction = np.reshape(stack2_test,(ytest.shape[0],ytest.shape[1]))

    return np.reshape(stack2,(y.shape[0],y.shape[1])), SPNNR_train_rmse, y_test_prediction , SPNNR_test_rmse
###########################################################################



from SPNNR_methods import NN2, lop_y_proba_NN2, Training_Parameters_Estimation

def Train_My_Potts_Cluster(Cluster_xtrain,Cluster_ytrain, Cluster_xtest,Cluster_ytest):

    
    Cluster_ytest_prediction = []
    SPNNR_test_rmse = []

    tf.reset_default_graph()     
    sess = tf.Session() 
    
    """ 
    
     hidden layer weights matrix is size l_0*l_1, 
     output layer weights matrix is size l_1*l_2,


     first biais/offset vector is size lb_0 (must be equal to l_1),
     second biais/offset vector is size lb_1 (must be equal to l_2)
    
    
     There are some conditions on the each layer size parameters
    
    """
    l_0 = Cluster_xtrain.shape[1] #(must be equal to xtrain.shape[1])
    l_1 = 3
    l_2 = Cluster_ytrain.shape[1]  #(must be equal to ytrain.shape[1])

    lb_0 = l_1 #(must be equal to l_1)
    lb_1 = l_2 #(must be equal to l_2)

    Continue_Principle = 0

    while (Continue_Principle == 0 ):

            try:

                SPNNR_method = SPNNR(Cluster_xtrain,Cluster_ytrain,l_0,l_1,l_2,lb_0,lb_1)

                #g = tf.Graph()
                #with tf.Session(graph=g) as sess:

                my_qw_evalb,my_xi_evalb, my_psi_bar_evalb, my_mu_bar_evalb =  SPNNR_method.initialisation_training(Cluster_xtrain,Cluster_ytrain)
                
                print(my_qw_evalb,my_xi_evalb, my_psi_bar_evalb, my_mu_bar_evalb)

                print("Training: For Parameters, my_qw_evalb,my_xi_evalb, my_psi_bar_evalb, my_mu_bar_evalb, it is, Done!")


                #train/test prediction
                print("This is my cluster test", Cluster_xtest)

                
                Cluster_ytrain_prediction, SPNNR_train_rmse, Cluster_ytest_prediction, SPNNR_test_rmse = SPNNR_training_test_prediction_rmse(my_qw_evalb,my_xi_evalb, my_psi_bar_evalb, my_mu_bar_evalb, Cluster_xtrain,Cluster_ytrain,Cluster_xtest,Cluster_ytest)
                
                print("train prediction, Done,test prediction, Done")
                #Compute Cluster proba ---> this will help to compute partition acceptation ratio

                W,b,Sigma = Training_Parameters_Estimation(my_qw_evalb,my_xi_evalb, my_psi_bar_evalb, my_mu_bar_evalb, Cluster_xtrain,Cluster_ytrain )
                
                print("parameters, Done")
                
                Cluster_proba = tf.exp(lop_y_proba_NN2(Cluster_xtrain,Cluster_ytrain,W,b,Sigma))
                print("probability, Done")

                Continue_Principle = 1
                
                #tf.reset_default_graph()

                print("Full!")

            except:

                #raise ValueError
                #print("Oops!  Random Initialisation in training does'nt fit.  Try again...")
                Continue_Principle = 0

    My_Cluster_proba = Cluster_proba.eval(session=sess)
    #sess.close()
    
    return My_Cluster_proba, Cluster_ytrain_prediction, Cluster_ytest_prediction, SPNNR_train_rmse, SPNNR_test_rmse



def Compute_Cluster_Train_Data (cluster, xtrain, ytrain):
    
    cluster_xtrain,Cluster_ytrain = xtrain[cluster], ytrain[cluster]
        
    return cluster_xtrain,Cluster_ytrain, cluster


def Compute_Cluster_Test_Data_allocation (xtrain, xtest, ytest, Partitions_Sets, partition_position):
    
    """ Compute Clusters Data for a given partition """
         
    #partition_position = Partitions_Sets 
    
    xtest_cluster_allocation = []
    
    for j in range(len(xtest)):
    
        New_observation = xtest[j]
        
        Cluster_index = New_Observation_Cluster(New_observation, xtrain, Partitions_Sets, partition_position)
        
        xtest_cluster_allocation.append(Cluster_index)
    
    #a dictionary to set each cluster  data
    
    dictionary_keys = np.unique(xtest_cluster_allocation)
    
    Test_Clusters_Data_Dictionary = {}
    
    for key in list(dictionary_keys) :
        
        clusters_data = [i for i in range(len(xtest_cluster_allocation)) if xtest_cluster_allocation[i] == key] 
        
        Test_Clusters_Data_Dictionary[key] = clusters_data
        
    return Test_Clusters_Data_Dictionary


def Compute_Cluster_Test_Data_ (c, Clusters_Data_Dictionary, xtest, ytest):
    
    Cluster_xtest = []
    
    Cluster_ytest = []
    
    mycluster_test_data_index = []
    
    My_Selected_Clusters_Keys = list(Clusters_Data_Dictionary.keys())
    
    if  c in My_Selected_Clusters_Keys:
        
        mycluster_test_data_index = Clusters_Data_Dictionary[c]
        
        Cluster_xtest = xtest[mycluster_test_data_index]
        
        Cluster_ytest = ytest[mycluster_test_data_index]
        
    return Cluster_xtest,Cluster_ytest, mycluster_test_data_index


def _Predictions_Data_For_Partitions (xtrain, ytrain, Partitions_Sets,partition_position, xtest,ytest):
    
    c_Cluster_xtest = []

    My_Partition = Partitions_Sets[partition_position]
    
    All_Clusters_Proba = np.ones(len(My_Partition))
    
    Train_Clusters_Data_ = np.zeros((len(xtrain),ytrain.shape[1])) 
    
    Test_Clusters_Data_ = np.zeros((len(xtest),ytest.shape[1])) 
    
    
    Test_Clusters_Data_Dictionary = Compute_Cluster_Test_Data_allocation(xtrain, xtest, ytest, Partitions_Sets, partition_position)
    
    for c, cluster in enumerate(My_Partition):
    
        #a cluster is representated by a list of observations indexes
        
        c_Cluster_xtrain, c_Cluster_ytrain, c_cluster = Compute_Cluster_Train_Data (cluster, xtrain, ytrain)
        
        
        #cluster_test_data_index is the index of allocated (xtest,ytest) data in training cluster c
        #this return also Cluster_xtest, and Cluster_ytest
        
        c_Cluster_xtest, c_Cluster_ytest, c_cluster_test_data_index = Compute_Cluster_Test_Data_ (c, Test_Clusters_Data_Dictionary, xtest, ytest)
        
        
        c_Cluster_proba, c_Cluster_ytrain_prediction, c_Cluster_ytest_prediction, c_SPNNR_train_rmse, c_SPNNR_test_rmse =Train_My_Potts_Cluster(c_Cluster_xtrain,c_Cluster_ytrain, c_Cluster_xtest,c_Cluster_ytest)
        
        
        Train_Clusters_Data_[cluster,:] = c_Cluster_ytrain_prediction
        
        
        All_Clusters_Proba[c] = float(c_Cluster_proba)
        
        
        #Cluster_xtest can be empty
        if  len(c_Cluster_xtest)>0:
        
            Test_Clusters_Data_[Test_Clusters_Data_Dictionary[c],:] = c_Cluster_ytest_prediction
        print("I am DONE for cluster number %i !"%c)

    """ Fuse all train predictions into one for the given partition """
    Partition_ytrain_prediction = Train_Clusters_Data_
    
    """ Fuse all test predictions into one for the given partition  """
    Partition_ytest_prediction =  Test_Clusters_Data_
        
    
    Partition_probability = All_Clusters_Proba.prod()
    
    
    return Partition_ytrain_prediction, Partition_ytest_prediction, Partition_probability


import time
start_time = time.time()

Y_train_predictions = np.empty((xtrain.shape[0],len(Partitions_Sets),ytrain.shape[1]))

Y_test_predictions = np.empty((xtest.shape[0],len(Partitions_Sets),ytest.shape[1]))

All_Partitions_Data = []

number_of_valides_partitions = 0

actual_probability = 1

List_of_Partitions_Probabilities = []

for h, Partition in enumerate(Partitions_Sets):
    
    
    Partition_ytrain_prediction, Partition_ytest_prediction, Partition_probability = _Predictions_Data_For_Partitions (xtrain, ytrain, Partitions_Sets,h, xtest,ytest)
    
    
    #if Partition_probability > 0 and float(Partition_probability/actual_probability) >=1 :  #I need to do this to control only accepted partitions
        
        
    Y_train_predictions[:,h] =  Partition_ytrain_prediction
 
    Y_test_predictions[:,h] =  Partition_ytest_prediction
     
    actual_probability =   Partition_probability 
    List_of_Partitions_Probabilities.append(Partition_probability)
     
    number_of_valides_partitions = number_of_valides_partitions + 1
    print("We are done for the %i-th Partition:"%h)





Y_train_predictions = np.delete(Y_train_predictions,np.where(~Y_train_predictions.any(axis=0))[0], axis=1)
Y_test_predictions = np.delete(Y_test_predictions,np.where(~Y_test_predictions.any(axis=0))[0], axis=1)

print("Final_Partition predictions are set up--- %s seconds ---" % (time.time() - start_time))

np.savetxt('All_Y_train_predictions.out', Y_train_predictions.mean(axis=1), delimiter=',', fmt='%.8f')
np.savetxt('All_Y_test_predictions.out', Y_test_predictions.mean(axis=1), delimiter=',', fmt='%.8f')
np.savetxt('List_of_Partitions_Probabilities.out', np.array(List_of_Partitions_Probabilities).reshape(1,-1), delimiter=',', fmt='%.8f')


#Partitions errors
Errors_train = Y_train_predictions.mean(axis=1) - ytrain
Errors_test =  Y_test_predictions.mean(axis=1) - ytest


Final_SPNNR_train_rmse = np.sqrt((LA.norm(Errors_train)**2)/xtrain.shape[0]) 
print("Final_Partition train error is : %f"%Final_SPNNR_train_rmse)

np.savetxt('Final_SPNNR_train_rmse.txt', Final_SPNNR_train_rmse.reshape(1,-1), delimiter =', ')

Final_SPNNR_test_rmse = np.sqrt((LA.norm(Errors_test)**2)/xtest.shape[0]) 
print("Final_Partition test error is : %f"%Final_SPNNR_test_rmse)

np.savetxt('Final_SPNNR_test_rmse.txt', Final_SPNNR_test_rmse.reshape(1,-1), delimiter =', ')

print('<*********************************Let us check the aRRMSE*********************************>')

ytrain_mean = np.tile(Y_train_predictions.mean(axis=1).mean(axis = 0), (xtest.shape[0], 1))

Errors_relative = ytrain_mean - ytest

Final_SPNNR_aRRMSE = 100*np.sqrt((LA.norm(Errors_test)**2)/(LA.norm(Errors_relative)**2))  

print("Final_Partition aRRMSE is : %f"%Final_SPNNR_aRRMSE)
np.savetxt('Final_SPNNR_aRRMSE.txt', Final_SPNNR_aRRMSE.reshape(1,-1), delimiter =', ')
