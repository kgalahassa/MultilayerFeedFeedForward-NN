# State-of-the-art Multivariate Regression with a General N_k Hidden Multi-Layer Feed Forward Neural Network Model
The code is of The Classical Neural Network Model introduced in [1].

We have designed two (2) types of model: a 1-Hidden-Layer, and a 4-Hidden-Layer; and have performed experiments with five (5) multivariate multiple regression datasets taken from the multiple-output benchmark datasets available in the Mulan project website [2]: Edm, Slump, Jura, Water Quality (wq), and Scpf. 

The train-test folder gives access to the train and test data for each of the dataset from the five (5) above.

In each Classical-Neural-Network folder, the main file of interest is the Notebook *.ipynb file; used for execution of the model. 
The folder "Potts-SHrinkage-Graphics-RandomForest" has been used for other experiments, including the comparisons with Random Forest Model. The Random Forest has been executed with each of the dataset in the <Dataset>-Potts-Partition-Analysis.ipynb file. 
  
  *For each dataset, the <Dataset>.arff file has to be loaded, and all the process is well described in the "Classical Neural Network Results on <Dataset> Datasets.ipynb" file.

[1] N. K. -A. Alahassa, "State-of-the-art Multivariate Regression with a General Nk Hidden Multi-Layer Feed Forward Neural Network Model," 2021 International Conference on Artificial Intelligence and Computer Science Technology (ICAICST), 2021, pp. 31-36, doi: 10.1109/ICAICST53116.2021.9497838.

[2] Tsoumakas, G., Spyromitros-Xioufis, E., Vilcek, J., & Vlahavas, I. (2020).  Datasets from mulan: A java libraryfor multi-label learning.URLhttp://mulan.sourceforge.net/datasets-mtr.html39
