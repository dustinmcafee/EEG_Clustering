Data can be found here:
https://archive.ics.uci.edu/ml/datasets/Epileptic+Seizure+Recognition

First source env.sh for correct python version and spark environment (THIS IS FOR DUSTIN'S ENVIRONMENT/MAY NOT WORK FOR YOU!):
. ./env.sh

There are three python files for this project:
data.py:	Centers and Projects dataset to first K Principal Components
kmeans.py:	Runs K-Means on the given K and dataset
exmax.py:	Runs Expectation Maximization assuming Gaussian Clusters given K and dataset

To standardize dataset (./input/data.csv) and project to K first Principal Components:
./data.py K
Output: Centered Data (output/Data.txt), Singular values (output/Singular_Values.txt),
	Percent Variance of each singular value (output/Singular_Values_Percent_Variance.txt),
	and Projected Dataset ont first K Principal Components (output/Data_Project.txt)

To run K-Means with K clusters on csv files input/train/TrainingData_Projected.txt and input/test/TestingData_Projected.txt with last column = correct category label
./kmeans.py K
Output: Categorized dataset/added category column (output/Data_Kmeans_k.txt and output/TestingData_Kmeans_k.txt)

To run Expectation Maximization (Gaussian Clusters) with K clusters on csv files input/train/TrainingData_Projected.txt and input/test/TestingData_Projected.txt with with last column = correct category label
./exmax.py K
Output: Categorized dataset/added category column (output/Data_EM_K.txt and output/TestingData_EM_k.txt)



If one wishes to use Jupyter Notebook (THIS IS FOR DUSTIN'S ENVIRONMENT/MAY NOT WORK FOR YOU!):
. ./jupyter_env.sh

There are three jupyter-notebook files corresponding to the three above files:
PCA.ipynb	<-->	data.py
kmeans.ipynb	<-->	kmeans.py
exmax.ipynb	<-->	exmax.py
