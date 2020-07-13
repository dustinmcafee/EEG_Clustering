#!/usr/bin/python3
"""
Name: Dustin Mcafee
Perform Expectation Maximization Assuming K Gaussian Clusters
"""
from pyspark import SparkContext		# For Spark mapreduce
import sys					# For system arguments
import numpy as np				# For fast matrix multiplication and linear algrebra
from numpy import genfromtxt			# For reading csv
from scipy.stats import multivariate_normal	# For multivariate normal probability distribution
from scipy.misc import logsumexp		# For use with map to sum likelihoods
from sklearn.metrics import silhouette_score	# For silhoutte score
import pandas as pd				# For fast computation of distance matrix
from scipy.spatial.distance import pdist,squareform

# Find the centroid that the `point` is closest to and return the centroid's ID
# The centroid ID in this case is simply its index in the `centroids` list
# Input: point: np.array(x,y)
#        centroids: [np.array(x1,y1), np.array(x2,y2), ..., np.array(xK,yK)], 
#                   where K is the number of clusters
# Return: clusterID
def getClosestCentroidID(point, centroids):
    distances = [np.sqrt(sum((point - centroid)**2)) for centroid in centroids]
    return np.argmin(distances)

# Given a point (i.e., (x,y) and a list of centroids (i.e., list of points),
# find the closest centroid and assign that cluster to the point
# Input: points_rdd: <<np.array(x1,y1), np.array(x2,y2), ... np.array(xN,yN)>>,
#                    where N is the number of lines in the file
#        centroids:  [np.array(x1,y1), np.array(x2,y2), ..., np.array(xK,yK)],
#                    where K is the number of clusters
# Return: RDD of clustered points: <<(clusterID, np.array(x1, y1)), (clusterID, np.array(x2, y2)), ...>>
def assignPointsToClosestCluster(points_rdd, centroids):
    return points_rdd.map(lambda x: (getClosestCentroidID(x, centroids), x))
    
# Sum the distance that each centroid moved by
# Input: old_centroids: [np.array(x1,y1), np.array(x2,y2), ..., np.array(xK,yK)],
#                       where K is the number of clusters
#        new_centroids: [np.array(x1,y1), np.array(x2,y2), ..., np.array(xK,yK)],
#                       where K is the number of clusters
# Return: sum of distances
def calculateChangeInCentroids(old_centroids, new_centroids):
    a = np.array(old_centroids, dtype=float)
    b = np.array(new_centroids, dtype=float)
    return sum([np.sqrt(sum((old - new)**2)) for old, new in zip(a, b)])

# Compute the distance matrix given a (normalized) data matrix
# Input: data: (normalized) N x D data matrix
# Return: distances: N x N symmetric Distance Matrix
def distance(data):
    DF_var = pd.DataFrame(data)
    distances = squareform(pdist(DF_var, metric='euclidean'))
    return distances

#Calculate Distance Matrix and calculate centroids using
# weighted probability proportional to the distances squared
# (i.e. kmeans++ method)
# Input: data: data matrix that contains the points (one per row)
#        K: number of clusters
# Return: centoids: [centroid1, centroid2, ..., centroidK]
#         dist_matrix: distance matrix of data points
def centroid(data, K):
    # Initialize Array Variables
    centroids = []
    used_indices = []

    # Create distance matrix
    dist_matrix = distance(data)
    indices = np.arange(np.size(data, 0))

    # Choose first index randomly
    index = np.random.choice(indices, 1)
    used_indices.append(index)

    # Find probable initial centroids
    for j in range(0, K):
        centroids.append(data[index][0])
        d = np.full((1, np.size(data, 0)), np.inf)

        # For each data point x, compute the distance between x and the nearest centroid that has already been chosen.
        for i in used_indices:
            dist = dist_matrix[i].copy()
            d = np.concatenate((d, dist))

        # Create probability distrubution from minimum distances
        min_prob = np.amin(d, axis=0)
        prob = np.square(min_prob)
        prob = min_prob / np.sum(min_prob)

        # Choose centroids by probability proportional to distance of nearest centroid
        index = np.random.choice(indices, 1, p=prob)
        used_indices.append(index)

    return centroids, dist_matrix

# Compute the Log PDF (Multivariate Normal) (i.e. the Log Likelihood)
# for one row of the data (the Kth row)
# Input: data: data matrix (N x D 2d list)
#	 k: number of clusters
#	 prob_mat: Pior Probabilities (N x k 2d list)
#	 means: mean of each cluster (list)
#	 covs: covariance of each cluster (list)
# Return: list of log likelihoods of x belonging to class k
def Log_Likelihood(data, k, prob_mat, means, covs):
	return [(np.log(prob_mat[i]) + multivariate_normal.logpdf(data[0],mean=means[i],cov=covs[i],allow_singular=True)) for i in range(k)]

# Map function to compute updated likelihood values
# Input: x: data row
#	 k: number of clusters
#	 probs: probabilities for the current row
#	 means: mean of each cluster (list)
#	 covs: covariance of each cluster (list)
# Return: [row, [likelihood of k==0, likelihood of k==1, ...]]
def map_LogLike(x, k, probs, means, covs):
	# Get log likelihoods for each k cluster
	result = []
	likelihood = 0
	# Compute the Likelihood (probability) for each cluster, k
	log_l = np.array(Log_Likelihood(x, k, probs, means, covs))
	for i in range(k):
		try:	# Could be zero
			log_l[i] = log_l[i] - logsumexp(log_l)
		except:
			pass
	try:	# Could be zero
		likelihood = np.exp(log_l)
	except:
		pass
	result.append(x[0])
	result.append(likelihood)
	return result

# Compute the new gaussian parameters given the data and prior probabilities
# Input: the_rdd: Data matrix of row attributes (RDD)
#        k: number of clusters
#        prob_mat: (N x k) probability matrix
#        means: means of each cluster
#        covs: Covariances of each cluster
# Return: rdd_LL: [data_points, log_likelihoods]
def Expectation(the_rdd, k, prob_mat, means, covs):
	rdd_LL = the_rdd.map(lambda x : map_LogLike(x, k, prob_mat, means, covs))
	return rdd_LL

#Maximization Step Map/Reduce Functions

# Reduce function to add the probabilities
# Input: K: number of clusters
#	 x: [0]: the data point (list)
#	    [1]: the probabilities (likelihoods) of [0] for each K (list)
#	 y: [0]: the other data point (list)
#	    [1]: the probabilities (likelihoods) of [0] for each K (list)
# Return: [0, [summed likelihoods (length k)]]
def reduce_add_Prob(K, x, y):
	return [0, [x[1][i] + y[1][i] for i in range(K)]]

# Map function to compute new partial means for each k (probability times data point = expected value)
# Input: K: number of clusters
#	 x: [0]: the data point (list)
#	    [1]: the probabilities (likelihoods) of [0] for each K (list)
# Return: [x[0], x[1], x[0]*[i] for i to k]
def map_Means(K, x):
	return x + [[x[0] * x[1][i] for i in range(K)]]

# Reduce function to Add means for each K (Add expected values from different clusters together) 
# Input: K: number of clusters
#	 x: output of map_Means (third element is a K * D 2d list)
#	 y: output of map_Means (third element is a K * D 2d list)
# Return: [0, 0, [summed expected values for each k (k seperate lists of length D)]]
def reduce_Means(K, x, y):
	return [0,0,[x[2][i] + y[2][i] for i in range(K)]]

# Map function to compute partial Covariances for each cluster
# Input: K: number of clusters
#	 x: output of map_Means (third element is a K * D 2d list)
#	 means: List of k mean arrays, all of size D
# Return: [output of map_Means, partial Covariance Matrix for each K cluster times N * probability of k]
def map_Covs(K, x, means):
	return x + [[x[1][i]*np.outer(x[0]-means[i],x[0]-means[i]) for i in range(K)]]

# Reduce function to add up the partial covariance matrices for each cluster
# Input: K: number of clusters
#	 x: output of map_Covs
#	 y: output of map_Covs
# Return: [0, 0, 0, [D x D covariance matrix for each k (times N times probability of k)]]
def reduce_Covs(K, x, y):
	return [0,0,0,[x[3][i] + y[3][i] for i in range(K)]]

def Maximization(the_rdd, k, N):
	# Compute Probabilities
	Nprob_mat = np.array(the_rdd.reduce(lambda x, y : reduce_add_Prob(k, x, y))[1]).clip(1e-10,np.inf)
	prob_mat = Nprob_mat / N

	# Compute Means 
	rdd_means = the_rdd.map(lambda x : map_Means(k, x))
	means_t = rdd_means.reduce(lambda x, y : reduce_Means(k, x, y))[2]
	means = [means_t[k]/Nprob_mat[k] for k in range(k)]	# Divide by N and the expected values to yeild the mean for each dimension, per k cluster

	# Compute Covariance Matrices
	rdd_covs = rdd_means.map(lambda x: map_Covs(k, x, means))
	covs_t = rdd_covs.reduce(lambda x, y : reduce_Covs(k, x, y))[3]
	covs = [covs_t[k]/Nprob_mat[k] for k in range(k)]	# Divide by N and the likelihoods for the correct covariance matrices
	return prob_mat, means, covs

# Compute the Sum of the Log Likelihoods the_rdd
# Input: the_rdd: The RDD of the data (N x D matrix)
#	 k: Number of clusters
#	 probs: Probability of each row belonging to each k cluster
#		(N x k)
#	 means: Means of each cluster
def lgsumexp_ll(the_rdd, k, probs, means, covs):
	return the_rdd.map(lambda x: logsumexp(np.array(Log_Likelihood(x, k, probs, means, covs)))).reduce(lambda x,y: x+y)

# Predicts the Cluster for each row of the RDD
# Input: the_rdd: the RDD of the data (N x D matrix)
#	 k: Number of clusters
#	 N: Number of observations
#	 probs: Probability of each row belonging to each k cluster
#		(N x k)
#	 means: Means of each cluster
#	 covs: Covariances of each cluster
# Returns: Numpy array of predictions for each row
def EM_predict_cluster(the_rdd, k, N, probs, means, covs):        
	return np.array(the_rdd.map(lambda x: np.array(Log_Likelihood(x, k, probs, means, covs))).map(lambda x: np.argmax(x)).take(N))

# Helper function for getPerformanceMetrics.
# Computes number of correct/incorrect predictions,
# as well as true positives/negatives, and false positives/negatives
# Input: validData: the valid list of categories
#	 predictions: the predicted list of categories
# Return: number of correct, incorrect, true positives,
#	  true negatives, false positives, and false negatives
def getPerformanceStatistics(validData, predictions):
	correct = 0
	incorrect = 0
	fNeg = 0
	fPos = 0
	tPos = 0
	tNeg = 0
	for x in range(len(validData)):
		if(validData[x] == predictions[x]):
			correct += 1
			if(validData[x] == 0):				#True negative
				tNeg +=1
			else:						#True positive
				tPos +=1
		else:
			incorrect += 1
			# 1 == seizure (positive), 0 == no seizure (negative)
			if(validData[x] == 1):				#False negative
				fNeg +=1
			else:						#False positive
				fPos +=1
	return correct, incorrect, tPos, tNeg, fPos, fNeg

#Report performance metrics from the validation dataset and list of predictions
#Returns tuple
def getPerformanceMetrics(validData, predictions):
	P, N, TP, TN, FP, FN = getPerformanceStatistics(validData, predictions)
	accuracy = (P / (P + N)) * 100.0
	if((TP + FN) == 0):
		TPR = np.inf
	else:
		TPR = TP / (TP + FN) * 100.0	#True positive rate; recall; sensitivity
	if((TP + FP) == 0):
		PPV = np.inf
	else:
		PPV = TP / (TP + FP) * 100.0	#Positive predictive value; precision
	if((TN + FP) == 0):
		TNR = np.inf
	else:
		TNR = TN / (TN + FP) * 100.0	#True negative rate; specificity
	if((PPV + TPR) == 0):
		F = np.inf
	else:
		F = 2 * PPV * TPR / (PPV + TPR)	#F1 Score
	return (P, N, TP, TN, FP, FN, accuracy, TPR, PPV, TNR, F)

def main():    
    #Command Line Argument
    k = int(sys.argv[1])
    if k <= 0:
        print("First arguement must be > 0")
        return
    print("Performing Expectation Maximization on Training Data for ", k, "clusters")

    #Load data matrix
    my_data = genfromtxt('input/train/TrainingData_Projected.txt', delimiter=',')
    testing_data = genfromtxt('input/test/TestingData_Projected.txt', delimiter=',')

    #Delete first row: header
    my_data = np.delete(my_data, 0, 0)
    #Delete first column: ID
    IDs = my_data[:,0].copy()
    IDs = IDs.reshape(np.size(my_data, 0),1)
    my_data = np.delete(my_data, 0, 1)
    #Collect Correct labels and delete label column (last column)
    labels = my_data[:,-1].copy()
    my_data = np.delete(my_data, -1, 1)

    # Create RDD from data
    sc = SparkContext.getOrCreate()
    points = sc.parallelize(my_data, numSlices=100)

    # Find Centroids (kmeans++ method)
    centroids, dist_matrix = centroid(my_data, k)
    # Assign each point to the centroid closest to it
    clustered_points = assignPointsToClosestCluster(points, centroids)
    points_arr = clustered_points.collect().copy()

    #Extract cluster labels
    label = [i[0] for i in points_arr]
    points_arr = [i[1] for i in points_arr]
    points_arr = np.vstack(points_arr)
    N = np.size(points_arr, 0)
    dimension = np.size(points_arr, 1)
    data = np.column_stack((points_arr, label))

    data_arranged = []
    pi_arr = np.zeros(k)
    for i in list(set(label)):
        data_arranged.append(data[np.logical_or.reduce([data[:,-1] == i])])
    iter = 0

    #Probabilities for each initial class
    for i in label:
        if ((data[:,-1] == i)[iter]):
            pi_arr[i] += 1
        iter += 1
    pi_arr = pi_arr / np.size(label, 0)

    #Covariances and mean for each initial class
    cov_arr = []
    mean_arr = []
    arr = list(range(1,dimension+1))
    for i in list(set(label)):
        cov_arr.append((np.dot(data_arranged[i][:,arr].T, data_arranged[i][:,arr]) / N).squeeze())
        mean_arr.append(np.mean(data_arranged[i][:,arr], 0))

    #data_points = data[:,arr].reshape(N,dimension).copy()
    #label = data[:,0].copy()
    data_points = data[:,range(0,dimension)].reshape(N,dimension).copy()

    # loop until parameters converge
    data_list = points.map(lambda x: [x])
    shift = sys.maxsize
    old_shift = sys.maxsize
    old_old_shift = sys.maxsize
    dist_centers = sys.maxsize
    epsilon = 0.005
    iters = 0
    ll_change = np.inf
    while(shift > epsilon and ll_change > epsilon):
        iters += 1

        #Expectation Step
        the_rdd = Expectation(data_list, k, pi_arr, mean_arr, cov_arr)

        #Maximization Step
        new_pi_arr, new_mean_arr, new_cov_arr = Maximization(the_rdd, k, N)

        # Calculate difference of Log Likelihood
        l = abs(lgsumexp_ll(data_list, k, pi_arr, mean_arr, cov_arr))
        if (iters == 1):
            ll_change = l
        else:
            ll_change = abs(l - ll_change)

        #Calculate Shift (Maximum Shift in Centroids)
        old_old_shift = old_shift
        old_shift = shift
        shift = calculateChangeInCentroids(mean_arr, new_mean_arr)

        #Update Variables
        mean_arr = new_mean_arr.copy()
        cov_arr = new_cov_arr.copy()
        pi_arr = new_pi_arr.copy()

        #Log Iteration
        print("Iteration:", iters, "Maximum Change in Centroids:", shift, "; Change in Sum of Log-Likelihoods:", float(ll_change))

        # Prevent oscillation
        if(abs(old_old_shift - shift) <= epsilon):
            break;

    predicted_labels = EM_predict_cluster(the_rdd, k, N, pi_arr, mean_arr, cov_arr)        

    print("Gaussian Expectation Maximization Converged after", iters, "iterations")
    if(not all(elem == predicted_labels[0] for elem in predicted_labels)):
        sil_in = silhouette_score(data_points, predicted_labels)
        print("Silhouette index for training Dataset:", float(sil_in))
    else:
        print("Gaussian Expectation Maximization Failed")

    #Prepend Row numbers for Plotting the Data
    new_data = np.matrix([])
    new_data = np.hstack((predicted_labels.reshape(N, 1), data_points))

    # Print metrics
    P, N, TP, TN, FP, FN, accuracy, TPR, PPV, TNR, F = getPerformanceMetrics(labels, predicted_labels)
    print('Training Set Performance Metrics:')
    print('Accuracy: ' + "{0:.2f}".format(round(float(accuracy),2)) + '%')
    print('True Positives: ' + str(TP))
    print('True Negatives: ' + str(TN))
    print('False Positives: ' + str(FP))
    print('False Negatives: ' + str(FN))
    print('True Positive Rate (sensitivity): ' + "{0:.2f}".format(round(float(TPR),2)) + '%')
    print('Positive Prediction Value (precision): ' + "{0:.2f}".format(round(float(PPV),2)) + '%')
    print('True Negative Rate (specificity): ' + "{0:.2f}".format(round(float(TNR),2)) + '%')
    print('F1 Score: ' + "{0:.2f}".format(round(float(F),2)) + '%')

    #Save output Files
    filename = "output/Data_EM_" + str(k) + ".txt"
    np.savetxt(filename, new_data, delimiter=',', fmt=' '.join(['%d,'] + ['%d,'] + ['%1.4f,']*(np.size(new_data, axis=1)-3) + ['%1.4f']))






    # <---Run on Testing Data--->

    print("Performing Expectation Maximization on Testing Data for ", k, "clusters")
    #Delete first row: header
    testing_data = np.delete(testing_data, 0, 0)
    #Delete first column: ID
    testing_data = np.delete(testing_data, 0, 1)
    #Collect labels and delete label column (last column)
    labels = testing_data[:,-1].copy()
    testing_data = np.delete(testing_data, -1, 1)

    # Create RDD from data
    sc = SparkContext.getOrCreate()
    test_points = sc.parallelize(testing_data, numSlices=50)
    N = np.size(testing_data, 0)

    # Assign each point to the centroid closest to it
    predicted_labels = EM_predict_cluster(test_points, k, N, pi_arr, mean_arr, cov_arr)
    sc.stop()

    sil_in = silhouette_score(testing_data, predicted_labels)
    print("Silhouette index for testing Dataset:", float(sil_in))

    #Prepend Categories for Plotting the Data
    N = np.size(testing_data, 0)
    predicted_labels = np.array(predicted_labels).reshape(N, 1)
    testing_data = np.hstack((predicted_labels, testing_data))

    # Print metrics
    P, N, TP, TN, FP, FN, accuracy, TPR, PPV, TNR, F = getPerformanceMetrics(labels, predicted_labels)
    print('Testing Set Performance Metrics:')
    print('Accuracy: ' + "{0:.2f}".format(round(float(accuracy),2)) + '%')
    print('True Positives: ' + str(TP))
    print('True Negatives: ' + str(TN))
    print('False Positives: ' + str(FP))
    print('False Negatives: ' + str(FN))
    print('True Positive Rate (sensitivity): ' + "{0:.2f}".format(round(float(TPR),2)) + '%')
    print('Positive Prediction Value (precision): ' + "{0:.2f}".format(round(float(PPV),2)) + '%')
    print('True Negative Rate (specificity): ' + "{0:.2f}".format(round(float(TNR),2)) + '%')
    print('F1 Score: ' + "{0:.2f}".format(round(float(F),2)) + '%')

    #Save output Files
    filename = "output/TestingData_EM_" + str(k) + ".txt"
    np.savetxt(filename, testing_data, delimiter=',', fmt='%3.4f')

main()
