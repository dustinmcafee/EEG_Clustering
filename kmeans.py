#!/usr/bin/python3
"""
Name: Dustin Mcafee
Class: COSC 528 - Introduction to Machine Learning
Assignment: Project 2 - Perform K-Means++ on dataset
"""

from pyspark import SparkContext
import sys
import numpy as np
import math
from numpy import genfromtxt
import os.path
import pandas as pd
from scipy.spatial.distance import pdist,squareform

# Add value1 and value 2
# Useful as a reduce function
def addValues(val1, val2):
    return val1 + val2

# Find the centroid that the point is closest to and return the centroid's index
# Input: point: point array(x, y)
#        centroids: array[array(x1,y1), array(x2,y2), ..., array(xk,yk)], 
#                   where k is the number of clusters
# Return: closest centroid index
def getClosestCentroid(point, centroids):
    distances = [np.sqrt(sum((point - centroid)**2)) for centroid in centroids]
    return np.argmin(distances)

# Given a point and a list of centroids, assign point to cluster of closest centroid
# Input: points_rdd: PipelinedRDD <<array(x1,y1), array(x2,y2), ... array(xN,yN)>>,
#                    where N is the number of lines in the file
#        centroids:  array[array(x1,y1), array(x2,y2), ..., array(xk,yk)],
#                    where k is the number of clusters
# Return: RDD of clustered points: <<(clusterID, np.array(x1, y1)), (clusterID, np.array(x2, y2)), ...>>
def assignPointsToClosestCluster(points_rdd, centroids):
    return points_rdd.map(lambda x: (getClosestCentroid(x, centroids), x))
    
# Sum the euclidean distance of two points
# Input: points1: array[array(x1,y1), array(x2,y2), ..., array(xk,yk)],
#        points2: array[array(x1,y1), array(x2,y2), ..., array(xk,yk)],
# Return: sum of distances
def sumDistances(points1, points2):
    return sum([np.sqrt(sum((one - two)**2)) for one, two in zip(points1, points2)])

# Calculate the mean coordinates of each cluster.
# Input: clustered_points_rdd: <<clustered_point1, clustered_point2, ..., clustered_pointN>>,
#                              where N is the number of clustered_points, and
#                              each clustered_point looks like (clusterID, array(x,y))
# Return: cluster means [centroid1, centroid2, ..., centroidK],
#         where K is the number of clusters, and
#         where each centroid is array(x,y)
def calculateClusterMeans(clustered_points_rdd):
    # Sum the xs and ys of all points in each cluster
    sum_points = clustered_points_rdd.reduceByKey(addValues)
    
    # Count the number of points in each cluster
    counts = clustered_points_rdd.countByKey()
    
    # Divide the x,y sums for each cluster by the number of points in each cluster
    cluster_means = []
    for key, value in sum_points.collect():
        n = counts.get(key)
        avg = value / n
        cluster_means.append(avg)
    
    return cluster_means

#Calculate Distance Matrix and calculate centroids using
# weighted probability proportional to the distances squared
# (i.e. kmeans++ method)
# Input: data: data matrix that contains the points (one per row)
#        K: number of clusters
# Return: [centroid1, centroid2, ..., centroidK]
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
    for j in range(0, K):
        centroids.append(data[index][0])
        d = np.full((1, np.size(data, 0)), np.inf)
        # For each data point x, compute the distance between x and the nearest centroid that has already been chosen.
        for i in used_indices:
            dist = dist_matrix[i]
            d = np.concatenate((d, dist))
        # Create probability distrubution from minimum distances
        min_prob = np.amin(d, axis=0)
        prob = np.square(min_prob)
        prob = min_prob / np.sum(min_prob)
        # Choose centroids by probability proportional to distance of nearest centroid
        index = np.random.choice(indices, 1, p=prob)
        used_indices.append(index)

    return centroids

# Cluster the points in the data matrix into K clusters using k-means clustering
# Input: data: data matrix that contains the points (point per row)
#        K: number of clusters
# Return: number of iterations, [centroid1, centroid2, ..., centroidK] and [clustered_point1, clustered_point2, ..., clustered_pointN]
#         where K is the number of clusters,
#         where N is the number of points,
#         where centroid is np.array(x,y), and
#         where each clustered_point is (clusterID, np.array(x,y))
def clusterData(data, K):
    # Choose K points as the centroids using K-means++
    centroids = centroid(data, K)

    # Create RDD from data
    sc = SparkContext.getOrCreate()
    points = sc.parallelize(data, numSlices=500)

    # Assign each point to the centroid closest to it
    clustered_points = assignPointsToClosestCluster(points, centroids)
    
    # Begin the iterative portion of k-means,
    # continue until the changes in centroids are very small (e.g., < .0001)
    change_in_centroids = np.inf
    old_change = 0
    iterations = 0
    while change_in_centroids > 0.005:
        old_centroids = centroids
        # Calculate the new centroids based on the means of the current clusters
        centroids = calculateClusterMeans(clustered_points)
        
        # Assign the points to the new centroids
        clustered_points = assignPointsToClosestCluster(points, centroids)
        
        # Calculate the change in the centroids since the last iteration
        change_in_centroids = sumDistances(old_centroids, centroids)
        if(change_in_centroids == old_change):
            break
        old_change = change_in_centroids
        iterations = iterations + 1
        print("Iteration:", iterations, "Change:", change_in_centroids)

    return iterations, centroids, clustered_points.collect()

# Compute the distance matrix given a (normalized) data matrix
# Input: data: (normalized) N x D data matrix
# Return: N x N symmetric Distance Matrix
def distance(data):
    DF_var = pd.DataFrame(data)
    distances = squareform(pdist(DF_var, metric='euclidean'))
    return distances
    
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
    print("Performing K-Means on Training Dataset for", k, "clusters")

    #Load data matrix
    my_data = genfromtxt("input/train/TrainingData_Projected.txt", delimiter=',')
    testing_data = genfromtxt("input/test/TestingData_Projected.txt", delimiter=',')

    #Delete first row: header
    my_data = np.delete(my_data, 0, 0)
    #Delete first column: ID
    my_data = np.delete(my_data, 0, 1)
    #Collect labels and delete label column (last column)
    dimensions = np.size(my_data, 1)
    labels = my_data[:,-1].copy()
    my_data = np.delete(my_data, -1, 1)

    #Initialize Variables
    dimensions = np.size(my_data, 1)
    N = np.size(my_data, 0)
    categories = np.zeros((N,1), dtype=int)
    arr = np.array([])
    for i in range(N):
        arr = np.append(arr, i)
    arr = arr.reshape(N,1)

    #Distance Matrix
    dist_matrix = distance(my_data)

    #K-means Clustering
    iterations, centroids, clustered_points = clusterData(my_data, k)
    print("Kmeans Converged after", iterations, "iterations")
    i = 0

    #Prepend Categories for Plotting the Data
    for elem in clustered_points:
        categories[i][0] = elem[0]
        i = i + 1
    my_data = np.hstack((categories, my_data))

    #Minimal Intercluster Distance
    min_inter = np.inf
    for i in range(0, N):
        elem = my_data[i]
        for j in range(i+1, N):
            if(not my_data[j][0] == elem[0]):
                if(dist_matrix[i][j] < min_inter):
                    min_inter = dist_matrix[i][j]
    print("Minimum Intercluster Distance:", min_inter)

    #Maximal Intracluster Distance
    max_intra = -np.inf
    for i in range(0, N):
        elem = my_data[i]
        for j in range(i+1, N):
            if(my_data[j][0] == elem[0]):
                if(dist_matrix[i][j] > max_intra):
                    max_intra = dist_matrix[i][j]
    print("Maximum Intracluster Distance:", max_intra)

    #Dunn Index
    dunn = min_inter / max_intra
    print("Dunn index (Min Intercluster Distance / Max Intracluster Dist):", dunn)

    # Print metrics
    P, N, TP, TN, FP, FN, accuracy, TPR, PPV, TNR, F = getPerformanceMetrics(labels, my_data[:,0])
    print('Training Set Performance Metrics:')
    print('Accuracy: ' + "{0:.2f}".format(round(accuracy,2)) + '%')
    print('True Positives: ' + str(TP))
    print('True Negatives: ' + str(TN))
    print('False Positives: ' + str(FP))
    print('False Negatives: ' + str(FN))
    print('True Positive Rate (sensitivity): ' + "{0:.2f}".format(round(TPR,2)) + '%')
    print('Positive Prediction Value (precision): ' + "{0:.2f}".format(round(PPV,2)) + '%')
    print('True Negative Rate (specificity): ' + "{0:.2f}".format(round(TNR,2)) + '%')
    print('F1 Score: ' + "{0:.2f}".format(round(F,2)) + '%')

    #Save output Files
    filename = "output/Data_Kmeans_" + str(k) + ".txt"
    np.savetxt(filename, my_data, delimiter=',', fmt='%3.4f')






    # <---Run on Testing Data--->
    print("Performing K-Means on Testing Dataset for", k, "clusters")

    #Delete first row: header
    testing_data = np.delete(testing_data, 0, 0)
    #Delete first column: ID
    testing_data = np.delete(testing_data, 0, 1)
    #Collect labels and delete label column (last column)
    test_labels = testing_data[:,-1].copy()
    testing_data = np.delete(testing_data, -1, 1)

    # Create RDD from data
    sc = SparkContext.getOrCreate()
    test_points = sc.parallelize(testing_data, numSlices=50)

    # Assign each point to the centroid closest to it
    test_clusters = assignPointsToClosestCluster(test_points, centroids).collect()

    #Prepend Categories for Plotting the Data
    N = np.size(testing_data, 0)
    cat = np.zeros((N,1), dtype=int)
    i = 0
    for elem in test_clusters:
        cat[i][0] = elem[0]
        i = i + 1
    testing_data = np.hstack((cat, testing_data))

    #Minimal Intercluster Distance
    min_inter = np.inf
    for i in range(0, N):
        elem = testing_data[i]
        for j in range(i+1, N):
            if(not testing_data[j][0] == elem[0]):
                if(dist_matrix[i][j] < min_inter):
                    min_inter = dist_matrix[i][j]
    print("Minimum Intercluster Distance:", min_inter)

    #Maximal Intracluster Distance
    max_intra = -np.inf
    for i in range(0, N):
        elem = testing_data[i]
        for j in range(i+1, N):
            if(testing_data[j][0] == elem[0]):
                if(dist_matrix[i][j] > max_intra):
                    max_intra = dist_matrix[i][j]
    print("Maximum Intracluster Distance:", max_intra)

    #Dunn Index
    dunn = min_inter / max_intra
    print("Dunn index (Min Intercluster Distance / Max Intracluster Dist):", dunn)

    # Print metrics
    P, N, TP, TN, FP, FN, accuracy, TPR, PPV, TNR, F = getPerformanceMetrics(test_labels, testing_data[:,0])
    print('Testing Set Performance Metrics:')
    print('Accuracy: ' + "{0:.2f}".format(round(accuracy,2)) + '%')
    print('True Positives: ' + str(TP))
    print('True Negatives: ' + str(TN))
    print('False Positives: ' + str(FP))
    print('False Negatives: ' + str(FN))
    print('True Positive Rate (sensitivity): ' + "{0:.2f}".format(round(TPR,2)) + '%')
    print('Positive Prediction Value (precision): ' + "{0:.2f}".format(round(PPV,2)) + '%')
    print('True Negative Rate (specificity): ' + "{0:.2f}".format(round(TNR,2)) + '%')
    print('F1 Score: ' + "{0:.2f}".format(round(F,2)) + '%')

    #Save output Files
    filename = "output/TestingData_Kmeans_" + str(k) + ".txt"
    np.savetxt(filename, testing_data, delimiter=',', fmt='%3.4f')


main()
