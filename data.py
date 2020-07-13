#!/usr/bin/python3
"""
Name: Dustin Mcafee
Standardize and Project data to Principal Components
"""

import csv
import numpy as np
import math
import sys
import os

# This is misleading, as this function does not load data: It splits data.
def loadData(dataset, split):
	#Copy the data as to not randomize the original set
	data = dataset.copy()
	np.random.shuffle(data)

	#Split the randomized dataset into a training dataset and a testing dataset
	valid, train = data[:split,:], data[split:,:]
	return train, valid

#This function loads the data from file.
def loadData2(filename):
	if(os.path.exists(filename)):
		data = np.genfromtxt(filename, delimiter=',')
	else:
		print("Second argument must be a valid input Training Dataset (numerical, csv) file with header row, first column is index column, and last column categorical (discrete-numerical)")
		sys.exit()
	return data

#Clean the data (Impute nan rows)
def imputeNAN(data, array):
	row_it = 0
	for row in data:
		col_it = 0
		num_imput = 0
		for elem in row:
			if(np.isnan(elem)):
				data[row_it, col_it] = array[col_it]
				num_imput = num_imput + 1
			col_it = col_it + 1
		if(num_imput > 0):
			print(num_imput, "Imputed in row", row_it)
		row_it = row_it + 1
	return data

#Standardize data
def standardize(data, mean):
	#Variables
	N = np.size(data, 0)
	dimensions = np.size(data, 1)

	# Center data around sample mean
	center_x = data.copy()
	center_x -= mean

	# Covariance matrix
	cov = (np.dot(center_x.T, center_x.conj()) / N).squeeze()

	# Standardize Data (z-normalize)
	D = np.sqrt(np.diag(cov))
	s = (dimensions, dimensions)
	dev = np.zeros(s)    #Standard Deviation Values
	i = 0
	for elem in D:
		dev[i,i] = elem
		i = i + 1

	DInv = np.linalg.inv(dev)
	standard_data = np.matmul(center_x, DInv).astype(float)

	return (cov, standard_data)


def main():
	PC = int(sys.argv[1])
	if(not len(sys.argv) == 3):
		return
	if(PC < 1):
		print("First input argument must be the number of Principal Components to project data onto")
		return

	my_data = loadData2('input/data.csv')

	#Delete first column: ID
	my_data = np.delete(my_data, 0, 1)
	#Delete header row
	my_data = np.delete(my_data, 0, 0)

	#Collect labels and delete label column (last column)
	dimensions = np.size(my_data, 1)
	labels = my_data[:,-1].copy()
	my_data = np.delete(my_data, -1, 1)
	for i in range(len(labels)):
#		labels[i] -= 1
		if(not labels[i] == 1):
			labels[i] = 0

	#Variables
	N = np.size(my_data, 0)
	labels = labels.reshape(N,1)
	dimensions = np.size(my_data, 1)
	print(dimensions, "Dimensions")
	print(N, "Observations")
	arr = np.array([])
	for i in range(N):
		arr = np.append(arr, i)
	arr = arr.reshape(N,1)

	#impute missing data
	mean = np.nanmean(my_data, axis=0)
	my_data = imputeNAN(my_data, mean)

	#Standardize (Z-Normalize) the Data
	cov, standard_data = standardize(my_data, mean)
	my_data = standard_data.copy()

	#Singular Value Decomposition
	u, s, vh = np.linalg.svd(my_data, full_matrices=True)
	ss = np.square(s)

	#Percentage of Variance for each k PC
	ss_sum = np.sum(ss)
	ss_percent = np.divide(ss,ss_sum) * 100
	sum = 0
	for i in range(0, PC):
		sum = sum + ss_percent[i]
		
	print("First", PC, "PC covers", sum, "percent of the Variance")

	#Project to first PCs
	v = np.transpose(vh)
	rang = range(0,PC)
	v_pc = v[:, rang]
	data_pc = np.matmul(my_data, v_pc)

	#Save output Files: Header row (numbered), First column ID, Last column Category Label
	head = np.array([])
	for i in range(dimensions+2):
		j = i
		if(j > 0):
			j = j - 1
		head = np.append(head, j)
	head = head.reshape(1,dimensions+2)
	dat_stand = np.vstack((head, np.hstack((arr, np.hstack((my_data, labels))))))
	np.savetxt("output/Data_Standardized.txt", dat_stand, delimiter=',', fmt='%3.4f') 
	np.savetxt("output/Singular_Values.txt", s, delimiter=',', fmt='%3.4f')
	np.savetxt("output/Singular_Values_Percent_Variance.txt", ss_percent, delimiter=',', fmt='%3.4f')
	head = np.array([])
	for i in range(PC+2):
		j = i
		if(j > 0):
			j = j - 1
		head = np.append(head, j)
	head = head.reshape(1,PC+2)
	dat_proj = np.vstack((head, np.hstack((arr, np.hstack((data_pc, labels))))))
	np.savetxt("output/Data_Project.txt", dat_proj, delimiter=',', fmt='%3.4f')

	# Load or generate training/testing data
	if(int(sys.argv[2]) == 1):
		trainData, testData = loadData(dat_proj, 2700)
		# Print Dataset sizes
		print(np.size(trainData, 1) - 2, "Dimensions")
		print(np.size(trainData, 0), "Triaining Set Observations")
		print(np.size(testData, 0), "Test Set Observations")
		np.savetxt("input/train/TrainingData_Projected.txt", np.matrix(trainData), delimiter=',', fmt=' '.join(['%3.4f,'] * (np.size(trainData, 1) - 1) + ['%d']))
		np.savetxt("input/test/TestingData_Projected.txt", np.matrix(testData), delimiter=',', fmt=' '.join(['%3.4f,'] * (np.size(testData, 1) - 1) + ['%d']))




main()
