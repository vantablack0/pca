import numpy as np
from matplotlib import pyplot
import sys

with open(str(sys.argv[1]),"r") as data:
    data_file = data.readlines()

for iter in range(len(data_file)):
    data_file[iter] = data_file[iter].strip().split(',')

data_file = [[float(data_file[i][j]) for j in range(len(data_file[i]))] for i in range(len(data_file))]


data_matrix = np.array(data_file)
mean = data_matrix.mean(axis=0)
std = data_matrix.std(axis=0)

rows,cols = data_matrix.shape

for row in range(rows):
    for col in range(cols):
        data_matrix[row][col] = (data_matrix[row][col]-mean[col])/std[col]

cov_matrix = np.cov(data_matrix.T)

eigenvalues,eigenvectors = np.linalg.eig(cov_matrix)

eigenvectors = eigenvectors[0:2]
eigenvectors = eigenvectors.T

transformed_data = data_matrix @ eigenvectors

x_data = list(transformed_data[:,0])
y_data = list(transformed_data[:,1])

figure = pyplot.figure()
pyplot.scatter(x_data,y_data)
pyplot.xlim(-15,15)
pyplot.ylim(-15,15)

figure.savefig("out.png")