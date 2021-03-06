import numpy as np
import random

# coordinates of clusters, and labels to be applied to each coordinate
def generate_data(clusters, labels, spread, pointsPerCluster):

	data = []
	for i in range(0, len(clusters)):
		for j in range(0, pointsPerCluster):
			newPoint = generate_data_point(clusters[i], labels[i], spread)
			data.append(newPoint)

	return data

# assembles a data point based on a starting coordinate, label, and spread from starting coordinate
def generate_data_point(baseCoord, label, spread):
	coord = np.array([])
	for value in baseCoord:
		newVal = value + (random.random() * spread) - (spread / 2)
		coord = np.append(coord, newVal)
	point = {'inputs': coord, 'label': [label]}
	return point