import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from network import Network
from trainer import Trainer
from generate_data import generate_data
from generate_graph import generate_shallow_network

def label_to_color(label):
	red = (label + 1) / 2
	green = 0
	blue = 1 - (label + 1) / 2
	alpha = 1
	return [red, green, blue, alpha]

def plot_data(data):
	xCoords = []
	yCoords = []
	labels = []
	for point in data:
		xCoords.append(point['inputs'][0])
		yCoords.append(point['inputs'][1])
		labels.append(point['label'][0])

	colors = []
	for label in labels:
		colors.append(label_to_color(label))

	plt.scatter(xCoords, yCoords, c=colors)

def plot_network(network):
	x = y = np.arange(-3.0, 3.0, 0.05)
	X, Y = np.meshgrid(x, y)
	Z = np.array([network.evaluate([x, y]) for (x,y) in zip(X.ravel(), Y.ravel())]).reshape(X.shape)

	levels = [0]

	plt.contour(X, Y, Z, levels, colors='black')

clusters = [[1, 1], [-1, -1], [1, -1], [-1, 1], [2, 0], [-2, 0], [-2, 2]]
labels = [-1, -1, 1, 1, -1, 1, -1]

data = generate_data(clusters, labels, 1, 20)
G = generate_shallow_network(2, 8, 1)
net = Network(G)
netTrainer = Trainer(data)

error = netTrainer.train(net)

plt.figure(1)
plot_data(data)
plot_network(net)
plt.xlabel('x1')
plt.ylabel('x2')

plt.figure(2)
plt.plot(error)

plt.show()