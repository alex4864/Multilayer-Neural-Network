import random

class Trainer:
	def __init__(self, trainingSet):
		self.trainingSet = trainingSet

	def train(self, network):
		errorSequence = []
		for i in range(100):
			errorSequence.append( self.epoch(network) )
		return errorSequence

	def epoch(self, network):
		error = 0
		unusedExamples = list(self.trainingSet)
		for i in range(len(self.trainingSet)):
			selection = random.randint(0, len(unusedExamples) - 1)
			trainingExample = unusedExamples.pop(selection)
			error += network.learn(trainingExample['inputs'], trainingExample['label'])
		return error