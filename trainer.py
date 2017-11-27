import random

class Trainer:
	def __init__(self, trainingSet):
		self.trainingSet = trainingSet

	def train(self, network):
		errorSequence = []
		for i in range(500):
			errorSequence.append( self.epoch(network) )

			if self.breakCondition(errorSequence):
				break

		return errorSequence

	# returns True if we should break (stop training), False otherwise
	def breakCondition(self, errorSequence):
		if len(errorSequence) < 40:
			return False

		recentAverage = 0
		for x in errorSequence[-20:]:
			recentAverage += x
		recentAverage = recentAverage / 20

		pastAverage = 0
		for x in errorSequence[-40:-20]:
			pastAverage += x
		pastAverage = pastAverage / 20

		improvement = pastAverage - recentAverage
		return improvement < .01

	def epoch(self, network):
		totalError = 0
		unusedExamples = list(self.trainingSet)
		for i in range(len(self.trainingSet)):
			selection = random.randint(0, len(unusedExamples) - 1)
			trainingExample = unusedExamples.pop(selection)
			totalError += network.learn(trainingExample['inputs'], trainingExample['label'])
		return totalError / len(self.trainingSet)