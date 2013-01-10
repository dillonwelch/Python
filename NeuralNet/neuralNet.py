# Dillon Welch
# 101-93-657
# CSC 475

# Library imports
import datetime
import math
import random
import re
import os
import time
import webbrowser

# Constants
amountOfBits = 5 				# Amount of bits to check parity for
epsilon = 0.03 					# If the value is within this threshold of 0 or 1, round it.
initialWeightsLowerBound = -1	# Lower bound on the random number generation.
initialWeightsUpperBound = 1	# Upper bound on the random number generation.
roundingThreshold = 0.4

generatingResults = False
printouts = False				# Whether or not to printout various debug info.
verbosePrintouts = False		# Whether or not to print out all debug info.

table1 = 1						# Use Table 1 Data.
full = 2						# Use Full Training Data.

# Character Constants
lineSeparator = "-----" 												# Divider between printout sections.
newline = "\n"															# Newline character.
noList = ['n','N','no','NO','No','nO'] 									# List of possible no combinations.
questionList = ['?'] 													# List of possible question mark combinations.
separator = ','															# Separator of values in training data file.
tab = "\t"																# Tab character.
yesList = ['y','Y','yes','Yes','yEs','yeS','YEs','YeS','yES','YES']		# List of possible yes combinations.

# Printout Strings.	
defaultWeightsFileName = "Weights.txt"						# Default name for the weights file.
defaultTestingFileName = "Testing.txt"						# Default name for the testing data file.
defaultTrainingFileName = "Test {0} Training Data.txt"		# Default name for the training data file.
fileParseError = "There was an error while parsing the file %s in %s. Exiting the program."
noInputOutputString = newline + "The amount of input and output nodes are not allowed to be changed currently, as they are constants for this assignment at 5 and 1 respectively. " + newline
unrecognizedInput = newline + "Unrecognized input."
trainingStartedString = "Training Started. . ." + newline
trainingDoneString =  "Training Done!" + newline + newline + "Testing. . . " + newline
yesNoString = "Input 'Y' for yes or 'N' for no:"

# Default NN settings
defaultEta = 0.4				# Default value of eta.
defaultMaxIterations = 2000		# Default max iterations.
defaultInputNodesCount = 5		# Default input nodes.
defaultHiddenLayersCount = 1	# Default hidden layers.
defaultHiddenNodesCount = [5]	# Default amount of nodes per hidden layer.
defaultOutputNodesCount = 1		# Default output nodes.

# TODO Add threshold to output

# Classes
# Note: In Python, all constructors and functions in a class have to have the parameter "self" as the first parameter. This is a reference to the instance of the class that is calling the function. It is automatically passed when the function is called.
# Note: __init__ is Python's version of a constructor.
# Note: range(a, b) iterates through each integer of the range [a, b).
class NeuralNet:	
	def backPropogate(self, overallInputList, row):
		''' Does the back propogation part of the algorithm. '''
		
		# List of all the calculated deltas, with the first element being the output deltas
		# and each following element being the deltas of the previous hidden layer.
		# So if there are 2 hidden layers, this will be [[output deltas], [hidden layer 2 deltas], [hidden layer 1 deltas]].
		overallDeltaList = [] 
		expectedValue = row[1] # The actual answer for this row of data.
		
		# Calculate the output deltas.
		outputDeltas = []
		for i in range(0, self.outputNodesCount):
			# Output Delta = Output * (1 - output) * (Actual value - output)
			outputDeltas.append(overallInputList[-1][i] * (1 - overallInputList[-1][i]) * (expectedValue - overallInputList[-1][i]))
		overallDeltaList.append(outputDeltas)
			
		# Calculate the deltas for each hidden layer, starting from the last and going towards the first.
		for i in range(0, self.hiddenLayersCount):
			deltaHList = []
			# Let h stand for the current node.
			# Let k be each of the next layer's nodes.
			# Delta(h) = Output(h) * (1 - output(h)) * Sum(Weight from h to each k * Delta(k))
			for j in range(0, self.hiddenNodesCount[self.hiddenLayersCount - 1 - i]):
				outputH = sigmoid(overallInputList[-3 - i][j])
				sum = 0
				for k in range(0, len(overallDeltaList[i])):
					sum += overallDeltaList[i][k] * self.weights[-1 - i][j][k]
				deltaHList.append(outputH * (1 - outputH) * sum)
				
			overallDeltaList.append(deltaHList)
			
		# Calculate the change in weights for all of the weights in the NN.
		for i in range(0, len(self.weights)):
			for j in range(0, len(self.weights[i])):
				for k in range(0, len(self.weights[i][j])):
					# If the ith layer is the input layer, use the input value for the current node as input.
					if i == 0:
						inputVal = row[0][j]
					# Otherwise, use the output of the ith layer as input.
					else:
						inputVal = sigmoid(overallInputList[i - 1][j])
						
					# Debug printing
					if printouts and verbosePrintouts:
						print overallInputList
						print "Self.weights[%d][%d][%d]: %f" % (i, j, k, self.weights[i][j][k])
						print "Self.eta: %f" % self.eta
						print "Delta[%d][%d]: %f" % ((-1 - i), k, overallDeltaList[-1 - i][k])
						print "Input: %f" % inputVal
						
					# Calculate the change in weight for the jth node of the ith layer 
					# going to the kth element of the (i + 1)th layer.
					# This is calculated by Weight(j)(k) = Weight(j)(k) + (Learning rate * Delta(k) * Input into the kth element.
					self.weights[i][j][k] += self.eta * overallDeltaList[-1 - i][k] * inputVal
					
					# Debug printing.
					if printouts and verbosePrintouts:
						print "New self.weights: %f" % self.weights[i][j][k]
						print newline
		
	def exportWeights(self):
		''' Export the weights into a file named by the current time in seconds since epoch. '''
		
		fileName = "Weights at " + str(time.mktime(time.localtime())) + ".txt"

		with open(fileName, 'w+') as f:
			f.write(str(self.inputNodesCount) + newline)
			f.write(str(self.hiddenLayersCount) + newline)
			for i in range(0, self.hiddenLayersCount):
				f.write(str(self.hiddenNodesCount[i]) + newline)
			f.write(str(self.outputNodesCount) + newline)
			for i in range(0, len(self.weights)):
				for j in range(0, len(self.weights[i])):
					string = ""
					for k in range(0, len(self.weights[i][j])):
						string += str(self.weights[i][j][k])
						if k + 1 < len(self.weights[i][j]):
							string += ", "
					string += newline
					f.write(string)
		
	def feedForward(self, row):
		''' Takes a row of input and feeds it forward through the NN, returning the result. '''
		
		# This will contain all of the input for each of the hidden layers and the output layer, plus the output of the output layer.
		overallInputList = [] 
		
		# For each of the hidden nodes in the first hidden layer, calculate the input values.
		input = 0
		inputList = []
		for i in range(0, self.hiddenNodesCount[0]):
			input = 0
			# The input value is the summation of the input node's value * the weight from that node to the current node.
			for j in range(0, self.inputNodesCount):
				# Weights from the input layer from node j to node i in the first hidden layer.
				input += row[0][j] * self.weights[0][j][i] 
			inputList.append(input)
		overallInputList.append(inputList)
		
		# For each of the rest of the hidden layers, calculate their input values.
		for i in range(1, self.hiddenLayersCount):
			input = 0
			inputList = []
			# For each of the hidden nodes of the current hidden layer, calculate their input value.
			for j in range(0, self.hiddenNodesCount[i]):
				# The input value is the summation of the previous hidden layer node's output * the weight from that node to the current node.
				for k in range(0, self.hiddenNodesCount[i - 1]):
					# Weights from the previous hidden layer from node k to node j in the current hidden layer.
					input += sigmoid(overallInputList[i - 1][k]) * self.weights[i][k][j]
				inputList.append(input)
			overallInputList.append(inputList)
		
		# For each node in the output layer, calculate their input values.
		input = 0
		inputList = []	
		for i in range(0, self.outputNodesCount):
			# The input value is the summation of the final hidden layer node's output * the weight from that node to the current output node.
			for j in range(0, self.hiddenNodesCount[-1]):
				# Weights from the final hidden layer from node j to node i in the output layer.
				input += sigmoid(overallInputList[-1][j]) * self.weights[-1][j][i]
			inputList.append(input)
		overallInputList.append(inputList)
		
		# Calculate the output of each of the nodes in the output layer
		inputList = []
		for i in range(0, len(overallInputList[-1])):
			# The output is just the result of the input plugged into the sigmoid function.
			inputList.append(sigmoid(overallInputList[-1][i])) 	
			
		overallInputList.append(inputList)
		
		return overallInputList
		
	def initializeWeights(self):
		''' Creates a list with an element for each list of weights.
			For example, if the NN is created with 2 hidden layers, the empty list will be [[], [], []]
			As there are weights from the input to the first hidden layer (The first element in the list),
			weights from hidden layer one to hidden layer two (The second element in the list), and
			weights from hidden layer two to the output layer.
			Each element will itself have as many elements as its current layer (an input layer of 5 would result in the first list element having 5 elements)
			and each of those has the weights wij to each node in the next layer. Confusing, I know.
		'''
		
		# Initialize the list.
		self.weights = []
		self.weights.append([])
		for i in range(1, self.hiddenLayersCount):
			self.weights.append([])
		self.weights.append([])
		
		# Set the random seed to the current time, so each time the weights are initialized in a program run they are unique.
		random.seed(datetime.datetime.now())
		
		# For each node in the input layer, create a list of weights to the first hidden input layer.
		for i in range(0, self.inputNodesCount):
			w = []
			# For each node in the first hidden layer, create a weight.
			for j in range(0, self.hiddenNodesCount[0]):
				w.append(random.triangular(initialWeightsLowerBound, initialWeightsUpperBound)) # a <= num <= b
			# Add the list of weights
			self.weights[0].append(w)
		
		# For each node in each hidden layer except the last layer, create a list of weights to the next hidden layer.
		for i in range(1, self.hiddenLayersCount):
			# For each node in the current hidden layer, create the list of weights to the next hidden layer.
			for j in range(0, self.hiddenNodesCount[i - 1]):
				w = []
				# For each node in the next hidden layer, create a weight.
				for k in range(0, self.hiddenNodesCount[i]):
					w.append(random.triangular(initialWeightsLowerBound, initialWeightsUpperBound))
				# Add the list of weights.
				self.weights[i].append(w)
			
		# For each node in the last hidden layer, create a list of weights to the output layer.
		for i in range(0, self.hiddenNodesCount[-1]):
			w = []
			# For each node in the output layer, create a weight
			for j in range(0, self.outputNodesCount):
				w.append(random.triangular(initialWeightsLowerBound, initialWeightsUpperBound))
			# Add the list of weights.
			self.weights[-1].append(w)
	
	def printValues(self):
		''' Prints out the values that define the neural net. '''

		print "Eta: %f" % self.eta
		print "Amount of input nodes: %d" % self.inputNodesCount
		print "Amount of hidden layers: %d" % self.hiddenLayersCount
		i = 1
		for count in self.hiddenNodesCount:
			print "Amount of nodes in hidden layer %d: %d" % (i, count)
			i += 1
		print "Amount of output nodes: %d" % self.outputNodesCount
		print "Amount of iterations: %d" % self.iterations
		
	def printWeights(self):
		''' Prints out all the weights of the neural net. '''
		
		# Prints the weights from the input layer to the first hidden layer
		print "Weights from input to hidden layer 1" + newline + lineSeparator + newline
		for i in range(0, len(self.weights[0])):
			print "Node %d" % (i + 1)
			for j in range(0, len(self.weights[0][i])):
				print "%sTo Hidden Layer %d Node %d: %f" % (tab, 1, (j + 1), self.weights[0][i][j])
				
		print newline
		
		# Prints out the weights of each hidden layer (except the last one) to the next hidden layer.
		for i in range(1, self.hiddenLayersCount):
			print "Weights from hidden layer %d to hidden layer %d%s%s%s" % (i, i + 1, newline, lineSeparator, newline)
			# Prints out the weights of the current hidden layer to the next one.
			for j in range(0, len(self.weights[i])):
				print "Node %d" % (j + 1)
				for k in range(0, len(self.weights[i][j])):
					print "%sTo Hidden Layer %d Node %d: %f" % (tab, i + 1, (k + 1), self.weights[i][j][k])
				print newline
		
		# Prints out the weights of the last hidden layer to the output layer. 
		print "Weights from hidden layer %d to output layer%s%s%s" % (self.hiddenLayersCount, newline, lineSeparator, newline)
		for i in range(0, len(self.weights[-1])):
			print "Node %d" % (i + 1)
			for j in range(0, len(self.weights[-1][i])):
				print "%sTo Output Layer Node %d: %f" % (tab, (j + 1), self.weights[-1][i][j])
			print newline
		
	def train(self):
		''' Train the NN with the given training data. '''
		
		inputFileList = readDataFile(self.trainingFile)
		for count in range(0, self.maxIterations):
			# Debug printing.
			if printouts and count != 0 and count % 1000 == 0:
				print count
				
			iter = 0
			# For each row in the training data, feed the input forward and calculate the change in weights by backpropogation.
			for row in inputFileList:			
				overallInputList = self.feedForward(row)
				self.backPropogate(overallInputList, row)
				
				# Debug printing.
				if printouts and count % 1000 == 0:
						print "At Count %d and Row %d, output was %f" % (count + 1, iter + 1, round(overallInputList[-1][0]))

				iter += 1
				
			self.iterations = count + 1
		
	def test(self):
		''' Test the NN with the given testing data. '''
		
		# Printout strings
		rowString = "Row %d: "
		testResultsString = "Expected output is %d and actual output is %f."
		autoTestString = rowString + testResultsString
		
		# If manual testing is turned off, load the testing data from the file and test each row.
		if not self.manualTesting:
			inputFileList = readDataFile(self.testingFile)
			iter = 0
			# For each row in the testing data, feed the input into the NN and show the result.
			for row in inputFileList:
					overallInputList = self.feedForward(row)
					
					print autoTestString % (iter + 1, row[1], round(overallInputList[-1][0]))
					iter += 1
		else: # Manual testing.
			# Printout strings.
			inputBin = "Please input a 5 digit binary number: "
			testAnother = "Would you like to test another number? " + yesNoString
			invalidLengthNumber = "This is not a valid length number." + newline
			nonBinaryNumber = "This is not a binary number." + newline
			doneWithTestingString = "Done with testing." + newline
			
			# Test the input value in the NN to see its results.
			repeat = True
			while repeat:
				# First, get the number and ensure it is valid.
				input = myRawInput(inputBin)
				validInput = False
				while not validInput:
					if not amountOfBits - 1 < len(input) < amountOfBits + 1:
						print invalidLengthNumber
						input = myRawInput(inputBin)
					result = binaryCheck(input)
					if not result:
						print nonBinaryNumber
						input = myRawInput(inputBin)
					else:
						validInput = True
				
				# Second, test it in the NN and print the results.
				row = [testingInputConversion(input), parityCheck(input)]
				
				overallInputList = self.feedForward(row)
				print testResultsString % (row[1], round(overallInputList[-1][0]))
				
				print newline
				
				# Third, check to see if the user wishes to do another test.
				input = myRawInput(testAnother)
				validInput = False
				while not validInput:
					if input in yesList:
						validInput = True
					elif input in noList:
						print doneWithTestingString
						validInput = True
						repeat = False
					else:
						print unrecognizedInput
						input = myRawInput(testAnother)
						
		self.exportWeights()
			
	def __init__ (self, eta = defaultEta, hiddenLayersCount = defaultHiddenLayersCount, hiddenNodesCount = defaultHiddenNodesCount, inputNodesCount = defaultInputNodesCount, outputNodesCount = defaultOutputNodesCount):
		''' learning rate, amount of input nodes, the value of each input node, amount of hidden layers, amount of nodes in each hidden layers, amount of output nodes '''

		self.eta = eta
		self.inputNodesCount = inputNodesCount
		self.hiddenLayersCount = hiddenLayersCount
		self.hiddenNodesCount = hiddenNodesCount
		self.outputNodesCount = outputNodesCount
		
		self.iterations = 0
		self.manualTesting = False
		self.maxIterations = defaultMaxIterations
		
# Functions
def amountOfDigits(str):
	''' Returns the amount of digits in a numeric string '''
	
	if str.isdigit() is False:
		return 0
	else:
		return len(list(str))
	
def binaryCheck(s):
	''' Test whether an input string is a binary number. '''
	
	for i in range(0, len(s)):
		if s[i] is not '0' and s[i] is not '1':
			return False
	return True
	
def defaultTest():
	''' Do a test on the NN with the default settings. '''
	
	nn = NeuralNet()
	nn.initializeWeights()
	nn.trainingFile = defaultTrainingFileName.format(full)
	nn.testingFile = defaultTestingFileName
	nn.maxIterations = 10
	
	print newline
	print trainingStartedString
	nn.train()
	print trainingDoneString
	nn.test()
	nn.exportWeights()
	
def testingInputConversion(s):
	''' Converts a binary string to a list of data. '''
	
	data = []
	for i in range(0, len(s)):
		data.append(int(s[i]))
	return data
	
def myRawInput(printout):
	''' My wrapper for raw_input() that handles exceptions '''
	
	try:
		result = raw_input(printout)
	except:
		print newline + "There was an issue with the input. The program will close now."	
		exit()
		
	if result == "exit()":
		print "Exiting the program!"
		exit()
	elif result == "42":
		print "This is not a Douglas Adams novel."
		exit()
	elif result == "import skynet":
		print "Soon..."
		exit()
	elif result == "import antigravity":
		webbrowser.open("http://xkcd.com/353/")
		exit()
	elif result == "help":
		printNeuralNetworkHelp()
		exit()
	elif result == "defaults":
		printGivenSettings()
		exit()
	elif result == "xor":
		xorTest()
		exit()
	elif result == "hack":
		defaultTest()
		exit()
	else:
		return result
		
def parityCheck(s):
	''' Test the parity of an input string. '''
	
	amountOfOnes = 0
	for i in range(0, len(s)):
		if s[i] == '1':
			amountOfOnes += 1
			
	return amountOfOnes % 2
		
def prettyPrintList(list):
	''' Prints out the items in a list '''

	i = 1
	maxDigits = amountOfDigits(str(len(list))) # The amount of digits in the number that is the size of the list
	for item in list:
		# Add in extra space if necessary so all the lines print out in the same column.
		prettify = ''
		digits = amountOfDigits(str(i))
		if(digits < maxDigits):
			prettify = " " * (maxDigits - digits)
		
		# Print the current item in the list
		print "Item %d: %s%s" % (i, prettify, item)
		i += 1

def printGivenSettings(eta = defaultEta, hiddenLayersCount = defaultHiddenLayersCount, hiddenNodesCount = defaultHiddenNodesCount, maxIterations = defaultMaxIterations):
	''' Print the given NN settings. '''
	
	print newline + "Settings" + newline + lineSeparator + newline
	print "Eta: %f" % eta
	print "Input Nodes: %d" % defaultInputNodesCount
	print "Hidden Layers: %d" % hiddenLayersCount
	for i in range(1, hiddenLayersCount + 1):
		print "Hidden Layer %d Nodes: %d" % (i, hiddenNodesCount[i - 1])
	print "Output Nodes: %d" % defaultOutputNodesCount	
	print "Max Iterations: %d" % maxIterations
	
def printNeuralNetworkHelp():
	print newline + newline
	print "Neural Network Help" + newline +  lineSeparator + newline + newline
	print "Eta is the learning rate of the neural network (NN). Setting this appropriately is like \"cooking rice\" if you will."
	print "Too high, and the NN will not train properly (the rice will burn). Too low, and the NN will take forever to train (the rice takes forever to cook)."
	print "It is suggested to keep this value between 0.3 and 0.5."
	print newline
	print "The amount of hidden layers is a factor in how well your NN trains. Each layer can have a different amount of nodes."
	print "Having 1 layer with 4 to 5 nodes has worked well."
	print "The amount of input and output nodes are not allowed to be changed currently, as they are constants for this assignment at 5 and 1 respectively."
	print newline
	
def readDataFile(trainingFileName):
	''' Reads in the training file '''
	
	try:
		# Open the file
		with open(trainingFileName, 'r') as f:
			lines = f.readlines() # Read all the lines efficiently.
			trainingData = [] # List to store each row of the training data.
			
			# Take in the training data line by line.
			# For all the examples given, row 1 of Table 1 Training Data is used. This row is "00001" as input with a correct answer of "1".
			# The row in the training data file looks like "0,0,0,0,1,1" and once the file is read it will look like [[0,0,0,0,1],1] in my program.
			for line in lines:
				trainingDataRow = [] # List for the entire row's data as [list of inputs, correct answer]. For example: [[0,0,0,0,1],1].
				trainingDataInput = [] # List for the input values. For example: [0,0,0,0,1]
				# Strip the newlines and split the line by the separator, in this case ',' (returns a list of each number in the row of training data. For example: ['0','0','0','0','1','1']).
				rawTrainingData = line.strip().split(separator) 

				# Create the list of the input values. For example: [0,0,0,0,1]
				for token in rawTrainingData[0:amountOfBits]: # The data from 0 <= k < amountOfBits. So if amountOfBits is 5, it will get the first five items of the row.
					trainingDataInput.append(int(token))

				# The input list is complete, add it to the training data list and then add the correct answer as well.
				trainingDataRow.append(trainingDataInput)
				trainingDataRow.append(int(rawTrainingData[amountOfBits]))
			
				# Add the row to the overall list of training data.
				trainingData.append(trainingDataRow)
			
			# Return the list of training data.
			return trainingData
	except:
		print fileParseError % (trainingFileName, "readDataFile()")
		
def readWeightsFile(weightsFileName):
	''' Reads in a weights file '''

	# Example: 
	#[[[input node 1 to hidden layer 1], [input node 2 to hidden layer 1]], [[hidden layer 1 node 1 to output layer], [hidden layer 1 node to to output layer]]]
	#[[[0.129952, -0.923123], [0.570345, -0.328932]], [[0.164732], [0.752621]]]
	
	try:
		# Open the file and parse the weights.
		with open(weightsFileName, 'r') as f:
			# Read in the data and clean it up.
			lines = f.readlines()
			lines = [line.strip().split(',') for line in lines]
			weights = [] 	# List to store the weights in.
			
			# First grab the stats about the NN.
			inputNodes = int(lines[0][0])
			amountOfHiddenLayers = int(lines[1][0])
			i = 2	# Counter for the line position in the file.
			hiddenNodesCount = []
			while i - 2 < amountOfHiddenLayers:
				hiddenNodesCount.append(int(lines[i][0]))
				i += 1
			outputNodes = int(lines[i][0])
			i += 1
			
			# Lists for the weights of each layer.
			inputLayerList = []
			hiddenLayersList = []
			outputList = []
			
			# Grab the weights going from the input layer to the first hidden layer.
			k = 0
			while k < inputNodes:
				inputLayerList.append([float(line) for line in lines[i]])
				k += 1
				i += 1
			weights.append(inputLayerList)
			
			# Grab the weights for all the hidden layers.
			k = 1
			while k < amountOfHiddenLayers:
				j = 0
				hiddenLayersList = []
				# Grab the weights for the current hidden layer.
				while j < hiddenNodesCount[k - 1]:
					hiddenLayersList.append([float(line) for line in lines[i]])
					j += 1
					i += 1
				k += 1
				weights.append(hiddenLayersList)
			
			# Grab the weights for the output layer.
			j = 0
			while j < hiddenNodesCount[-1]:
				outputList.append([float(line) for line in lines[i]])
				j += 1
				i += 1	
			weights.append(outputList)
			
			return weights
	except:
		print fileParseError % (weightsFileName, "readWeightsFile()")
		
def round(x):
	''' Check to see if a number should be rounded up to 1 or down to 0. '''
	if x >= 1 - epsilon:
		return 1
	elif x <= 0 + epsilon:
		return 0
	elif x < roundingThreshold:
		return 0
	else:
		return 1
		
def sigmoid(x):
	''' The sigmoid function '''
	if x > 100:
		return 1
	elif x < -100:
		return 0
	
	try:
		return 1/(1 + math.exp(-x))
	except:
		print "Math error!"
		print x
		exit()
		
def userInput():
	''' Deals with user input for training/testing the NN. '''
	
	halfTraining = False	# False = Use Table 1 Training data. True = Use Full Training Data.
	randomWeights = False	# False = Load weights from a file. True = Randomly generate weights.
	manualTesting = False	# False = Allow for manual entry of testing values. True = Use entries from a file.
	useDefaults = False		# False = Use custom NN settings. True = Use default NN settings.
	weightsFile = ""		# The name of the weights file to use ("" if randomly generating).
	testingFile = ""		# The name of the testing data file to use ("" if manually testing).
	eta = 0					# Value for eta (0 if using defaults).
	maxIterations = 0		# The max amount of iterations to train the NN on (0 if using defaults)
	hiddenLayersCount = 0	# Amount of hidden layers (0 if using defaults).
	hiddenNodesCount = []	# Amount of hidden nodes in each layer ([] if using defaults).
	
	# Printout Strings
	inputNotNumber = "Your input was not a number."
	inputNotPositiveNumber = "Your input was not a positive number."
	inputFile = "Input the name of the file: "
	fileNotFound = "The file %s does not exist."
	trainingDataString = "Input 'Y' to use Table 1 (half of the possible inputs) for training data, or 'N' full set of possible inputs: "
	weightsString = "Input 'Y' to load weights in from a file (Default: {0}). Input 'N' to have them be randomly generated: ".format(defaultWeightsFileName)
	testingString = "Input 'Y' to load in testing data from a file (Default: {0}). Input 'N' to manually test the NN by typing input: ".format(defaultTestingFileName)
	table1Yes = "You want to use Table 1!" + newline
	table1No = "You want to use the full set!" + newline
	randomWeightsString = "You want to randomly generate weights." + newline
	manualTestingString = "You want to do manual testing." + newline
	settingsString = "Input 'Y' to use your own NN node/layer settings. Input 'N' to use the default settings. Input '?' to display default settings: "
	etaInputString = "Input the value of eta: "
	iterationsString = "Input the amount of max iterations: "
	hiddenLayersString = "Input the amount of hidden layers, a number between 1 and 5 (inclusive): "
	hiddenLayersErrorString = "Your input was not a number between 1 and 5 (inclusive)."
	hiddenNodeString = "Input the amount of nodes for hidden layer %d: "
	correctValuesCheck = "Are these values correct?"
	repeatCustomInput = "Ok, let's do this again..."
	defaultSettingsString = "You want to use the default settings." + newline	
	
	# Ask whether to use Table 1 training data or full training data.
	training = myRawInput(trainingDataString)
	done = False
	while not done:
		if training in yesList:
			print table1Yes
			halfTraining = True
			done = True
		elif training in noList:
			print table1No
			done = True
		else:
			print unrecognizedInput
			training = myRawInput(trainingDataString)
		
	# Ask whether to load weights from a file or randomly generate weights.
	testing = myRawInput(weightsString)
	done = False
	while not done:
		if testing in yesList:
			weightsFile = myRawInput(inputFile)
			fileExists = False
			while not fileExists:
				try:
					with open(weightsFile) as f:
						fileExists = True
						done = True
						pass
				except:
					print fileNotFound % weightsFile + newline
					weightsFile = myRawInput(inputFile)
					
		elif testing in noList:
			print randomWeightsString
			randomWeights = True
			done = True
		else:
			print unrecognizedInput
			testing = myRawInput(weightsString)
	
	# Ask whether to do manual testing or load testing data from a file.
	testing = myRawInput(testingString)
	done = False
	while not done:
		if testing in noList:
			print manualTestingString
			manualTesting = True
			done = True
		elif testing in yesList:
			testingFile = myRawInput(inputFile)
			fileExists = False
			while not fileExists:
				try:
					with open(testingFile) as f:
						print newline
						fileExists = True
						done = True
						pass
				except:
					print fileNotFound % testingFile + newline
					testingFile = myRawInput(inputFile)
		else:
			print unrecognizedInput
			testing = myRawInput(testingString)
			
	# Ask whether to use custom NN settings or default settings 	.
	settings = myRawInput(settingsString)
	done = False
	while not done:
		if settings in questionList:
			printGivenSettings()
			settings = myRawInput(settingsString)
		elif settings in yesList:
			print noInputOutputString
			
			# Get the value of eta.
			correctEta = False
			while not correctEta:
				try:
					eta = float(myRawInput(etaInputString))
					correctEta = True
				except SystemExit:
					exit()
				except:
					print inputNotNumber
					
			# Get the amount of max iterations.
			print newline
			correctIterations = False
			while not correctIterations:
				try:
					maxIterations = int(myRawInput(iterationsString))
					correctIterations = True
				except SystemExit:
					exit()
				except:
					print inputNotNumber 
			
			# Get the amount of hidden layers.
			print newline
			correctLayers = False
			while not correctLayers:
				try:
					hiddenLayersCount = int(myRawInput(hiddenLayersString))
					if not 1 <= hiddenLayersCount <= 5:
						raise
					correctLayers = True
				except SystemExit:
					exit()
				except:
					print hiddenLayersErrorString
			
			# Get the amount of nodes in each layer.
			print newline
			for i in range(1, hiddenLayersCount + 1):
				correctNodes = False
				while not correctNodes:
					try:
						hiddenNodes = int(myRawInput(hiddenNodeString % i))
						if hiddenNodes < 1:
							raise
						correctNodes = True
					except SystemExit:
						exit()
					except:
						print inputNotPositiveNumber
						
				hiddenNodesCount.append(hiddenNodes)
			
			# Ask if these values are correct.
			print newline
			print correctValuesCheck
			printGivenSettings(eta, hiddenLayersCount, hiddenNodesCount)
			print newline
			input = myRawInput(yesNoString)
			doneAgain = False
			while not doneAgain:
				if input in yesList:
					doneAgain = True
					done = True
				elif input in noList:
					print newline
					print repeatCustomInput
					doneAgain = True
				else:
					print unrecognizedInput
					input = myRawInput(yesNoString)
					
			print newline
		elif settings in noList:
			print defaultSettingsString
			useDefaults = True
			done = True	
			
	# Make the NN with either default or custom settings.		
	if useDefaults:
		nn = NeuralNet(defaultEta, defaultHiddenLayersCount, defaultHiddenNodesCount)
	else:
		nn = NeuralNet(eta, hiddenLayersCount, hiddenNodesCount)
			
	# Set the training data file.
	if halfTraining:
		nn.trainingFile = defaultTrainingFileName.format(table1)
	else:
		nn.trainingFile = defaultTrainingFileName.format(full)
		
	# Set the testing data file.
	nn.manualTesting = manualTesting
	nn.testingFile = ""
	if not manualTesting:
		nn.testingFile = testingFile
		
	# Set the weights.
	if randomWeights:
		nn.initializeWeights()
	else:
		nn.weights = readWeightsFile(weightsFile)
		
	print trainingStartedString
	nn.train()
	print trainingDoneString
	nn.test()
			
def xorTest():
	''' Do a test on the NN with XOR. '''
	
	xorTraining = "XOR Training.txt"
	xorTesting = "XOR Testing.txt"
	xorWeights = "XOR Weights.txt"

	global amountOfBits
	amountOfBits = 2
	
	eta = 0.5
	hiddenLayersCount = 1
	hiddenNodesCount = [2]
	inputNodesCount = 2
	outputNodesCount = 1
	
	nn = NeuralNet(eta, hiddenLayersCount, hiddenNodesCount, inputNodesCount, outputNodesCount)
	nn.weights = readWeightsFile(xorWeights)
	nn.trainingFile = xorTraining
	nn.testingFile = xorTesting
	nn.maxIterations = 2000
	
	print newline
	print trainingStartedString
	nn.train()
	print trainingDoneString
	nn.test()
	
# Main
def main():
	# Code to generate results for the report
	if generatingResults:
		for j in range(0, 3):
			print "Weight weight set %d. . . " % (j + 1)
			nn = NeuralNet()
			nn.initializeWeights()
			print "With Table 1 training data. . . "
			nn.trainingFile = defaultTrainingFileName.format(table1)
			nn.testingFile = defaultTestingFileName
			initialWeights = nn.weights
			nn.printWeights()
			nn.exportWeights()
			nn.maxIterations = 2000
			nn.eta = 0.3
			
			for i in range(0, 5):
				nn.weights = initialWeights
				print newline
				print "With a max iterations of %d" % nn.maxIterations
				print "With an eta of %f. . ." % nn.eta
				print newline
				print trainingStartedString
				nn.train()
				print trainingDoneString
				nn.test()
				nn.eta += 0.05
				
			print newline
			print "With Full training data. . . "
			nn.trainingFile = defaultTrainingFileName.format(full)
			nn.eta = 0.3
			for i in range(0, 5):
				nn.weights = initialWeights
				print newline
				print "With a max iterations of %d" % nn.maxIterations
				print "With an eta of %f. . ." % nn.eta
				print newline
				print trainingStartedString
				nn.train()
				print trainingDoneString
				nn.test()
				nn.eta += 0.05
			
			print newline
			
	else:
		# Printout Strings.
		tryAgainString = "Would you like to create another NN? " + yesNoString
		exitString = "Thanks for playing!"
		
		# Get the data from the user and train/test the NN.
		userInput()
		
		# Do it again until the user decides to exit.
		tryAgain = True
		print newline
		result = myRawInput(tryAgainString)
		while tryAgain:
			if result in yesList:
				userInput()
				print newline
				result = myRawInput(tryAgainString)
			elif result in noList:
				tryAgain = False
				print exitString
				exit()
			else:
				print unrecognizedInput
				result = myRawInput(tryAgainString)
	
# Protects main from being run if this file is imported into another script.
if __name__ == "__main__":
	main()