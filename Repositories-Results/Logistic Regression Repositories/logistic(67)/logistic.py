# Adam Martini
# 2-10-13
# Update 1-5-14: added feature scaling and simplified numpy operations to a more efficient
# vectorized implementation.
# 
# usage : python logistic.py <train_file> <test_file> <learning_rate> <regularization_term>
# 	Notes:
# 	- The training file and testing file are assumed to be in csv format. 
# 
# python version : 2.7.6

import sys
import numpy as np
import mlutils as mlu

class LogisticRegressionClassifier():
	"""
	This class is responsible for all logistic regression classifier operations.  These operations include
	building a model from training data, class prediction from testing data, and printing the model.
	"""
	def __init__(self, alpha, lmbda, maxiter):
		self.alpha = float(alpha) # learning rate for gradient ascent
		self.lmbda = float(lmbda) # regularization constant
		self.epsilon = 0.00001 # convergence measure
		self.maxiter = int(maxiter) # the maximum number of iterations through the data before stopping
		self.threshold = 0.5 # the class prediction threshold

	def __str__(self):
		return "<Logistic Regression Classifier Instance: alpha=" + str(self.alpha) + ">\n"

	def fit(self, X_, y):
		"""
		This function optimizes the parameters for the logistic regression classification model from training 
		data using learning rate alpha and regularization constant lmbda
		@post: parameter(theta) optimized by gradient descent
		"""
		X = self.add_ones(X_) # prepend ones to training set for theta_0 calculations
		
		# initialize optimization arrays
		self.n = X.shape[1] # the number of features
		self.m = X.shape[0] # the number of instances
		self.probas = np.zeros(self.m, dtype='float') # stores probabilities generated by the logistic function
		self.theta = np.zeros(self.n, dtype='float') # stores the model theta

		# iterate through the data at most maxiter times, updating the theta for each feature
		# also stop iterating if error is less than epsilon (convergence tolerance constant)
		print "iter | magnitude of the gradient"
		for iteration in xrange(self.maxiter):
			# calc probabilities
			self.probas = self.get_proba(X)

			# calculate the gradient and update theta
			gw = (1.0/self.m) * np.dot(X.T, (self.probas - y))
			g0 = gw[0] # save the theta_0 gradient calc before regularization
			gw += ((self.lmbda * self.theta) / self.m) # regularize using the lmbda term
			gw[0] = g0 # restore regularization independent theta_0 gradient calc
			self.theta -= self.alpha * gw # update parameters
			
			# calculate the magnitude of the gradient and check for convergence
			loss = np.linalg.norm(gw)
			if self.epsilon > loss:
				break
			
			print iteration, ":", loss

	def get_proba(self, X):
		return 1.0 / (1 + np.exp(- np.dot(X, self.theta)))

	def predict_proba(self, X):
		"""
		Returns the set of classification probabilities based on the model theta.
		@parameters: X - array-like of shape = [n_samples, n_features]
		    		 The input samples.
		@returns: y_pred - list of shape = [n_samples]
				  The probabilities that the class label for each instance is 1 to standard output.
		"""
		X_ = self.add_ones(X)
		return self.get_proba(X_)

	def predict(self, X):
		"""
		Classifies a set of data instances X based on the set of trained feature theta.
		@parameters: X - array-like of shape = [n_samples, n_features]
		    		 The input samples.
		@returns: y_pred - list of shape = [n_samples]
				  The predicted class label for each instance.
		"""
		y_pred = [proba > self.threshold for proba in self.predict_proba(X)]
		return np.array(y_pred)

	def add_ones(self, X):
		# prepend a column of 1's to dataset X to enable theta_0 calculations
		return np.hstack((np.zeros(shape=(X.shape[0],1), dtype='float') + 1, X))

	def print_model(self, features, model_file):
		"""
		Deprecated: This method is no longer relevant as loader functions do not load features.  Future
		versions of the software with include model storage methods.
		Wite the parameter values corresponding to each feature to the given model file
		"""
		with open(model_file, 'w') as mf:
			for i in xrange(self.n):
				if i == 0:
					mf.write('%f\n' % (self.theta[i]))
				else:
					mf.write('%s %f\n' % (features[i-1], self.theta[i]))


def main(train_file, test_file, alpha=0.01, lmbda=0, maxiter=100):
	"""
	Manages files and operations for logistic regression model creation, training, and testing.
	@parameters: alpha - the learning rate for gradient descent
				 lmbda - the regularization term
				 model_file - the name of the file to store the final classification model
	"""
	# open and load csv files
	X_train, y_train = mlu.load_csv(train_file, True)
	X_test, y_test = mlu.load_csv(test_file, True)
	y_train = y_train.flatten() 
	y_test = y_test.flatten()

	# scale features to encourage gradient descent convergence
	X_train = mlu.scale_features(X_train, 0.0, 1.0)
	X_test = mlu.scale_features(X_test, 0.0, 1.0)

	# create the logistic regression classifier using the training data
	LRC = LogisticRegressionClassifier(alpha, lmbda, maxiter)
	print "\nCreated a logistic regression classifier =", LRC

	# fit the model to the loaded training data
	print "Fitting the training data...\n"
	LRC.fit(X_train, y_train)

	# predict the results for the test data
	print "Generating probability prediction for the test data...\n"
	y_pred = LRC.predict(X_test)

	### print the classification results ###
	print "The probabilities for each instance in the test set are:\n"
	for prob in LRC.predict_proba(X_test):
		print prob
	# print simple precision metric to the console
	print('Accuracy:  ' + str(mlu.compute_accuracy(y_test, y_pred)))
	
	# TODO: implement model storage methods to store parameters in hdf.
	# write the model to the model file
	# LRC.print_model(features, model_file)


if __name__ == '__main__':
	"""
	The main function is called when logistic.py is run from the command line with arguments.
	"""
	args = sys.argv[1:] # get arguments from the command line
	main( *args )