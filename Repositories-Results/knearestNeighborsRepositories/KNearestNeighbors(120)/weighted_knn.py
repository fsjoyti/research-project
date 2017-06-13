import sys
import csv
import pandas
import numpy
import math

#Proximity Functions
#NOTE: Euclidean distance is a measure of dissimilarity
def Euclidean(record, row):
	'''
	Calculates Euclidean distance between two records with an arbitrary, identical number of dimensions
	'''
	distance = 0
	for i in range(0,len(record)):
			distance += abs(record[i]-row[i])**2
	distance = math.sqrt(distance)
	return distance

#NOTE: Cosine similarity is a measure of similarity
def Cosine(record, row):
	'''
	Calculates Cosine similarity of two records with an arbitrary number of dimensions
	'''
	#Initialize values
	dot_product = 0
	record_length = 0
	row_length = 0

	for i in range(0,len(record)):
		record_length += record[i]**2
		row_length += row[i]**2
		dot_product += record[i]*row[i]

	#take the square root of the sum of squares
	record_length = math.sqrt(record_length)
	row_length = math.sqrt(row_length)

	#return cosine similarity
	return dot_product/(record_length*row_length)

def Binary(element1, element2, similarity):
	if element1 == element2:
		if similarity:
			return 1
		else:
			return 0
	else:
		if similarity:
			return 0
		else:
			return 1

def WorkclassComparison(element1, element2, similarity):
	if "gov" in element1:
		if "gov" in element2:
			if similarity:
				return 1
			else:
				return 0
		
	elif "Self-emp" in element1:
		if "Self-emp" in element2:
			if similarity:
				return 1
			else:
				return 0

	return Binary(element1, element2, similarity)

def MaritalComparison(element1, element2, similarity):
	if "Married" in element1:
		if "Married" in element2:
			if similarity:
				return 1
			else:
				return 0
		else:
			return Binary(element1, element2, similarity)
	
	else:
		if "Married" not in element2:
			if similarity:
				return 1
			else:
				return 0
		else:
			return Binary(element1, element2, similarity)


def kNNClassifier(kNearest, k, weighted, measure):
	'''
	Determines the most popular class out of k records and calculates the posterior probability of that class.
	Requires len(kNearest) > 0
	'''
	class_voting = {}

	#Total number of votes to be used in posterior probability calculation
	total = k #k by default
	if weighted:
		total = 0

	#For each of k nearest neighbors,
	for neighbor in kNearest:
		#Determine the size of the vote this neighbor gets
		vote = 1 #1 by default
		if weighted:
			if measure == "E":
				#Size of vote is equal to the distance of the nearest neighbor divided by the proximity of the current neighbor
				if neighbor[1] == 0:
					vote = 1
				else:
					vote = kNearest[0][1]/neighbor[1]
			else:
				#Size of vote is equal to the similarity of the current neighbor divided by the proximity of the most similar neighbor
				vote = neighbor[1]/kNearest[0][1]
			total += vote
		#If this neighbor's class has already been voted for by another neighbor,
		if neighbor[2] in class_voting:
			#Add this neighbor's vote to its class
			class_voting[neighbor[2]] = class_voting[neighbor[2]]+vote
		else:
			#Add class to dictionary with one vote
			class_voting[neighbor[2]] = vote
	
	#Determine "most popular" class
	predicted_class = max(class_voting, key=class_voting.get)

	#Calculate posterior probability
	posterior = class_voting[predicted_class]/total

	return [predicted_class, posterior]


#Initialization
#initialize k to a default value of 5
k = 5

#If user has provided a value for k, replace the initial value
if len(sys.argv) > 1:
	k = int(sys.argv[1])

#By default, this program will calculate a Euclidean distance as the proximity measure
iris_measure = "E"
#If user has provided an argument specifying the type of proximity measure to use, save that preference
if len(sys.argv) > 2:
	#If user enters "C", change measure to Cosine Similarity
	if sys.argv[2] == "C":
		iris_measure = "C"
	#If user has entered anything other than the two options ("E" and "C"), output a warning message to console.
	elif sys.argv[2]!="E":
		print("Warning: user has entered invalid argument for proximity measure. The default measure (Euclidean distance) will be used.")

#By default, both data sets will use the same proximity measure
income_measure = iris_measure
#If user has provided a secondary measure to use,
if len(sys.argv) > 3:
	#record user preference
	if sys.argv[3] == "E":
		income_measure = "E"
	elif sys.argv[3] == "C":
		income_measure = "C"
	else:
		#If user has entered anything other than the two options ("E" and "C"), output a warning message to console.
		if iris_measure == "E":
			print("Warning: user has entered invalid argument for secondary proximity measure. By default, the primary measure (Euclidean distance) will be used.")
		else:
			print("Warning: user has entered invalid argument for secondary proximity measure. By default, the primary measure (Cosine similarity) will be used.")

#Iris Data
#Open test data with CSV reader
with open('Iris_Test.csv', "rt") as iris_test_data:
	iris_test = csv.reader(iris_test_data)
	iris_test = list(iris_test)

	with open('Iris_Test.csv', "rt") as panda_iris_test_data:
		#Open test data with pandas library to provide ability to read entire columns at a time
		panda_iris_test = pandas.read_csv(panda_iris_test_data)

		#Open training data with CSV reader
		with open('Iris.csv', "rt") as iris_train_data:
			iris_train = csv.reader(iris_train_data)
			iris_train = list(iris_train)

			with open('Iris.csv', "rt") as panda_iris_train_data:
				#Open training data with pandas library to provide ability to read entire columns at a time
				panda_iris_train = pandas.read_csv(panda_iris_train_data)

				#Create (or overwrite) CSV file to hold output
				with open('iris_out.csv',"w") as output_file:
					output = csv.writer(output_file, dialect='excel')

					#Create header row for output CSV file
					output.writerow(['ID', 'Actual Class', 'Predicted Class', 'Posterior Probability'])

					#Create pandas dataframes for each column in test Iris data
					test_sepal_length = panda_iris_test.sepal_length
					test_sepal_width = panda_iris_test.sepal_width
					test_petal_length = panda_iris_test.petal_length
					test_petal_width = panda_iris_test.petal_width

					#Create pandas dataframes for each column in training Iris data
					train_sepal_length = panda_iris_train.sepal_length
					train_sepal_width = panda_iris_train.sepal_width
					train_petal_length = panda_iris_train[' petal_length']
					train_petal_width = panda_iris_train[' petal_width']

					#Concatenate pandas dataframes for each column to create combined Iris data
					sepal_length = pandas.concat([test_sepal_length, train_sepal_length])
					sepal_width = pandas.concat([test_sepal_width, train_sepal_width])
					petal_length = pandas.concat([test_petal_length, train_petal_length])
					petal_width = pandas.concat([test_petal_width, train_petal_width])

					#pre-formatting test Iris data
					normalized_test = []
					
					for row in iris_test[1:]:
						#min-max normalization of each column in combined Iris data set
						normal_sl = (float(row[0]) - sepal_length.min().item())/(sepal_length.max().item()-sepal_length.min().item())
						normal_sw = (float(row[1]) - sepal_width.min().item())/(sepal_width.max().item()-sepal_width.min().item())
						normal_pl = (float(row[2]) - petal_length.min().item())/(petal_length.max().item()-petal_length.min().item())
						normal_pw = (float(row[3]) - petal_width.min().item())/(petal_width.max().item()-petal_width.min().item())
						#Save normalized rows with class
						normalized_test.append([normal_sl, normal_sw, normal_pl, normal_pw, row[4]])

					

					#pre-formatting training Iris data
					normalized_train = []
					for row in iris_train[1:]:
						#min-max normalization of each column in combined Iris data set
						normal_sl = (float(row[0]) - sepal_length.min().item())/(sepal_length.max().item()-sepal_length.min().item())
						normal_sw = (float(row[1]) - sepal_width.min().item())/(sepal_width.max().item()-sepal_width.min().item())
						normal_pl = (float(row[2]) - petal_length.min().item())/(petal_length.max().item()-petal_length.min().item())
						normal_pw = (float(row[3]) - petal_width.min().item())/(petal_width.max().item()-petal_width.min().item())
						#Save normalized rows with class
						normalized_train.append([normal_sl, normal_sw, normal_pl, normal_pw, row[4]])

					#Initialize record ID (Iris data does not assign IDs)
					record_id = 1
					#For each test record
					for record in normalized_test:
						kNearest = []

						#Initialize row ID
						row_id = 1
						#Compare test record to each row in training data set
						for row in normalized_train:
							if iris_measure == "E":
								#Calculate Euclidean distance between current test record and this row of training data
								proximity = Euclidean(record[0:4], row[0:4])
								#Add record ID, proximity, and class for training row as tuple to list
								kNearest.append((str(row_id), proximity, row[4]))
								#Sort list by proximity in ascending order (nearest first)
								kNearest = sorted(kNearest, key=lambda record:record[1])
							else:
								#Calculate Cosine similarity of record test record and this row of training data
								proximity = Cosine(record[0:4], row[0:4])
								#Add record ID, proximity, and class for training row as tuple to list
								kNearest.append((str(row_id), proximity, row[4]))
								#Sort list by proximity in descending order (most similar first)
								kNearest = sorted(kNearest, key=lambda record:record[1], reverse=True)
							#Ensure that list includes no more than k records
							del kNearest[k:]

							#Increment row ID
							row_id += 1

						#Add record ID and actual class to result row
						result = [record_id, record[4]]

						#Predict class of record from k-Nearest neighbors
						result += kNNClassifier(kNearest, k, True, iris_measure)
							
						#Write row to output file
						output.writerow(result)

						#Increment record ID
						record_id += 1

#Income Data
#Open test data with CSV reader
with open('income_te.csv', "rt") as income_test_data:
	income_test = csv.reader(income_test_data)
	income_test = list(income_test)

	with open('income_te.csv', "rt") as panda_income_test_data:
		#Open test data with pandas library to provide ability to read entire columns at a time
		panda_income_test = pandas.read_csv(panda_income_test_data)

		#Open training data with CSV reader
		with open('income_tr.csv', "rt") as income_train_data:
			income_train = csv.reader(income_train_data)
			income_train = list(income_train)

			with open('income_tr.csv', "rt") as panda_income_train_data:
				#Open training data with pandas library to provide ability to read entire columns at a time
				panda_income_train = pandas.read_csv(panda_income_train_data)

				#Create (or overwrite) CSV file to hold output
				with open('income_out.csv',"w") as output_file:
					output = csv.writer(output_file, dialect='excel')

					#Create header row for output CSV file
					output.writerow(['ID', 'Actual Class', 'Predicted Class', 'Posterior Probability'])

					#Create pandas dataframes for each continuous attribute's column in test Income data
					test_age = panda_income_test.age
					test_fnlwgt = panda_income_test.fnlwgt
					test_education = panda_income_test.education_cat
					test_gain = panda_income_test.capital_gain
					test_loss = panda_income_test.capital_loss
					test_hpw = panda_income_test.hour_per_week

					#Create pandas dataframes for each continuous attribute's column in training Income data
					train_age = panda_income_train.age
					train_fnlwgt = panda_income_train.fnlwgt
					train_education = panda_income_train.education_cat
					train_gain = panda_income_train.capital_gain
					train_loss = panda_income_train.capital_loss
					train_hpw = panda_income_train.hour_per_week

					#Concatenate pandas dataframes for each column to create combined Income data
					age = pandas.concat([test_age, train_age])
					fnlwgt = pandas.concat([test_fnlwgt, train_fnlwgt])
					education = pandas.concat([test_education, train_education])
					gain = pandas.concat([test_gain, train_gain])
					loss = pandas.concat([test_loss, train_loss])
					hpw = pandas.concat([test_hpw, train_hpw])

					#pre-formatting test Income data
					normalized_test = []

					for row in income_test[1:]:
						#min-max normalization of each column
						normal_age = (float(row[1]) - age.min().item())/(age.max().item()-age.min().item())
						normal_fnlwgt = (float(row[3]) - fnlwgt.min().item())/(fnlwgt.max().item()-fnlwgt.min().item())
						normal_education = (float(row[5]) - education.min().item())/(education.max().item()-education.min().item())
						normal_gain = (float(row[11]) - gain.min().item())/(gain.max().item()-gain.min().item())
						normal_loss = (float(row[12]) - loss.min().item())/(loss.max().item()-loss.min().item())
						normal_hpw = (float(row[13]) - hpw.min().item())/(hpw.max().item()-hpw.min().item())
						#Save normalized row
						normalized_test.append([normal_age, normal_fnlwgt, normal_education, normal_gain, normal_loss, normal_hpw])

					#pre-formatting training Income data
					normalized_train = []

					for row in income_train[1:]:
						#min-max normalization of each column
						normal_age = (float(row[1]) - age.min().item())/(age.max().item()-age.min().item())
						normal_fnlwgt = (float(row[3]) - fnlwgt.min().item())/(fnlwgt.max().item()-fnlwgt.min().item())
						normal_education = (float(row[5]) - education.min().item())/(education.max().item()-education.min().item())
						normal_gain = (float(row[11]) - gain.min().item())/(gain.max().item()-gain.min().item())
						normal_loss = (float(row[12]) - loss.min().item())/(loss.max().item()-loss.min().item())
						normal_hpw = (float(row[13]) - hpw.min().item())/(hpw.max().item()-hpw.min().item())
						#Save normalized row
						normalized_train.append([normal_age, normal_fnlwgt, normal_education, normal_gain, normal_loss, normal_hpw])


					record_number = 1
					#For each test record
					for record in normalized_test:
						#Retrieve corresponding row from full data set
						income_record = income_test[record_number]

						kNearest = []

						#Initialize row ID
						row_number = 1
						#Compare test record to each row in training data set
						for row in normalized_train:
							#Retrieve corresponding row from full data set
							income_row = income_train[row_number]

							if iris_measure == "E":
								#For continuous attributes, calculate Euclidean distance between current record and this row
								euclidean = Euclidean(record, row)

								#Compute dissimilarities for categorical attributes
								workclass = WorkclassComparison(income_record[2], income_row[2], False)
								#Drop Education categorical attribute (index 4) since education_cat has already been included in continuous measurement
								marital = MaritalComparison(income_record[6], income_row[6], False)
								occupation = Binary(income_record[7], income_row[7], False)
								relationship = Binary(income_record[8], income_row[8], False)
								race = Binary(income_record[9], income_row[9], False)
								gender = Binary(income_record[10], income_row[10], False)
								country = Binary(income_record[14], income_row[14], False)

								#Compute average dissimilarity by weighting each attribute
								#out of 13 total attributes, 6 were included in the Euclidean distance. Weight accordingly.
								proximity = ((6/13)*euclidean)+((1/13)*(workclass+marital+occupation+relationship+race+gender+country))

								#Add record ID, proximity, and class for row as tuple to list
								kNearest.append((str(income_row[0]), proximity, income_row[15]))

								#Sort list by proximity in ascending order (nearest first)
								kNearest = sorted(kNearest, key=lambda record:record[1])
							else:
								#For continuous attributes, calculate Cosine similarity between current record and this row
								cos = Cosine(record, row)

								#Compute similarities for categorical attributes
								workclass = WorkclassComparison(income_record[2], income_row[2], True)
								#Drop Education categorical attribute (record[4]) since education_cat has already been included in continuous measurement
								marital = MaritalComparison(income_record[6], income_row[6], True)
								occupation = Binary(income_record[7], income_row[7], True)
								relationship = Binary(income_record[8], income_row[8], True)
								race = Binary(income_record[9], income_row[9], True)
								gender = Binary(income_record[10], income_row[10], True)
								country = Binary(income_record[14], income_row[14], True)

								#Compute average similarity by weighting each attribute
								#out of 13 total attributes, 6 were included in the Cosine similarity. Weight accordingly.
								proximity = ((6/13)*cos)+((1/13)*(workclass+marital+occupation+relationship+race+gender+country))

								#Add record ID, proximity, and class for row as tuple to list
								kNearest.append((str(income_row[0]), proximity, income_row[15]))

								#Sort list by proximity in descending order (most similar first)
								kNearest = sorted(kNearest, key=lambda record:record[1], reverse=True)

							#Ensure that list includes no more than k records
							del kNearest[k:]

							row_number += 1

						#Add record ID and actual class to result row
						result = [income_record[0], income_record[15]]

						#Predict class of record from k-Nearest neighbors
						result += kNNClassifier(kNearest, k, True, income_measure)
							
						#Write row to output file
						output.writerow(result)

						#Increment record ID
						record_number += 1