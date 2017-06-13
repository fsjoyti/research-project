from sklearn.neighbors import KNeighborsClassifier
import csv
import numpy
import sys

k = 5
if len(sys.argv) > 1:
	k = int(sys.argv[1])


#Income Data
#Open test data with CSV reader
with open('Iris_Test.csv', "rt") as iris_test_data:
	iris_test = csv.reader(iris_test_data)
	iris_test = list(iris_test)

	#Open training data with CSV reader
	with open('Iris.csv', "rt") as iris_train_data:
		iris_train = csv.reader(iris_train_data)
		iris_train = list(iris_train)

		training_records = []
		training_classes = []
		for record in iris_train[1:]:
			training_records.append(record[0:4])
			training_classes.append(record[4])

		knn_iris = KNeighborsClassifier(n_neighbors=k)
		knn_iris.fit(training_records, training_classes)

		test_records = []
		test_classes = []
		for record in iris_test[1:]:
			test_records.append(record[0:4])
			test_classes.append(record[4])

		print("Iris Accuracy:", knn_iris.score(test_records, test_classes))

#Income Data
conversion = {}
x = 1
categorical_indices = [2, 4, 6, 7, 8, 9, 10, 14]

#Open test data with CSV reader
with open('income_te.csv', "rt") as income_test_data:
	income_test = csv.reader(income_test_data)
	income_test = list(income_test)

	#Open training data with CSV reader
	with open('income_tr.csv', "rt") as income_train_data:
		income_train = csv.reader(income_train_data)
		income_train = list(income_train)

		training_records = []
		training_classes = []
		for record in income_train[1:]:
			for i in categorical_indices:
				if record[i] in conversion:
					record[i] = conversion[record[i]]
				else:
					conversion[record[i]] = x
					record[i] = x
					x += 1
			training_records.append(record[1:15])
			training_classes.append(record[15])

		knn = KNeighborsClassifier(n_neighbors=k)
		knn.fit(training_records, training_classes)

		test_records = []
		test_classes = []
		for record in income_test[1:]:
			for i in categorical_indices:
				if record[i] in conversion:
					record[i] = conversion[record[i]]
				else:
					conversion[record[i]] = x
					record[i] = x
					x += 1
			test_records.append(record[1:15])
			test_classes.append(record[15])

		print("Income Accuracy:", knn.score(test_records, test_classes))