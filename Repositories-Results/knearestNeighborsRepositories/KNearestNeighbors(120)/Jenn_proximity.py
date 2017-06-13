<<<<<<< HEAD
import sys
import csv
import pandas
import numpy
import math
from itertools import islice

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
#Open input data with CSV reader
with open('Iris.csv', "rt") as iris_data:
	iris = csv.reader(iris_data)

	with open('Iris.csv', "rt") as panda_iris_data:
		#Open input data with pandas library to provide ability to read entire columns at a time
		panda_iris = pandas.read_csv(panda_iris_data)

		#Create (or overwrite) CSV file to hold output
		with open('iris_out.csv',"w") as output_file:
			output = csv.writer(output_file, dialect='excel')

			#Create header for output CSV file
			header_row = ['ID']

			for i in range(1, k+1):
				header_row += [str(i), str(i)+' prox.']

			output.writerow(header_row)

			#Create pandas dataframes for each column in Iris data
			sepal_length = panda_iris.sepal_length
			sepal_width = panda_iris.sepal_width
			petal_length = panda_iris[' petal_length']
			petal_width = panda_iris[' petal_width']

			#pre-formatting Iris data
			normalized = []
			for row in islice(iris, 1, None):
				#min-max normalization of each column
				normal_sl = (float(row[0]) - sepal_length.min().item())/(sepal_length.max().item()-sepal_length.min().item())
				normal_sw = (float(row[1]) - sepal_width.min().item())/(sepal_width.max().item()-sepal_width.min().item())
				normal_pl = (float(row[2]) - petal_length.min().item())/(petal_length.max().item()-petal_length.min().item())
				normal_pw = (float(row[3]) - petal_width.min().item())/(petal_width.max().item()-petal_width.min().item())
				#Save normalized rows
				normalized.append([normal_sl, normal_sw, normal_pl, normal_pw])

			#Initialize record ID (Iris data does not assign IDs)
			record_id = 1
			for record in normalized:
				nearest = []

				#Initialize row ID
				row_id = 1
				for row in normalized:
					if row_id != record_id:
						if iris_measure == "E":
							#Calculate Euclidean distance between current record and this row
							proximity = Euclidean(record, row)
							#Add proximity and record ID for row as tuple to list
							nearest.append((str(row_id), proximity))
							#Sort list by proximity in ascending order (nearest first)
							nearest = sorted(nearest, key=lambda record:record[1])
						else:
							#Calculate Cosine similarity of record row and this row
							proximity = Cosine(record, row)
							#Add proximity and record ID for row as tuple to list
							nearest.append((str(row_id), proximity))
							#Sort list by proximity in descending order (most similar first)
							nearest = sorted(nearest, key=lambda record:record[1], reverse=True)
						#Ensure that list includes no more than k records
						del nearest[k:]

					#Increment row ID
					row_id += 1

				result = [record_id]
				#For each of k nearest neighbors,
				for i in range(0,k):
					#Add ID and proximity measure to result row
					result += [nearest[i][0], nearest[i][1]]
				#Write row to output file
				output.writerow(result)

				#Increment record ID
				record_id += 1

#Income Data
#Open input data with CSV reader
with open('income_tr.csv', "rt") as income_data:
	income = csv.reader(income_data)
	income = list(income)

	with open('income_tr.csv', "rt") as panda_income_data:
		#Open input data with pandas library to provide ability to read entire columns at a time
		panda_income = pandas.read_csv(panda_income_data)

		#Create (or overwrite) CSV file to hold output
		with open('income_out.csv',"w") as output_file:
			output = csv.writer(output_file, dialect='excel')

			#Create header for output CSV file
			header_row = ['ID']

			for i in range(1, k+1):
				header_row += [str(i), str(i)+' prox.']

			output.writerow(header_row)

			#Create pandas dataframes for each continuous attribute's column in Income data
			age = panda_income.age
			fnlwgt = panda_income.fnlwgt
			gain = panda_income.capital_gain
			loss = panda_income.capital_loss
			hpw = panda_income.hour_per_week

			#pre-formatting continuous Income data
			normalized = []
			for row in islice(income, 1, None):
				#min-max normalization of each column
				normal_age = (float(row[1]) - age.min().item())/(age.max().item()-age.min().item())
				normal_fnlwgt = (float(row[3]) - fnlwgt.min().item())/(fnlwgt.max().item()-fnlwgt.min().item())
				normal_gain = (float(row[11]) - gain.min().item())/(gain.max().item()-gain.min().item())
				normal_loss = (float(row[12]) - loss.min().item())/(loss.max().item()-loss.min().item())
				normal_hpw = (float(row[13]) - hpw.min().item())/(hpw.max().item()-hpw.min().item())
				#Save normalized rows
				normalized.append([normal_age, normal_fnlwgt, normal_gain, normal_loss, normal_hpw])

			record_number = 1;
			for record in normalized:
				#Retrieve corresponding row from full data set
				income_record = income[record_number]
				#Initialize list of k-nearest
				nearest = []

				row_number = 1;
				for row in normalized:
					#Retrieve corresponding row from full data set
					income_row = income[row_number]

					#If this record is not the same as the record it is being compared to (based on ID)
					if income_row[0] != income_record[0]:
						if income_measure == "E":
							#For continuous attributes, calculate Euclidean distance between current record and this row
							euclidean = Euclidean(record, row)

							#Compute dissimilarities for categorical attributes
							workclass = Binary(income_record[2], income_row[2], False)
							education = Binary(income_record[4], income_row[4], False)
							education_cat = abs(int(income_record[5])-int(income_row[5]))/15	#15 is the total number of distinct values held by this attribute
							marital = Binary(income_record[6], income_row[6], False)
							occupation = Binary(income_record[7], income_row[7], False)
							relationship = Binary(income_record[8], income_row[8], False)
							race = Binary(income_record[9], income_row[9], False)
							gender = Binary(income_record[10], income_row[10], False)
							country = Binary(income_record[14], income_row[14], False)

							#Average together dissimilarity by weighting attributes
							#out of 14 total attributes, 5 were included in the Euclidean proximity. Weight accordingly.
							proximity = ((5/14)*euclidean)+((1/14)*(workclass+education+education_cat+marital+occupation+relationship+race+gender+country))

							#Add proximity and record ID for row as tuple to list
							nearest.append((str(income_row[0]), proximity))

							#Sort list by proximity in ascending order (nearest first)
							nearest = sorted(nearest, key=lambda record:record[1])
						else:
							#For continuous attributes, calculate Cosine similarity of current record and this row
							cos = Cosine(record, row)

							#Compute similarities for categorical attributes
							workclass = Binary(income_record[2], income_row[2], True)
							education = Binary(income_record[4], income_row[4], True)
							education_cat = 1-(abs(int(income_record[5])-int(income_row[5]))/15)	#15 is the total number of distinct values held by this attribute
							marital = Binary(income_record[6], income_row[6], True)
							occupation = Binary(income_record[7], income_row[7], True)
							relationship = Binary(income_record[8], income_row[8], True)
							race = Binary(income_record[9], income_row[9], True)
							gender = Binary(income_record[10], income_row[10], True)
							country = Binary(income_record[14], income_row[14], True)

							#Average together dissimilarity by weighting attributes
							#out of 14 total attributes, 5 were included in the Cosine proximity. Weight accordingly.
							proximity = ((5/14)*cos)+((1/14)*(workclass+education+education_cat+marital+occupation+relationship+race+gender+country))

							#Add proximity and record ID for row as tuple to list
							nearest.append((str(income_row[0]), proximity))

							#Sort list by proximity in descending order (most similar first)
							nearest = sorted(nearest, key=lambda record:record[1], reverse=True)

						#Ensure that list includes no more than k records
						del nearest[k:]

					row_number += 1

				#initialize output row with record's row ID
				result = [income_record[0]]
				#For each of k nearest neighbors,
				for i in range(0,k):
					#Add ID and proximity measure to result row
					result += [nearest[i][0], nearest[i][1]]
				#Write row to output file
				output.writerow(result)

				record_number += 1
=======
import csv
import operator
import math

#attribute categories 
nominal_income = ['workclass', 'marital_status', 'occupation', 'race', 'gender', 'native_country']
ordinal_income = ['education']
ordinal_income_conversion = {' Preschool': 0,' 1st-4th':1,' 5th-6th':2, ' 7th-8th':3, ' 9th':4, ' 10th':5, ' 11th':6, ' 12th':7, ' HS-grad':8,' Prof-school':9, ' Assoc-acdm':10, ' Assoc-voc':11, ' Some-college':12, ' Bachelors':13, ' Masters':14, ' Doctorate':15}
ratio_income = ['age','education_cat', 'capital-gain', 'capital-loss', 'hour_per_week']

#nominal_iris = ['class']
ratio_iris = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']

#outputs tables to 
def output_proximity_table(k):

	#output files 
	output_file_income = open("output_income.csv", "w")
	output_file_iris = open("output_iris.csv", "w")

	#csv writers
	writer_income = csv.writer(output_file_income)
	writer_iris = csv.writer(output_file_iris)


	#set up cvs readers
	with open('./income_tr.csv', 'r') as income_csvfile:
		with open('./Iris.csv', 'r') as iris_csvfile:
			income_reader = csv.DictReader(income_csvfile)
			iris_reader = csv.DictReader(iris_csvfile)

			#list of dictionairesn
			income_reader = list(income_reader)
			iris_reader = list(iris_reader)

	#income

	#set up output file
	cols = ['Transaction ID']
	for x in range(1,k+1): 
		cols.extend([str(x), str(x) + '-prox'])
	writer_income.writerow(cols)

	for row1 in income_reader: #loop through the file
		id1 = row1['ID']
		row_to_write = [id1] #row that will be writen to output file
		closest_records = [] #list of tuples of closest other records ID's and prox_value
		
		for row2 in income_reader: 
			id2 = row2['ID']

			if id2 != id1:

				#separate the nominal, ratio, and ordinal attributes for each record 
				row1_nominal = {key:val for (key,val) in row1.items() if key in nominal_income}
				row2_nominal = {key:val for (key,val) in row2.items() if key in nominal_income}
				row1_ratio = {key:val for (key,val) in row1.items() if key in ratio_income}
				row2_ratio = {key:val for (key,val) in row2.items() if key in ratio_income}
				row1_ordinal = {key:val for (key,val) in row1.items() if key in ordinal_income}
				row2_ordinal= {key:val for (key,val) in row2.items() if key in ordinal_income}

				#numerator of similarity calculation
				num = cos_similarity(row1_ratio, row2_ratio, ratio_income) #weighted cosine similarity

				#demoninator of similarity calculation
				dem = len(ratio_income)

				#for each ordinal attribute, add the similarity if both values are present
				for att in ordinal_income:
					sim = ordinal_similarity(ordinal_income_conversion[row1_ordinal[att]],ordinal_income_conversion[row2_ordinal[att]], len(ordinal_income_conversion))
					if sim != -1:
						num += sim
						dem += 1

				#for each nominal attribute, add the similarity if both values are present
				for att in nominal_income:
					sim = similarity(row1_nominal[att], row2_nominal[att])
					if sim != -1:
						num += sim
						dem += 1 
				proximity = float(num)/float(dem) 

				#determine if it is in the top k proximities 
				if(len(closest_records)<k):
					closest_records.append((id2,proximity)) #add to list if there aren't k values in it yet
				else:
					#sort records
					closest_records = sorted(closest_records, key=operator.itemgetter(1)) 
					if proximity>closest_records[0][1]:
						closest_records[0] = (id2, proximity)

		closest_records = sorted(closest_records, key=operator.itemgetter(1))
		closest_records.reverse()
		for (ke,v) in closest_records:
			row_to_write.extend([ke,v])
		writer_income.writerow(row_to_write)

	#iris

	#set up output file
	cols = ['ID']
	for x in range(1,k+1): 
		cols.extend([str(x), str(x) + '-prox'])
	writer_iris.writerow(cols)

	id1 = 1 #id not defined in dataset, use line number
	for row1 in iris_reader:
		row_to_write = [id1]
		closest_records = []

		id2 = 1
		#get the closest records to row
		for row2 in iris_reader:
			if id1 != id2:
				row1_ratio = {key:val for (key,val) in row1.items() if key in ratio_iris}
				row2_ratio = {key:val for (key,val) in row2.items() if key in ratio_iris}

				proximity = cos_similarity(row1_ratio, row2_ratio, ratio_iris)/len(ratio_iris)

				if(len(closest_records)<k):
					closest_records.append((id2,proximity))
				else:
					#sort records
					closest_records = sorted(closest_records, key=operator.itemgetter(1)) 
					if proximity>closest_records[0][1]:
						closest_records[0] = (id2, proximity)
			id2 +=1
		closest_records = sorted(closest_records, key=operator.itemgetter(1))
		closest_records.reverse()
		for (ke,v) in closest_records:
			row_to_write.extend([ke,v])
		writer_iris.writerow(row_to_write)
		id1 +=1

	output_file_income.close();
	output_file_iris.close();
				
#returns 1 if values are the same, 0 if they are different, and -1 if either value is missing
def similarity(val1, val2):
	if val1=='?' or val2=='?':
		return -1
	if val1 == val2:
		return 1
	else:
		return 0

#returns the weighted similarity of two vectors
def cos_similarity(val1, val2, attribute_list):
	val1_mag = 0
	val2_mag = 0
	dot = 0
	for attribute in attribute_list:
		x1 = val1.get(attribute, 0)
		x2 = val2.get(attribute, 0)
		if x1 == '?':
			x1 = 0
		if x2 == '?':
			x2 = 0
		x1 = float(x1)
		x2 = float(x2)

		dot += (x1 * x2)
		val1_mag += x1*x1
		val2_mag += x2*x2

	return (dot * len(attribute_list))/ (math.sqrt(val1_mag)*math.sqrt(val2_mag))

#returns ordinal similarity, if either of the values are missing returns -1
def ordinal_similarity(val1, val2, n):
	if val1 == '?' or val2 == '?':
		return -1
	else:
		return 1 - (abs(val1 - val2)/float(n-1))

#main method, change parameter if needed
def main():
	output_proximity_table(5)

if __name__ == '__main__':
	main()

>>>>>>> 6486bcf61be16589d2b6303bd5a3559a35906831
