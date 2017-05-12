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

