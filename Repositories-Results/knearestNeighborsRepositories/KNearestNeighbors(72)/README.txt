File name: knn.py
Authors: Jennifer Barry and Claudia Moeller

Dependencies:
	This program expects python3 to be installed.
	This program requires the "NumPy" library for Python.
		To install, use the command "sudo apt-get install python-pip"
	This program requires the "pandas" library for Python.
		To install, use the command "sudo apt-get install python3-pandas"

Input Data:
	Input data files should be placed in the same directory as the program file.

Execution:
	Once dependencies have been resolved, the program can be run from a linux command line with the following command:
		python3 knn.py <optional value of k> <optional indicator for preferred measurement> <secondary indicator>
	
		<optional value of k> is expected to be an integer value specifying the number of "nearest neighbors" to include in the output.

		<optional indicator for preferred measurement> is expected to be either:
			- The letter E (capitalized) for Euclidean distance
			- The letter C (capitalized) for Cosine similarity
			This argument specifies the preferred proximity measurement to be applied.
		
		<secondary indicator> follows the same rules as <optional indicator for preferred measurement> above, except it will only be applied to the Income data set.
			The value chosen for the first indicator will only be used on the Iris data set.
		
		
		NOTE: While all bracketed arguments listed above are optional, they must be in the order listed if they are included.
			For example, the program will fail if only a secondary indicator is provided without including the k value and first indicator.
			Examples of proper input include:
				- python3 knn.py
				- python3 knn.py 3
				- python3 knn.py 10 C
				- python3 knn.py 6 E C

Behavior:
	The program takes as input four data sets:
		- the Iris training data set (Iris.csv)
		- the Iris test data set (Iris_Test.csv)
		- the Income training data set (income_tr.csv)
		- the Income test data set (income_te.csv)
	For each test data set, the proximity of each record to all records in the respective training data set is calculated.
	For each test record, the "k" nearest training records are examined to determine which class occurs most frequently.
		The most frequent class among the k-nearest neighbors is predicted to be the test record's class.
	Upon completion, the program outputs a new data set containing each record's ID, its actual class, the class predicted by its neighbors, and the posterior probability of the predicted class.
		The posterior probability is calculated as the number of neighbors whose class is the predicted class divided by k total neighbors.

	The program accepts a number of parameters.
		- The value for k can be changed by the user. By default, it is 5.
		- The proximity measure used to calculate how near a given record is to another can be chosen by the user.
			- There are two options: Euclidean distance (a measure of dissimilarity), and Cosine similarity.
			- By default, the program will calculate Euclidean distance in determining proximity.
		- The proximity measure for the second data set (the Income data set) can be chosen independently of the first data set.
			- By default, the second data set will use the same proximity measure as the first data set.

Output Data:
	Output data files will be stored in CSV format in the same directory as the program file.
	The output files will be named "iris_out.csv" and "income_out.csv" for the Iris and Income data sets, respectively.
