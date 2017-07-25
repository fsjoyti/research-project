#include "KNearestNeighbors.h"

KNearestNeighbors::KNearestNeighbors(std::string filename) {
	std::ifstream ifs;
	ifs.open(filename, std::ifstream::in);

	while (ifs.good()) {
		std::string line;
		getline(ifs, line);
		std::istringstream buffer(line);
		double feature;
		FeaturedVector featuredVector;

		while (buffer >> feature) {
			if (buffer.peek() == ',') {
				buffer.ignore();
			}
			featuredVector.push_back(feature);
		}
		this->trainingSet.push_back(featuredVector);
	}
	this->displayTrainingSet();
	ifs.close();
}

KNearestNeighbors::KNearestNeighbors(std::vector<FeaturedVector> &vector) {
	this->trainingSet = vector;
}

void KNearestNeighbors::loadInputData(std::string filename) {
	std::ifstream ifs;
	ifs.open(filename, std::ifstream::in);

	while (ifs.good()) {
		std::string line;
		getline(ifs, line);
		std::istringstream buffer(line);
		double feature;
		FeaturedVector featuredVector;

		while (buffer >> feature) {
			if (buffer.peek() == ',') {
				buffer.ignore();
			}
			featuredVector.push_back(feature);
		}
		this->trainingSet.push_back(featuredVector);
	}
	this->displayTrainingSet();
	ifs.close();
}

double KNearestNeighbors::euclidianDistance(FeaturedVector &v1, FeaturedVector &v2) {
	assert(v1.size() == v2.size());
	double sum = 0;
	for (unsigned i = 0; i < v1.size(); ++i) {
		sum += (v1[i] - v2[i]) * (v1[i] - v2[i]);
	}
	return sqrt(sum);
}

double KNearestNeighbors::mostCommonElement(std::vector<double> v) {
	if (!v.empty()) {
		std::sort(v.begin(), v.end());
		double mostCommon = v[0];
		unsigned times = 0;

		for (unsigned i = 0; i < v.size(); ++i) {
			double   tempMostCommon = (i > 0) ? v[i - 1] : v[0];
			unsigned tempTimes = 0;

			unsigned j = (i > 0) ? (i - 1) : 0;
			while (v[j] == tempMostCommon) {
				++tempTimes;
				++j;
				++i;
			}

			if (tempTimes > times) {
				mostCommon = tempMostCommon;
				times = tempTimes;
			}
		}

		return mostCommon;
	} else {
		return -1;
	}
}

double KNearestNeighbors::predict(FeaturedVector &v, unsigned k) {
	if (k < this->trainingSet.size()) {

		std::vector<FeaturedVector> &data = this->trainingSet;
		std::vector<std::pair<double, double>> dist;

		for (unsigned i = 0; i < data.size(); ++i) {
			FeaturedVector v0 = data[i];
			FeaturedVector v1(&v0[0], &v0[v0.size() - 1]);

			double euclidianDist = euclidianDistance(v1, v);
			std::pair<double, double> p(euclidianDist, v0.back());
			dist.push_back(p);
		}

		std::sort(dist.begin(), dist.end(), [](auto & left, auto & right) {
			return left.first < right.first;
		});

		std::vector<double> votes;
		for (unsigned i = 0; i < k; ++i) {
			votes.push_back(dist[i].second);
		}

		double mostCommon = mostCommonElement(votes);
		return mostCommon;

	} else {
		return -1;
	}
}

void KNearestNeighbors::displayVector (const std::vector<double> &v) const {
	for (auto &feature : v) {
		std::cout << feature << " ";
	}
	std::cout << std::endl;
}

void KNearestNeighbors::displayTrainingSet() const {
	for (auto a : this->trainingSet) {
		displayVector(a);
	}
}