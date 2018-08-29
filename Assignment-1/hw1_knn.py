from __future__ import division, print_function

from typing import List, Callable

import numpy
import scipy


############################################################################
# DO NOT MODIFY ABOVE CODES
############################################################################

class KNN:

	def __init__(self, k: int, distance_function) -> float:
		self.k = k
		self.distance_function = distance_function
		self.training_list = None
		self.labels_list = None

	def train(self, features: List[List[float]], labels: List[int]):
		self.training_list = features
		self.labels_list = labels

	def predict(self, features: List[List[float]]) -> List[int]:
		classification = []
		for i in features:
			dist = []
			for (x,y) in zip(self.training_list, self.labels_list):
				point_dist = self.distance_function(x,i)
				dist.append((point_dist, y))
			dist.sort(key=lambda x: x[0])
			size = len(dist)
			voting_points = []
			dist = dist[:self.k]
			for a,b in dist:
				voting_points.append(b)
			votes = 0
			point_classification = -1
			for x in voting_points:
				voting_points.remove(x)
				curr_votes = 1
				for y in voting_points:
					if x == y:
						curr_votes+=1
						voting_points.remove(y)
				if(curr_votes > votes):
					votes = curr_votes
					point_classification = x
			classification.append(point_classification)
		return classification

if __name__ == '__main__':
	print(numpy.__version__)
	print(scipy.__version__)
