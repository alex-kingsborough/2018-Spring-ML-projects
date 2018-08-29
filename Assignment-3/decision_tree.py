import numpy as np
from typing import List
from classifier import Classifier

class DecisionTree(Classifier):
	def __init__(self):
		self.clf_name = "DecisionTree"
		self.root_node = None

	def train(self, features: List[List[float]], labels: List[int]):
		# init.
		assert(len(features) > 0)
		self.feautre_dim = len(features[0])
		num_cls = np.max(labels)+1

		# build the tree
		self.root_node = TreeNode(features, labels, num_cls)
		if self.root_node.splittable:
			self.root_node.split()

		return
		
	def predict(self, features: List[List[float]]) -> List[int]:
		y_pred = []
		for feature in features:
			y_pred.append(self.root_node.predict(feature))
		return y_pred

	def print_tree(self, node=None, name='node 0', indent=''):
		if node is None:
			node = self.root_node
		print(name + '{')
		if node.splittable:
			print(indent + '  split by dim {:d}'.format(node.dim_split))
			for idx_child, child in enumerate(node.children):
				self.print_tree(node=child, name= '  '+name+'/'+str(idx_child), indent=indent+'  ')
		else:
			print(indent + '  cls', node.cls_max)
		print(indent+'}')


class TreeNode(object):
	def __init__(self, features: List[List[float]], labels: List[int], num_cls: int):
		self.features = features
		self.labels = labels
		self.children = []
		self.num_cls = num_cls

		count_max = 0
		for label in np.unique(labels):
			if self.labels.count(label) > count_max:
				count_max = labels.count(label)
				self.cls_max = label # majority of current node

		if len(np.unique(labels)) < 2:
			self.splittable = False
		else:
			self.splittable = True

		self.dim_split = None # the dim of feature to be splitted

		self.feature_uniq_split = None # the feature to be splitted


	def split(self):
		def conditional_entropy(branches: List[List[int]]) -> float:
			'''
			branches: C x B array, 
					  C is the number of classes,
					  B is the number of branches
					  it stores the number of 
			'''
			########################################################
			# TODO: compute the conditional entropy
			########################################################
			
			tot_entrop = 0
			total = 0
			for i in range(len(branches[0])):
				for j in range(len(branches)):
					total += branches[j][i]

			for i in range(len(branches[0])):
				branch_entrop = 0
				branch_tot = 0
				for j in range(len(branches)):
					branch_tot += branches[j][i]
				for j in range(len(branches)):
					if(branches[j][i]/branch_tot != 0):
						entrop = np.log2(branches[j][i]/branch_tot)
						entrop *= branches[j][i]/branch_tot
						branch_entrop += entrop
				branch_entrop = branch_entrop*(branch_tot/total)
				tot_entrop -= branch_entrop
			return tot_entrop

		
		if(len(self.features[0]) == 0):
			self.splittable = False
			return

		entro_list = []
		for idx_dim in range(len(self.features[0])):
		############################################################
		# TODO: compare each split using conditional entropy
		#       find the 
		############################################################
			
			num_labels = len(np.unique(self.labels))
			a = np.matrix(self.features)[:,0]
			s = set()
			for i in a:
				s.add(i[0,0])
			num_features = len(s)
			labels_list = []
			for i in np.unique(self.labels):
				labels_list.append(i)


			base = np.zeros((num_labels, num_features))

			for curr in range(len(self.features)):
				label = labels_list.index(self.labels[curr])
				feat = int(self.features[curr][idx_dim])
				base[label,feat] = base[label,feat] + 1


			entro_list.append(conditional_entropy(base.tolist()))


		############################################################
		# TODO: split the node, add child nodes
		############################################################
		min_val = entro_list[0]
		min_index = 0
		for i in range(len(entro_list)):
			if(min_val > entro_list[i]):
				min_val = entro_list[i]
				min_index = i
		self.dim_split = min_index


		a = np.matrix(self.features)[:,min_index]
		s = set()
		for i in a:
			s.add(i[0,0])
		s = list(s)
		self.feature_uniq_split = s
		for j in range(num_features):
			labels2 = []
			feat2 = []
			for i in range(len(self.features)):
				if(len(self.features[i]) > 0):
					if(self.features[i][min_index] == j):
						new_list = list(self.features[i])
						new_list.pop(min_index)
						feat2.append(new_list)
						labels2.append(self.labels[i])

			self.children.append(TreeNode(feat2, labels2, len(np.unique(labels2))))




		# split the child nodes
		for child in self.children:
			if child.splittable:
				child.split()

		return

	def predict(self, feature: List[int]) -> int:
		if self.splittable:
			# print(feature)
			idx_child = self.feature_uniq_split.index(feature[self.dim_split])
			feature = feature[:self.dim_split] + feature[self.dim_split+1:]
			return self.children[idx_child].predict(feature)
		else:
			return self.cls_max



