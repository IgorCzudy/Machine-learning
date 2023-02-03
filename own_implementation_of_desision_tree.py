from collections import defaultdict
import numpy as np 
import pandas as pd
import treelib

class Node(object):
	def __init__(self):
		self.children = defaultdict(lambda: Node())
		self.node_value = None # is None if its leaf Przechowuje wartość o ile wierzchołek jest liściem
		self.best_feature_to_split = None # name of feature in node 
		self.name = None # use ony for visuaization 

        
	def _entropy_count(self, y: pd.Series) -> int:
		probabiltiy = y.value_counts()/y.value_counts().sum()
		return -(probabiltiy * np.log2(probabiltiy)).sum()

	def _conditional_entropy_count(self, x:pd.Series, y:pd.Series) -> int:
		
		conditional_entropy = 0
		for features_value in x.unique():
			filtered_y = y.loc[x==features_value]
			conditional_entropy += (filtered_y.count()/y.count()) * self._entropy_count(filtered_y)
		return conditional_entropy

	def _information_gain_count(self, entropy: int, contitional_entropy: int) -> int:
		return entropy - contitional_entropy


	def perform_split(self, data: pd.DataFrame, Y: pd.Series, node_name = 'r'):
		# use only for vizualization 
		self.name = node_name
	
		features_and_entropy = {}
		for feature in data.columns:
			features_and_entropy[feature] = self._information_gain_count(self._entropy_count(Y),
						self._conditional_entropy_count(data[feature], Y))

		
		self.best_feature_to_split = max(features_and_entropy, key=features_and_entropy.get)

		# if uzyskano poprawę funkcji celu (bądź inny, zaproponowany przez Ciebie warunek):
		if features_and_entropy[self.best_feature_to_split] > 0:
			for value in data[self.best_feature_to_split].unique():
				data_with_result = pd.concat([data, Y], axis=1)

				Y_rec = data_with_result.loc[data_with_result[self.best_feature_to_split] == value][Y.name]
				X_rec = data.loc[data[self.best_feature_to_split] == value]
				
				self.children[value].perform_split(X_rec, Y_rec, self.name + str(value))
		else:
			# deciding class type by mode  
			self.node_value = int(Y.mode())


	def predict(self, example: pd.Series):
		# try and exept becouse there is a possibility that in test data are some new fetures values 
		try:
			if self.node_value == None:
				atrybut_to_split = example[self.best_feature_to_split]
				return self.children[atrybut_to_split].predict(example)
		except KeyError:
			print(f"There is no path in graph for fetures: {self.best_feature_to_split} and values: {example[self.best_feature_to_split]}")
			return None 
		return self.node_value
		

# TODO model working for non discrete values  
def age_distretization(df: pd.DataFrame):
	d = {range(0, 20): "young", range(20, 40): "middle", range(40, 100): "old"}

	df['Age'] = df['Age'].apply(lambda x: next((v for k, v in d.items() if x in k), 0))

	return df


def show_node(node: Node, tree: treelib.Tree, from_attr="root", parent=None):
	if node.node_value != None:
		tree.create_node(f"{from_attr} -> [{node.node_value}]", node.name, parent=parent)
	else:
		tree.create_node(f"{from_attr} -> {node.best_feature_to_split}", node.name, parent=parent)
		for attr_value, child_node in node.children.items():
			show_node(child_node, tree, attr_value, node.name)


if __name__ == "__main__":	

	df = pd.read_csv("titanic-homework.csv")
	df = df.drop(columns = ["Name", "PassengerId"])
	df = age_distretization(df)

	test = df.iloc[70:,:]
	df = df.iloc[:70,:]
	learn = df.drop(columns=["Survived"])
	print('Data loading complete!')
	

	tree_root = Node()
	tree = tree_root.perform_split(learn, df["Survived"])
	print('Training complete!')
	
	# show the tree
	tree = treelib.Tree()
	show_node(tree_root, tree)
	tree.show()

	# accuracy count 
	sum = 0 
	for _, element in test.iterrows():
		el_to_check = element.drop(columns=["Survived"])
		if tree_root.predict(el_to_check) == element["Survived"]:
			sum+=1
	print("accuracy: ", sum/len(test))
	# accuracity:  0.8333333333333334
