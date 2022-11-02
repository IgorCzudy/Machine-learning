from collections import defaultdict
import numpy as np 
import pandas as pd


class Node(object):
	def __init__(self):
		self.children = defaultdict(lambda: Node()) # Typ: {wartość cechy: Node}, wierzchołeki znajdujące się w odnogach
		self.node_value = None # Przechowuje wartość o ile wierzchołek jest liściem
        
	def entropy_count(self, y: pd.Series):
		y_value_counts = y.value_counts()
		return -(y_value_counts/y_value_counts.sum() * np.log2(y_value_counts/y_value_counts.sum())).sum()

	def conditional_entropy_count(self, x:pd.Series, y:pd.Series):

		xy = pd.concat([x, y], axis=1)
		
		summ = 0
		for features_value in x.unique():
			summ += (y.loc[x==features_value].count()/y.count()) * self.entropy_count(y.loc[x==features_value])
		return summ

	def information_gain_count(self, entropy: int, contitional_entropy: int):
		return entropy - contitional_entropy


	def perform_split(self, data: pd.DataFrame, Y: pd.Series):
		# Znajdź najlepszy podział data
		
		s = {}
		for feature in data.columns:
			s[feature] = self.information_gain_count(self.entropy_count(Y), self.conditional_entropy_count(data[feature], Y))

		
		best_feature_to_split = max(s, key=s.get)

		# if uzyskano poprawę funkcji celu (bądź inny, zaproponowany przez Ciebie warunek):
		if s[best_feature_to_split] > 0:
			for value in data[best_feature_to_split].unique():
				xy = pd.concat([data, Y], axis=1)
				self.children[value].perform_split(data.loc[data[best_feature_to_split] == value], xy.loc[xy[best_feature_to_split] == value]["Survived"])
		else:
			# najcześtrz wartośc z survived 
			self.node_value = int(Y.mode())

		return self.children


		# if uzyskano poprawę funkcji celu (bądź inny, zaproponowany przez Ciebie warunek):
			#podziel dane na części d1, d2, d3, ... dla każdej możliwej wartości wybranej cechy
		#	for value in możliwe wartości wybranej cechy:
		#		self.children[value].perform_split( odpowiednie d*)
		#else:
			#obecny Node jest liściem, zapisz jego odpowiedź

	def predict(self, example):
		pass
		"""
		if not Node jest liściem:
			return self.children[wartość cechy po której robimy podział].predict(example)
		return self.node_value = zwróć wartość (Node jest liściem)
		"""
		
		
def age_distretization(df: pd.DataFrame):
	d = {range(0, 20): "young", range(20, 40): "middle", range(40, 100): "old"}

	df['Age'] = df['Age'].apply(lambda x: next((v for k, v in d.items() if x in k), 0))

	return df


if __name__ == "__main__":	
	### Implementacja wczytywania danych i losowy podział na dane uczące i testowe
	
	df = pd.read_csv("titanic-homework.csv")
	df = df.drop(columns = ["Name", "PassengerId"])
	df = age_distretization(df)

	#print(Node().entropy_count(df["Survived"]))
	#print(Node().conditional_entropy_count(df["Pclass"], df["Survived"]))


	# print("Pclass", Node().information_gain_count(Node().entropy_count(df["Survived"]), Node().conditional_entropy_count(df["Pclass"], df["Survived"])))
	# print("Sex", Node().information_gain_count(Node().entropy_count(df["Survived"]), Node().conditional_entropy_count(df["Sex"], df["Survived"])))
	# print("Age", Node().information_gain_count(Node().entropy_count(df["Survived"]), Node().conditional_entropy_count(df["Age"], df["Survived"])))
	# print("SibSp", Node().information_gain_count(Node().entropy_count(df["Survived"]), Node().conditional_entropy_count(df["SibSp"], df["Survived"])))
	# print("Parch", Node().information_gain_count(Node().entropy_count(df["Survived"]), Node().conditional_entropy_count(df["Parch"], df["Survived"])))


	#data = ... 
	#test_data=...
	#print('Data loading complete!')

	tree_root = Node()
	tree = tree_root.perform_split(df.drop(columns=["Survived"]), df["Survived"])
	print('Training complete!')

	### Implementacja zmierzenia trafności klasyfikacji (!) na danych testowych i uczących np.
	# for element in test_data:
	#      y = tree_root.predict(element)
	
