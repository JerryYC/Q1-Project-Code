import os
import json
import pickle

import shap
import sklearn
from sklearn.model_selection import train_test_split

def train_model(config):
	X_train,X_test,Y_train,Y_test = train_test_split(*shap.datasets.iris(), test_size=0.2, random_state=0)
	knn = sklearn.neighbors.KNeighborsClassifier()
	knn.fit(X_train, Y_train)

	log_path = os.path.join(config["log_path"], f'{config["name"]}.json')
	with open(log_path, 'w') as f:
		json.dump({"Model Accuracy":knn.score(X_test, Y_test)}, f, indent=4)

	model_path = os.path.join(config["model_path"], f'{config["name"]}.sav')
	pickle.dump(knn, open(model_path, 'wb'))
	print(f'model saved at {model_path}')

 
	