import pickle
import os 

from sklearn.inspection import PartialDependenceDisplay
from sklearn.model_selection import train_test_split

import shap


def explain_knn(config):
	# load the model from disk
	print(f'Model Loaded from {config["model_path"]}')
	knn = pickle.load(open(config["model_path"], 'rb'))
	X_train,X_test,Y_train,Y_test = train_test_split(*shap.datasets.iris(), test_size=0.2, random_state=0)
	out_path = os.path.join(config["out_path"],f'{config["name"]}.png')
	PartialDependenceDisplay.from_estimator(knn, X_test, [0,1,2], target=2, kind='both').figure_.savefig(out_path)
	print(f'Explanation saved at {config["out_path"]}')