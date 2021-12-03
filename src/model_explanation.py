from model import text_classifier, iris_knn
from explainer import lime_explainer, shap_explainer, pdp_explainer

def train_model(config):
	if config["train_type"] == "NN_TEXT_CLASSIFIER":
		print('Start training nn text classifer')
		text_classifier.train_model(config["train"])

	if config["train_type"] == "KNN_IRIS":
		print('Start training KNN Iris classifier')
		iris_knn.train_model(config["train"])

def explain_model(config):
	if config["explain_type"] == "LIME" and config["train_type"] == "NN_TEXT_CLASSIFIER":
		print('Start generating explanation with LIME')
		lime_explainer.explain_text_nn(config["explain"])
	if config["explain_type"] == "SHAP" and config["train_type"] == "NN_TEXT_CLASSIFIER":
		print('Start generating explanation with SHAP')
		shap_explainer.explain_text_nn(config["explain"])

	if config["explain_type"] == "LIME" and config["train_type"] == "KNN_IRIS":
		print('Start generating explanation with LIME')
		lime_explainer.explain_knn(config["explain"])

	if config["explain_type"] == "PDP" and config["train_type"] == "KNN_IRIS":
		print('Start generating explanation with PDP')
		pdp_explainer.explain_knn(config["explain"])