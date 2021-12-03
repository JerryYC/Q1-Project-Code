import numpy as np
import os
import pickle

import torch
from torchtext.datasets import AG_NEWS
from torchtext.data.functional import to_map_style_dataset
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from torch.utils.data import DataLoader

import shap
from sklearn.model_selection import train_test_split
from lime.lime_text import LimeTextExplainer
from lime.lime_tabular import LimeTabularExplainer

def softmax(scores):
    return np.exp(scores) / np.exp(scores).sum()

def predict_multiple(model, text_pipeline, inputs):
    rt = []
    for text in inputs:
        with torch.no_grad():
            text = torch.tensor(text_pipeline(text))
            output = model(text, torch.tensor([0]))
        rt.append(softmax(output[0].numpy()))
    return np.array(rt)

def explain_text_nn(config):

	############################################### DATA/MODEL PREPARATION ###############################################

	print(f'Model Loaded from {config["model_path"]}')
	model = torch.jit.load(config["model_path"])
	train_iter = AG_NEWS(split='train')
	tokenizer = get_tokenizer('basic_english')
	def yield_tokens(data_iter):
	    for _, text in data_iter:
	        yield tokenizer(text)
	vocab = build_vocab_from_iterator(yield_tokens(train_iter), specials=["<unk>"])
	vocab.set_default_index(vocab["<unk>"])
	text_pipeline = lambda x: vocab(tokenizer(x))
	data_iter = AG_NEWS(split = "test")
	dataset = to_map_style_dataset(data_iter)
	ids = np.random.randint(0, len(dataset) ,config["num_sample"])

	############################################### LIME ###############################################

	limeexplainer = LimeTextExplainer(class_names=["World", "Sports", "Business", "Sci/Tec"])
	
	for i in ids:
		print(f'generating explanation for test data {i}')
		label, text = dataset[i]
		exp = limeexplainer.explain_instance(text, lambda x: predict_multiple(model, text_pipeline, x), 
			                                 num_features=config["num_features"], labels=[0,1,2,3])
		out_path = os.path.join(config["out_path"],f'{config["name"]} {i}.html')
		exp.save_to_file(out_path)
		print(f'Explanation saved at {out_path}')


def explain_knn(config):
	# load the model from disk
	print(f'Model Loaded from {config["model_path"]}')
	knn = pickle.load(open(config["model_path"], 'rb'))
	X_train,X_test,Y_train,Y_test = train_test_split(*shap.datasets.iris(), test_size=0.2, random_state=0)
	limeexplainer = LimeTabularExplainer(X_train.to_numpy(), feature_names = X_train.columns, class_names = [0,1,2], mode = 'classification')
	exp = limeexplainer.explain_instance(X_test.iloc[0,:], knn.predict_proba, labels = [0,1,2])
	out_path = os.path.join(config["out_path"],f'{config["name"]}.html')
	exp.save_to_file(out_path)
	print(f'Explanation saved at {out_path}')

