import numpy as np
import os

import torch
from torchtext.datasets import AG_NEWS
from torchtext.data.functional import to_map_style_dataset
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from torch.utils.data import DataLoader

import shap
from shap import Explainer

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

	############################################### SHAP ###############################################

	shapexplainer = Explainer(lambda x: predict_multiple(model, text_pipeline, x),
		shap.maskers.Text(r"\w+"), output_names=["World", "Sports", "Business", "Sci/Tec"])
	
	shap_values = shapexplainer([dataset[id][1] for id in ids])

	for i in range(len(ids)):
		print(f'generating explanation for test data {i}')
		out = shap.plots.text(shap_values[i], display = False)
		out_path = os.path.join(config["out_path"],f'{config["name"]} {ids[i]}.html')
		with open(out_path, 'w') as f:
			f.write(out)
		print(f'Explanation saved at {out_path}')


