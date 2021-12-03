import os
import time 
import json

import torch
from torchtext.datasets import AG_NEWS
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from torch.utils.data import DataLoader
from torch import nn
from torch.utils.data.dataset import random_split
from torchtext.data.functional import to_map_style_dataset

device = torch.device("cpu")

def train_model(config):

	if os.path.exists(os.path.join(config["model_path"],f'{config["name"]}.pt')):
		print("Model already exist!")
		return

	############################################### DATA PIPELINE ###############################################

	train_iter = AG_NEWS(split='train')
	tokenizer = get_tokenizer('basic_english')
	def yield_tokens(data_iter):
	    for _, text in data_iter:
	        yield tokenizer(text)
	vocab = build_vocab_from_iterator(yield_tokens(train_iter), specials=["<unk>"])
	vocab.set_default_index(vocab["<unk>"])
	text_pipeline = lambda x: vocab(tokenizer(x))
	label_pipeline = lambda x: int(x) - 1

	def collate_batch(batch):
	    label_list, text_list, offsets = [], [], [0]
	    for (_label, _text) in batch:
	         label_list.append(label_pipeline(_label))
	         processed_text = torch.tensor(text_pipeline(_text), dtype=torch.int64)
	         text_list.append(processed_text)
	         offsets.append(processed_text.size(0))
	    label_list = torch.tensor(label_list, dtype=torch.int64)
	    offsets = torch.tensor(offsets[:-1]).cumsum(dim=0)
	    text_list = torch.cat(text_list)
	    return label_list.to(device), text_list.to(device), offsets.to(device)

	############################################ INITIALIZE MODEL/DATASET ############################################

	train_iter = AG_NEWS(split='train')
	num_class = len(set([label for (label, text) in train_iter]))
	vocab_size = len(vocab)
	emsize = 64
	model = TextClassificationModel(vocab_size, emsize, num_class).to(device)
	# Hyperparameters
	EPOCHS = config["epoch"] # epoch
	LR = config["lr"]  # learning rate
	BATCH_SIZE = config["batch_size"] # batch size for training
	  
	criterion = torch.nn.CrossEntropyLoss()
	optimizer = torch.optim.SGD(model.parameters(), lr=LR)
	scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.1)
	total_accu = None

	train_iter, test_iter = AG_NEWS()
	train_dataset = to_map_style_dataset(train_iter)
	test_dataset = to_map_style_dataset(test_iter)
	num_train = int(len(train_dataset) * 0.95)
	split_train_, split_valid_ = \
	    random_split(train_dataset, [num_train, len(train_dataset) - num_train])

	train_dataloader = DataLoader(split_train_, batch_size=BATCH_SIZE,
	                              shuffle=True, collate_fn=collate_batch)
	valid_dataloader = DataLoader(split_valid_, batch_size=BATCH_SIZE,
	                              shuffle=True, collate_fn=collate_batch)
	test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE,
	                             shuffle=True, collate_fn=collate_batch)

	############################################### TRAINING ###############################################
	
	log = {}
	for epoch in range(1, EPOCHS + 1):
	    epoch_start_time = time.time()
	    accu_train = train(model, criterion, optimizer, train_dataloader)
	    accu_val = evaluate(model, criterion, valid_dataloader)
	    if total_accu is not None and total_accu > accu_val:
	        scheduler.step()
	    else:
	        total_accu = accu_val
	    print('-' * 59)
	    print('| end of epoch {:3d} | time: {:5.2f}s | '
	          'valid accuracy {:8.3f} '.format(epoch,
	                                           time.time() - epoch_start_time,
	                                           accu_val))
	    print('-' * 59)

	    log[f'Epoch {epoch}'] = {
	    	"Training Accuracy": accu_train,
	    	"Validation Accuracy": accu_val
	    }

	############################################### SAVE MODEL ###############################################

	
	with open(os.path.join(config["log_path"], config["name"] + 'accuracy.json'), 'w') as f:
		json.dump(log, f, indent=4)

	scripted_model = torch.jit.script(model)
	model_path = os.path.join(config["model_path"],f'{config["name"]}.pt')
	scripted_model.save(model_path)
	print(f'Model saved at {model_path}')


def train(model, criterion, optimizer, dataloader):
    model.train()
    total_acc, total_count = 0, 0
    start_time = time.time()

    for idx, (label, text, offsets) in enumerate(dataloader):
        optimizer.zero_grad()
        predicted_label = model(text, offsets)
        loss = criterion(predicted_label, label)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
        optimizer.step()
        total_acc += (predicted_label.argmax(1) == label).sum().item()
        total_count += label.size(0)
    return total_acc/total_count

def evaluate(model, criterion, dataloader):
    model.eval()
    total_acc, total_count = 0, 0
    with torch.no_grad():
        for idx, (label, text, offsets) in enumerate(dataloader):
            predicted_label = model(text, offsets)
            loss = criterion(predicted_label, label)
            total_acc += (predicted_label.argmax(1) == label).sum().item()
            total_count += label.size(0)
    return total_acc/total_count



class TextClassificationModel(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_class):
        super(TextClassificationModel, self).__init__()
        self.embedding = nn.EmbeddingBag(vocab_size, embed_dim, sparse=True)
        self.fc = nn.Linear(embed_dim, num_class)
        self.init_weights()

    def init_weights(self):
        initrange = 0.5
        self.embedding.weight.data.uniform_(-initrange, initrange)
        self.fc.weight.data.uniform_(-initrange, initrange)
        self.fc.bias.data.zero_()

    def forward(self, text, offsets):
        embedded = self.embedding(text, offsets)
        return self.fc(embedded)
