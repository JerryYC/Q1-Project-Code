import sys
import os

def mkdir(d):
	if not os.path.exists(d):
		os.makedirs(d)

def setup_env():
	mkdir("src/data")
	mkdir("src/data/model")
	mkdir("src/data/out")
	mkdir("src/data/dataset")
	mkdir("src/data/log")

def get_data(config):
	pass