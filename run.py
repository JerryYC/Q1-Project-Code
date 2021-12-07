#!/usr/bin/env python

import sys
import os
import json

sys.path.insert(0, 'src')

from utils import setup_env, get_data
from model_explanation import train_model, explain_model
from causal_inference import hotel_cancellation, twins
# from causal_inference import generate_analysis



def main(target):
    '''
    Runs the main project pipeline logic, given the targets.
    targets must contain: 'data', 'analysis', 'model'. 
    
    `main` runs the targets in order of data=>analysis=>model.
    '''

    ### Setup
    setup_env()

    if 'test' == target:
        print("Running test pipeline")
        ### Get Data
        get_data("test")

        ### XAI Test
        with open('config/test/model_explanation.json') as fh:
            test_XAI_cfg = json.load(fh)
        explain_model(test_XAI_cfg)

        ### CI
        with open('config/test/causal_inference.json') as fh:
            test_CI_cfg = json.load(fh)
        generate_analysis(test_CI_cfg)

    if 'XAI' == target:
        print("Running model explanation pipeline")
        with open(sys.argv[2]) as fh:
            XAI_cfg = json.load(fh)

        print(f'Model: {XAI_cfg["train_type"]}')
        print(f'Data: {XAI_cfg["data"]}')
        print(f'Explanation method: {XAI_cfg["explain_type"]}')

        if XAI_cfg["data"]:
            get_data(XAI_cfg["data"])

        train_model(XAI_cfg)
        explain_model(XAI_cfg)

    if 'CI' == target:
        print("Running causal inference explanation pipeline")
        get_data("CI")

        with open(sys.argv[2])  as fh:
            CI_cfg = json.load(fh)

        if CI_cfg["type"] == "hotel":
            hotel_cancellation.main(CI_cfg)
        if CI_cfg["type"] == "twins":
            twins.main(CI_cfg)

    return

if __name__ == '__main__':
    # run via:
    # python main.py data features model
    target = sys.argv[1]
    main(target)