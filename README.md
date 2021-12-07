# Causal Inference and Explainable AI

This is a repository that contains code for DSC180A section B06's Q1 Project: Causal Inference and Explainable AI. 

We conducted experiments for causal inference using the DoWhy library on two examples: hotel cancellation and mortality of twins. In the hotel cancellation example, we tried to make causal inference on the effect of having a different room assigned before check in on booking cancellation. In the mortality of twins example, we tried to make causal inference on the effect of weights on the mortality of twins. 

We also conducted experiments for Explainable AI using LIME, SHAP, and the Partial Dependence Plot(PDP) on two machine learning models: Text Classification Neural Network and K-Nearest-Neighbor Model. In this part, we tried to generate global explanations for the K-Nearest-Neighbor Model using PDP and generate local explanations for both machine learning model using LIME and SHAP.

## Authors
- [Jerry (Yung-Chieh) Chan](https://github.com/JerryYC)
- Apoorv Pochiraju
- Zhendong Wang
- Yujie Zhang


## Retrieving the data for Causal Inference:
If you plan to run the casaul inference pipeline, please follow those steps to retrieve the data:

* Hotel Cancellation: 

https://raw.githubusercontent.com/rfordatascience/tidytuesday/master/data/2020/2020-02-11/hotels.csv 

* Twins: 

The covariates data has 46 features: https://raw.githubusercontent.com/AMLab-Amsterdam/CEVAE/master/datasets/TWINS/twin_pairs_X_3years_samesex.csv

The outcome data contains mortality of the lighter and heavier twin: https://raw.githubusercontent.com/AMLab-Amsterdam/CEVAE/master/datasets/TWINS/twin_pairs_Y_3years_samesex.csv

The treatment data contains weight in grams of both the twins: https://raw.githubusercontent.com/AMLab-Amsterdam/CEVAE/master/datasets/TWINS/twin_pairs_T_3years_samesex.csv


## Running the project

* To install the dependencies, run the following command from the root directory of the project: `pip install -r requirements.txt`
* To run the casaul inference or model explanation pipeline, run `python run [target] [config]`. The following table list the configuations and the corresponding experiment.  

 target | config | experiment |
| :---: | :---: | :---: |
| test | None | Run XAI on a light model and CI on a simple dataset to make sure pipeline is working |
| XAI | config/model＿explanation/nn_text_cls_lime.json | Train and explain text classification neural network with LIME |
| XAI | config/model＿explanation/nn_text_cls_shap.json | Train and explain text classification neural network with SHAP |
| XAI | config/model＿explanation/knn_iris_lime.json | Train and explain classification KNN with LIME |
| XAI | config/model＿explanation/knn_iris_pdp.json | Train and explain classification KNN with PDP |

## Reference

* Lundberg, S. M., & Lee, S.-I. (2017). A Unified Approach to Interpreting Model Predictions. In I. Guyon, U. V. Luxburg, S. Bengio, H. Wallach, R. Fergus, S. Vishwanathan, & R. Garnett (Eds.), Advances in Neural Information Processing Systems 30 (pp. 4765–4774). Curran Associates, Inc. http://papers.nips.cc/paper/7062-a-unified-approach-to-interpreting-model-predictions.pdf
* Ribeiro, M. T., Singh, S., & Guestrin, C. (2016). “Why Should I Trust You?”: Explaining the Predictions of Any Classifier. Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining, San Francisco, CA, USA, August 13-17, 2016, 1135–1144.
* Zhao, Qingyuan, and Trevor Hastie. “Causal interpretations of black-box models.” Journal of Business & Economic Statistics, to appear. (2017)
