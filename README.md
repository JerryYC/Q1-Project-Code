# Causal Inference and Explainable AI

This is a repository ... TODO (Short introduction)

## Authors
- [Jerry (Yung-Chieh) Chan](https://github.com/JerryYC)
- TODO


## Retrieving the data for Causal Inference:
If you plan to run the casaul inference pipeline, please follow those steps to retrieve the data:

* TODO


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
