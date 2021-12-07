import logging.config


logging.config.dictConfig(DEFAULT_LOGGING)
# Disabling warnings output
import warnings
from sklearn.exceptions import DataConversionWarning, ConvergenceWarning
# warnings.filterwarnings(action='ignore', category=DataConversionWarning)
# warnings.filterwarnings(action='ignore', category=ConvergenceWarning)
# warnings.filterwarnings(action='ignore', category=UserWarning)
warnings.filterwarnings("ignore")

import dowhy
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import Image, display


#The covariates data has 46 features

############################################## FEATURE ENGINEERING ###############################################


def feature_engineering():
    x = pd.read_csv("https://raw.githubusercontent.com/AMLab-Amsterdam/CEVAE/master/datasets/TWINS/twin_pairs_X_3years_samesex.csv")

    #The outcome data contains mortality of the lighter and heavier twin
    y = pd.read_csv("https://raw.githubusercontent.com/AMLab-Amsterdam/CEVAE/master/datasets/TWINS/twin_pairs_Y_3years_samesex.csv")

    #The treatment data contains weight in grams of both the twins
    t = pd.read_csv("https://raw.githubusercontent.com/AMLab-Amsterdam/CEVAE/master/datasets/TWINS/twin_pairs_T_3years_samesex.csv")

    #_0 denotes features specific to the lighter twin and _1 denotes features specific to the heavier twin



    lighter_columns = ['pldel', 'birattnd', 'brstate', 'stoccfipb', 'mager8',
           'ormoth', 'mrace', 'meduc6', 'dmar', 'mplbir', 'mpre5', 'adequacy',
           'orfath', 'frace', 'birmon', 'gestat10', 'csex', 'anemia', 'cardiac',
           'lung', 'diabetes', 'herpes', 'hydra', 'hemo', 'chyper', 'phyper',
           'eclamp', 'incervix', 'pre4000', 'preterm', 'renal', 'rh', 'uterine',
           'othermr', 'tobacco', 'alcohol', 'cigar6', 'drink5', 'crace',
           'data_year', 'nprevistq', 'dfageq', 'feduc6', 'infant_id_0',
           'dlivord_min', 'dtotord_min', 'bord_0',
           'brstate_reg', 'stoccfipb_reg', 'mplbir_reg']
    heavier_columns = [ 'pldel', 'birattnd', 'brstate', 'stoccfipb', 'mager8',
           'ormoth', 'mrace', 'meduc6', 'dmar', 'mplbir', 'mpre5', 'adequacy',
           'orfath', 'frace', 'birmon', 'gestat10', 'csex', 'anemia', 'cardiac',
           'lung', 'diabetes', 'herpes', 'hydra', 'hemo', 'chyper', 'phyper',
           'eclamp', 'incervix', 'pre4000', 'preterm', 'renal', 'rh', 'uterine',
           'othermr', 'tobacco', 'alcohol', 'cigar6', 'drink5', 'crace',
           'data_year', 'nprevistq', 'dfageq', 'feduc6',
           'infant_id_1', 'dlivord_min', 'dtotord_min', 'bord_1',
           'brstate_reg', 'stoccfipb_reg', 'mplbir_reg']

    data = []

    for i in range(len(t.values)):

        #select only if both <=2kg
        if t.iloc[i].values[1]>=2000 or t.iloc[i].values[2]>=2000:
            continue

        this_instance_lighter = list(x.iloc[i][lighter_columns].values)
        this_instance_heavier = list(x.iloc[i][heavier_columns].values)

        #adding weight
        this_instance_lighter.append(t.iloc[i].values[1])
        this_instance_heavier.append(t.iloc[i].values[2])

        #adding treatment, is_heavier
        this_instance_lighter.append(0)
        this_instance_heavier.append(1)

        #adding the outcome
        this_instance_lighter.append(y.iloc[i].values[1])
        this_instance_heavier.append(y.iloc[i].values[2])
        data.append(this_instance_lighter)
        data.append(this_instance_heavier)

    cols = [ 'pldel', 'birattnd', 'brstate', 'stoccfipb', 'mager8',
       'ormoth', 'mrace', 'meduc6', 'dmar', 'mplbir', 'mpre5', 'adequacy',
       'orfath', 'frace', 'birmon', 'gestat10', 'csex', 'anemia', 'cardiac',
       'lung', 'diabetes', 'herpes', 'hydra', 'hemo', 'chyper', 'phyper',
       'eclamp', 'incervix', 'pre4000', 'preterm', 'renal', 'rh', 'uterine',
       'othermr', 'tobacco', 'alcohol', 'cigar6', 'drink5', 'crace',
       'data_year', 'nprevistq', 'dfageq', 'feduc6',
       'infant_id', 'dlivord_min', 'dtotord_min', 'bord',
       'brstate_reg', 'stoccfipb_reg', 'mplbir_reg','wt','treatment','outcome']
    df = pd.DataFrame(columns=cols,data=data)
    return df

############################################## CREATING CAUSAL MODEL ###############################################

def create_model(df):
    model=dowhy.CausalModel(
        data = df,
        treatment='treatment',
        outcome='outcome',
        common_causes='gestat10'
        )
    return model

############################################## IDENITFYING CAUSAL EFFECT ###############################################

def identify_causal_effect(model):
    identified_estimand = model.identify_effect(proceed_when_unidentifiable=True)
    print(identified_estimand)
    return

############################################## ESTIMATING CAUSAL EFFECT ###############################################

def estimate_effect_linear_regression(model):
    estimate = model.estimate_effect(identified_estimand,
        method_name="backdoor.linear_regression", test_significance=True)

    print(estimate)
    print("ATE", np.mean(data_1["outcome"])- np.mean(data_0["outcome"]))
    print("Causal Estimate is " + str(estimate.value))
    return


def estimate_effect_propensity_score(model):
    estimate = model.estimate_effect(identified_estimand,
        method_name="backdoor.propensity_score_matching")

    print("Causal Estimate is " + str(estimate.value))

    print("ATE", np.mean(data_1["outcome"])- np.mean(data_0["outcome"]))
    return

############################################## REFUTING RANDOM COMMON CAUSE ###############################################

def refute_random_common_cause(model):
    refute1_results=model.refute_estimate(identified_estimand, estimate,
        method_name="random_common_cause")
    print(refute1_results)
    return

############################################## REFUTING PLACEBO TREATMENT ###############################################

def refute_placebo_treatment(model):
    refute2_results=model.refute_estimate(identified_estimand, estimate,
        method_name="placebo_treatment_refuter")
    print(refute2_results)
    return

############################################## REFUTING DATA SUBSET ###############################################


def refute_data_subset(model):
    refute3_results=model.refute_estimate(identified_estimand, estimate,
        method_name="data_subset_refuter")
    print(refute3_results)
    return



def main():

    data = feature_engineering()
    model = create_model(data)
    identify_causal_effect(model)
    estimate_effect_linear_regression(model)
    estimate_effect_propensity_score(model)
    refute_random_common_cause(model)
    refute_placebo_treatment(model)
    refute_data_subset(model)

if __name__=="__main__":
    main()
