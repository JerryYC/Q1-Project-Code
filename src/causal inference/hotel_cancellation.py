# Setup
import logging.config
DEFAULT_LOGGING = {
    'version': 1,
    'disable_existing_loggers': False,
    'loggers': {
        '': {
            'level': 'INFO',
        },
    }
}

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


dataset = pd.read_csv('https://raw.githubusercontent.com/Sid-darthvader/DoWhy-The-Causal-Story-Behind-Hotel-Booking-Cancellations/master/hotel_bookings.csv')

############################################## FEATURE ENGINEERING ###############################################

def feature_engineer(dataset):
    # Total stay in nights
    dataset['total_stay'] = dataset['stays_in_week_nights']+dataset['stays_in_weekend_nights']
    # Total number of guests
    dataset['guests'] = dataset['adults']+dataset['children'] +dataset['babies']
    # Creating the different_room_assigned feature
    dataset['different_room_assigned']=0
    slice_indices =dataset['reserved_room_type']!=dataset['assigned_room_type']
    dataset.loc[slice_indices,'different_room_assigned']=1
    # Deleting older features
    dataset = dataset.drop(['stays_in_week_nights','stays_in_weekend_nights','adults','children','babies'
                            ,'reserved_room_type','assigned_room_type'],axis=1)
    dataset.isnull().sum() # Country,Agent,Company contain 488,16340,112593 missing entries
    dataset = dataset.drop(['agent','company'],axis=1)
    # Replacing missing countries with most freqently occuring countries
    dataset['country']= dataset['country'].fillna(dataset['country'].mode()[0])

    dataset = dataset.drop(['reservation_status','reservation_status_date','arrival_date_day_of_month'],axis=1)
    dataset = dataset.drop(['arrival_date_year'],axis=1)
    dataset = dataset.drop(['distribution_channel'], axis=1)
    dataset['different_room_assigned']= dataset['different_room_assigned'].replace(1,True)
    dataset['different_room_assigned']= dataset['different_room_assigned'].replace(0,False)
    dataset['is_canceled']= dataset['is_canceled'].replace(1,True)
    dataset['is_canceled']= dataset['is_canceled'].replace(0,False)
    dataset.dropna(inplace=True)
    return dataset

############################################## CREATING CAUSAL MODEL ###############################################
def create_model(dataset):
    causal_graph = """
    digraph {
    age[label="Age"];
    experience[label="Experience"];
    income[label="Income"];
    family[label="Family"];
    ccavg[label="Credit Card Spending"];
    edu[label="Education"];
    mortgage[label="Mortgage"];
    personal_loan[label="Personal Loan"];
    securities_account[label="Securities Account"];
    cd_account[label="CD Account"];
    online[label="Online"];
    credit_card[label="Credit Card"];
    age -> experience; age -> income; family -> income; age -> ccavg;
    age -> edu; age -> mortgage; age -> personal_loan; age -> family;
    experience -> income; experience -> personal_loan;
    income -> ccavg; income -> mortgage; income -> personal_loan; income -> securities_account;
    income -> cd_account; income -> credit_card;
    family -> ccavg; family -> personal_loan;
    ccavg -> personal_loan; edu -> personal_loan;
    edu -> income;
    mortgage, securities_account, cd_account, online, credit_card -> personal_loan
    }
    """

    model= dowhy.CausalModel(
        data = dataset,
        graph=causal_graph.replace("\n", " "),
        treatment='edu',
        outcome='personal_loan')
    return model

############################################## IDENITFYING CAUSAL EFFECT ###############################################

def identify_causal_effect(model):
    identified_estimand = model.identify_effect(proceed_when_unidentifiable=True)
    print(identified_estimand)
    return

############################################## ESTIMATING CAUSAL EFFECT ###############################################
def estimate_effect(model):
    estimate = model.estimate_effect(identified_estimand,
                                 method_name="backdoor.propensity_score_stratification",target_units="ate")
    print(estimate)
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
