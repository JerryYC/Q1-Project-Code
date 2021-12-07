# Setup
import logging.config

# Disabling warnings output
import warnings
from sklearn.exceptions import DataConversionWarning, ConvergenceWarning
# warnings.filterwarnings(action='ignore', category=DataConversionWarning)
# warnings.filterwarnings(action='ignore', category=ConvergenceWarning)
# warnings.filterwarnings(action='ignore', category=UserWarning)


import dowhy
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import Image, display



############################################## FEATURE ENGINEERING ###############################################

def feature_engineering():
    dataset = pd.read_csv('https://raw.githubusercontent.com/Sid-darthvader/DoWhy-The-Causal-Story-Behind-Hotel-Booking-Cancellations/master/hotel_bookings.csv')

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
    causal_graph = """digraph {
    different_room_assigned[label="Different Room Assigned"];
    is_canceled[label="Booking Cancelled"];
    booking_changes[label="Booking Changes"];
    previous_bookings_not_canceled[label="Previous Booking Retentions"];
    days_in_waiting_list[label="Days in Waitlist"];
    lead_time[label="Lead Time"];
    market_segment[label="Market Segment"];
    country[label="Country"];
    U[label="Unobserved Confounders"];
    is_repeated_guest;
    total_stay;
    guests;
    meal;
    hotel;
    U->different_room_assigned; U->is_canceled;U->required_car_parking_spaces;
    market_segment -> lead_time;
    lead_time->is_canceled; country -> lead_time;
    different_room_assigned -> is_canceled;
    country->meal;
    lead_time -> days_in_waiting_list;
    days_in_waiting_list ->is_canceled;
    previous_bookings_not_canceled -> is_canceled;
    previous_bookings_not_canceled -> is_repeated_guest;
    is_repeated_guest -> is_canceled;
    total_stay -> is_canceled;
    guests -> is_canceled;
    booking_changes -> different_room_assigned; booking_changes -> is_canceled;
    hotel -> is_canceled;
    required_car_parking_spaces -> is_canceled;
    total_of_special_requests -> is_canceled;
    country->{hotel, required_car_parking_spaces,total_of_special_requests,is_canceled};
    market_segment->{hotel, required_car_parking_spaces,total_of_special_requests,is_canceled};
    }"""

    model= dowhy.CausalModel(
        data = dataset,
        graph=causal_graph.replace("\n", " "),
        treatment='different_room_assigned',
        outcome='is_canceled')
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


def main(config):
    warnings.filterwarnings("ignore")
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
    dataset = feature_engineering()
    model = create_model(dataset)
    identify_causal_effect(model)
    estimate_effect(model)
    if config["refute"]:
        refute_random_common_cause(model)
        refute_placebo_treatment(model)
        refute_data_subset(model)

if __name__=="__main__":
    main()
