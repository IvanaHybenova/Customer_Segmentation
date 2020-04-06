# -*- coding: utf-8 -*-
"""
Created on Sun Apr  5 17:15:48 2020

@author: ivana
"""

import os
import psycopg2
from sqlalchemy import create_engine
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn_pandas import DataFrameMapper
from customer_segmentation_helper import *
import pickle
import warnings
warnings.filterwarnings("ignore")
import sys

def main(argv):
    #### Define parameters to load saved model, scaler and new customers
    
    # path to the dataset
    path_to_file_data = sys.argv[1]
    
    # path to the model
    path_to_file_model = sys.argv[2]
    
    # path to the mapper of the scaler
    path_to_file_scaler_mapper = sys.argv[3]
    
    # Boolean value, whether dataset should be pulled from the database (PostgreSQL)
    from_database = False
    
    ## Credentials to create connection to the database with psycopg2 and sqlalchemy
    #database_credentials = pd.DataFrame({
    #    'host': [os.environ['HOST_WAREHOUSE']],
    #    'database': [os.environ['NAME_WAREHOUSE']],
    #    'user': [os.environ['USER_WAREHOUSE']],
    #    'password': [os.environ['PASSWORD_WAREHOUSE']],
    #    'engine': [os.environ['ENGINE_WAREHOUSE']]
    #})
    
    ## SQL query to download customers table
    ## Make sure to download the whole table as the original one will get replaced, not only specified features will be used
    #sql_query =  """
    #               SELECT *
    #               FROM customers
    #             """     
    
    
    # categorical variables
    cat_features = ['gender']
    
    # Names of numerical features
    numerical_features = ['age', 'annual_income_thousands', 'spending_score']
    
    #### Load data, model and feature scaler
    data, segmented_data, model, scaler_mapper = load_data_model(from_database=False, path_to_file_data=path_to_file_data, 
                                                                 path_to_file_scaler_mapper = path_to_file_scaler_mapper, 
                                                                 path_to_file_model = path_to_file_model)
    
    #### Check whether there are new customers
    if len(data) == 0:
        raise SystemExit("There are no new customers to assign segment to!")
    
    #### Preprocess data
    # encoding of new_data
    new_data, cat_dummy = dummy_encode(data, cat_cols = cat_features)
    print('Data with extra dummy columns:')
    new_data.head()    
    print('Data with scaled features:')
    new_data, numerical_scaled = preprocess_data(new_data, dummy_cols = cat_dummy, num_cols = numerical_features, scaler_mapper = scaler_mapper)
    
    #### Assign segments to the new customers
    print('Data with assigned segments and segment origin:')
    new_data = assign_segments(model = model, data = new_data, dummy_cols = cat_dummy, scaled_cols = numerical_scaled)
    
    ### Save data
    save_data(from_database = from_database, path_to_file_data = path_to_file_data, data = new_data, segmented_data = segmented_data)
        
if __name__ == "__main__":
   main(sys.argv[1:])
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    