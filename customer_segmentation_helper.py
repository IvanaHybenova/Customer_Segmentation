# -*- coding: utf-8 -*-
"""
Created on Sun Apr  5 16:47:01 2020

@author: ivana
"""

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn_pandas import DataFrameMapper
import pickle
import psycopg2
from sqlalchemy import create_engine
import warnings
warnings.filterwarnings("ignore")

'''CLUSTER ANALYSIS'''

class Data:
    def __init__(self, from_database=False, path_to_file=None, sql_query=None, database_credentials = None, cat_cols=None, num_cols = None):
        """
        The class containes all the functions for pulling the data, data preprocessing, clustering and saving the results.

        Parameters
        ----------
        from_database : bool
            If yes, the data will be pulled from and saved to a PostgreSQL database. Train model and scaler will be saved to that database, too.
        path_to_file : str
            Path to the csv file with the data, which will be used if from_database argument equals False.
        sql_query : str
            A string containing a select statement used to download data from the database. Needed if from_database argument eaquals True.
        database_credentials : DataFrame
            A dataframe with strings of host, database, user, password, and string to create an engin. Needed if from_database argument eaquals True.
        cat_cols : array 
            An array of names of columns in the data that are categorical.
        num_cols : array 
            An array of names of columns in the data that are numerical.

        Returns
        -------
        DataFrame
            DataFrame with all the original columns, extended by scaled numerical columns that are in num_cals array and scaled dummy columns of all 
            categorical columns from cat_cols array.
            
        """
        
        # create new copies instead of references
        self.cat_cols = cat_cols
        self.num_cols = num_cols
        self.from_database = from_database
        self.path_to_file = path_to_file
        self.sql_query = sql_query
        self.database_credentials = database_credentials
        self.data, self.dummy_cols, self.scaled_cols, self.scaler_mapper = self._create_data()
        self.kmeans = None
                   
    def elbow_method(self, k):
        """
        Calculates and plot within clusters sum of squares for number of clusters in range from 1 to k.
        
        The functions first loop through the whole range of number of clusters and stores calculated wcss
        for each number in a array. The array is passed to _plot_elbow_method function, that plots stored results.

        Parameters
        ----------
        k : int
            Max number of clusters to calculate within clusters sum of squares for.

        """
            
        # array to save within clusters sum of squares
        wcss = []  
        for i in range(1, k+1):
            kmeans = KMeans(n_clusters = i, init = 'k-means++', max_iter = 300, n_init = 10, random_state = 0)
            kmeans.fit(self.data[self.dummy_cols + self.scaled_cols])
            wcss.append(kmeans.inertia_)  # intertia computes wcss
            
        self._plot_elbow_method(k = k, wcss = wcss)
    
    def get_clusters(self, n_clusters):    
        """
        Runs KMeans clustering adds 'segment' column with the cluster number the datapoint belongs to and label 
        to 'segment_origin' column implying that the datapoint was used for clustering to the data dataframe. Saves fitted KMeans model.
        
        The function first fits and predicts KMeans algorithm with passed number of clusters. The segment and segment_column are
        added to the original dataset. Stored model is saved in kmeans variable. Function also displays the dataset extended by the results.

        Parameters
        ----------
        n_clusters : int
           Chosen optimal number of clusters.
     
        """
   
        # Applying k-means to the passed dataset
        kmeans = KMeans(n_clusters = n_clusters, init = 'k-means++', max_iter = 300, n_init = 10, random_state = 0)
        self.data['segment'] = kmeans.fit_predict(self.data[self.dummy_cols + self.scaled_cols])
        self.data['segment_origin'] = 'created'
        self.kmeans = kmeans
        print('Dataframe with added segment label:')
        print()
        try:
            display(self.data.head())
        except:
            print(self.data.head())
    
    def visualize_clusters(self):
        """
        Uses PCA to visualize clusters on 2-dimensional chart.
        
        Function calls _run_pca function that returns 2 components, that are then plotted. 

        """

        self._run_pca()
        plt.figure(figsize=(15, 10))
        plt.scatter(x='x', y='y', data=self.data, alpha=0.75, c='segment')
        plt.xlabel('PC1', fontsize=15)
        plt.ylabel('PC2', fontsize=15)
        plt.title('Segments in 2-dimensional space', fontdict = {'fontsize' : 20})
        plt.show()
        
    def describe_clusters(self, ymax = 100):
        """
        Prints out percentage of each category and plots distribution of each numerical feature for each cluster.
        
        Functions loops through each cluster and subset data for that cluster only. For that subset prints number of the cluster
        and number of customers in the cluster. Then calls _cat_features_summary function to get and proportion of categories
        for categorical variables. Then it creates data frame with min, max, mean value and quantiles for each numerical variable
        and loops through each numerical feature to get the values and puts them in the dataframe, which is then displayed.
        Eventually function plots histogram for each numerical variable. All charts has the same y axis limit and x axis limit
        is everytime maximum value of that numerical feature, so the charts are comparable across the clusters.
        
        Parameters
        ----------
        ymax : int
           Limit of y axis, applied to all charts created by this function.
     
        """
        
        for cluster in self.data['segment'].sort_values().unique():
            data_cluster = self.data[self.data['segment'] == cluster].copy()
            print('Segment: ' + str(cluster) + '    ' + 'Number of Customers: ' + str(len(data_cluster)))
            self._cat_features_summary(data_cluster)
            cluster_describe_df = pd.DataFrame(columns = ['measure'])
            cluster_describe_df['measure'] = ['Min value', '25 %', '50 %', 'Mean', '75 %', 'Max value']
            for feature in self.num_cols:   
                cluster_describe_df[feature]= [round(data_cluster[feature].min(),0), round(data_cluster[feature].quantile(.25),0), round(data_cluster[feature].quantile(.50),0), round(data_cluster[feature].mean(),0), 
                                               round(data_cluster[feature].quantile(.75),0), round(data_cluster[feature].min(),0)]
            try:
                display(cluster_describe_df)
            except:
                print(cluster_describe_df)
            plt.figure(figsize = (30,3))
            i = 1
            for feature in self.num_cols:  
                plt.subplot(1, len(self.num_cols), i)
                i += 1
                plt.hist(data_cluster[feature], bins = 20)
                plt.title('Distribution of ' + str(feature), fontdict = {'fontsize': 15})
                plt.xlim(0, self.data[feature].max())
                plt.ylim(0,ymax)
                plt.ylabel('Counts', fontsize=10)
                plt.xlabel(feature, fontsize=10)
            plt.show()
            
    def save_segmented_data_model(self):
        """
        Saves dataset with added segments and column labeling them as the cases that were used to create segments, scaler and model.
        
        The function first drops all the extra columns in the original dataframe, then evaluates value of from_database attribute, 
        if that is True, it saves the dataset with original columns and results to the customers table, that is replaced if that 
        already exists. The model and scaler are added to models table.
        If from_database attribute equals false, the data, model and scaler are saved a csv file, where the data file replace the original one,
        and model and scaler are saved in current working directory.
        
        Parameters
        ----------
        ymax : int
           Limit of y axis, applied to all charts created by this function.
           
        """
        self.data.drop(self.dummy_cols + self.scaled_cols +['x','y'], axis = 1, inplace = True)
        
        if self.from_database:
                # replacing the original dataframe to the warehouse
                engine = create_engine(engine_link = self.database_credentials['engine'].item())  
                print("Overwriting database in the warehouse")
                self.data.to_sql('customers', engine, if_exists='replace',index=False, method = 'multi')    
                engine.dispose()
                
                
                # saving the model and scaler_mapper                
                conn = psycopg2.connect(host = self.database_credentials['host'].item(),
                                        database = self.database_credentials['database'].item(), 
                                        user = self.database_credentials['user'].item(), 
                                        password = self.database_credentials['password'].item())
                cur = conn.cursor()

                #### # Assuming you have a postgres table with columns model_name, model_file, scaler_file
                pickled_model = pickle.dumps(self.kmeans)
                pickled_scaler_mapper = pickle.dumps(self.scaler_mapper)
                sql = "INSERT INTO models (model_name, model_file, scaler_filer)  VALUES(%s, %s, %s)"
                cur.execute(sql, ('customer_segmentation', psycopg2.Binary(pickled_model), psycopg2.Binary(pickled_scaler_mapper)))
                conn.commit() 
                
                cur.close()
                conn.close()
        else:
            self.data.to_csv(self.path_to_file, index = False)
            pickle.dump(self.kmeans, open('model.pkl', 'wb'))
            pickle.dump(self.scaler_mapper, open('scaler_mapper.pkl', 'wb'))
    
    def _create_data(self):
        """
        Loads dataframe from a csv file or a database, dummy encode categorical variables and scale all the columns for clustering.
        
        The function based on value of from_database attribute either pulls data from a database utilizing passed sql query
        and dataframe with host, database, user and password, or loads a csv file from passed path and display first 5 rows.
        Then the function calls _dummy_encode function and diplays the first 5 rows with dummy columns for visual check.
        After that calls _scale_data function and displays the first 5 rows with scalled dummy features and extended by
        scaled numerical features.
        
        Returns
        -------
        data: DataFrame
            DataFrame with all the loaded columns extended by dummy encoded and scaled categorical columns and scaled numerical columns.
        dummy_cols: array
            Array of names of dummy columns created from passed categorical columns.
        scaled_cols: array
            Array of names scaled numerical columns.
        scaler_mapper: obj
            Fitted standard scaler, wrapped in DataFrameMapper, so that its output is dataframe instead of a numpy array.     
            
        """
       
        if self.from_database:
            conn = psycopg2.connect(host = self.database_credentials['host'].item(),
                                    database = self.database_credentials['database'].item(), 
                                    user = self.database_credentials['user'].item(), 
                                    password = self.database_credentials['password'].item())

            # creates a new cursor used to execute SELECT statements
            cur = conn.cursor()
            # creating the query 
            postgreSQL_select_Query = self.sql_query
            # quering the data
            cur.execute(postgreSQL_select_Query)
            data = cur.fetchall() 

            # puting data into a dataframe
            data = pd.DataFrame.from_records(data, columns = ['customer_id', 'gender', 'age', 'annual_income_thousands', 'spending_score'])
            
            cur.close()
            conn.close()

        else:
            data = pd.read_csv(self.path_to_file)
        print('Data from the csv file/database:')
        try:
            display(data.head())
        except:
            print(data.head())
        print()
        data, dummy_cols = self._dummy_encode(data)
        print('Data with extra dummy columns:')
        try:
            display(data.head())
        except:
            print(data.head())
        print()
        data, scaled_cols, scaler_mapper = self._scale_data(data, dummy_cols)
        print('Data with scaled features:')
        try:
            display(data.head())
        except:
            print(data.head())
        
        return data, dummy_cols, scaled_cols, scaler_mapper
    
    def _dummy_encode(self, data):
        """
        Adds to the passed data dummy columns for each categorical variable.
        
        The function loops through each categorical column specified in cat_cols attribute and create dummy columns for each
        category and stores names of created columns in an array. Each created column is added to the original dataframe.
        
        Parameters
        -------
        data: DataFrame
            DataFrame with the data.
        
        Returns
        -------
        data: DataFrame
            DataFrame with all the loaded columns extended by dummy encoded and scaled categorical columns and scaled numerical columns
        dummy_cols: array
            Array of names of dummy columns created from passed categorical columns.
            
        """
       
        dummy_cols = []
        for col in self.cat_cols:
            data_col_dummy = pd.get_dummies(data[col])
            dummy_cols.extend(data_col_dummy.columns)
            data = pd.concat([data, data_col_dummy], axis = 1, sort = False)
        return data, dummy_cols
    
    def _scale_data(self, data, dummy_cols):
        """
        Scale the dummy columns and add scaled numerical columns to the original dataframe.
        
        The function first loop through each numerical column and create a copy of it with '_scaled' ending in the name.
        All these created names are added to an array. Afterwards are all dummy columns and duplicates of numerical columns
        scalled and replaced in the original dataframe.
        
        Parameters
        -------
        data: DataFrame
            DataFrame with the data.
        dummy_cols: array
            Array of names of dummy columns created from passed categorical columns.
        
        Returns
        -------
        data: DataFrame
            DataFrame with all the loaded columns extended by dummy encoded and scaled categorical columns and scaled numerical columns
        scaled_cols: array
            Array of names scaled numerical columns.
        scaler_mapper: obj
            Fitted standard scaler, wrapped in DataFrameMapper, so that its output is dataframe instead of a numpy array.   
            
        """

        scaled_cols = []
        for col in self.num_cols:
            column = str(col + '_scaled')
            scaled_cols.append(column)
            data[column] = data[col]
        df = data[dummy_cols + scaled_cols]
        scaler_mapper = DataFrameMapper([(df.columns, StandardScaler())])
        scaled_features = scaler_mapper.fit_transform(df.copy())
        data[dummy_cols + scaled_cols] = pd.DataFrame(scaled_features, index=df.index, columns=df.columns)

        return data, scaled_cols, scaler_mapper

    def _plot_elbow_method(self, k, wcss):
        """
        Plots visualization based on which optimal number of clusters can be picked.
                
        Parameters
        -------
        k: int
            Max number of clusters to calculate within clusters sum of squares for.
        wcss: array
            Array of calculated within cluster sum of squares for each number of clusters in rage from 1 to k.
            
        """
            
        plt.figure(figsize=(15,10))
        plt.plot(range(1, k+1), wcss)
        plt.title('The Elbow Method', fontdict = {'fontsize': 20})
        plt.xlabel('Number of clusters', fontsize=15)
        plt.ylabel('WCSS', fontsize=15)
        plt.show()
        
    def _run_pca(self):
        """
        Runs PCA, to transform the features into 2 components, which coordinates are added to the data dataframe.
        
        The function initializes PCA with 2 components and fits and transformed dummy and numerical columns after scalling.
        The values of components are added to 'x' and 'y' column in the original dataframe.
     
        """
        
        # Principal component separation to create a 2-dimensional picture
        pca = PCA(n_components = 2)
        self.data['x'] = pca.fit_transform(self.data[self.scaled_cols + self.dummy_cols])[:,0]
        self.data['y'] = pca.fit_transform(self.data[self.scaled_cols + self.dummy_cols])[:,1]
        self.data = self.data.reset_index(drop = True)
        
    def _cat_features_summary(self, data_cluster):
        """
        Creates a table with proportion of values of categorical features of given cluster.
        
        The function loops through all categorical columns in the passed subset of data. It groups dataframe by given feature,
        to get counts for each category and adds 'percentage' column as well. At the end of the loop dataframe with counts
        and percentages is displayed.
        
        Parameters
        -------
        data_cluster: DataFrame
            Data of a given cluster only.
          
        """
    
        for cat_feature in self.cat_cols:
            data_cluster['count'] = data_cluster[cat_feature]
            data_cluster_grouped = data_cluster[[cat_feature, 'count']].groupby(cat_feature).count().reset_index()
            data_cluster_grouped['percentage'] = round(data_cluster_grouped['count'].astype(int) / data_cluster_grouped['count'].astype(int).sum() * 100,0)
            try:
                display(data_cluster_grouped)
            except:
                print(data_cluster_grouped)
            
        
        
        
'''ASSIGNING CLUSTERS TO THE NEW CUSTOMERS'''

def load_data_model(from_database=False, path_to_file_data=None, path_to_file_scaler_mapper = None,  path_to_file_model = None, sql_query=None, database_credentials = None):
    """
    Loads the data, model and scaler mapper; data are split to new customers and already segmented customers.
    
    The function first checks whether from_database paramter is True. If yes, the data, model and scaler are pulled from the database,
    where data are pulled based on passed sql query but query for model and scaler is hardcoded. If the from_database parameter equals False,
    data, model and scaler are loaded from passed paths for each file. Eventually are data split into to two dataframes, where new_data dataframe
    stores data, where segment needs to be assigned and spending_score is calculated, which means there are no missing values in the features.
    Dataframe segmented_data contains 
    
    Parameters
    ----------
    from_database : bool
        If yes, the data will be pulled from and saved to a PostgreSQL database. Train model and scaler will be pulled to that database, too.
    path_to_file : str
        Path to the csv file with the data, which will be used if from_database argument equals False.    
    sql_query : str
        A string containing a select statement used to download data from the database. Needed if from_database argument eaquals True.
    database_credentials : DataFrame
        A dataframe with strings of host, database, user, password, and string to create an engin. Needed if from_database argument eaquals True.
    

    Returns
    -------
    new_data: DataFrame
        DataFrame with customers, that does not have segment yet and spending_score value is not null.
    segmented_data: DataFrame
        DataFrame with customers, that has already segment or spending_score value is null.
    model: Obj
        Pulled Kmeans model.
    scaler_mapper: Obj
        Pulled standard scaler, wrapped in DataFrameMapper, so that its output is dataframe instead of a numpy array.     

        
    """
        
    if from_database:
        conn = psycopg2.connect(host = database_credentials['host'].item(),
                                database = database_credentials['database'].item(), 
                                user = database_credentials['user'].item(), 
                                password = database_credentials['password'].item())

        # creates a new cursor used to execute SELECT statements
        cur = conn.cursor()
        # creating the query 
        postgreSQL_select_Query = sql_query
        # quering the data
        cur.execute(postgreSQL_select_Query)
        data = cur.fetchall() 
        # puting data into a dataframe
        data = pd.DataFrame.from_records(data, columns = ['customer_id', 'gender', 'age', 'annual_income_thousands', 'spending_score'])
        
        new_data = data[data['segment'].isnull()].copy()
        segmented_data = data[data['segment'].notnull()].copy()
        
        # Loading the model
        postgreSQL_select_Query = """
                          select model_file
                          from models
                          where model = 'customer_segmentation'
                        """
  
        cur.execute(postgreSQL_select_Query)
        data = cur.fetchone() 
        model = pickle.loads(data[0])
        
        # Loading the scaler
        postgreSQL_select_Query = """
                          select scaler_file
                          from models
                          where model = 'customer_segmentation'
                        """
        # quering the data
        cur.execute(postgreSQL_select_Query)
        data = cur.fetchone() 
        scaler_mapper = pickle.loads(data[0])       
        
        
        cur.close()
        conn.close()
        
    else:
        data = pd.read_csv(path_to_file_data)
        model = pickle.load(open(path_to_file_model, 'rb'))
        scaler_mapper = pickle.load(open(path_to_file_scaler_mapper, 'rb'))
        
    new_data = data[(data['spending_score'].notnull()) & (data['segment'].isnull())].copy()
    segmented_data = data[(data['segment'].notnull()) | (data['spending_score'].isnull())].copy()
    
    return new_data, segmented_data, model, scaler_mapper

def dummy_encode(data, cat_cols):
    """
    Adds to dataframe dummy columns for each categorical variable and returns array with their names.
    
    The function loops through each categorical column specified in cat_cols attribute and create dummy columns for each
    category and stores names of created columns in an array. Each created column is added to the original dataframe.
     
    Parameters
    ----------
    data : DataFrame
        Data with customers, that doesn not have assigned segment yet.
    cat_cols : array 
        An array of names of columns in the data that are categorical.

    Returns
    -------
    data: DataFrame
        DataFrame with all the loaded columns extended by dummy encoded columns.
    dummy_cols: array
        Array of names of dummy columns created from passed categorical columns.
        
    """

    dummy_cols = []
    for col in cat_cols:
        data_col_dummy = pd.get_dummies(data[col])
        dummy_cols.extend(data_col_dummy.columns)
        data = pd.concat([data, data_col_dummy], axis = 1, sort = False)
    return data, dummy_cols

def preprocess_data(data, dummy_cols, num_cols, scaler_mapper):
    """
    Scales created dummy columns for categorical data and numerical columns.
    
    The function first loop through each numerical column and create a copy of it with '_scaled' ending in the name.
    All these created names are added to an array. Afterwards are all dummy columns and duplicates of numerical columns
    scalled, utilizing loaded scaler and replaced in the original dataframe.
 
    
    Parameters
    ----------
    data : DataFrame
        Data with customers, that doesn not have assigned segment yet.
    dummy_cols: array
        Array of names of dummy columns created from passed categorical columns.
    num_cols : array 
        An array of names of columns in the data that are numerical.
    scaler_mapper: Obj
        Pulled standard scaler, wrapped in DataFrameMapper, so that its output is dataframe instead of a numpy array.     
    

    Returns
    -------
    data: DataFrame
        DataFrame with scaled dummy columns and extended by scaled numerical columns.
    scaled_cols: array
        An array of names of scaled numerical columns.
            
    """
        
    scaled_cols = []
    for col in num_cols:
        column = str(col + '_scaled')
        scaled_cols.append(column)
        data[column] = data[col]
    df = data[dummy_cols + scaled_cols]
    scaled_features = scaler_mapper.transform(df.copy())
    data[dummy_cols + scaled_cols] = pd.DataFrame(scaled_features, index=df.index, columns=df.columns)
    try:
        display(data.head())
    except:
        print(data.head())
    return data, scaled_cols 

       
def assign_segments(model, data, dummy_cols, scaled_cols):
    """
    Combine dataset with assigned segments and column labeling them as the cases were an already existed cluster were assigned.
    
    The function get predictions based on preprocessed columns and assign segment to each datapoint utilizing the loaded KMeans model.
    The 'segment_origin' column is labeled with 'assigned' indicating that the  datapoint was not used to form clusters.
    Afterwards are dropped dummy columns and scaled numerical columns.
    
    Parameters
    ----------
    model : Obj
        Pulled Kmeans model.
    data : DataFrame
        Data extended by preprocessed columns.
    dummy_cols: array
        Array of names of dummy columns created from passed categorical columns.
    scaled_cols: array
        An array of names of scaled numerical columns.
     

    Returns
    -------
    data: DataFrame
        DataFrame with populated 'segment' column with the cluster number the datapoint belongs to and label 
        'segment_origin' with label implying that the datapoint was to an already existed cluster.
   
    """
    
    new_data = data[dummy_cols + scaled_cols].copy()
    data['segment'] = model.predict(new_data)
    data['segment_origin'] = 'assigned'
    data.drop(dummy_cols + scaled_cols, axis = 1, inplace = True)
    try:
        display(data.head())
    except:
        print(data.head())
    return data

def save_data(from_database=False, path_to_file_data=None, database_credentials=None, data=None, segmented_data=None):
    """
    Saves both already segmented customers and new customers with assigned segments as a one table again.
    
    The function first concatinate segmented_data dataframe and data with recently assigned segments. Then based on from_database paramter
    is the dataframe either loaded to the database replacing the existed customers table, or saved as a csv file replacing the original one.
    
    Parameters
    ----------
    from_database : bool
            If yes, the data will be pulled from and saved to a PostgreSQL database. Train model and scaler will be saved to that database, too.
    path_to_file : str
        Path to the csv file with the data, which will be replaced if from_database argument equals False.
    database_credentials : DataFrame
        A dataframe with strings of host, database, user, password, and string to create an engin. Needed if from_database argument eaquals True.
    data: DataFrame
        DataFrame with customers, that just got assigned a cluster.
    segmented_data: DataFrame
        DataFrame with customers, that already  had a segment or spending_score value was null.
        
    """
    
    data = pd.concat([segmented_data, data])
    if from_database:
        engine = create_engine(engine_link = database_credentials['engine'].item())  
        data.to_sql('customers', engine, if_exists = 'replace', index = False, method = 'multi')    
        engine.dispose()
    else:
        data.to_csv(path_to_file_data, index = False)
        
    return print('Segments were assigned and saved')
