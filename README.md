# Customer_Segmentation
Repository holds end-to-end clustering project, with solution for deployment using PostgreSQL database and Airflow. 

### Problem definition
A mall holds some information about its clients, e.g. gender, age, aunnual income and spending score. These clients have subscribed to the membership card. When the clients subscribed to the membership card, they provided all the information. Beause they have this card they use it to buy all sorts of things in the mall and therefore the mall has the purchase history of each of its client member. That is how they obtained the spending score. Spending score is a score that the mall computed for each of their clients based on severlal criteria including for example thier income, the number of times per week they shop in the mall and of course the amount of money they spend in a year. And based on all this they computed this metric that takes values between 1 and 100, so that the closer the spending score is to 1, the less the client spends and the closed the spending score is to 100, the more the client spends.

The goal of this project is to find customer segments based on all these information, describe them and analyze spending score in each segment. Solution should be delivered in a form, in which it allows easily repeat the segmentation excercise and allow quick and automated labeling of each new customer.

### Dataset
Dataset - Mall_Customers.csv has 200 unique rows with 5 columns

![image](https://user-images.githubusercontent.com/31499140/78266332-f261bf80-7505-11ea-98da-644fbaf9f188.png)

### Clustering
The main part of the project is happening in __Customer_Segmentation.ipynb__. The dataset notebook is pre-set to work with the __Mall_Customers.csv__, but for production there is commented out part in cell with input parameters with details to download and save the data to a Postgresql database alongside with the model and standard scaler.

### Deployment
Nootebook __New_data_segments.ipynb__ is for assigning segments to new customers (if there are any) based on existing clusters. 
Output of this notebook is table with both already segmented customers and new customers labeled accordingly.

![image](https://user-images.githubusercontent.com/31499140/78268942-3b674300-7509-11ea-9910-4d8e051e2479.png)


It is pre-set to work with Mall_Customers-New.csv, but again there is commented out part of the code for downloading data, model and scaler from a PostgreSQL database.

File __customer_segments_DAG.py__ is for the deployment with Airflow server. It has task to execute the notebook New_data_segments.ipynb, that is scheduled to run every night.

The notebook has a snippet of code, that checks whether there are some new customers at all and stop the execution of the code, if there are not any.

### Presentation 
Project presentation is in the attached __Customer_Segmentation.pptx__ powerpoint presentation - read explanation under each slide for full understanding :)

File __Customer_Segmentation-Result.html__ is part of the presentation.


