#First, understanding the problem statement - 
#Segmentation means grouping various entities together based on certain similar properties. Here, customer segmentation means grouping 
#customers together based on similar features or properties as given in the dataset. 
#This helps companies in understanding and predicting their customers’ characteristics, behaviour and needs. They can launch products or 
#enhance features accordingly. Companies can target particular sectors for more revenue. This leads to overall market enhancement of the company.  

#worflow for the project 
#1. load customer data
#2. data preprocessing 
#3. data analysis 
#4. calculate optimum number of clusters 
#5. k means clustering 
#6. visualizing the clusters 

#load libraries 
#linear algebra - for building matrices
import numpy as np 

#data preprocessing and exploration 
import pandas as pd

#data visualisation and analysis 
import matplotlib.pyplot as plt
import seaborn as sns 
from sklearn.cluster import KMeans 

#loading the data 
#load dataset from csv file to pandas dataframe
customer_data = pd.read_csv(r'customer_data.csv')

#we need to be familiar with the data before working on it
#view first five rows of the dataframe 
customer_data.head()
#the data contains following columns - 
#1. CustomerID - identification number alloted to the customer 
#2. Gender - Male or Female 
#3. Age 
#4. Annual Income - in k$
#5. Spending score - score alloted to the customers based on their spending habits 

#view the total number of rows and columns in the data 
#the data has 200 rows (201 data points) and 5 columns (5 features)
customer_data.shape

#more information on data structure 
customer_data.info()

#check if there are any missing values 
customer_data.isnull().sum()
#there are no missing values 

#choosing annual income column and spending score column 
X = customer_data.iloc[:,[3,4]].values

#choosing ideal number of clusters
#this is done by calculating wcss - within cluster sum of squares 

#finding wcss value for different number of clusters 
wcss = []

for i in range(1,11):
    kmeans = KMeans(n_clusters=i, init='k-means++', random_state=40)
    kmeans.fit(X)
    
    wcss.append(kmeans.inertia_)

#plot an elbow graph 
sns.set()
plt.plot(range(1,11), wcss)
plt.title('the elbow point graph')
plt.xlabel('number of clusters')
plt.ylabel('wcss')
plt.show()

#we see that optimum number of clusters = 5
#0,1,2,3,4

#training the k means clustering model 
kmeans = KMeans(n_clusters=5, init='k=means++', random_state=0)

#return a label for each datapoint based on their cluster
Y = kmeans.fit_predict(X)
print (Y)

#visualizing all the clusters 
#plotting all the clusters and their centroids 

plt.figure(figsize=(8,8))
plt.scatter(X[Y==0,0], X[Y==0,1], s=50, c='green', label='Cluster 1')
plt.scatter(X[Y==1,0], X[Y==1,1], s=50, c='red', label='Cluster 2')
plt.scatter(X[Y==2,0], X[Y==2,1], s=50, c='yellow', label='Cluster 3')
plt.scatter(X[Y==3,0], X[Y==3,1], s=50, c='violet', label='Cluster 4')
plt.scatter(X[Y==4,0], X[Y==4,1], s=50, c='blue', label='Cluster 5')

#plot the centroid 
plt.scatter(kmeans.cluster_centers_[:,0], kmeans.cluster_centers_[:,1], c='cyan', label='centroids')

plt.title('customer groups')
plt.xlabel('annual income')
plt.ylabel('spending score ')
plt.show()
