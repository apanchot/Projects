# -*- coding: utf-8 -*-
"""
Created on Mon Sep 30 11:37:02 2019

@author: gsmfa
"""
### Packeges ###
import pandas as pd
import numpy as np
import sklearn as sl
import seaborn as sns
import matplotlib.pyplot as plt
from statsmodels.graphics.gofplots import qqplot
from numpy import random
from scipy.cluster.vq import kmeans, vq
#, whiten
from sklearn import preprocessing
#import statsmodels.api as sm
import scipy.stats as stats


### Importing data ###
df = pd.read_csv('D:/Mestrado/Materias/Programming_for_Data_Science/Final_Project/API_ST.INT.ARVL_DS2_en_csv_v2_103871.csv', skiprows = lambda x: x in [0, 2])
df_md1 = pd.read_csv('D:/Mestrado/Materias/Programming_for_Data_Science/Final_Project/Metadata_Indicator_API_ST.INT.ARVL_DS2_en_csv_v2_103871.csv')
df_md2 = pd.read_csv('D:/Mestrado/Materias/Programming_for_Data_Science/Final_Project/Metadata_Country_API_ST.INT.ARVL_DS2_en_csv_v2_103871.csv')
df_extra = pd.read_csv('D:/Mestrado/Materias/Programming_for_Data_Science/Final_Project/WDIData.csv')



### Information about Costumer, GNI... drop some columns   
df_extra = df_extra.drop(df_extra.columns[[range(4,56)]], axis=1)
df_extra = df_extra.drop(df_extra.columns[[-1,-2]], axis=1)

### get the max in a range of 6 years
df_extra['data_inf'] = df_extra[["2012","2013","2014","2015", "2016", "2017"]].max(axis=1)

### drop the years and keep the max
df_extra = df_extra.drop(df_extra.columns[[range(4,10)]], axis=1)

extra_null = df_extra.data_inf.isnull().groupby([df_extra['Indicator Name']]).sum().astype(int).reset_index(name='count')

###creating a table for each indicator
df_GNI = df_extra.loc[df_extra['Indicator Name'] == "GNI per capita, Atlas method (current US$)"]
df_ITR = df_extra.loc[df_extra['Indicator Name'] == "International tourism, receipts (current US$)"]
df_ITRP = df_extra.loc[df_extra['Indicator Name'] == "International tourism, receipts (% of total exports)"]
df_CHE = df_extra.loc[df_extra['Indicator Name'] == "Current health expenditure per capita (current US$)"]
df_FAR = df_extra.loc[df_extra['Indicator Name'] == "Forest area (sq. km)"]
df_LAR = df_extra.loc[df_extra['Indicator Name'] == 'Land area (sq. km)']



### rename columns
df_GNI.columns = ['Country Name', 'Country Code', 'Indicator Name', 'indicator Code', 'GNI']
df_ITR.columns = ['Country Name', 'Country Code', 'Indicator Name', 'indicator Code', 'ITR']
df_ITRP.columns = ['Country Name', 'Country Code', 'Indicator Name', 'indicator Code', 'ITRP']
df_CHE.columns = ['Country Name', 'Country Code', 'Indicator Name', 'indicator Code', 'CHE']
df_FAR.columns = ['Country Name', 'Country Code', 'Indicator Name', 'indicator Code', 'FAR']
df_LAR.columns = ['Country Name', 'Country Code', 'Indicator Name', 'indicator Code', 'LAR']



### checking null values
df_GNI.isna().sum()
df_ITR.isna().sum()
df_ITRP.isna().sum()
df_CHE.isna().sum()
df_FAR.isna().sum()
df_LAR.isna().sum()


### Cleaning Data ###

#1 Checking null Values by column
qtt_null_col = df.isna().sum()
df_info = df.describe()
#No values for column 1960 until 1994 and 2018


#2 Drop empty columns
tourism_db = df.drop(df.columns[[range(4,39)]], axis=1)
tourism_db = tourism_db.drop(tourism_db.columns[[-1,-2]], axis=1)

#3 Recount null values by column
qtt_null_col = tourism_db.isna().sum()

#4 Checking null values by row
qtt_null_row = tourism_db.isnull().sum(axis=1)

#5 Drop rows with more than 15% of null values
x = list(qtt_null_row[qtt_null_row > 5].index)
tourism_db = tourism_db.drop(tourism_db.index[[x]])
qtt_null_row = tourism_db.isnull().sum(axis=1)

# total of 43 countries was droped by the analysis
droped_countries = df.iloc[x,0:2]

# Join information about Region, Income and notes 
tourism_db = pd.merge(tourism_db, df_md2, on='Country Code', how='left')


tourism_db = pd.merge(tourism_db, df_GNI[['Country Code','GNI']], on='Country Code', how='left')
tourism_db = pd.merge(tourism_db, df_ITR[['Country Code','ITR']], on='Country Code', how='left')
tourism_db = pd.merge(tourism_db, df_ITRP[['Country Code','ITRP']], on='Country Code', how='left')
tourism_db = pd.merge(tourism_db, df_CHE[['Country Code','CHE']], on='Country Code', how='left')
tourism_db = pd.merge(tourism_db, df_LAR[['Country Code','LAR']], on='Country Code', how='left')



tourism_db = tourism_db.drop(tourism_db.columns[[2,3,-8,-7,-6]], axis=1)

tourism_db.isna().sum()

#reorder the columns
tourism_db = tourism_db[['Country Name', 'Country Code', 'Region', 'IncomeGroup', '1995', 
 '1996', '1997', '1998', '1999', '2000', '2001', '2002', '2003', '2004', '2005',
 '2006', '2007', '2008', '2009', '2010', '2011', '2012', '2013', '2014', '2015', 
 '2016', '2017', 'GNI', 'ITR', 'ITRP', 'CHE', 'LAR']]

tourism_db = tourism_db.dropna(subset=['Region'], how='all')
tourism_db.isna().sum()


#AVG calculation of indicators by region and income to fill remaining NULL values with the same caracteristics
GNI_avg = tourism_db.groupby(by = ['Region', 'IncomeGroup'])['GNI'].mean().reset_index(name = "avg GNI")
ITR_avg = tourism_db.groupby(by = ['Region', 'IncomeGroup'])['ITR'].mean().reset_index(name = "avg ITR")
ITRP_avg = tourism_db.groupby(by = ['Region', 'IncomeGroup'])['ITRP'].mean().reset_index(name = "avg ITRP")
CHE_avg = tourism_db.groupby(by = ['Region', 'IncomeGroup'])['CHE'].mean().reset_index(name = "avg CHE")





tourism_db = pd.merge(tourism_db, GNI_avg[['Region', 'IncomeGroup', "avg GNI"]], on=['Region', 'IncomeGroup'], how='left')
tourism_db = pd.merge(tourism_db, ITR_avg[['Region', 'IncomeGroup', "avg ITR"]], on=['Region', 'IncomeGroup'], how='left')
tourism_db = pd.merge(tourism_db, ITRP_avg[['Region', 'IncomeGroup', "avg ITRP"]], on=['Region', 'IncomeGroup'], how='left')
tourism_db = pd.merge(tourism_db, CHE_avg[['Region', 'IncomeGroup', "avg CHE"]], on=['Region', 'IncomeGroup'], how='left')


tourism_db['GNI'].fillna(tourism_db['avg GNI'], inplace=True)
tourism_db['ITR'].fillna(tourism_db['avg ITR'], inplace=True)
tourism_db['ITRP'].fillna(tourism_db['avg ITRP'], inplace=True)
tourism_db['CHE'].fillna(tourism_db['avg CHE'], inplace=True)

tourism_db.isna().sum()


#aux1 = list(range(1995,2017,1))
#aux2 = list(range(1996,2018,1))
#col1 = []
#col2 = []

#for i in aux1:
#    col1.append(str(i))

#for i in aux2:
#    col2.append(str(i))

#for i in col1 and j in col2:
#    if (tourism_db.loc[1,['2000']] == None) == True:
#        tourism_db[i].fillna(tourism_db[j], inplace=True) 
        
    

tourism_db['1996'].fillna(tourism_db['1995'], inplace=True)
tourism_db['1997'].fillna(tourism_db['1996'], inplace=True)
tourism_db['1998'].fillna(tourism_db['1997'], inplace=True)
tourism_db['1999'].fillna(tourism_db['1998'], inplace=True)
tourism_db['2000'].fillna(tourism_db['1999'], inplace=True)
tourism_db['2001'].fillna(tourism_db['2000'], inplace=True)
tourism_db['2002'].fillna(tourism_db['2001'], inplace=True)
tourism_db['2003'].fillna(tourism_db['2002'], inplace=True)
tourism_db['2004'].fillna(tourism_db['2003'], inplace=True)
tourism_db['2005'].fillna(tourism_db['2004'], inplace=True)
tourism_db['2006'].fillna(tourism_db['2005'], inplace=True)
tourism_db['2007'].fillna(tourism_db['2006'], inplace=True)
tourism_db['2008'].fillna(tourism_db['2007'], inplace=True)
tourism_db['2009'].fillna(tourism_db['2008'], inplace=True)
tourism_db['2010'].fillna(tourism_db['2009'], inplace=True)
tourism_db['2011'].fillna(tourism_db['2010'], inplace=True)
tourism_db['2012'].fillna(tourism_db['2011'], inplace=True)
tourism_db['2013'].fillna(tourism_db['2012'], inplace=True)
tourism_db['2014'].fillna(tourism_db['2013'], inplace=True)
tourism_db['2015'].fillna(tourism_db['2014'], inplace=True)
tourism_db['2016'].fillna(tourism_db['2015'], inplace=True)
tourism_db['2017'].fillna(tourism_db['2016'], inplace=True)



tourism_db['1998'].fillna(tourism_db['1999'], inplace=True)
tourism_db['1997'].fillna(tourism_db['1998'], inplace=True)
tourism_db['1996'].fillna(tourism_db['1997'], inplace=True)
tourism_db['1995'].fillna(tourism_db['1996'], inplace=True)


#fill Sudan Area external information
tourism_db['LAR'].fillna(1886068, inplace=True)

tourism_db['inc_2y'] = tourism_db['2017']/tourism_db.iloc[:,24:26].mean(axis=1) - 1 
tourism_db['inc_5y'] = tourism_db['2017']/tourism_db.iloc[:,21:26].mean(axis=1) - 1
tourism_db['inc_10y'] = tourism_db['2017']/tourism_db.iloc[:,16:26].mean(axis=1) - 1
tourism_db['inc_15y'] = tourism_db['2017']/tourism_db.iloc[:,11:26].mean(axis=1) - 1
tourism_db['2017_ApA'] = tourism_db['2017']/tourism_db['LAR']



tourism_db.set_index('Country Name', inplace=True)



tourism_db1 = tourism_db.drop(tourism_db.columns[[range(3,25)]], axis=1)
tourism_db1.drop(tourism_db1.columns[[-6,-7,-8,-9]], axis=1, inplace=True)


corr_matrix = tourism_db1.corr()

###Corr between GNI and CHE  = 0.92 drop CHE
###Corr between ITR and 2017 = 0.78 drop ITR 
###Drop inc_15y 
 
tourism_db1.drop(tourism_db1.columns[[5,7,-2]], axis=1, inplace=True)


#tourism_db.drop(tourism_db.columns[[-2,-3]], axis=1, inplace=True)



###Histogram with the distribution of the variables
sns.distplot(tourism_db1['2017'])
sns.distplot(tourism_db1['GNI'])
sns.distplot(tourism_db1['ITRP'])
sns.distplot(tourism_db1['inc_2y'])
sns.distplot(tourism_db1['inc_5y'])
sns.distplot(tourism_db1['inc_10y'])
sns.distplot(tourism_db1['2017_ApA'])



#### With Log in variables.
sns.distplot(np.log(tourism_db1['2017']))
sns.distplot(np.log(tourism_db1['GNI']))
sns.distplot(np.log(tourism_db1['ITRP']))
sns.distplot(np.log(tourism_db1['inc_2y']+1))
sns.distplot(np.log(tourism_db1['inc_5y']+1))
sns.distplot(np.log(tourism_db1['inc_10y']+1))
sns.distplot(np.log(tourism_db1['2017_ApA']))





###Checking Outliers
stats.probplot(np.log(tourism_db1['2017']), dist="norm", plot=plt)
sns.boxplot( y=np.log(tourism_db1['2017']))
tourism_db1[np.log(tourism_db1['2017'])<9]

outliers =  tourism_db1.loc[["Tuvalu", "Kiribati", "Marshall Islands"]]
tourism_db1.drop(["Tuvalu", "Kiribati", "Marshall Islands"], inplace = True)

#outliers =  tourism_db1.loc[["Tuvalu", "Kiribati", "Marshall Islands","France","Spain","United States","China","Italy"]]
#tourism_db1.drop(["Tuvalu", "Kiribati", "Marshall Islands","France","Spain","United States","China","Italy"], inplace = True)



stats.probplot(np.log(tourism_db1['inc_10y']+1), dist="norm", plot=plt)
sns.boxplot( y=np.log(tourism_db1['inc_10y']+1) )
tourism_db1[np.log(tourism_db1['inc_10y']+1)<-0.3]['inc_10y']
tourism_db1[np.log(tourism_db1['inc_10y']+1)> 0.8]['inc_10y']

outliers = outliers.append(tourism_db1.loc[['Belarus','Bhutan','Iceland','Japan','Paraguay',"Angola", "Venezuela, RB", "Yemen, Rep.",'Burkina Faso','Bangladesh','Ukraine']])
tourism_db1.drop(['Belarus','Bhutan','Iceland','Japan','Paraguay',"Angola", "Venezuela, RB", "Yemen, Rep.",'Burkina Faso','Bangladesh','Ukraine'], inplace = True)


stats.probplot(np.log(tourism_db1['2017_ApA']), dist="norm", plot=plt)
sns.boxplot(y=np.log(tourism_db1['2017_ApA']))
tourism_db1[np.log(tourism_db1['2017_ApA'])> 10]['2017_ApA']

outliers = outliers.append(tourism_db1.loc[["Macao SAR, China", "Monaco", "Hong Kong SAR, China"]])
tourism_db1.drop([ "Hong Kong SAR, China", "Macao SAR, China", "Monaco",], inplace = True)




tourismScaled = pd.concat([np.log(tourism_db1['2017']),np.log(tourism_db1['GNI']),np.log(tourism_db1['ITRP']), np.log(tourism_db1['inc_2y']+1), np.log(tourism_db1['inc_5y']+1), np.log(tourism_db1['inc_10y']+1), np.log(tourism_db1['2017_ApA'])], axis =1)

#tourismScaled = tourism_db1[['2017',"GNI", 'ITRP','inc_2y', 'inc_10y', '2017_ApA']]
mm_scaler = preprocessing.MinMaxScaler()
X_train_minmax = mm_scaler.fit_transform(tourismScaled)
tourismScaled = pd.DataFrame(mm_scaler.transform(tourismScaled))

tourismScaled.columns = ['scaled_2017',"scaled_GNI", 'scaled_ITRP', 'scaled_inc_2y', 'scaled_inc_5y', 'scaled_inc_10y', 'scaled_2017_ApA']




#tourism_db1['scaled_2017'] = whiten(tourism_db1['2017'])
#tourism_db1['scaled_GNI'] = whiten(tourism_db1['GNI'])
#tourism_db1['scaled_ITRP'] = whiten(tourism_db1['ITRP'])
#tourism_db1['scaled_inc_2y'] = whiten(tourism_db1['inc_2y'])
#tourism_db1['scaled_inc_5y'] = whiten(tourism_db1['inc_5y'])
#tourism_db1['scaled_inc_10y'] = whiten(tourism_db1['inc_10y'])

sns.distplot(tourismScaled['scaled_2017'])
sns.distplot(tourismScaled['scaled_inc_10y'])
sns.distplot(tourismScaled['scaled_2017_ApA'])

stats.probplot(tourismScaled['scaled_2017'], dist="norm", plot=plt)
stats.probplot(tourismScaled['scaled_inc_5y'], dist="norm", plot=plt)
stats.probplot(tourismScaled['scaled_2017_ApA'], dist="norm", plot=plt)




tourism_db1 = pd.concat([tourism_db1.reset_index(drop=False), tourismScaled], axis=1)
tourism_db1.set_index('Country Name', inplace=True)



random.seed(2000)
#elbow grafic
distortions = []
num_clusters = range(1, 15)

# Create a list of distortions from the kmeans function
for i in num_clusters:
    cluster_centers, distortion = kmeans(tourism_db1[["scaled_2017", "scaled_GNI", 'scaled_ITRP', 'scaled_inc_10y']], i)
    distortions.append(distortion)

# Create a data frame with two lists - num_clusters, distortions
elbow_plot = pd.DataFrame({'num_clusters': num_clusters, 'distortions': distortions})

# Creat a line plot of num_clusters and distortions
sns.lineplot(x='num_clusters', y='distortions', markers=True, data = elbow_plot)
plt.xticks(num_clusters)
plt.show()


# Set up a random seed in numpy
random.seed(2000)

# Fit the data into a k-means algorithm
cluster_centers,_ = kmeans(tourism_db1[["scaled_2017", "scaled_GNI", 'scaled_ITRP', 'scaled_inc_5y', 'scaled_2017_ApA']], 8)

# Assign cluster labels
tourism_db1['cluster_labels'], _ = vq(tourism_db1[["scaled_2017", "scaled_GNI", 'scaled_ITRP', 'scaled_inc_5y', 'scaled_2017_ApA']], cluster_centers)

# Display cluster centers 
center = tourism_db1[["scaled_2017", "scaled_GNI", 'scaled_ITRP', 'scaled_inc_5y', 'scaled_2017_ApA','cluster_labels']].groupby('cluster_labels').mean()

centerRV = tourism_db1[["2017", "GNI", 'ITRP', 'inc_5y', '2017_ApA','cluster_labels']].groupby('cluster_labels').mean()

sns.scatterplot(x="scaled_2017", y="scaled_inc_2y", hue="cluster_labels", data=tourism_db1)
plt.show()



tourism_db1[["scaled_2017", "scaled_GNI", 'scaled_ITRP', 'scaled_inc_2y', 'scaled_inc_10y']].mean()
tourism_db1




# Set up a random seed in numpy
random.seed(2000)

# Fit the data into a k-means algorithm
cluster_centers,_ = kmeans(tourism_db1[["scaled_2017", 'scaled_inc_10y', "scaled_2017_ApA"]], 6)

# Assign cluster labels
tourism_db1['cluster_labels_b'], _ = vq(tourism_db1[["scaled_2017", 'scaled_inc_10y', "scaled_2017_ApA"]], cluster_centers)

# Display cluster centers 
centerB = tourism_db1[["scaled_2017", 'scaled_inc_10y', "scaled_2017_ApA", 'cluster_labels_b']].groupby('cluster_labels_b').mean()

centerRVB = tourism_db1[["2017", 'inc_10y', '2017_ApA', 'LAR', 'GNI', 'ITRP', 'cluster_labels_b']].groupby('cluster_labels_b').mean().reset_index()

sns.scatterplot(x="scaled_2017", y="scaled_2017_ApA", hue="cluster_labels_b", data=tourism_db1, legend = False)
plt.show()

sns.scatterplot(x="2017", y="2017_ApA", hue="cluster_labels_b", data=centerRVB, legend = False)
plt.show()

sns.scatterplot(x="2017", y="2017_ApA", hue="cluster_labels_b", data=centerRVB, legend = False)
plt.show()
sns.scatterplot(x="2017", y="inc_5y", hue="cluster_labels_b", data=centerRVB, legend = False)
plt.show()
sns.scatterplot(x="2017_ApA", y="inc_5y", hue="cluster_labels_b", data=centerRVB, legend = False)
plt.show()


tourism_db1[["2017", 'inc_10y', '2017_ApA']].mean()

import geopandas as gpd
shapefile = 'data/countries_110m/ne_110m_admin_0_countries.shp'


