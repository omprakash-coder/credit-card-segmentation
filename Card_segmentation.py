#!/usr/bin/env python
# coding: utf-8

# In[55]:


#Basic python library which need to import
import pandas as pd
import numpy as np


# In[56]:


# Date stuff
# we have imported datetime module using import datetime statement
from datetime import datetime
from datetime import timedelta


# In[57]:


#Library for Nice graphing
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.formula.api as sn
get_ipython().run_line_magic('matplotlib', 'inline')


# In[58]:


#Library for statistics operation
import scipy.stats as stats


# In[59]:


# Date Time library
from datetime import datetime
#!pip install --user scikit-learn
#!pip install scikit-learn
#installed necessary libraries
import sys
get_ipython().system('{sys.executable} -m pip install numpy')
get_ipython().system('{sys.executable} -m pip install scipy')
get_ipython().system('{sys.executable} -m pip install sklearn')

import sklearn


# In[60]:


#Machine learning Library
import statsmodels.api as sm # use for statistical models, as well as for conducting statistical tests
from sklearn import metrics # how the performance of machine learning algorithms is measured and compared.
from sklearn.model_selection import train_test_split # splitting data arrays into two subsets: for training data and for testing data
from sklearn.linear_model import LinearRegression # answer whether and how some phenomenon influences the other
from sklearn.ensemble import RandomForestRegressor #fit the model on the data. 
from sklearn.tree import DecisionTreeRegressor # decision-making tool that uses a flowchart-like tree structure
from sklearn.ensemble import AdaBoostRegressor #it starts fitting the regressor with the dataset and adjusts the weights according to error rate
from sklearn.ensemble import GradientBoostingRegressor # it allows for the optimization of arbitrary differentiable loss functions
from sklearn.svm import SVC, LinearSVC # it is to fit to the data you provide, returning a "best fit"
from sklearn.metrics import mean_squared_error as mse # it is an estimator measures the average 
from sklearn.metrics import mean_absolute_error, mean_squared_error


# In[62]:


# Ignore warnings
import warnings
warnings.filterwarnings('ignore')


# In[63]:


# Settings
pd.set_option('display.max_columns', None)# to see maximum column in output page
np.set_printoptions(threshold=sys.maxsize)# to print the full NumPy array, without truncation
np.set_printoptions(precision=3)#determine the way floating point numbers, arrays and other NumPy objects 
sns.set(style="darkgrid") # styling figures with
plt.rcParams['axes.labelsize'] = 14 # seting the size of axes
plt.rcParams['xtick.labelsize'] = 12 # increase or reduce the font size in x-axis 
plt.rcParams['ytick.labelsize'] = 12 # increase or reduce the font size in y-axis


# In[64]:


import os
os.chdir("D:/Data Science Edwisor/Project")
os.getcwd()


# In[65]:


# reading data into dataframe
credit= pd.read_csv("credit-card-data.csv")


# In[66]:


credit.head()


# In[67]:


credit.info() # summary of dataframe


# In[68]:


# Find the total number of missing values in the dataframe
print ("\nMissing values :  ", credit.isnull().sum().values.sum())

# printing total numbers of Unique value in the dataframe. 
print ("\nUnique values :  \n",credit.nunique())


# In[69]:


credit.shape


# In[70]:


# Intital descriptive analysis of data.
credit.describe()


# In[71]:


#Missing value treating by inputing them with median
credit.isnull().any()


# In[73]:


# CREDIT_LIMIT  and MINIMUM_PAYMENTS has missing values so we need to remove with median.
credit['CREDIT_LIMIT'].fillna(credit['CREDIT_LIMIT'].median(),inplace=True)
credit['CREDIT_LIMIT'].count()


credit['MINIMUM_PAYMENTS'].median()
credit['MINIMUM_PAYMENTS'].fillna(credit['MINIMUM_PAYMENTS'].median(),inplace=True)


# In[74]:


# Now again check the missing values.

credit.isnull().any()


# In[75]:


# Monthly_avg_purchase calculation
credit['Monthly_avg_purchase']=credit['PURCHASES']/credit['TENURE']


# In[77]:


print(credit['Monthly_avg_purchase'].head(),'\n ',
credit['TENURE'].head(),'\n', credit['PURCHASES'].head())


# In[81]:


# Monthly_cash_advance Amount calulation
credit['Monthly_cash_advance']=credit['CASH_ADVANCE']/credit['TENURE']


# In[82]:


#counting zeros in ONEOFF_PURCHASES 
credit[credit['ONEOFF_PURCHASES']==0]['ONEOFF_PURCHASES'].count()


# In[83]:


# Purchases by type (one-off, installments)
credit.loc[:,['ONEOFF_PURCHASES','INSTALLMENTS_PURCHASES']]


# In[84]:


# Find customers ONEOFF_PURCHASES and INSTALLMENTS_PURCHASES details
credit[(credit['ONEOFF_PURCHASES']==0) & (credit['INSTALLMENTS_PURCHASES']==0)].shape


# In[85]:


credit[(credit['ONEOFF_PURCHASES']>0) & (credit['INSTALLMENTS_PURCHASES']>0)].shape


# In[86]:


credit[(credit['ONEOFF_PURCHASES']>0) & (credit['INSTALLMENTS_PURCHASES']==0)].shape


# In[87]:


credit[(credit['ONEOFF_PURCHASES']==0) & (credit['INSTALLMENTS_PURCHASES']>0)].shape


# In[88]:


#As per above detail we found out that there are 4 types of purchase behaviour in the data set. 
#So we need to derive a categorical variable based on their behaviour
def purchase(credit):# creating purchase as method
    if (credit['ONEOFF_PURCHASES']==0) & (credit['INSTALLMENTS_PURCHASES']==0):
        return 'none'
    if (credit['ONEOFF_PURCHASES']>0) & (credit['INSTALLMENTS_PURCHASES']>0):
         return 'both_oneoff_installment'
    if (credit['ONEOFF_PURCHASES']>0) & (credit['INSTALLMENTS_PURCHASES']==0):
        return 'one_off'
    if (credit['ONEOFF_PURCHASES']==0) & (credit['INSTALLMENTS_PURCHASES']>0):
        return 'istallment'


# In[89]:


#applying purchase
credit['purchase_type']=credit.apply(purchase,axis=1)


# In[90]:


credit['purchase_type'].value_counts()


# In[91]:


#Limit_usage (balance to credit limit ratio ) credit card utilization
#Lower value implies cutomers are maintaing thier balance properly. Lower value means good credit score
credit['limit_usage']=credit.apply(lambda x: x['BALANCE']/x['CREDIT_LIMIT'], axis=1)


# In[92]:


credit['limit_usage'].head()


# In[93]:


#Payments to minimum payments ratio etc.
credit['PAYMENTS'].isnull().any()
credit['MINIMUM_PAYMENTS'].isnull().value_counts()


# In[94]:


#getting the details of MINIMUM_PAYMENTS
credit['MINIMUM_PAYMENTS'].describe()


# In[95]:


#calculating the minimum_pay
credit['payment_minpay']=credit.apply(lambda x:x['PAYMENTS']/x['MINIMUM_PAYMENTS'],axis=1)


# In[96]:


credit['payment_minpay']


# In[100]:


#Extreme value Treatment
#Since there are variables having extreme values so I am doing log-transformation on the dataset to remove outlier effect
# log tranformation
cr_log=credit.drop(['CUST_ID','purchase_type'],axis=1).applymap(lambda x: np.log(x+1))


# In[101]:


cr_log.describe()


# In[102]:


col=['BALANCE','PURCHASES','CASH_ADVANCE','TENURE','PAYMENTS','MINIMUM_PAYMENTS','PRC_FULL_PAYMENT','CREDIT_LIMIT']
cr_pre=cr_log[[x for x in cr_log.columns if x not in col ]]


# In[103]:


cr_pre.columns


# In[104]:


cr_log.columns


# In[105]:


# Insights from KPIs
# Average payment_minpayment ratio for each purchse type

x=credit.groupby('purchase_type').apply(lambda x: np.mean(x['payment_minpay']))
type(x)
x.values


# In[106]:


fig,ax=plt.subplots()
ax.barh(y=range(len(x)), width=x.values,align='center')
ax.set(yticks= np.arange(len(x)),yticklabels = x.index);
plt.title('Mean payment_minpayment ratio for each purchse type')


# In[107]:


credit.describe()


# In[108]:


credit[credit['purchase_type']=='n']


# In[109]:


credit.groupby('purchase_type').apply(lambda x: np.mean(x['Monthly_cash_advance'])).plot.barh()

plt.title('Average cash advance taken by customers of different Purchase type : Both, None,Installment,One_Off')


# In[110]:


# Customers who don't do either one-off or installment purchases take more cash on advance
credit.groupby('purchase_type').apply(lambda x: np.mean(x['limit_usage'])).plot.barh()


# In[112]:


#Original dataset with categorical column converted to number type.
cre_original=pd.concat([credit,pd.get_dummies(credit['purchase_type'])],axis=1)


# In[113]:


# Preparing Machine learning algorithm
# We do have some categorical data which need to convert with the help of dummy creation
# creating Dummies for categorical variable
cr_pre['purchase_type']=credit.loc[:,'purchase_type']
pd.get_dummies(cr_pre['purchase_type'])


# In[114]:


# Now merge the created dummy with the original data frame
cr_dummy=pd.concat([cr_pre,pd.get_dummies(cr_pre['purchase_type'])],axis=1)


# In[116]:


l=['purchase_type']


# In[117]:


cr_dummy=cr_dummy.drop(l,axis=1)
cr_dummy.isnull().any()


# In[119]:


cr_dummy.info()


# In[120]:


cr_dummy.head(3)


# In[121]:


sns.heatmap(cr_dummy.corr())


# In[122]:


# Standardrizing data
# To put data on the same scale
from sklearn.preprocessing import  StandardScaler


# In[123]:


sc=StandardScaler()


# In[124]:


cr_dummy.shape


# In[125]:


cr_scaled=sc.fit_transform(cr_dummy)


# In[126]:


cr_scaled


# In[127]:


# Applying PCA
# With the help of principal component analysis we will reduce features
from sklearn.decomposition import PCA


# In[128]:


cr_dummy.shape


# In[129]:


#We have 17 features so our n_component will be 17.
pc=PCA(n_components=17)
cr_pca=pc.fit(cr_scaled)


# In[130]:


#Lets check if we will take 17 component then how much varience it explain. Ideally it should be 1 i.e 100%
sum(cr_pca.explained_variance_ratio_)


# In[131]:


var_ratio={}
for n in range(2,18):
    pc=PCA(n_components=n)
    cr_pca=pc.fit(cr_scaled)
    var_ratio[n]=sum(cr_pca.explained_variance_ratio_)


# In[132]:


var_ratio


# In[133]:


# Since 6 components are explaining about 90% variance so we select 6 components
pc=PCA(n_components=6)


# In[134]:


p=pc.fit(cr_scaled)


# In[135]:


cr_scaled.shape


# In[136]:


p.explained_variance_


# In[140]:


np.sum(p.explained_variance_)


# In[141]:


#np.sum(p.explained_variance_)


# In[142]:


var_ratio


# In[143]:


pd.Series(var_ratio).plot()


# In[144]:


# Since 5 components are explaining about 87% variance so we select 5 components
cr_scaled.shape


# In[148]:


# converting string to integer type....[a,b,c] to [1,2,3]
pc_final=PCA(n_components=6).fit(cr_scaled)
reduced_cr=pc_final.fit_transform(cr_scaled)


# In[149]:


dd=pd.DataFrame(reduced_cr)


# In[150]:


dd.head()


# In[151]:


# So initially we had 17 variables now its 5 so our variable go reduced
dd.shape


# In[154]:


#storing index to col_list
col_list=cr_dummy.columns


# In[155]:


col_list


# In[156]:


pd.DataFrame(pc_final.components_.T, columns=['PC_' +str(i) for i in range(6)],index=col_list)


# In[ ]:


# So above data gave us eigen vector for each component we had all eigen vector value very small
# we can remove those variable bur in our case its not


# In[159]:


# Factor Analysis : variance explained by each component- 
pd.Series(pc_final.explained_variance_ratio_,index=['PC_'+ str(i) for i in range(6)])


# In[160]:


#Clustering
# Based on the intuition on type of purchases made by customers and their distinctive behavior exhibited based on the
# purchase_type (as visualized above in Insights from KPI) , I am starting with 4 clusters.
from sklearn.cluster import KMeans


# In[161]:


km_4=KMeans(n_clusters=4,random_state=123)


# In[162]:


km_4.fit(reduced_cr)


# In[163]:


km_4.labels_


# In[164]:


pd.Series(km_4.labels_).value_counts()


# In[165]:


# Here we donot have known k value so we will find the K. To do that we need to take a cluster range between 1 and 21.
# Identify Cluster Error

cluster_range = range( 1, 21 )
cluster_errors = []

for num_clusters in cluster_range:
    clusters = KMeans( num_clusters )
    clusters.fit( reduced_cr )
    cluster_errors.append( clusters.inertia_ )# clusters.inertia_ is basically cluster error here.


# In[166]:


clusters_df = pd.DataFrame( { "num_clusters":cluster_range, "cluster_errors": cluster_errors } )
clusters_df[0:21]


# In[167]:


# allow plots to appear in the notebook
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
plt.figure(figsize=(12,6))
plt.plot( clusters_df.num_clusters, clusters_df.cluster_errors, marker = "o" )


# In[168]:


# From above graph we will find elbow range. here it is 4,5,6
# Silhouette Coefficient
from sklearn import metrics


# In[169]:


# calculate SC(Super Conductivity) for K=3 through K=12
k_range = range(2, 21)
scores = []
for k in k_range:
    km = KMeans(n_clusters=k, random_state=1)
    km.fit(reduced_cr)
    scores.append(metrics.silhouette_score(reduced_cr, km.labels_))


# In[170]:


scores


# In[171]:


# plot the results
plt.plot(k_range, scores)
plt.xlabel('Number of clusters')
plt.ylabel('Silhouette Coefficient')
plt.grid(True)


# In[173]:


color_map={0:'r',1:'b',2:'g',3:'y'}
label_color=[color_map[l] for l in km_4.labels_]
plt.figure(figsize=(7,7))
plt.scatter(reduced_cr[:,0],reduced_cr[:,1],c=label_color,cmap='Spectral',alpha=0.1)


# In[174]:


# It is very difficult to draw iddividual plot for cluster, so we will use pair plot which will provide us all graph in one shot.
#To do that we need to take following steps
df_pair_plot=pd.DataFrame(reduced_cr,columns=['PC_' +str(i) for i in range(6)])


# In[175]:


df_pair_plot['Cluster']=km_4.labels_ #Add cluster column in the data frame


# In[176]:


df_pair_plot.head()


# In[177]:


#pairwise relationship of components on the data
sns.pairplot(df_pair_plot,hue='Cluster', palette= 'Dark2', diag_kind='kde',size=1.85)


# In[167]:


# Now we have done here with priciple component now we need to come bring our 
#original data frame and we will merge the cluster with them
# To interprate result we need to use our data frame


# In[178]:


# Key performace variable selection . here i am taking varibales which we will use in derving new KPI. 
#We can take all 17 variables but it will be difficult to interprate.So are are selecting less no of variables.

col_kpi=['PURCHASES_TRX','Monthly_avg_purchase','Monthly_cash_advance','limit_usage','CASH_ADVANCE_TRX',
         'payment_minpay','both_oneoff_installment','istallment','one_off','none','CREDIT_LIMIT']


# In[179]:


cr_pre.describe()


# In[180]:


# Conactenating labels found through Kmeans with data 
cluster_df_4=pd.concat([cre_original[col_kpi],pd.Series(km_4.labels_,name='Cluster_4')],axis=1)
#cluster_df_4=pd.concat([cr_pre.describe[col_kpi],pd.Series(km_4.labels_,name='Cluster_4')],axis=1)


# In[181]:


cluster_df_4.head()


# In[182]:


# Mean value gives a good indication of the distribution of data. So we are finding mean value for each variable for each cluster
cluster_4=cluster_df_4.groupby('Cluster_4').apply(lambda x: x[col_kpi].mean()).T
cluster_4


# In[183]:


fig,ax=plt.subplots(figsize=(15,10))
index=np.arange(len(cluster_4.columns))

cash_advance=np.log(cluster_4.loc['Monthly_cash_advance',:].values)
credit_score=(cluster_4.loc['limit_usage',:].values)
purchase= np.log(cluster_4.loc['Monthly_avg_purchase',:].values)
payment=cluster_4.loc['payment_minpay',:].values
installment=cluster_4.loc['istallment',:].values
one_off=cluster_4.loc['one_off',:].values


bar_width=.10
b1=plt.bar(index,cash_advance,color='b',label='Monthly cash advance',width=bar_width)
b2=plt.bar(index+bar_width,credit_score,color='m',label='Credit_score',width=bar_width)
b3=plt.bar(index+2*bar_width,purchase,color='k',label='Avg purchase',width=bar_width)
b4=plt.bar(index+3*bar_width,payment,color='c',label='Payment-minpayment ratio',width=bar_width)
b5=plt.bar(index+4*bar_width,installment,color='r',label='installment',width=bar_width)
b6=plt.bar(index+5*bar_width,one_off,color='g',label='One_off purchase',width=bar_width)

plt.xlabel("Cluster")
plt.title("Insights")
plt.xticks(index + bar_width, ('Cl-0', 'Cl-1', 'Cl-2', 'Cl-3'))
plt.legend()


# In[184]:


# Clusters are clearly distinguishing behavior within customers
# Percentage of each cluster in the total customer base
s=cluster_df_4.groupby('Cluster_4').apply(lambda x: x['Cluster_4'].value_counts())
print (s),'\n'

per=pd.Series((s.values.astype('float')/ cluster_df_4.shape[0])*100,name='Percentage')
print ("Cluster -4 "),'\n'
print (pd.concat([pd.Series(s.values,name='Size'),per],axis=1))


# In[185]:


# Finding behaviour with 5 Clusters:
km_5=KMeans(n_clusters=5,random_state=123)
km_5=km_5.fit(reduced_cr)
km_5.labels_


# In[186]:


pd.Series(km_5.labels_).value_counts()


# In[187]:


plt.figure(figsize=(7,7))
plt.scatter(reduced_cr[:,0],reduced_cr[:,1],c=km_5.labels_,cmap='Spectral',alpha=0.5)
plt.xlabel('PC_0')
plt.ylabel('PC_1')


# In[188]:


cluster_df_5=pd.concat([cre_original[col_kpi],pd.Series(km_5.labels_,name='Cluster_5')],axis=1)


# In[189]:


# Finding Mean of features for each cluster
cluster_df_5.groupby('Cluster_5').apply(lambda x: x[col_kpi].mean()).T


# In[190]:


# Conclusion With 5 clusters :
# So we don't have quite distinguishable characteristics with 5 clusters,
s1=cluster_df_5.groupby('Cluster_5').apply(lambda x: x['Cluster_5'].value_counts())
print (s1)


# In[191]:


# percentage of each cluster

print ("Cluster-5"),'\n'
per_5=pd.Series((s1.values.astype('float')/ cluster_df_5.shape[0])*100,name='Percentage')
print (pd.concat([pd.Series(s1.values,name='Size'),per_5],axis=1))


# In[192]:


# Finding behavior with 6 clusters:
km_6=KMeans(n_clusters=6).fit(reduced_cr)
km_6.labels_


# In[193]:


color_map={0:'r',1:'b',2:'g',3:'c',4:'m',5:'k'}
label_color=[color_map[l] for l in km_6.labels_]
plt.figure(figsize=(7,7))
plt.scatter(reduced_cr[:,0],reduced_cr[:,1],c=label_color,cmap='Spectral',alpha=0.5)


# In[194]:


cluster_df_6 = pd.concat([cre_original[col_kpi],pd.Series(km_6.labels_,name='Cluster_6')],axis=1)


# In[195]:


six_cluster=cluster_df_6.groupby('Cluster_6').apply(lambda x: x[col_kpi].mean()).T
six_cluster


# In[196]:


fig,ax=plt.subplots(figsize=(15,10))
index=np.arange(len(six_cluster.columns))

cash_advance=np.log(six_cluster.loc['Monthly_cash_advance',:].values)
credit_score=(six_cluster.loc['limit_usage',:].values)
purchase= np.log(six_cluster.loc['Monthly_avg_purchase',:].values)
payment=six_cluster.loc['payment_minpay',:].values
installment=six_cluster.loc['istallment',:].values
one_off=six_cluster.loc['one_off',:].values

bar_width=.10
b1=plt.bar(index,cash_advance,color='b',label='Monthly cash advance',width=bar_width)
b2=plt.bar(index+bar_width,credit_score,color='m',label='Credit_score',width=bar_width)
b3=plt.bar(index+2*bar_width,purchase,color='k',label='Avg purchase',width=bar_width)
b4=plt.bar(index+3*bar_width,payment,color='c',label='Payment-minpayment ratio',width=bar_width)
b5=plt.bar(index+4*bar_width,installment,color='r',label='installment',width=bar_width)
b6=plt.bar(index+5*bar_width,one_off,color='g',label='One_off purchase',width=bar_width)

plt.xlabel("Cluster")
plt.title("Insights")
plt.xticks(index + bar_width, ('Cl-0', 'Cl-1', 'Cl-2', 'Cl-3','Cl-4','Cl-5'))

plt.legend()


# In[197]:


cash_advance=np.log(six_cluster.loc['Monthly_cash_advance',:].values)
credit_score=list(six_cluster.loc['limit_usage',:].values)
cash_advance


# In[198]:


#Checking performance metrics for Kmeans
#I am validating performance with 2 metrics Calinski harabaz and Silhouette score
from sklearn.metrics import calinski_harabaz_score,silhouette_score


# In[199]:


score={}
score_c={}
for n in range(3,10):
    km_score=KMeans(n_clusters=n)
    km_score.fit(reduced_cr)
    score_c[n]=calinski_harabaz_score(reduced_cr,km_score.labels_)
    score[n]=silhouette_score(reduced_cr,km_score.labels_)


# In[200]:


pd.Series(score).plot()


# In[201]:


pd.Series(score_c).plot()


# In[ ]:




