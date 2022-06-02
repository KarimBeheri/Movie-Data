#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

from sklearn.model_selection import train_test_split
#feature Selection
from sklearn.feature_selection import SelectKBest , f_regression
#for Scaling 
from sklearn import preprocessing
from matplotlib import pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from sklearn.metrics import r2_score,mean_squared_error


# In[2]:


data=pd.read_csv(r'C:\Users\karim\Downloads\tmdb-movies (2) (1).csv')


# In[3]:


cor_matrix  = data.corr().abs()
cor_matrix


# In[4]:


upper_tri = cor_matrix.where (np.triu(np.ones(cor_matrix.shape),k=1).astype(np.bool))
upper_tri


# In[5]:


to_drop =[column for column in upper_tri.columns if any(upper_tri[column])  >(0.5)]
to_drop


# In[6]:


data=data.drop(
    [
     'id',
     'overview',
     'imdb_id',
     'director',
     'keywords',
     'tagline',
     'homepage',
     'original_title',
     'release_date',
     ],
    axis=1
    )


# In[7]:


data.head()


# In[8]:


print(data.isnull().sum().sort_values())


# In[9]:


data.info()


# In[10]:


data=data.drop_duplicates()


# In[11]:


data['profit']=data.revenue_adj-data.budget_adj


# In[12]:


data=data.drop(
['revenue_adj','budget_adj'
],axis=1)


# In[13]:


data.dropna(subset = ["production_companies"], inplace=True)


# In[14]:


data.dropna(subset = ["cast"], inplace=True)


# In[15]:


data.dropna(subset = ["genres"], inplace=True)


# In[16]:


dumies1=data['genres'].str.get_dummies(sep="|")
data=data.drop(['genres'],axis=1)


# In[17]:


dumies2=data['production_companies'].str.get_dummies(sep="|")
data=data.drop(['production_companies'],axis=1)


# In[18]:


dumies3=data['cast'].str.get_dummies(sep="|")
data=data.drop(['cast'],axis=1)


# In[19]:


data=pd.concat([data,dumies1],axis='columns')
data=pd.concat([data,dumies2],axis='columns')
data=pd.concat([data,dumies3],axis='columns')


# In[20]:


x=data.drop(['profit'],axis=1)
y=data['profit'] 


# In[21]:


data.describe()


# In[22]:


scaler = preprocessing.MinMaxScaler()
names = data.columns
d= scaler.fit_transform(data)
scaled_df = pd.DataFrame(d, columns=names)
print(scaled_df.head())


# In[23]:


data.shape


# In[24]:


selector = SelectKBest(score_func = f_regression , k=17)
selector.fit(x,y)
cols = selector.get_support(indices = True)
x = x.iloc[:,cols]
print("Left Columns are :")
print(x.columns)


# In[403]:


x_train,x_test,y_train,y_test=train_test_split(x , y,test_size=.30)


# In[404]:


model = LinearRegression().fit(x_train, y_train)


# In[405]:


print(model.score(x_train, y_train))


# In[406]:


print(model.score(x_test, y_test))


# In[354]:


poly = PolynomialFeatures(degree=3, include_bias=False)


# In[221]:


X_poly = poly.fit_transform(x)


# In[222]:


lin_reg2 = LinearRegression()
lin_reg2.fit(X_poly,y)


# In[223]:


lin_reg2.score(X_poly,y)


# In[111]:


from sklearn.ensemble import GradientBoostingRegressor
model = GradientBoostingRegressor(random_state=0).fit(x_train, y_train)
print(model.score(x_train, y_train))
print(model.score(x_test, y_test))


# In[112]:


from sklearn.ensemble import RandomForestRegressor
model = RandomForestRegressor(random_state=0).fit(x_train, y_train)
print(model.score(x_train, y_train))
print(model.score(x_test, y_test))


# In[ ]:


#data.info()


# In[ ]:




