#!/usr/bin/env python
# coding: utf-8

# In[95]:


#Importing Libraries for my project
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[96]:


import warnings 
warnings.filterwarnings("ignore")


# In[97]:


#Data set

df = pd.read_csv("Iris.csv")


# In[98]:


#Data set Visualisation


df.head()


# In[99]:


df.shape


# data frame description>>>

# In[100]:


df.describe()


# In[101]:


#DataFrame information
df.info


# In[102]:


df.Species.value_counts


# In[103]:


#Data Exploration
#Data Pre Processing

x = df[['Id', 'SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']]
y = df['Species']


# In[104]:


#Take x as a variable
x


# In[113]:


#Take y is a variable 
y


# In[106]:


#Splitting the data

import sklearn
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.20,random_state=42)


# In[107]:


#Linear Regresion Model
#Random Forest  Model
from sklearn.linear_model import LogisticRegression


# In[108]:


#This is a Logistic Regression Model
lr = LogisticRegression()


#Training the model
lr.fit(x,y)
#Using Training sets to train the model
lr.fit(x_train,y_train)


# In[109]:


#Prediction
Prediction = lr.predict(x)


# In[110]:


#Compare the Prediction with the Actual
Scores = pd.DataFrame({'Actual':y, 'Prediction':Prediction})
Scores.head


# In[111]:


#Exe
y_test_hat = lr.predict(x_test)


# In[112]:


from sklearn.metrics import accuracy_score
print(accuracy_score(y_test,y_test_hat)*100,'%')


# #### this project is done by Samudranil Dutta ####
# 
# 

# # THANK YOU...!!!

# In[ ]:




