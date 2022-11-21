#!/usr/bin/env python
# coding: utf-8

# In[1]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load


# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import sklearn.preprocessing as preprocessing
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, FormatStrFormatter

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# In[2]:


xtrain=pd.read_csv("./xtrain.csv")
ytrain=pd.read_csv("./ytrain.csv")


# In[3]:


test=pd.read_csv("./xtest.csv")


# In[4]:


lr1 = LogisticRegression()
lr1.fit(xtrain, ytrain.values.ravel())


# In[5]:


te=test


# In[6]:


ypred=lr1.predict(te)


# In[7]:


test["isFraud"]=ypred
ypred=test["isFraud"]
ypred.to_csv("f2.csv")
test=test.drop(['isFraud'],axis=1,inplace=False)


# In[8]:


# #Applying PCA

# scaler = StandardScaler()
# X_scaled = scaler.fit_transform(xtrain)


# In[9]:


# pca = PCA(n_components=24)
# principalComponents = pca.fit_transform(X_scaled)
# print(pca.explained_variance_ratio_)


# In[10]:


# lr2 = LogisticRegression()
# lr2.fit(principalComponents, ytrain)


# In[11]:


# principalComponents1 = pca.fit_transform(test)


# In[12]:


# ypred=lr2.predict(principalComponents1)


# In[13]:


# test["isFraud"]=ypred
# ypred=test["isFraud"]
# ypred.to_csv("f2.csv")
# test=test.drop(['isFraud'],axis=1,inplace=False)


# In[14]:


from collections import Counter
Counter(ypred)


# In[ ]:




