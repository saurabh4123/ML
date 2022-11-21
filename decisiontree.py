#!/usr/bin/env python
# coding: utf-8

# In[1]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

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

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# In[2]:


xtrain=pd.read_csv("./xtrain.csv")
ytrain=pd.read_csv("./ytrain.csv")
test=pd.read_csv("./xtest.csv")


# In[3]:


from sklearn import tree
dtc = tree.DecisionTreeClassifier()
dtc = dtc.fit(xtrain, ytrain.values.ravel())


# In[4]:


ypred=dtc.predict(test)


# In[5]:


test["isFraud"]=ypred
ypred=test["isFraud"]
ypred.to_csv("f(DTC).csv")
test=test.drop(['isFraud'],axis=1,inplace=False)


# In[6]:


from collections import Counter


# In[7]:


Counter(ypred)


# In[ ]:




