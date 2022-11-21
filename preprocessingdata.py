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


train=pd.read_csv("./train.csv")


# In[3]:


test=pd.read_csv("./test.csv")


# In[4]:


pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)


# In[5]:


def missing(dff):
    print (round((dff.isnull().sum() * 100/ len(dff)),2).sort_values(ascending=False))


# In[6]:


missing(train)


# In[7]:


missing(test)


# **        **Missing values removal(threshold-50%)****

# In[8]:


limitPer = len(train) * .50
train.dropna(thresh=limitPer, axis=1,inplace=True)
limitPer1 = len(test) * .50
test.dropna(thresh=limitPer1, axis=1,inplace=True)


# In[ ]:





# In[9]:


train.duplicated().sum()


# In[10]:


test.duplicated().sum()


# In[11]:


n = train.nunique(axis=0)
n


# In[12]:


ntest = test.nunique(axis=0)
ntest


# In[13]:


wt=train["isFraud"].value_counts(normalize=True).to_frame()
wt.plot.bar()
wt.T


# In[14]:


cat_features = ['ProductCD', 'card1','card2','card3','card4','card5','card6', 'addr1','addr2', 'P_emaildomain',
                'M1', 'M2', 'M3', 'M4', 'M6']

num_features = [x for x in train.columns.values[2:] if x not in cat_features]


# In[15]:


print('Categorical features :', len(cat_features))
print('Numerical features : ',len(num_features))


# In[16]:


plt.figure(figsize=(12,6))
train_ProductCD = (train.groupby(['isFraud'])['ProductCD']
                     .value_counts(normalize=True)
                     .rename('percentage')
                     .mul(100)
                     .reset_index()
                     .sort_values('ProductCD'))
sns.barplot(x="ProductCD", y="percentage", hue="isFraud", data=train_ProductCD, palette=["#FFD500", "#005BBB"]);


# In[17]:


train[cat_features]=train[cat_features].fillna(train.mode().iloc[0])
test[cat_features]=test[cat_features].fillna(test.mode().iloc[0])


# In[18]:


train[num_features]=train[num_features].fillna(train.median())
test[num_features]=test[num_features].fillna(test.median())


# In[19]:


train.isnull().sum().sum()


# In[20]:


test.isnull().sum().sum()


# In[21]:


def heatmap(data, row_labels, col_labels, ax=None,
            cbar_kw={}, cbarlabel="", **kwargs):
    
    if not ax:
        ax = plt.gca()

    # Plot the heatmap
    im = ax.imshow(data, **kwargs)

    # Create colorbar
    cbar = ax.figure.colorbar(im, ax=ax, **cbar_kw)
    cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom")

    # We want to show all ticks...
    ax.set_xticks(np.arange(data.shape[1]))
    ax.set_yticks(np.arange(data.shape[0]))
    # ... and label them with the respective list entries.
    ax.set_xticklabels(col_labels)
    ax.set_yticklabels(row_labels)

    # Let the horizontal axes labeling appear on top.
    ax.tick_params(top=True, bottom=False,
                   labeltop=True, labelbottom=False)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=-30, ha="right",
             rotation_mode="anchor")

    # Turn spines off and create white grid.
    for edge, spine in ax.spines.items():
        spine.set_visible(False)

    ax.set_xticks(np.arange(data.shape[1]+1)-.5, minor=True)
    ax.set_yticks(np.arange(data.shape[0]+1)-.5, minor=True)
    ax.grid(which="minor", color="w", linestyle='-', linewidth=3)
    ax.tick_params(which="minor", bottom=False, left=False)

    return im, cbar
def annotate_heatmap(im, data=None, valfmt="{x:.2f}",
                     textcolors=["black", "white"],
                     threshold=None, **textkw):
   
    if not isinstance(data, (list, np.ndarray)):
        data = im.get_array()

    # Normalize the threshold to the images color range.
    if threshold is not None:
        threshold = im.norm(threshold)
    else:
        threshold = im.norm(data.max())/2.

    # Set default alignment to center, but allow it to be
    # overwritten by textkw.
    kw = dict(horizontalalignment="center",
              verticalalignment="center")
    kw.update(textkw)

    # Get the formatter in case a string is supplied
    if isinstance(valfmt, str):
        valfmt = matplotlib.ticker.StrMethodFormatter(valfmt)

    # Loop over the data and create a `Text` for each "pixel".
    # Change the text's color depending on the data.
    texts = []
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            kw.update(color=textcolors[int(im.norm(data[i, j]) > threshold)])
            text = im.axes.text(j, i, valfmt(data[i, j], None), **kw)
            texts.append(text)

    return texts


# In[22]:


plt.figure(figsize=(15,18))
plt.subplot(3,3,1)
train_m1 = (train.groupby(['isFraud'])['M1']
                     .value_counts(normalize=True)
                     .rename('percentage')
                     .mul(100)
                     .reset_index()
                     .sort_values('M1'))
sns.barplot(x="M1", y="percentage", hue="isFraud", palette=['#90EE90', '#FA8072'], data=train_m1)
plt.title('TrainM1')
plt.subplot(3,3,2)
train_m2 = (train.groupby(['isFraud'])['M2']
                     .value_counts(normalize=True)
                     .rename('percentage')
                     .mul(100)
                     .reset_index()
                     .sort_values('M2'))
sns.barplot(x="M2", y="percentage", hue="isFraud", palette=['#90EE90', '#FA8072'], data=train_m2)
plt.title('TrainM2')
plt.subplot(3,3,3)
train_m3 = (train.groupby(['isFraud'])['M3']
                     .value_counts(normalize=True)
                     .rename('percentage')
                     .mul(100)
                     .reset_index()
                     .sort_values('M3'))
sns.barplot(x="M3", y="percentage", hue="isFraud", palette=['#90EE90', '#FA8072'], data=train_m3)
plt.title('TrainM3')
plt.subplot(3,3,4)
train_m4 = (train.groupby(['isFraud'])['M4']
                     .value_counts(normalize=True)
                     .rename('percentage')
                     .mul(100)
                     .reset_index()
                     .sort_values('M4'))
sns.barplot(x="M4", y="percentage", hue="isFraud", palette=['#90EE90', '#FA8072'], data=train_m4)
plt.title('TrainM4')
plt.subplot(3,3,5)
train_m6 = (train.groupby(['isFraud'])['M6']
                     .value_counts(normalize=True)
                     .rename('percentage')
                     .mul(100)
                     .reset_index()
                     .sort_values('M6'))
sns.barplot(x="M6", y="percentage", hue="isFraud", palette=['#90EE90', '#FA8072'], data=train_m6)
plt.title('TrainM6')


# In[23]:


train["M4"].unique()


# In[24]:


m_cols = [c for c in train if c[0] == 'M']
train[m_cols].head()


# In[25]:


m_col = train[['M1','M2','M3','M4','M6']]


# In[26]:


(train[m_cols] == 'T').sum().plot(kind='bar', color='#87CEFA', title='Count of T by M column', figsize=(15, 2))
plt.show()
(train[m_cols] == 'F').sum().plot(kind='bar', color='#87CEFA', title='Count of F by M column',figsize=(15, 2))
plt.show()
(train[m_cols].isna()).sum().plot(kind='bar', color='#87CEFA', title='Count of NaN by M column',figsize=(15, 2))
plt.show()


# In[27]:


train.groupby('M4')['TransactionID'].count().plot(kind='bar',
                                                  color='#87CEFA',
                                                  title='Count of values for M4',
                                                  figsize=(15, 3))
plt.show()


# In[28]:


C_cols = [c for c in train if c[0] == 'C']
train[C_cols].head()


# In[29]:


cor_c = train[['C1','C2','C3','C4','C5','C6','C7','C8','C9','C10','C11','C12','C13','C14','isFraud']]


# In[30]:


corr_matrix = cor_c.corr()
plt.figure(figsize=(15, 8))

im, _ = heatmap(corr_matrix, cor_c, cor_c, 
                cmap="YlGn", vmin=-1, vmax=1,
                cbarlabel="correlation coeff.")


def func(x, pos):
    return "{:.2f}".format(x).replace("0.", ".").replace("1.00", "")

annotate_heatmap(im, valfmt=matplotlib.ticker.FuncFormatter(func), size=10)


plt.tight_layout()
plt.show()




# In[31]:


d_cols = [c for c in train if c[0] == 'D']
train[d_cols].head()


# In[32]:


plt.figure(figsize=(15,18))
plt.subplot(3,3,1)
train_d1 = (train.groupby(['isFraud'])['D1']
                     .value_counts(normalize=True)
                     .rename('percentage')
                     .mul(.100)
                     .reset_index()
                     .sort_values('D1'))
sns.barplot(x="D1", y="percentage", hue="isFraud", palette=['#90EE90', '#FA8072'], data=train_d1)
plt.title('TrainD1')



# In[33]:


def describe(datatrain,datatest,feature):
    d = pd.DataFrame(columns=[feature,'Train','TrainFraud','TrainLegit','Test'])
    d[feature] = ['count','mean','std','min','25%','50%','75%','max','unique','NaN','NaNshare']
    for i in range(0,8):
        d['Train'].iloc[i] = datatrain[feature].describe().iloc[i]
        d['TrainFraud'].iloc[i]=datatrain[datatrain['isFraud']==1][feature].describe().iloc[i]
        d['TrainLegit'].iloc[i]=datatrain[datatrain['isFraud']==0][feature].describe().iloc[i]
        d['Test'].iloc[i]=datatest[feature].describe().iloc[i]
    d['Train'].iloc[8] = len(datatrain[feature].unique())
    d['TrainFraud'].iloc[8]=len(datatrain[datatrain['isFraud']==1][feature].unique())
    d['TrainLegit'].iloc[8]=len(datatrain[datatrain['isFraud']==0][feature].unique())
    d['Test'].iloc[8]=len(datatest[feature].unique())
    d['Train'].iloc[9] = datatrain[feature].isnull().sum()
    d['TrainFraud'].iloc[9] = datatrain[datatrain['isFraud']==1][feature].isnull().sum()
    d['TrainLegit'].iloc[9] = datatrain[datatrain['isFraud']==0][feature].isnull().sum()
    d['Test'].iloc[9]=datatest[feature].isnull().sum()
    d['Train'].iloc[10] = datatrain[feature].isnull().sum()/len(datatrain)
    d['TrainFraud'].iloc[10] = datatrain[datatrain['isFraud']==1][feature].isnull().sum()/len(datatrain[datatrain['isFraud']==1])
    d['TrainLegit'].iloc[10] = datatrain[datatrain['isFraud']==0][feature].isnull().sum()/len(datatrain[datatrain['isFraud']==0])
    d['Test'].iloc[10]=datatest[feature].isnull().sum()/len(datatest)
    return d


# In[34]:


transactionAmtDescribe = describe(train,test,'TransactionAmt')


# In[35]:


transactionAmtDescribe


# In[36]:


l=[99.9,99.91,99.92,99.93,99.94,99.95,99.96,99.97,99.98,99.99]
for i in l:
    print('train',np.percentile(train['TransactionAmt'],i))
    print('test',np.percentile(test['TransactionAmt'],i))


# In[37]:


train['LogTransactionAmt'] = np.log(train['TransactionAmt'])
test['LogTransactionAmt'] = np.log(test['TransactionAmt'])


# In[38]:


plt.figure(figsize=(9,6))
plt.subplot(1,2,1)
sns.distplot(train[train['isFraud']==0]['LogTransactionAmt'])
sns.distplot(train[train['isFraud']==1]['LogTransactionAmt'])
plt.legend(['legit','fraud'])
plt.title('Train')
plt.subplot(1,2,2)
sns.distplot(test['LogTransactionAmt'])


# In[39]:


plt.figure(figsize=(12,6))
plt.subplot(1,2,1)
train_card4 = (train[~train['card4'].isnull()].groupby(['isFraud'])['card4']
                     .value_counts(normalize=True)
                     .rename('percentage')
                     .mul(100)
                     .reset_index()
                     .sort_values('card4'))
sns.barplot(x="card4", y="percentage", hue="isFraud", data=train_card4)
plt.title('Train')
plt.subplot(1,2,2)
test_card4 =test[~test['card4'].isnull()]['card4'].value_counts(normalize=True).mul(100).rename('percentage')\
.reset_index()
sns.barplot(x="index", y="percentage", data=test_card4)
plt.title('Test')


# In[40]:


plt.figure(figsize=(12,6))
plt.subplot(1,2,1)
train_card6 = (train[~train['card6'].isnull()].groupby(['isFraud'])['card6']
                     .value_counts(normalize=True)
                     .rename('percentage')
                     .mul(100)
                     .reset_index()
                     .sort_values('card6'))
sns.barplot(x="card6", y="percentage", hue="isFraud", data=train_card6)
plt.title('Train')
plt.subplot(1,2,2)
test_card6 =test[~test['card6'].isnull()]['card6'].value_counts(normalize=True).mul(100).rename('percentage')\
.reset_index()
sns.barplot(x="index", y="percentage", data=test_card6)
plt.title('Test')


# In[41]:


plt.figure(figsize=(15,6))
plt.subplot(1,2,1)
train_card4 = (train[~train['card4'].isnull()].groupby(['isFraud'])['card4']
                     .value_counts(normalize=True)
                     .rename('percentage')
                     .mul(100)
                     .reset_index()
                     .sort_values('card4'))
sns.barplot(x="card4", y="percentage", hue="isFraud", data=train_card4)
plt.title('Train')
plt.subplot(1,2,2)
test_card4 =test[~test['card4'].isnull()]['card4'].value_counts(normalize=True).mul(100).rename('percentage')\
.reset_index()
sns.barplot(x="index", y="percentage", data=test_card4)
plt.title('Test')


# In[42]:


plt.figure(figsize=(12,6))
plt.subplot(1,2,1)
train_card6 = (train[~train['card6'].isnull()].groupby(['isFraud'])['card6']
                     .value_counts(normalize=True)
                     .rename('percentage')
                     .mul(100)
                     .reset_index()
                     .sort_values('card6'))
sns.barplot(x="card6", y="percentage", hue="isFraud", data=train_card6)
plt.title('Train')
plt.subplot(1,2,2)
test_card6 =test[~test['card6'].isnull()]['card6'].value_counts(normalize=True).mul(100).rename('percentage')\
.reset_index()
sns.barplot(x="index", y="percentage", data=test_card6)
plt.title('Test')


# In[43]:


# train['P_emaildomain'].value_counts()[:10]


# In[44]:


# def returnfirst(email):
#     return email.split(".")[0]


# In[45]:


# train['first'] = train[~train['P_emaildomain'].isnull()]['P_emaildomain'].apply(returnfirst)


# In[46]:


# test['first'] = test[~test['P_emaildomain'].isnull()]['P_emaildomain'].apply(returnfirst)


# In[47]:


# train_email = (train.groupby(['isFraud'])['first']
#                      .value_counts(normalize=True)
#                      .rename('percentage')
#                      .mul(100)
#                      .reset_index()
#                      .sort_values('first'))


# In[48]:


# plt.figure(figsize=(22,10))
# sns.barplot(x="first", y="percentage", hue="isFraud", data=train_email)
# plt.xticks(rotation=90)


# In[49]:


D_cols = [c for c in train if c[0] == 'D']
train[D_cols].head()


# In[50]:


D_c = train[['D1','D2','D3','D4','D10','D11','D15','isFraud']];
f = D_c.corr()


# In[51]:


plt.figure(1,figsize=(12,12))
sns.heatmap(f,annot=True)
plt.title('D train')


# In[52]:


train["M4"].unique()


# In[53]:


train["P_emaildomain"].unique()


# In[54]:


colsToHotEncode=["ProductCD","card4","card6","M4"]


# In[55]:


train=pd.get_dummies(train,columns=colsToHotEncode)


# In[56]:


test=pd.get_dummies(test,columns=colsToHotEncode)


# In[57]:


train["M1"].replace(['T', 'F'], [1, 0], inplace=True)
train["M2"].replace(['T', 'F'], [1, 0], inplace=True)
train["M3"].replace(['T', 'F'], [1, 0], inplace=True)
train["M6"].replace(['T', 'F'], [1, 0], inplace=True)


# In[58]:


test["M1"].replace(['T', 'F'], [1, 0], inplace=True)
test["M2"].replace(['T', 'F'], [1, 0], inplace=True)
test["M3"].replace(['T', 'F'], [1, 0], inplace=True)
test["M6"].replace(['T', 'F'], [1, 0], inplace=True)


# In[59]:


frq=train.groupby("P_emaildomain").size()
frq
frqtest=test.groupby("P_emaildomain").size()
frqtest


# In[60]:


frq_dist=frq/len(train)
frq_dist
frq_dist_test=frqtest/len(test)
frq_dist_test


# In[61]:


train["P_emailDomainFreq"] = train.P_emaildomain.map(frq_dist)
test["P_emailDomainFreq"] = test.P_emaildomain.map(frq_dist_test)


# In[62]:


train.drop("P_emaildomain",axis=1,inplace=True)
test.drop("P_emaildomain",axis=1,inplace=True)


# In[63]:


#for removing columns with only one unique value if any
for col in train.columns:
    if len(train[col].unique()) == 1:
        train.drop(col,inplace=True,axis=1)
for col in test.columns:
    if len(test[col].unique()) == 1:
        test.drop(col,inplace=True,axis=1)


# In[64]:


class OutlierRemoval: 
    def __init__(self, lower_quartile, upper_quartile):
        self.lower_whisker = lower_quartile - 1.5*(upper_quartile - lower_quartile)
        self.upper_whisker = upper_quartile + 1.5*(upper_quartile - lower_quartile)
    def removeOutlier(self, x):
        return (x if x <= self.upper_whisker and x >= self.lower_whisker else (self.lower_whisker if x < self.lower_whisker else (self.upper_whisker)))


# In[65]:


for i in num_features:
    num_var_outlier_remover = OutlierRemoval(train[i].quantile(0.25), train[i].quantile(0.75))
    train[i] = train[i].apply(num_var_outlier_remover.removeOutlier)
    num_var_outlier_remover_test = OutlierRemoval(test[i].quantile(0.25), test[i].quantile(0.75))
    test[i] = test[i].apply(num_var_outlier_remover_test.removeOutlier)    


# In[66]:


train.describe() #removed outliers


# In[67]:


test.describe() #removed outliers


# In[68]:


# train.corr().isFraud


# In[69]:


ytrain=train["isFraud"]
xtrain=train
xtrain.drop("isFraud",inplace=True,axis=1)


# In[70]:


xtrain=xtrain.drop(['TransactionID'],axis=1,inplace=False)


# In[71]:


test=test.drop(['TransactionID'],axis=1,inplace=False)


# In[72]:


from imblearn.over_sampling import SMOTE
oversample = SMOTE()
xtrain, ytrain = oversample.fit_resample(xtrain, ytrain)


# In[73]:


xtrain.to_csv("xtrain.csv",index=False)


# In[74]:


ytrain.to_csv("ytrain.csv",index=False)


# In[75]:


test.to_csv("xtest.csv",index=False)


# In[76]:


# #Applying PCA

# scaler = StandardScaler()
# X_scaled = scaler.fit_transform(xtrain)


# In[77]:


# #Finding parameters(components) for PCA

# candidate_components=np.arange(1,100,5)
# variances=[]
# for candidate in candidate_components:
#     pca = PCA(n_components=candidate)
#     principalComponents = pca.fit_transform(X_scaled)
#     variances.append(np.sum(pca.explained_variance_ratio_))
    
# plt.figure(figsize = (5, 5))
# plt.plot(candidate_components,variances)
# plt.xlabel('Number of components')
# plt.ylabel('Variance')
# plt.title('There will usually be an elbow in the curve, where the explained variance stops growing fast',fontsize=7);


# In[78]:


# pca = PCA(n_components=24)
# principalComponents = pca.fit_transform(X_scaled)
# print(pca.explained_variance_ratio_)


# In[79]:


# s = x.select_dtypes(np.number).lt(0).any()\
#       .reindex(x.columns, fill_value=False)\
# #       .rename_axis("col").reset_index(name='isnegative')

