#!/usr/bin/env python
# coding: utf-8

# In[6]:


import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt


# In[7]:


os.getcwd()


# In[8]:


df= pd.read_csv('bitcoin_dataset1.csv')
df


# In[36]:


df.iloc[1023:1024,1:2]


# In[16]:


df=df.copy()


# In[9]:


df.describe()


# In[10]:


df.head()


# In[11]:


df.tail()


# In[12]:


df.nunique()


# In[13]:


df.isnull().sum()


# In[14]:


df.isnull()


# In[15]:


import seaborn as sns
sns.heatmap(df.isnull(),yticklabels= False,cmap='viridis' )


# In[17]:


df1=df.fillna(method='bfill')


# In[18]:


df1.isnull().sum()


# In[20]:


X=df1[["btc_market_cap","btc_n_transactions","btc_miners_revenue","btc_cost_per_transaction","btc_difficulty","btc_hash_rate","btc_cost_per_transaction_percent"]]
X


# 

# In[23]:


y=df1["btc_market_price"]
y


# In[25]:


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3)


# In[26]:


from sklearn.linear_model import LinearRegression
model=LinearRegression()
train=model.fit(X_train,y_train)


# In[27]:


train.coef_


# In[28]:


train.intercept_


# In[29]:


train.score(X_train,y_train)


# In[30]:


pred=train.predict(X_test)
pred


# In[31]:


from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error


# In[32]:


MSE=mean_squared_error(y_test,pred)
MSE


# In[33]:


MAE=mean_absolute_error(y_test,pred)
MAE


# In[34]:


plt.figure(figsize=(30,20))
sns.heatmap(df.corr(),annot=True)


# In[35]:


df.corr()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




