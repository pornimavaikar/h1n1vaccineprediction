#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")


# In[3]:


data=pd.read_csv("h1n1_vaccine_prediction.csv")
data.head(5)


# In[14]:


#to predict whether the the person is vaccinated or not
#classifier
#dependent variable = h1n1_vaccine


# In[15]:


data.sample(25)


# In[16]:


data.isnull().sum()


# In[17]:


data.dtypes


# In[18]:


data.shape


# In[19]:


sns.countplot(x="sex",hue ="h1n1_vaccine",data=data)


# In[20]:


sns.countplot(x="no_of_children",hue ="h1n1_vaccine",data=data)


# In[21]:


sns.countplot(x="no_of_adults",hue ="h1n1_vaccine",data=data)


# In[22]:


sns.countplot(x="age_bracket",hue ="h1n1_vaccine",data=data)


# In[23]:


sns.countplot(x="qualification",hue="h1n1_vaccine",data=data)


# In[24]:


plt.figure(figsize=(30,25))
sns.pairplot(data,hue="h1n1_vaccine")
plt.show()


# In[25]:


correlation_mat=data.corr()


# In[26]:


g=sns.heatmap(correlation_mat,  vmax=.3, center=0,
            square=True, linewidths=1, cbar_kws={"shrink": .5}, annot=True, fmt='.2f', cmap='Spectral')
sns.despine()
g.figure.set_size_inches(30,25)
    
plt.show()


# In[ ]:





# In[ ]:





# In[27]:


x = data.drop(["has_health_insur"],axis=1,inplace = True)


# In[94]:


str_cols = data.select_dtypes(include = 'object').columns
data[str_cols].head()


# In[28]:


for col in data.columns:
    if data[col].isnull().sum() and data[col].dtypes != 'object':
        data[col].loc[(data[col].isnull())] = data[col].median()
for col in data.columns:
    if data[col].isnull().sum() and data[col].dtypes == 'object':
        data[col].loc[(data[col].isnull())] = data[col].mode().max()


# In[29]:


data.isnull().sum()


# In[30]:


mode1=data["age_bracket"].mode()
data["agw_bracket"]=data["age_bracket"].fillna(mode1)


# In[31]:


data=pd.get_dummies(data,columns=["sex"])


# In[32]:


data.dtypes


# In[33]:


data=pd.get_dummies(data,columns=["income_level","race","marital_status","housing_status","employment","census_msa","agw_bracket"])


# In[34]:


data.dtypes


# In[43]:


data=pd.get_dummies(data,columns=["age_bracket","qualification"])


# In[44]:


data.dtypes


# In[45]:


from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier,RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier


from sklearn.metrics import classification_report,accuracy_score,confusion_matrix
from sklearn.metrics import f1_score
from sklearn.model_selection import cross_val_score

#data processing function
from sklearn.model_selection import train_test_split


# In[46]:


X=data.drop(["unique_id","h1n1_vaccine","chronic_medic_condition","cont_child_undr_6_mnths"],axis=1)
y=data["h1n1_vaccine"]


# In[47]:


X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=0)


# In[ ]:





# In[48]:


Abc = AdaBoostClassifier()
Abc.fit(X_train,y_train)


# In[49]:


Abc.score(X_train,y_train)


# In[50]:


Abc.score(X_test,y_test)


# In[51]:


Gbc = GradientBoostingClassifier()
Gbc.fit(X_train,y_train)


# In[52]:


Gbc.score(X_train,y_train)


# In[53]:


Gbc.score(X_test,y_test)


# In[54]:


dtc = DecisionTreeClassifier(criterion = 'gini',splitter='best',max_depth=7)
dtc.fit(X_train,y_train)


# In[55]:


dtc.score(X_train,y_train)


# In[56]:


dtc.score(X_test,y_test)


# In[57]:


rfc = RandomForestClassifier(n_estimators =100,max_depth = 7)
rfc.fit(X_train,y_train)


# In[58]:


rfc.score(X_train,y_train)


# In[59]:


rfc.score(X_test,y_test)


# In[60]:


lr=LogisticRegression()
lr.fit(X_train,y_train)


# In[61]:


lr.score(X_train,y_train)


# In[63]:


lr.score(X_test,y_test)


# In[64]:


from sklearn import metrics


# In[67]:


predictions = rfc.predict(X_test) #randomforestclassifier


# In[68]:


cm=metrics.confusion_matrix(y_test,predictions,labels=[1,0])

df_cm=pd.DataFrame(cm,index=[i for i in ["1","0"]],
                  columns=[i for  i in ["Predict 1","Predict 0"]])
sns.heatmap(df_cm,annot=True,fmt='g')


# In[69]:


print(metrics.classification_report(y_test,predictions))


# In[70]:


predictions1 = dtc.predict(X_test) #decisintreeclassifier


# In[71]:


cm=metrics.confusion_matrix(y_test,predictions1,labels=[1,0])

df_cm=pd.DataFrame(cm,index=[i for i in ["1","0"]],
                  columns=[i for  i in ["Predict 1","Predict 0"]])
sns.heatmap(df_cm,annot=True,fmt='g')


# In[72]:


print(metrics.classification_report(y_test,predictions1))


# In[73]:


predictions2 = lr.predict(X_test)#logistic regression


# In[74]:


cm=metrics.confusion_matrix(y_test,predictions2,labels=[1,0])

df_cm=pd.DataFrame(cm,index=[i for i in ["1","0"]],
                  columns=[i for  i in ["Predict 1","Predict 0"]])
sns.heatmap(df_cm,annot=True,fmt='g')


# In[75]:


print(metrics.classification_report(y_test,predictions2))


# In[76]:


predictions3 = Abc.predict(X_test) #adaboostclassifier


# In[77]:


cm=metrics.confusion_matrix(y_test,predictions3,labels=[1,0])

df_cm=pd.DataFrame(cm,index=[i for i in ["1","0"]],
                  columns=[i for  i in ["Predict 1","Predict 0"]])
sns.heatmap(df_cm,annot=True,fmt='g')


# In[78]:


print(metrics.classification_report(y_test,predictions3))


# In[79]:


predictions4 = Gbc.predict(X_test)#Gradient boosting classsifier


# In[80]:


cm=metrics.confusion_matrix(y_test,predictions4,labels=[1,0])

df_cm=pd.DataFrame(cm,index=[i for i in ["1","0"]],
                  columns=[i for  i in ["Predict 1","Predict 0"]])
sns.heatmap(df_cm,annot=True,fmt='g')


# In[81]:


print(metrics.classification_report(y_test,predictions4))


# In[ ]:




