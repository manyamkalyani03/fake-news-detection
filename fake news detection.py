#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
import re
import string


# In[ ]:


data_fake = pd.read_csv('Fake.csv')
data_true = pd.read_csv('True.csv')


# In[3]:


data_fake.head()


# In[4]:


data_fake.head()


# In[5]:


data_true.head()


# In[6]:


data_true.tail()


# In[7]:


data_fake["class"]=0
data_true["class"]=1


# In[8]:


data_fake.shape,data_true.shape


# In[9]:


data_fake_manual_testing=data_fake.tail(10)
for i in range(23480,23470,-1):
    data_fake.drop([i],axis=0,inplace=True)

data_true_manual_testing=data_true.tail(10)
for i in range(21416,21406,-1):
    data_true.drop([i],axis=0,inplace=True)


# In[10]:


data_fake.shape,data_true.shape 


# In[11]:


data_fake_manual_testing['class']=0
data_fake_manual_testing['class']=1


# In[12]:


data_fake_manual_testing.head(10)


# In[13]:


data_true_manual_testing.head(10)


# In[14]:


data_merge=pd.concat([data_fake,data_true],axis=0)
data_merge.head(10)


# In[15]:


data_merge.columns


# In[16]:


data=data_merge.drop(['title','subject','date'],axis=1)  


# In[17]:


data.isnull().sum()


# In[18]:


data=data.sample(frac=1)


# In[19]:


data.head()


# In[20]:


data.reset_index(inplace=True)
data.drop(['index'],axis=1,inplace=True)


# In[21]:


def wordopt(text):
    text=text.lower()
    text=re.sub('\[.*?\]','',text)
    text=re.sub("\\W"," ",text)
    text=re.sub('https?://\S+www\.\S+','',text)
    text=re.sub('<.*?>+','',text)
    text=re.sub('[%s]'%re.escape(string.punctuation),'',text)
    text=re.sub('\n','',text)
    text=re.sub('\W\d\W*','',text)
    return text


# In[22]:


data['text']=data['text'].apply(wordopt)


# In[23]:


x=data['text']
y=data['class']


# In[24]:


x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.25)


# In[25]:


from sklearn.feature_extraction.text import TfidfVectorizer

vectorization=TfidfVectorizer()
xv_train=vectorization.fit_transform(x_train)
xv_test=vectorization.transform(x_test)


# In[26]:


from sklearn.linear_model import LogisticRegression

LR=LogisticRegression()
LR.fit(xv_train,y_train)


# In[27]:


pred_lr=LR.predict(xv_test)


# In[28]:


LR.score(xv_test,y_test)


# In[29]:


print(classification_report(y_test,pred_lr))


# In[34]:


from sklearn.tree import DecisionTreeClassifier

DT=DecisionTreeClassifier()
DT.fit(xv_train,y_train)


# In[35]:


pred_dt=DT.predict(xv_test)


# In[36]:


DT.score(xv_test,y_test)


# In[37]:


print(classification_report(y_test,pred_dt))


# In[39]:


from sklearn.ensemble import GradientBoostingClassifier

GB=GradientBoostingClassifier(random_state=0)
GB.fit(xv_train,y_train)


# In[40]:


pred_gb=GB.predict(xv_test)


# In[41]:


GB.score(xv_test,y_test)


# In[42]:


print(classification_report(y_test,pred_gb))


# In[43]:


from sklearn.ensemble import RandomForestClassifier

RF=RandomForestClassifier(random_state=0)
RF.fit(xv_train,y_train)


# In[44]:


pred_rf=RF.predict(xv_test)


# In[45]:


RF.score(xv_test,y_test)


# In[46]:


print(classification_report(y_test,pred_rf))


# In[48]:


def output(n):
    if n==0:
        return"fake news"
    elif n==1:
        return"not fake news"
def testing(news):
    testing_news={"text":[news]}
    new_def_test=pd.DataFrame(testing_news)
    new_def_test["text"]=new_def_test["text"].apply(wordopt)
    new_x_test=new_def_test["text"]
    new_xv_test=vectorization.transform(new_x_test)
    pred_LR=LR.predict(new_xv_test)
    pred_DT=DT.predict(new_xv_test)
    pred_GR=GR.predict(new_xv_test)
    pred_RF=RF.predict(new_xv_test)
    return print("\n\nLR prediction:{} \nDT prediction:{} \nGBC prediction:{} \nRFC prediction:{}". format(output(pred_LR[0]),
                                                                                                           output(pred_GBC[0]),
                                                                                                           output(pred_RFC[0])))


# In[ ]:


news=string(input())
testing(news)


# In[ ]:




