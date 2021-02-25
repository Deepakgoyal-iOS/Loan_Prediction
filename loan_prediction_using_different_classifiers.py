
# coding: utf-8

# In[55]:


import pandas as pd
data=pd.read_csv("https://s3-api.us-geo.objectstorage.softlayer.net/cf-courses-data/CognitiveClass/ML0101ENv3/labs/loan_train.csv")
data.head()


# In[56]:


data.drop(['Unnamed: 0','Unnamed: 0.1','effective_date','due_date'],axis=1,inplace=True)
data.head()


# In[19]:


data.dtypes


# In[11]:


data.isnull().sum()


# In[57]:


from sklearn.preprocessing import LabelEncoder
label_encoder=LabelEncoder()
data.iloc[:,5:]=label_encoder.fit_transform(data.iloc[:,5:])
data.head()


# In[58]:


data.iloc[:,2]=label_encoder.fit_transform(data.iloc[:,2])
data.iloc[:,1]=label_encoder.fit_transform(data.iloc[:,1])
data.iloc[:,4]=label_encoder.fit_transform(data.iloc[:,4])
data.iloc[:,0]=label_encoder.fit_transform(data.iloc[:,0])
data.head()


# In[59]:


data_corr=data.corr()
import seaborn as sns
#sns.heatmap(data_corr,Xticklabels=False,yticklabels=false,annot=True,cmap='Blues')
sns.heatmap(data_corr,annot=True)


# In[61]:


sns.boxplot(y='age',x='loan_status',data=data)


# In[65]:


max_age=data[data['loan_status']==1]
max_age=max_age['age'].max()
max_age


# In[66]:


index_of_age=data[data['age']==51].index
data.drop(index_of_age,axis=0,inplace=True)


# In[67]:


sns.boxplot(y='age',x='loan_status',data=data)


# In[69]:


index_of_age=data[data['age']==50].index
data.drop(index_of_age,axis=0,inplace=True)
sns.boxplot(y='age',x='loan_status',data=data)


# In[73]:


from sklearn.model_selection import train_test_split
features=data.iloc[:,1:]
target=data.iloc[:,0]
X_train,X_test,Y_train,Y_test=train_test_split(features,target,test_size=0.3,random_state=0)
X_train.head()


# In[76]:


from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
X_train=sc.fit_transform(X_train)
X_test=sc.fit_transform(X_test)
X_train=pd.DataFrame(X_train)
X_test=pd.DataFrame(X_test)
X_train.head()


# In[78]:


#Model Selection
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
neigh=KNeighborsClassifier(n_neighbors=3)
neigh=neigh.fit(X_train,Y_train)
ypredict=neigh.predict(X_test)
accuracy_score(Y_test,ypredict)


# In[85]:


list_of_accuracy=[]
for i in range(1,40):
    neigh=KNeighborsClassifier(n_neighbors=i)
    neigh=neigh.fit(X_train,Y_train)
    ypredict=neigh.predict(X_test)
    e=accuracy_score(Y_test,ypredict)
    list_of_accuracy.append(e)
    
acc_data=pd.DataFrame(list_of_accuracy)


# In[86]:


import matplotlib.pyplot as plt
acc_data.plot(kind="line")


# In[98]:


from sklearn.metrics import jaccard_similarity_score
from sklearn.metrics import f1_score
from sklearn.metrics import log_loss
neigh=KNeighborsClassifier(n_neighbors=16)
neigh=neigh.fit(X_train,Y_train)
ypredict=neigh.predict(X_test)
yprobability=neigh.predict_proba(X_test)
jaccard_index=jaccard_similarity_score(Y_test,ypredict)
F1_score=f1_score(Y_test,ypredict,average='weighted')
logloss=log_loss(Y_test,yprobability)
print(jaccard_index)
print(F1_score)
print(logloss)


# In[108]:


from sklearn.tree import DecisionTreeClassifier
list_of_accuracy.clear()
for i in range(1,9):
    dtc=DecisionTreeClassifier(criterion='entropy',max_depth=i)
    dtc=dtc.fit(X_train,Y_train)
    ypredict=dtc.predict(X_test)
    e=accuracy_score(Y_test,ypredict)
    list_of_accuracy.append(e)
    
acc_data=pd.DataFrame(list_of_accuracy)
import matplotlib.pyplot as plt
acc_data.plot(kind="line")


# In[111]:


dtc=DecisionTreeClassifier(criterion='entropy',max_depth=2)
dtc=dtc.fit(X_train,Y_train)
ypredict=dtc.predict(X_test)
yprobability=dtc.predict_proba(X_test)
jaccard_index=jaccard_similarity_score(Y_test,ypredict)
F1_score=f1_score(Y_test,ypredict,average='weighted')
logloss=log_loss(Y_test,yprobability)
print(jaccard_index)
print(F1_score)
print(logloss)


# In[124]:


from sklearn.svm import SVC
SVM_model=SVC(kernel='rbf')
SVM_model=SVM_model.fit(X_train,Y_train)
ypredict=SVM_model.predict(X_test)
print(accuracy_score(Y_test,ypredict))
SVM_model=SVC(kernel='linear')
SVM_model=SVM_model.fit(X_train,Y_train)
ypredict=SVM_model.predict(X_test)
print(accuracy_score(Y_test,ypredict))
SVM_model=SVC(kernel='sigmoid')
SVM_model=SVM_model.fit(X_train,Y_train)
ypredict=SVM_model.predict(X_test)
print(accuracy_score(Y_test,ypredict))


# In[129]:


SVM_model=SVC(kernel='rbf',probability=True)
SVM_model=SVM_model.fit(X_train,Y_train)
ypredict=SVM_model.predict(X_test)
yprobability=SVM_model.predict_proba(X_test)
jaccard_index=jaccard_similarity_score(Y_test,ypredict)
F1_score=f1_score(Y_test,ypredict,average='weighted')
logloss=log_loss(Y_test,yprobability)
print(jaccard_index)
print(F1_score)
print(logloss)


# In[132]:


from sklearn.linear_model import LogisticRegression
lr=LogisticRegression(C=0.01)
lr=lr.fit(X_train,Y_train)
ypredict=lr.predict(X_test)
yprobability=lr.predict_proba(X_test)
jaccard_index=jaccard_similarity_score(Y_test,ypredict)
F1_score=f1_score(Y_test,ypredict,average='weighted')
logloss=log_loss(Y_test,yprobability)
print(jaccard_index)
print(F1_score)
print(logloss)

