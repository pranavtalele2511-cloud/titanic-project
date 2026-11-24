# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
import joblib
from sklearn.neighbors import KNeighborsClassifier


# In[2]:

titanic=pd.read_csv("titanic.csv")


# In[3]:


titanic


# In[4]:


titanic.info()


# In[5]:


titanic.isnull().sum()


# In[6]:


titanic.describe()


# In[7]:


titanic.duplicated().sum()


# In[8]:


titanic["Age"]=titanic["Age"].fillna(titanic["Age"].mean())


# In[9]:


titanic.drop(["Cabin","PassengerId","Name","Ticket"],axis=1,inplace=True)


# In[ ]:





# In[10]:


titanic.isnull().sum()


# In[11]:


titanic["Survived"].value_counts().plot(kind="bar")


# In[12]:


b=titanic.groupby("Sex")["Survived"].sum().plot(kind="bar")


# In[13]:


titanic_encoded=pd.get_dummies(
titanic,
columns=['Sex','Embarked'],
drop_first=True,
)


# In[14]:


titanic_encoded


# In[15]:


titanic['FamilySize']=titanic['SibSp']+titanic['Parch'] +1


# In[16]:


titanic['IsAlone']=(titanic['FamilySize']==1).astype(int)


# In[17]:


titanic=pd.get_dummies(
titanic,
columns=['Sex','Embarked'],
drop_first=True,
)


# In[18]:


titanic


# In[19]:


x=titanic.drop("Survived",axis=1)


# In[20]:


y=titanic["Survived"]


# In[21]:


x


# In[22]:


y


# In[23]:


x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.20,random_state=42)

scaler=StandardScaler()
#x_train_scaled=scaler.fit_transform(x_train)
#x_test_scaled=scaler.transform(x_test)


# In[24]:


logistic_model=LogisticRegression(max_iter=1000)


# In[25]:


logistic_model.fit(x_train,y_train)


# In[26]:


prediction=logistic_model.predict(x_test)
if prediction[0]==1:
    print("predicted: Survived")
else:
    print("predicted: Did Not Survived")


# In[27]:


knn_model = KNeighborsClassifier(n_neighbors=5)


# In[28]:


knn_model.fit(
x_train,
y_train
)


# In[ ]:


knn_pred = knn_model.predict(x_test)


# In[ ]:


prediction=knn_model.predict(x_test)
if prediction[0]==1:
    print("predicted: Survived")
else:
    print("predicted: Did Not Survived")


# In[ ]:


svm_model = SVC(
kernel='rbf',
random_state=42
)


# In[ ]:


svm_model.fit(
x_train,
y_train
)


# In[ ]:


svm_pred = svm_model.predict(x_test)


# In[ ]:


prediction=svm_model.predict(x_test)
if prediction[0]==1:
    print("predicted: Survived")
else:
    print("predicted: Did Not Survived")


# In[ ]:


joblib.dump({
    logistic_model:" log",
    knn_model: "knn",
    svm_model:"svm"
}, "all_models.pkl")


# In[ ]:




