#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np 
import pandas as pd 
import seaborn as sns 
train = pd.read_csv('C://Users/Dream Walker/Desktop/train.csv')
test = pd.read_csv('C://Users/Dream Walker/Desktop/test.csv')


# In[2]:


train.head()


# In[3]:


train.drop(['Cabin', 'Name', 'Ticket'], axis=1, inplace=True)


# In[4]:


train.isnull().sum()


# In[5]:


train['Age']= train['Age'].fillna(train['Age'].mean())


# In[6]:


train[train['Embarked'].isnull()]


# In[7]:


train['Embarked'].value_counts()


# In[8]:


train.dropna(subset=['Embarked'],axis = 0 ,inplace = True)


# In[9]:


train = pd.get_dummies(data=train, columns=['Embarked'])
train = pd.get_dummies(data=train, columns=['Sex'])


# In[10]:


train.head()


# In[11]:


train['Fare'] = train['Fare'].astype(int, copy=True)
train['Age'] = train['Age'].astype(int, copy=True)


# In[12]:


train.head()


# In[13]:


import matplotlib.pyplot as plt 
plt.hist(train['Age'])
plt.show()


# In[14]:


bins = np.linspace(min(train['Age']), max(train['Age']), 9 )
label = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']
train['New_age']= pd.cut(train['Age'], bins, labels=label, include_lowest=True)
train = pd.get_dummies(data=train, columns=['New_age'])


# In[15]:


train.drop(['Age'], axis=1, inplace=True)


# In[16]:


train.head()


# In[17]:


import matplotlib.pyplot as plt 
plt.hist(train['Fare'])
plt.show()


# In[18]:


bins = np.linspace(min(train['Fare']), max(train['Fare']),  11)
label = ['AB', 'BC', 'CD', 'DE', 'EF', 'FG', 'GH', 'HI', 'IJ', 'JK']
train['New_Fare']= pd.cut(train['Fare'], bins, labels=label, include_lowest=True)
train = pd.get_dummies(data=train, columns=['New_Fare'])
train.drop(['Fare'], axis=1, inplace=True)


# In[19]:


train.set_index('PassengerId',inplace = True)
train.head()


# In[20]:


test.drop(['Cabin', 'Name', 'Ticket'], axis=1, inplace=True)
test['Age']= test['Age'].fillna(test['Age'].mean())
test.dropna(subset=['Embarked'],axis = 0 ,inplace = True)
test = pd.get_dummies(data=test, columns=['Embarked'])
test = pd.get_dummies(data=test, columns=['Sex'])


# In[21]:


test['Fare']= test['Fare'].fillna(test['Fare'].mean())
test['Fare'] = test['Fare'].astype(int, copy=True)
test['Age'] = test['Age'].astype(int, copy=True)
bins = np.linspace(min(test['Age']), max(test['Age']), 9 )
label = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']
test['New_age']= pd.cut(test['Age'], bins, labels=label, include_lowest=True)
test = pd.get_dummies(data=test, columns=['New_age'])
test.drop(['Age'], axis=1, inplace=True)
bins = np.linspace(min(test['Fare']), max(test['Fare']),  11)
label = ['AB', 'BC', 'CD', 'DE', 'EF', 'FG', 'GH', 'HI', 'IJ', 'JK']
test['New_Fare']= pd.cut(test['Fare'], bins, labels=label, include_lowest=True)
test = pd.get_dummies(data=test, columns=['New_Fare'])
test.drop(['Fare'], axis=1, inplace=True)


# In[22]:


test.set_index('PassengerId',inplace = True)
test.head()


# In[25]:


X_train = train.drop(['Survived'], axis=1)
X_train.head()


# In[27]:


Y_train= train['Survived']
Y_train.head()


# In[28]:


from sklearn import ensemble
params = {'n_estimators': 500, 'max_depth': 4, 'min_samples_split': 2,
          'learning_rate': 0.01, 'loss': 'ls'}
model = ensemble.GradientBoostingRegressor(**params)

model.fit(X_train, Y_train)


# In[40]:


Survived_new = model.predict(X_train)
from sklearn.metrics import r2_score
print(r2_score(Y_train, Survived_new))


# In[41]:


from sklearn.svm import SVR
regressor = SVR(kernel='rbf', gamma='scale')
regressor.fit(X_train, Y_train)
Survivedi_new = model.predict(X_train)
from sklearn.metrics import r2_score
print(r2_score(Y_train, Survived_new))


# In[59]:


from sklearn.naive_bayes import GaussianNB
gaussian = GaussianNB()
gaussian.fit(X_train, Y_train)
acc_gaussian = round(gaussian.score(X_train, Y_train) * 100, 2)
acc_gaussian


# In[58]:


from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier
decision_tree = DecisionTreeClassifier()
decision_tree.fit(X_train, Y_train)
acc_decision_tree = round(decision_tree.score(X_train, Y_train) * 100, 2)
acc_decision_tree


# In[57]:


from sklearn.ensemble import RandomForestClassifier
random_forest = RandomForestClassifier(n_estimators=100)
random_forest.fit(X_train, Y_train)
random_forest.score(X_train, Y_train)
acc_random_forest = round(random_forest.score(X_train, Y_train) * 100, 2)
acc_random_forest


# In[61]:


sgd = SGDClassifier()
sgd.fit(X_train, Y_train)
acc_sgd = round(sgd.score(X_train, Y_train) * 100, 2)
acc_sgd


# In[1]:


import numpy as np 
import seaborn as sns 
import pandas as pd 
pp = pd.read_csv('C://Users/Dream Walker/Desktop/house/train.csv')
qq = pd.read_csv('C://Users/Dream Walker/Desktop/house/test.csv')
pp.info()


# In[3]:


from sklearn.preprocessing import Imputer


# In[ ]:




