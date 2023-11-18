#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
filename = 'pima-indians-diabetes.csv'
col_names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age',
'class']
data = pd.read_csv(filename, names = col_names)
print("File loaded successfully")


# In[2]:


print(data.head(20))                       


# In[3]:


print(data.shape)                    #shape method                                          


# In[5]:


types=data.dtypes
print(types)                          #data types of attributes in data


# In[6]:


description=data.describe()
print(description)                      # describe method


# In[8]:


class_counts=data.groupby('class').size()
print(class_counts)                                                  #groupby method


# In[11]:


correlations=data.corr(method='pearson')                            # finding correalations between attributes
print(correlations)


# In[12]:


skew=data.skew()
print(skew)                                        #Skew of Univariate Distributions


# # Data using visuliazition

# # Histogram

# In[13]:


from matplotlib import pyplot
data.hist()
pyplot.show


# data.plot(
# kind = 'box',
# subplots = True,
# layout = (3,3),
# sharex = False,
# sharey = False
# )
# pyplot.show()

# # Box & Whisker plot

# In[23]:


data.plot(
kind = 'box',
subplots = True,
layout = (3,3),
sharex = False,
sharey = False
)
pyplot.show()


# # Multivariable plots
# 
# 

# In[ ]:


import numpy


# In[24]:


correlations = data.corr()
fig = pyplot.figure()
ax = fig.add_subplot(111)
cax = ax.matshow(correlations, vmin=-1, vmax=1)
fig.colorbar(cax)
ticks = numpy.arange(0,9,1)
ax.set_xticks(ticks)
ax.set_yticks(ticks)
ax.set_xticklabels(col_names)
ax.set_yticklabels(col_names)
pyplot.show()


# # Scatter Plot Matrix

# In[26]:


from pandas.plotting import scatter_matrix
scatter_matrix(data)
pyplot.show()


# # Rescale the data 

# In[35]:


import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from numpy import set_printoptions

filename = 'pima-indians-diabetes.csv'
col_names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age',
'class']

dataframe = pd.read_csv(filename, names = col_names)

array = dataframe.values
X = array[:,0:8]
Y = array[:,8]
scaler = MinMaxScaler(feature_range = (0, 1))
rescaledX = scaler.fit_transform(X)
set_printoptions(precision = 3)
print(rescaledX[0:5,:])


# In[36]:


from sklearn.preprocessing import StandardScaler
from numpy import set_printoptions
dataframe = pd.read_csv(filename, names = col_names)
array = dataframe.values
X = array[:,0:8]
Y = array[:,8]
scaler = StandardScaler().fit(X)
rescaledX = scaler.transform(X)
set_printoptions(precision = 3)
print(rescaledX[0:5,:])


# In[37]:


from sklearn.preprocessing import Normalizer
dataframe = pd.read_csv(filename, names = col_names)
array = dataframe.values
X = array[:,0:8]
Y = array[:,8]
scaler = Normalizer().fit(X)
normalizedX = scaler.transform(X)
set_printoptions(precision=3)
print(normalizedX[0:5,:])


# # Binarize data

# In[38]:


from sklearn.preprocessing import Binarizer
from numpy import set_printoptions
dataframe = pd.read_csv(filename, names = col_names)
array = dataframe.values
                                                               # separate array into input and output components
X = array[:, 0:8]
Y = array[:, 8]
binarizer = Binarizer(threshold = 0.0).fit(X)
binaryX = binarizer.transform(X)
                                                              # summarize transformed data
set_printoptions(precision=3)
print(binaryX[0:5,:])


# # Univariable selection

# In[39]:


from numpy import set_printoptions
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
dataframe = pd.read_csv(filename, names = col_names)
array = dataframe.values
X = array[:,0:8]
Y = array[:,8]
test = SelectKBest(score_func = chi2, k = 4)
fit = test.fit(X, Y)
                                                                  # summarize scores
set_printoptions(precision = 3)
print(fit.scores_)
features = fit.transform(X)
                                                                # summarize selected features
print(features[0:5,:])


# In[41]:


from sklearn.ensemble import ExtraTreesClassifier
dataframe = pd.read_csv(filename, names = col_names)
array = dataframe.values
X = array[:, 0:8]
Y = array[:, 8]
model = ExtraTreesClassifier()
model.fit(X, Y)
print(model.feature_importances_)


# # Training and Testing Datasert 

# In[42]:


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
dataframe = pd.read_csv(filename, names = col_names)
array = dataframe.values
X = array[:, 0:8]
Y = array[:, 8]
test_size = 0.33
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size =
test_size)
model = LogisticRegression(solver = 'lbfgs', max_iter = 3000)
model.fit(X_train, Y_train)
result = model.score(X_test, Y_test)
print("Accuracy:", result*100.0)


# #  K fold cross validation

# In[43]:


from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression

dataframe = pd.read_csv(filename, names = col_names)
array = dataframe.values
X = array[:, 0:8]
Y = array[:, 8]
num_folds = 10
kfold = KFold(n_splits = num_folds)
model = LogisticRegression(solver = 'lbfgs', max_iter = 3000)
results = cross_val_score(model, X, Y, cv = kfold)
print("Accuracy:", results.mean()*100.0)


# In[45]:


from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression

dataframe = pd.read_csv(filename, names = col_names)
array = dataframe.values
X = array[:, 0:8]
Y = array[:, 8]
kfold = KFold(n_splits = 10)
model = LogisticRegression(solver = 'lbfgs', max_iter = 3000)
scoring = 'accuracy'
results = cross_val_score(model, X, Y, cv = kfold, scoring = scoring)
print("Accuracy: ", results.mean())


# In[46]:


dataframe = pd.read_csv(filename, names = col_names)
array = dataframe.values
X = array[:, 0:8]
Y = array[:, 8]
kfold = KFold(n_splits = 10)
model = LogisticRegression(solver = 'lbfgs', max_iter = 3000)
scoring = 'roc_auc'
results = cross_val_score(model, X, Y, cv = kfold, scoring = scoring)
print("AUC:", results.mean()*100)


# In[49]:


import pandas as pd
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
filename = 'pima-indians-diabetes.csv'
col_names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age',
'class']
dataframe = pd.read_csv(filename, names = col_names)
array = dataframe.values
X = array[:, 0:8]
Y = array[:, 8]
kfold = KFold(n_splits = 10)
model = LogisticRegression(solver = 'lbfgs', max_iter = 3000)
scoring = 'accuracy'
results = cross_val_score(model, X, Y, cv = kfold, scoring = scoring)
print("Accuracy: ", results.mean())


# In[54]:


import pandas as pd
from matplotlib import pyplot
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
filename = 'pima-indians-diabetes.csv'
col_names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age',
'class']
dataframe = pd.read_csv(filename, names = col_names)
array = dataframe.values
X = array[:, 0:8]
y = array[:, 8]


models = []
models.append(('LR', LogisticRegression(solver = 'lbfgs', max_iter
= 3000)))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))
models.append(('SVM', SVC()))
results = []
names = []
scoring = 'accuracy'
for name, model in models:
    kfold = KFold(n_splits = 10)
    cv_results = cross_val_score( model,X,y,cv = kfold,scoring = scoring)
results.append(cv_results)
names.append(name)
msg = (name, cv_results.mean(),cv_results.std())
print(msg)
fig = pyplot.figure()
fig.suptitle('Algorithm Comparison')
ax = fig.add_subplot(111)
pyplot.boxplot(results)
ax.set_xticklabels(names)
pyplot.show()


# In[ ]:




