#!/usr/bin/env python
# coding: utf-8

# In[2]:


#!git clone https://github.com/hussnain-imtiaz/POI_Analysis.git
#!mv /content/POI_Analysis/final_project /content
#!mv /content/POI_Analysis/tools /content


# # Installing all the required tools:
# Any missing tool(s) can be installed by using pip command. 
# 

# In[ ]:


#!pip install pycaret
#!pip install numpy
#!pip install pandas matplotlib seaborn sklearn


# # 2. Getting the Tools Ready:

# In[1]:


import pandas as pd
import sys
import numpy as np
import pickle
from pycaret.classification import * 
import seaborn as sns
import matplotlib.pyplot as plt

from final_project.tester import dump_classifier_and_data
from tools.feature_format import featureFormat, targetFeatureSplit


# Provided to give you a starting point. Try a variety of classifiers.
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from final_project.tester import test_classifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier


# # 3. Loading the data:

# In[2]:


### Load the dictionary containing the dataset
with open("/content/final_project/final_project_dataset.pkl", "rb") as data_file:
    data_dict = pickle.load(data_file)


# # 4. Getting the data Ready:

# In the intial run we will take all the available features and then based on analysis report we will select the best ones. 

# In[3]:


### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
features_list = ['poi', 'salary', 'deferral_payments', 'total_payments', 'loan_advances',
                 'bonus', 'restricted_stock_deferred', 'deferred_income',
                 'total_stock_value', 'expenses', 'exercised_stock_options',
                 'other', 'long_term_incentive', 'restricted_stock', 'director_fees',
                 'to_messages', 'from_poi_to_this_person', 'from_messages',
                 'from_this_person_to_poi', 'shared_receipt_with_poi'] 


# In[4]:


### Store to my_dataset for easy export below.
my_dataset = data_dict

### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys = True)

### converting the data to df for some analysis and exploration
my_data = pd.DataFrame(data, columns=features_list)
my_data.head()


# ### i. Exploratory Data Analysis

# In[5]:


# inspect info
my_data.info()


# **We see no missing values because they were imputed with '0'. But it doesnt mean that there is no missing values. We will find their % later.**

# #### (A). Total Number of data points and their distribution across different classes (poi/non-poi)

# In[6]:


### how many data points we have?
len(my_data)


# In[7]:


### distribution across different classes
plt.title('Data Points(Obs) per class')
plt.xlabel('Class/Label')
plt.ylabel('Count')
my_data['poi'].value_counts().plot.bar()
plt.xticks([0, 1], labels=['Non-POI', 'POI'], rotation=45)
plt.show()


# #### (B). Looking up for the missing values in availble features and filtereing the ones with less than 50% missing values. 

# In[8]:


### find how many values are missing in each column
filtered_features_missing = []
missing_count = []
for col in features_list[1:]:
  missing_vals = len(my_data[col][np.where(my_data[col]==0)[0]])
  missing_count.append(missing_vals)
  total_vals = len(my_data[col])
  ptc_missing = missing_vals / total_vals
  if ptc_missing < 0.5:
    filtered_features_missing.append(col)
  print(f'{col} = {ptc_missing:.2f}%')


# In[9]:


### Visualizing the missing values (by count)
plt.figure(figsize=(10,5))
sns.barplot(x=features_list[1:], y=missing_count)
plt.xticks(rotation=90)
plt.title('Missing Values per feature out of 140.')
plt.xlabel('Feature')
plt.ylabel('Missing value count')
plt.show()


# In[10]:


### some filtered features those are having less than 50% missing values
filtered_features_missing


# #### (C). Finding the correlation between dependent and independant variable

# In[11]:


my_data.corr().iloc[0]


# In[12]:


### find the correlation in all_features
plt.figure(figsize=(10, 10))
sns.heatmap(my_data.corr());


# **There is no super corelation.**

# #### (D). Is the Data Balanced?

# In[13]:


pd.DataFrame(data_dict).T['poi'].value_counts()


# **Data is very much imbalanced and more biased to class 0.**

# ### ii. Data Preparation
# We will prepare the data for better analysis and modeling. We will
# 1. Remove the outliers
# 2. Create new feature(s)
# 3. Select higly correlated features

# #### 1. Outliers detection and removal

# **Inspecting with box plot**

# In[14]:


### Task 2: Remove outliers

for i in range(1, len(features_list)):
  plt.figure(figsize = (10,20))
  plt.subplot(10, 2, i)
  sns.boxplot(data=my_data[features_list[i]])
  plt.ticklabel_format(style='plain', axis='y')
  plt.xlabel(features_list[i])
  plt.tight_layout()


# **Using IQR**

# In[15]:


q1 = my_data.quantile(0.25)
q3 = my_data.quantile(0.75)
iqr = q3 - q1


# In[16]:


iqr


# In[17]:


(my_data < (q1 - 1.5 * iqr)) | (my_data > (q3 + 1.5 * iqr))


# `True` in the data frame above represents the presence of outliers.

# In[18]:


((my_data > (q3 + 1.5 * iqr)) | (my_data < (q1 - 1.5 * iqr))).any(axis=1)


# In[19]:


### removal of oultliers based on iqr
my_data_outliers_rm = my_data[((my_data > (q3 + 1.5 * iqr)) | (my_data < (q1 - 1.5 * iqr))).any(axis=1)]


# In[20]:


my_data_outliers_rm.reset_index(inplace=True, drop=True)


# In[21]:


my_data_outliers_rm


# In[22]:


### Task 3: Create new feature(s)
my_data_outliers_rm['pct_amount_paid_from_bonus'] = my_data_outliers_rm['bonus'] / my_data_outliers_rm['total_payments']
my_data_outliers_rm.fillna(0, inplace=True)


# #### 3. Finding and selecting more correlated features

# In[54]:


### selecting only highly correlated featues
highly_corr_feats = []
for label, value in my_data_outliers_rm.corr().iloc[0, 1:].items():
  if abs(value) >= 0.01:
    print(label, value)
    highly_corr_feats.append(label)

highly_corr_feats.insert(0, 'poi')
highly_corr_feats


# In[55]:


my_data_selected = my_data_outliers_rm[highly_corr_feats]
my_data_selected


# In[56]:


## corelation in filtered features
plt.figure(figsize=(10, 10))
sns.heatmap(my_data_selected.corr());


# # Modeling experiments
# 1. Take the selected features from data as and find the best/highly accurate model.

# In[57]:


my_data_selected


# In[58]:


experiment = setup(my_data_selected, target='poi')  
best_model = compare_models(sort='F1')


# In[59]:


best_model


# #### best model found in first experiment
# AdaBoostClassifier(algorithm='SAMME.R', base_estimator=None, learning_rate=1.0,
#                    n_estimators=50, random_state=8895)

# In[60]:


from sklearn.ensemble import AdaBoostClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score


# In[67]:


scaller = MinMaxScaler()

best_model = AdaBoostClassifier(algorithm='SAMME.R', base_estimator=None, learning_rate=1.0,
                   n_estimators=50, random_state=8895)


# #### Model training and evaluation

# In[68]:


### testing the best model with tester code
### splitting the selected data into X(features-independent vars) and y (labels-dependent vars)
labels, features = targetFeatureSplit(my_data_selected.values)

### ### scalling the data
features_scalled  = scaller.fit_transform(features)


### split the data in train and test
X_train, X_test, y_train, y_test = train_test_split(features_scalled, labels, test_size=0.2, random_state=42)


# In[69]:


### train the model 
best_model.fit(X_train, y_train)


# In[70]:


### evaluate the model on testing data
### accuracy
preds = best_model.predict(X_test)

acc = best_model.score(X_test, y_test)
print('Accuracy=', acc)

prec = precision_score(y_test, preds)
print('Precision=', prec)

recall = recall_score(y_test, preds)
print('Recall=', recall)


# In[71]:


my_data_selected.columns


# In[72]:


### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html


precision, recall, accuracy = test_classifier(best_model, my_data_selected, folds=500)
print(f'Precision: {precision}')
print(f'Recall: {recall}')
print(f'Accuracy: {accuracy}')
print()


# In[ ]:


#### saving the classifier
dump_classifier_and_data(best_model, my_dataset, features_list)


# In[ ]:




