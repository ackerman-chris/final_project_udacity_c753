

import pandas as pd
import sys
import numpy as np
import pickle
sys.path.append("../tools/")
from tester import dump_classifier_and_data
from feature_format import featureFormat, targetFeatureSplit
from tester import test_classifier




### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "rb") as data_file:
    data_dict = pickle.load(data_file)



### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
features_list = ['poi', 'salary', 'deferral_payments', 'total_payments', 'loan_advances',
                 'bonus', 'restricted_stock_deferred', 'deferred_income',
                 'total_stock_value', 'expenses', 'exercised_stock_options',
                 'other', 'long_term_incentive', 'restricted_stock', 'director_fees',
                 'to_messages', 'from_poi_to_this_person', 'from_messages',
                 'from_this_person_to_poi', 'shared_receipt_with_poi']

### Store to my_dataset for easy export below.
my_dataset = data_dict

### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys = True)

### converting the data to df for some analysis and exploration
my_data = pd.DataFrame(data, columns=features_list)
my_data.head()



### Task 2: Remove outliers

my_data[my_data.salary == max(my_data.salary)]

outlier_index = my_data[my_data.salary == max(my_data.salary)].index

my_data_selected = my_data.drop(index=outlier_index, axis=1).reset_index(drop=True)

### Task 3: Create new feature(s)
my_data_selected['pct_amount_paid_from_bonus'] = my_data_selected['bonus'] / my_data_selected['total_payments']
my_data_selected.fillna(0, inplace=True)



print("""#### best model found in the experiments
AdaBoostClassifier()
""")

from sklearn.ensemble import AdaBoostClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score

scaller = MinMaxScaler()

best_model = AdaBoostClassifier(n_estimators=200, random_state=88)

# """#### Model training and evaluation"""

### testing the best model with tester code
### splitting the selected data into X(features-independent vars) and y (labels-dependent vars)
labels, features = targetFeatureSplit(my_data_selected.values)

### ### scalling the data
features_scalled  = scaller.fit_transform(features)


### split the data in train and test
X_train, X_test, y_train, y_test = train_test_split(features_scalled, labels, test_size=0.4, random_state=10)

### train the model 
best_model.fit(X_train, y_train)

### evaluate the model on testing data
### accuracy
preds = best_model.predict(X_test)

print('[INFO] Model Performance without tunning and Cross Validation')
print('-----------------------')
acc = best_model.score(X_test, y_test)
print('Accuracy=', acc)
prec = precision_score(y_test, preds)
print('Precision=', prec)
recall = recall_score(y_test, preds)
print('Recall=', recall)
print('-----------------------')
print()


print('[INFO] Model Performance without tunning But with Cross Validation')
print('[INFO] Performing 100 fold. Please wait')
precision, recall, accuracy = test_classifier(best_model, my_data, folds=100)
print('-----------------------')
print(f'Precision: {precision}')
print(f'Recall: {recall}')
print(f'Accuracy: {accuracy}')
print('-----------------------')
print()

print('[INFO] Tunnning The model....... Please Wait')
# """### Tuning using the gridsearch cv"""

from sklearn.model_selection import GridSearchCV

best_model = AdaBoostClassifier(random_state=8895)

params = {
    'base_estimator': [None], 
    'learning_rate' : [0.1, 1.0, 0.01],
    'n_estimators': [20, 50, 100],
    'algorithm': ['SAMME', 'SAMME.R']
    
}

grid_cv = GridSearchCV(best_model, params, scoring='f1', verbose=1)

grid_cv.fit(features_scalled, labels)

best_model = AdaBoostClassifier(algorithm='SAMME.R', base_estimator=None, learning_rate=1.0,
                   n_estimators=50, random_state=8895)



## evaluating tuned model with tester script
print()
print('[INFO] Model Performance after tunning and with Cross Validation')
print('[INFO] Performing 100 fold. Please wait')
precision, recall, accuracy = test_classifier(best_model, my_data, folds=100)
print('-----------------------')
print(f'Precision: {precision}')
print(f'Recall: {recall}')
print(f'Accuracy: {accuracy}')
print('-----------------------')
print()
print('Saving the model.')
#### saving the classifier
dump_classifier_and_data(best_model, my_dataset, features_list)
print('Saved.')

# ### loading the saved model
# loaded_model = pickle.load(open('my_classifier.pkl', 'rb'))

# ## evaluating loaded model with tester script
# precision, recall, accuracy = test_classifier(loaded_model, my_data, folds=500)
# print(f'Precision: {precision}')
# print(f'Recall: {recall}')
# print(f'Accuracy: {accuracy}')
# print()

