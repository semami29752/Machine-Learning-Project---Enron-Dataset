'''
    the code for graphical analysis of the features and
    returning the top k features based on their F-score.
'''
import sys
import pickle

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt



from feature_format import featureFormat, targetFeatureSplit
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.feature_selection import SelectKBest
from operator import itemgetter

### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)

#Convert the dataset to pandas dataframe for easy analysis
enron_df = pd.DataFrame(data_dict.values(), index = data_dict.keys())
enron_df = enron_df.replace(to_replace="NaN", value=0)  #replace NaN values with 0
enron_df = enron_df.loc[(enron_df!=0).any(axis=1)]   #Drop rows with all 0's (1 row was dropped)

### Remove outliers
enron_df = enron_df.drop(["THE TRAVEL AGENCY IN THE PARK", "TOTAL"])    #Drop the non employee rows from the dataset


### Define a function to create a new feature, bonus to salary ratio
def ratio(row):
    if row['salary'] == 0 or row['bonus'] == 0:
        return 0
    else:
        return float(row['bonus'])/float(row['salary'])

### Add a new feature to the enron df
enron_df["bonus_salary_ratio"] = enron_df.apply(ratio, axis = 1)


### Graphical Analysis of the new added feature
### Enter the target variables in var1 or var2
var1 = "bonus_salary_ratio"
var2 = "bonus"
plt.scatter(enron_df[var1], enron_df[var2], c = enron_df["poi"], s=15)
plt.tight_layout()
plt.xlabel(var1)
plt.ylabel(var2)
plt.show()
sns.boxplot(x=enron_df["poi"], y =  enron_df[var1])
plt.show()


### Convert the updated dataframe to dictionary
my_dataset = enron_df.to_dict('index')

features_list = ['poi','salary' , 'deferral_payments', 'total_payments', 'loan_advances',
                 'bonus', 'restricted_stock_deferred', 'deferred_income', 'total_stock_value',
                 'expenses', 'exercised_stock_options', 'other', 'long_term_incentive',
                 'restricted_stock', 'director_fees', 'to_messages', 'from_poi_to_this_person',
                 'from_messages', 'from_this_person_to_poi', 'shared_receipt_with_poi', 'bonus_salary_ratio']


### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)


def return_top_features(n_splits, k):

    '''
    Return the F-scores for the top k features in the dataset
    First argument is the number of folds in StratifiedShuffleSplit object
    Second argument is the number of top features selected
    '''
    score = []
    sss = StratifiedShuffleSplit(n_splits = n_splits, random_state = 42)
    for train_indices, test_indices in sss.split(features, labels):
        top_feature = []
        features_train = [features[i] for i in train_indices]
        features_test =  [features[i] for i in test_indices]
        labels_train =   [labels[i] for i in train_indices]
        labels_test =    [labels[i] for i in test_indices]
        selector = SelectKBest(k=k)
        selector.fit(features_train, labels_train)
        score.append(selector.scores_)
        score_array = np.array(score)
        average_scores = score_array.mean(axis = 0)
        top_indices = selector.get_support(indices = True)
        for i in top_indices:
            top_feature.append((features_list[1:][i], selector.scores_[i]))
        top_feature = sorted(top_feature, key = itemgetter(1), reverse = True)
        print top_feature
    print "---------------------------"
    print "TOP FEATURES' AVERAGE SCORES:"
    print "---------------------------"
    for i in np.argsort(average_scores)[:len(average_scores)-(k+1):-1]:
        print features_list[1:][i], average_scores[i]

return_top_features(10,5)
