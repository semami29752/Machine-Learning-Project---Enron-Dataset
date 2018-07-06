import sys
import pickle
import pandas as pd
sys.path.append("../tools/")


from feature_format import featureFormat, targetFeatureSplit
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.feature_selection import SelectKBest
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import precision_score, recall_score, make_scorer
from tester import dump_classifier_and_data

### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)

### Convert the dataset to pandas dataframe for easy analysis
enron_df = pd.DataFrame(data_dict.values(), index = data_dict.keys())
enron_df = enron_df.replace(to_replace="NaN", value=0)  #replace NaN values with 0
enron_df = enron_df.loc[(enron_df!=0).any(axis=1)]   #Drop rows with all 0's (1 row was dropped)

### Remove outliers
enron_df = enron_df.drop(["THE TRAVEL AGENCY IN THE PARK", "TOTAL"])    #Drop the non employee rows from the dataset

### Define a function to create a new feature; "bonus to salary ratio"
def ratio(row):
    if row['salary'] == 0:
        return 0
    else:
        return float(row['bonus'])/float(row['salary'])

### Add the new feature to the enron df
enron_df["bonus_salary_ratio"] = enron_df.apply(ratio, axis = 1)

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


###### GradienBoosting Algorithm ########
#classifier = GradientBoostingClassifier()
#parameters = dict(selector__k = [4,5,6],
                    #classifier__n_estimators = [10,20,30,40,50,60],
                    #classifier__min_samples_leaf = [2,4,6,8,10,12],
                    #classifier__min_samples_split = [2,4,6,8])

###### RandomForest Algorithm ########
#classifier = RandomForestClassifier()
#parameters = dict(selector__k = [4,5,6],
                    #classifier__n_estimators = [10,20,30,40,50,60,70,80],
                    #classifier__min_samples_leaf = [2,4,6,8,10,12,14,16,18,20],
                    #classifier__min_samples_split = [2,4,6])

###### Final Classifier ########
classifier = GaussianNB()
parameters = dict(selector__k = [3,4,5,6])

### Build the pipline
selector = SelectKBest()
steps = [('selector', selector), ('classifier', classifier)]
pipeline = Pipeline(steps)

### Define the evalution metrics
scorers = {'precision_score': make_scorer(precision_score), 'recall_score': make_scorer(recall_score)}

### Use GridSearchCV to tune the parameters
stratified = StratifiedShuffleSplit(n_splits = 500, random_state = 42)
grid = GridSearchCV(pipeline, parameters, cv = stratified, scoring = scorers, refit = 'recall_score')
grid.fit(features, labels)
print grid.best_params_
clf = grid.best_estimator_
print "-------"
print "MEAN RECALL SCORES IN ORDER: ", (sorted(grid.cv_results_['mean_test_recall_score'], reverse = True))
print "MEAN PRECISION SCORES IN ORDER: ", (sorted(grid.cv_results_['mean_test_precision_score'], reverse = True))


### Dump your classifier, dataset, and features_list
dump_classifier_and_data(clf, my_dataset, features_list)
