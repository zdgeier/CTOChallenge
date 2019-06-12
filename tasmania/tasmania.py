# Load libraries
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from pandas.plotting import scatter_matrix
from sklearn import model_selection
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

# Read in data from csv
training_dataset_path = ".\\contest-train.csv"
target_dataset_path = ".\\contest-test.csv"

train_base = pd.read_csv(training_dataset_path, encoding='Latin-1', index_col=0)
target_base = pd.read_csv(target_dataset_path, encoding='Latin-1', index_col=0)

# Drop unrelated columns
cols_to_drop = ['ID', 'Course Name', 'Course URL / Code', 'Certification URL','Consolidated Course Name', 'Assigned To', 'Request Status', 'Start Date', 'End Date', 'Start Mo/Yr', 'End Mo/Yr', 'Start FY', 'End FY', 'Individual Travel Hours', 'Rqst Tot Labor Hrs', 'Airfare', 'Hotel', 'Per Diem', 'Other', 'Estimated Individual Travel', 'Misc Expenses', 'Catering', 'Facility Rental', 'Direct Other Expenses', 'Describe Other Expenses', 'Direct Expense Impact', 'Rqst NPR Alloc', 'Rqst NPR OH', 'Cancel No Response', 'Created', 'Retroactive Start Date', 'Duplicates', 'Reporting Status']
categorical_columns = ['Training Source', 'Home Office/Metro Area', 'Organization Number', 'Organization', 'Capability', 'Function 2', 'Career Level', 'Function', 'Function Name', 'Title','Training Type', 'Training Provider', 'Training Delivery Type', 'Training Location', 'Vendor Name', 'Conference Name', 'Course or Event Name', 'Certification Type', 'Certification Name', 'Is there a course with this certification?', 'Activity', 'Support Group', 'Business Justification', 'What % of the conference is business development?', 'Travel Required']

combined_base = pd.concat([train_base, target_base], axis=0).drop(cols_to_drop, axis=1)
combined_dummies = pd.get_dummies(data = combined_base, columns = categorical_columns, drop_first = True)

# Get rows with the category column filled out (training set)
combined_with_category = combined_dummies.dropna(subset=['Category'])
X = combined_with_category.drop(columns='Category')
Y = combined_with_category['Category'].copy()

train, test, train_labels, test_labels =  train_test_split(X, Y, test_size = .02)

# I got the best results with LR
lr = LogisticRegression(solver='liblinear', multi_class='auto')
lr_model = lr.fit(train, train_labels)
lr_preds = lr.predict(test)
print ("Logistic Regression Accuracy: ", accuracy_score(test_labels, lr_preds))

# Test out the model on the target dataset
target_test = combined_dummies[combined_dummies.Category.isnull()]
target_test = target_test.drop(['Category'], axis = 1)

predictions_test = lr.predict(target_test)

for i, prediction in enumerate(predictions_test):
    target_base.loc[i, 'Category'] = prediction

target_base.to_csv('results.csv')
