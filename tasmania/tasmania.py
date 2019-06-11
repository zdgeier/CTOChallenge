# Load libraries
import pandas as pd
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
import seaborn as sns

# Load dataset
train_path = ".\contest-train.csv"
out_path = ".\contest-test.csv"

train_base = pd.read_csv(train_path, encoding='Latin-1')
full_base = pd.read_csv(out_path, encoding='Latin-1')


cols_to_drop = ['ID', 'Course Name', 'Course URL / Code', 'Certification URL','Consolidated Course Name', 'Assigned To', 'Function 2', 'Request Status', 'Start Date', 'End Date', 'Start Mo/Yr', 'End Mo/Yr', 'Start FY', 'End FY', 'Individual Travel Hours', 'Rqst Tot Labor Hrs', 'Airfare', 'Hotel', 'Per Diem', 'Other', 'Estimated Individual Travel', 'Misc Expenses', 'Catering', 'Facility Rental', 'Direct Other Expenses', 'Describe Other Expenses', 'Direct Expense Impact', 'Rqst NPR Alloc', 'Rqst NPR OH', 'Cancel No Response', 'Created', 'Retroactive Start Date', 'Duplicates', 'Reporting Status']
categorical_columns = ['Training Source', 'Home Office/Metro Area', 'Organization Number', 'Organization', 'Capability', 'Career Level', 'Function', 'Function Name', 'Title','Training Type', 'Training Provider', 'Training Delivery Type', 'Training Location', 'Vendor Name', 'Conference Name', 'Course or Event Name', 'Certification Type', 'Certification Name', 'Is there a course with this certification?', 'Activity', 'Support Group', 'Business Justification', 'What % of the conference is business development?', 'Travel Required']

train_base = train_base.drop(cols_to_drop, axis=1)
out_base = full_base.drop(cols_to_drop, axis=1)

full_set = pd.concat([train_base,out_base], axis=0)
x_full_preprocessed = pd.get_dummies(data = full_set, columns = categorical_columns, drop_first = True)

#remove rows where Category is NaN
x_train_fin = x_full_preprocessed.dropna(subset=['Category'])
#output portion should be records where Category is NaN
x_out_fin = x_full_preprocessed[x_full_preprocessed.Category.isnull()]
x_out_fin = x_out_fin.drop(['Category'], axis = 1)

#create a map for Category values
categories = x_train_fin['Category'].unique()
category_map = dict()
i = 0
for cat in categories:
    category_map [cat] = i
    i  = i+1

#split the training portion - setting very low for final pass, use .2 for reasonable accuracy test
train, test =  train_test_split(x_train_fin, test_size = .02)

#separate Category
x_train = train.drop(['Category'], axis = 1)
y_train = pd.DataFrame(data=train, columns = ['Category'])

x_test = test.drop(['Category'], axis = 1)
y_test = pd.DataFrame(data=test, columns = ['Category'])

#apply the map to Category
y_train_preprocessed = pd.DataFrame(data = y_train['Category'].map(category_map))
y_test_preprocessed = pd.DataFrame(data = y_test['Category'].map(category_map))

# Test options and evaluation metric
seed = 56
C = 8.5
scoring = 'accuracy'


LR = LogisticRegression(C=C)
LR.fit(x_train, y_train_preprocessed.values.ravel())
predictions = LR.predict(x_test)    
print (accuracy_score(y_test_preprocessed.values.ravel(), predictions))

predictions_test = LR.predict(x_out_fin)
# print (category_map)
# print(predictions_test)
# print (predictions_test.shape)

i = 1
for row in predictions_test:
    cat_val = list(category_map.keys())[list(category_map.values()).index(row)]
    full_base.ix[i,'Category'] = cat_val
    i = i+1
    
full_base.to_csv('results.csv')  
