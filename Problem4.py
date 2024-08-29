
import pandas as pd
from sklearn import svm
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.preprocessing import LabelEncoder

train = pd.read_csv(r"train_resized.csv")
test = pd.read_csv(r"test_resized.csv")

# q1
tr36 = train[(train['label'] == 3) | (train['label'] == 6)]
te36 = test[(test['label'] == 3) | (test['label'] == 6)]
set1 = tr36.copy()
set2 = te36.copy()
set1['label'] = set1['label'].astype('category')
set2['label'] = set2['label'].astype('category')
grid_q1 = {'C': [0.0001, 0.001, 0.1, 1, 10, 100, 1000]}
svm_linear = svm.SVC(kernel='linear')
fit_q1 = GridSearchCV(svm_linear, grid_q1, scoring='accuracy', cv=5)
fit_q1.fit(set1.drop(columns=['label']), set1['label'])
print(fit_q1.best_params_)
yhat_q1 = fit_q1.predict(set2.drop(columns=['label']))
print(confusion_matrix(set2['label'], yhat_q1))
print(accuracy_score(set2['label'], yhat_q1))

# q2
grid_q2 = {'C': [0.1, 1, 5, 10, 15, 20, 100], 'gamma': [0.0001, 0.001, 0.01, 0.1, 1]}
svm_radial = svm.SVC(kernel='rbf')
fit_q2 = GridSearchCV(svm_radial, grid_q2, scoring='accuracy', cv=5)
fit_q2.fit(set1.drop(columns=['label']), set1['label'])
print(fit_q2.best_params_)
yhat_q2 = fit_q2.predict(set2.drop(columns=['label']))
print(confusion_matrix(set2['label'], yhat_q2))
print(accuracy_score(set2['label'], yhat_q2))

# q4
tr1258 = train[train['label'].isin([1, 2, 5, 8])]
te1258 = test[test['label'].isin([1, 2, 5, 8])]
set1 = tr1258.copy()
set2 = te1258.copy()
set1['label'] = set1['label'].astype('category')
set2['label'] = set2['label'].astype('category')
grid_q3 = {'C': [0.001, 0.1, 0.5, 1, 5, 10, 15, 100]}
fit_q3 = GridSearchCV(svm_linear, grid_q3, scoring='accuracy', cv=5)
fit_q3.fit(set1.drop(columns=['label']), set1['label'])
print(fit_q3.best_params_)
yhat_q3 = fit_q3.predict(set2.drop(columns=['label']))
print(confusion_matrix(set2['label'], yhat_q3))
print(accuracy_score(set2['label'], yhat_q3))

# q5
set1 = train.iloc[:1000].copy()
set1['label'] = set1['label'].astype('category')
grid_q4 = {'C': [0.01, 0.1, 1, 5, 10, 15, 20], 'gamma': [0.0001, 0.001, 0.01, 0.1, 1]}
fit_q4 = GridSearchCV(svm_radial, grid_q4, scoring='accuracy', cv=5)
fit_q4.fit(set1.drop(columns=['label']), set1['label'])
print(fit_q4.best_params_)
set1 = train.iloc[:10000].copy()
set2 = test.copy()
set1['label'] = set1['label'].astype('category')
set2['label'] = set2['label'].astype('category')
fit_q4 = GridSearchCV(svm_radial, grid_q4, scoring='accuracy', cv=5)
fit_q4.fit(set1.drop(columns=['label']), set1['label'])
print(fit_q4.best_params_)
yhat_q4 = fit_q4.predict(set2.drop(columns=['label']))
print(confusion_matrix(set2['label'], yhat_q4))
print(accuracy_score(set2['label'], yhat_q4))