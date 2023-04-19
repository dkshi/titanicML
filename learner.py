import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import cross_val_score
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GridSearchCV
from preprocessor import Preprocessor
from saver import save

preprocessor = Preprocessor('data/train.csv', 'data/test.csv')
data = preprocessor.process()
X = data['X_train']
y = data['y_train']
X_test = data['X_test']


def get_mae_score(clf, X_train, X_valid, y_train, y_valid):
    clf.fit(X_train, y_train)
    return mean_absolute_error(clf.predict(X_valid), y_valid)


def get_roc_auc_score(clf, X_train, X_valid, y_train, y_valid):
    clf.fit(X_train, y_train)
    return roc_auc_score(y_valid, clf.predict(X_valid))


X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=42)
temp_clf = RandomForestClassifier(random_state = 0)
scores = cross_val_score(temp_clf, X_train, y_train, cv=10, scoring='accuracy')

print(scores.mean())
print(scores.std())
print(get_mae_score(temp_clf, X_train, X_valid, y_train, y_valid))
print(get_roc_auc_score(temp_clf, X_train, X_valid, y_train, y_valid))

temp_clf.fit(X_train, y_train)

importances = pd.DataFrame({'Name': temp_clf.feature_names_in_, 'Importance': temp_clf.feature_importances_})\
.sort_values('Importance', ascending=False)

print(importances)

params_grid = { "criterion" : ["gini", "entropy"],
               "min_samples_leaf" : [1, 5, 10, 25, 50, 70],
               "min_samples_split" : [2, 4, 10, 12, 16, 18, 25, 35],
               "n_estimators": [100, 400, 700, 1000, 1500]}

search_clf = RandomForestClassifier()
search = GridSearchCV(search_clf, params_grid)

#{'criterion': 'gini',
# 'min_samples_leaf': 1,
# 'min_samples_split': 10,
# 'n_estimators': 100}


#best_clf = search.best_estimator
best_clf = RandomForestClassifier(criterion='gini', min_samples_leaf=1, min_samples_split=10, n_estimators=100)
best_clf.fit(X, y)

save('gender_submission.csv', X_test, best_clf.predict(X_test.drop('PassengerId', axis=1)))

