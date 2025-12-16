from sklearn import svm, datasets
from sklearn.model_selection import GridSearchCV
iris = datasets.load_iris()
parameters = {'kernel':('linear', 'rbf'), 'C':[1, 10]}
svc = svm.SVC()
clf = GridSearchCV(svc, parameters)
clf.fit(iris.data, iris.target)

print("Best parameters found:", clf.best_params_)
print("Best cross-validation score:", clf.best_score_)
print("Best estimator:", clf.best_estimator_)

import pandas as pd

results = pd.DataFrame(clf.cv_results_)
print(results[['params', 'mean_test_score', 'rank_test_score']])
print(results.sort_values(by='mean_test_score', ascending=False))

y_pred = clf.predict(iris.data)