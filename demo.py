from xgboost import XGBClassifier

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

data = load_iris()
# print(data)
X_train, X_test, y_train, y_test = train_test_split(data['data'], data['target'], test_size=.2)

# bst = XGBClassifier(n_estimators=2, max_depth=2, learning_rate=1, objective='binary:logistic')

# print(data['target'])
# print(data['target'])
print(y_train.shape)

# bst.fit(X_train, y_train)

# print("Xtest")
# print(X_test)

# preds = bst.predict(X_test)

# print("preds")
# print(preds)
# print(y_test)