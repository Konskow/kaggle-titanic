import pandas as pd
import numpy as np

training_set = pd.read_csv('./data/train.csv')
test_set = pd.read_csv('./data/test.csv')

X_train = training_set.iloc[:, 2:]
y_train = training_set.iloc[:, 1].values
X_train = X_train.reindex(sorted(X_train.columns), axis=1)

X_test = test_set.iloc[:, 1:]
ids = test_set.iloc[:, 0].values
X_test = X_test.reindex(sorted(X_test.columns), axis=1)

# Taking care of missing data
from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0)

imputer.fit(X_train[['Age']])
X_train[['Age']] = imputer.transform(X_train[['Age']])
X_train[['Cabin']] = X_train[['Cabin']].replace(".*", True, regex=True)
X_train[['Cabin']] = X_train[['Cabin']].replace(np.nan, False)
X_train = X_train.fillna(method='ffill')
X_train = X_train.drop(['Name', 'Ticket'], axis=1)


imputer.fit(X_test[['Age']])
X_test[['Age']] = imputer.transform(X_test[['Age']])
X_test[['Cabin']] = X_test[['Cabin']].replace(".*", True, regex=True)
X_test[['Cabin']] = X_test[['Cabin']].replace(np.nan, False)
X_test = X_test.fillna(method='ffill')
X_test = X_test.drop(['Name', 'Ticket'], axis=1)

# Encoding categorical data
X_train = pd.get_dummies(X_train, columns=['Cabin', 'Embarked', 'Sex', 'Pclass'])
X_test = pd.get_dummies(X_test, columns=['Cabin', 'Embarked', 'Sex', 'Pclass'])


#Drop one of the dummy column from each category to avoid dummy trap
X_train = X_train.drop(['Cabin_False', 'Embarked_S', 'Sex_female', 'Pclass_3'], axis=1)
X_test = X_test.drop(['Cabin_False', 'Embarked_S', 'Sex_female', 'Pclass_3'], axis=1)

#feature scaling
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
X_train = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns)
X_test = pd.DataFrame(scaler.fit_transform(X_test), columns=X_test.columns)


X_train = X_train.values
X_test = X_test.values

#from sklearn.model_selection import train_test_split
#X_train_cv, X_test_cv, y_train_cv, y_test_cv = train_test_split(X_train, y_train, test_size = 0.2, random_state = 42)


from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state=42)

#from sklearn.svm import SVC
#classifier = SVC(kernel = 'linear', random_state = 42)

#from sklearn.naive_bayes import GaussianNB
#classifier = GaussianNB()

#from sklearn.ensemble import RandomForestClassifier
#classifier = RandomForestClassifier(n_estimators=19, criterion='entropy', random_state=42)


# metrics
from sklearn.model_selection import cross_val_score
scores = cross_val_score(classifier, X_train, y_train, cv=5)
print('Mean accuracy = %.3f' % scores.mean())
print('Std= %.3f' % scores.std())


classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)

y_pred = y_pred.reshape(len(y_pred),1)
ids = ids.reshape(len(ids),1)
result = np.concatenate((ids,y_pred), axis=1)

np.savetxt("result.csv", result, delimiter=",", fmt='%i', header='PassengerId,Survived')