import pandas as pd
import numpy as np
from sklearn.preprocessing import Imputer
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression


training_set = pd.read_csv('./data/train.csv')
test_set = pd.read_csv('./data/test.csv')

training_set_men = training_set.loc[training_set['Sex'] == 'male']
training_set_women = training_set.loc[training_set['Sex'] == 'female']

test_set_men = test_set.loc[training_set['Sex'] == 'male']
test_set_women = test_set.loc[training_set['Sex'] == 'female']

def fill_missing_data(X):
    imputer = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0)
    imputer.fit(X[['Age']])
    X[['Age']] = imputer.transform(X[['Age']])
    X[['Cabin']] = X[['Cabin']].replace(".*", True, regex=True)
    X[['Cabin']] = X[['Cabin']].replace(np.nan, False)
    X = X.fillna(method='ffill')
    
    return X

def encode_categorical_data(X):
    X[['Sex']] = X[['Sex']].replace('male', True)
    X[['Sex']] = X[['Sex']].replace('female', False)
    X = pd.get_dummies(X, columns=['Cabin', 'Embarked', 'Pclass'])
    X = X.drop(['Cabin_False', 'Embarked_S', 'Pclass_3'], axis=1)

    return X

def scale_features(X):
    scaler = MinMaxScaler()
    X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
    
    return X

def prepare_dataset(data_set, features):
    X = data_set.reindex(sorted(data_set.columns), axis=1)
    
    # Taking care of missing data
    X = fill_missing_data(X)
    X = X.drop(['Name', 'Ticket'], axis=1)
    
    # Encoding categorical data
    X = encode_categorical_data(X)

    #Choose features
    X = X[features]  
    
    #feature scaling
    X = scale_features(X)
    
    return X.values

def get_classifier(X, y, metrics=True):
    classifier = LogisticRegression(random_state=42)
    #from sklearn.ensemble import RandomForestClassifier
    #classifier = RandomForestClassifier(n_estimators=22, criterion='entropy', random_state=42)

    # metrics
    if metrics:
        from sklearn.model_selection import cross_val_score
        scores = cross_val_score(classifier, X, y, cv=5)
        print('Mean accuracy = %.3f' % scores.mean())
        print('Std= %.3f' % scores.std())
    
    classifier.fit(X, y)
    
    return classifier

#from sklearn.svm import SVC
#classifier = SVC(kernel = 'linear', random_state = 42)

#from sklearn.naive_bayes import GaussianNB
#classifier = GaussianNB()
#from sklearn.ensemble import RandomForestClassifier
#classifier = RandomForestClassifier(n_estimators=22, criterion='entropy', random_state=42)

    

data_set = training_set
features = ['Age', 'Sex', 'Pclass_1', 'Pclass_2', 'Fare', 'Parch']

X_train = prepare_dataset(data_set.iloc[:, 2:], features)
y_train = data_set.iloc[:, 1].values


classifier = get_classifier(X_train, y_train, metrics=True)

X_test = prepare_dataset(test_set.iloc[:, 1:], features)

ids = test_set.iloc[:, 0].values
ids = ids.reshape(len(ids), 1)
y_pred = classifier.predict(X_test) 
y_pred = y_pred.reshape(len(y_pred), 1)

result = np.concatenate((ids, y_pred), axis=1)


np.savetxt("result.csv", result, delimiter=",", fmt='%i', header='PassengerId,Survived')