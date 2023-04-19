import pandas as pd
from sklearn.preprocessing import OrdinalEncoder


class Preprocessor:
    def __init__(self, train, test):
        self.train = pd.read_csv(train)
        self.test = pd.read_csv(test)

    def process(self):
        X = self.train.drop(["PassengerId", "Ticket", "Survived"], axis=1)
        X_test = self.test.drop(["Ticket"], axis=1)
        y = self.train.Survived

        # Data preprocessing
        data = [X, X_test]
        enc = OrdinalEncoder()
        for dataset in data:
            # Getting type of cabin from cabin number
            dataset['Cabin'] = dataset['Cabin'].fillna(0)
            dataset['Cabin'] = [str(i)[0] for i in dataset['Cabin']]

            # Filling NA in column Embarked with the most common value
            common = dataset['Embarked'].describe().top
            dataset['Embarked'] = dataset['Embarked'].fillna(common)

            # Filling NA in Age with mean
            mean_age = dataset['Age'].mean()
            dataset['Age'] = dataset['Age'].fillna(mean_age)

            # Taking title from name
            dataset['Name'] = [i.split('.')[0].split(',')[-1] for i in dataset['Name']]
            dataset['Name'] = dataset['Name'].replace(['Lady', 'the Countess', 'Capt', 'Col', 'Don', 'Dr', \
                                                       'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare', regex=True)
            dataset['Name'] = dataset['Name'].replace('Mlle', 'Miss', regex=True)
            dataset['Name'] = dataset['Name'].replace('Ms', 'Miss', regex=True)
            dataset['Name'] = dataset['Name'].replace('Mme', 'Mrs', regex=True)

            # Converting object columns to numeric
            s = (dataset.dtypes == 'object')
            object_cols = list(s[s].index)
            dataset[object_cols] = enc.fit_transform(dataset[object_cols]).astype(int)

            dataset.rename(columns={'Name': 'Title', 'Cabin': 'Cabin_type'}, inplace=True)

            # Creating age groups
            dataset['Age'] = dataset['Age'].astype(int)
            dataset.loc[dataset['Age'] <= 11, 'Age'] = 0
            dataset.loc[(dataset['Age'] > 11) & (dataset['Age'] <= 18), 'Age'] = 1
            dataset.loc[(dataset['Age'] > 18) & (dataset['Age'] <= 22), 'Age'] = 2
            dataset.loc[(dataset['Age'] > 22) & (dataset['Age'] <= 27), 'Age'] = 3
            dataset.loc[(dataset['Age'] > 27) & (dataset['Age'] <= 33), 'Age'] = 4
            dataset.loc[(dataset['Age'] > 33) & (dataset['Age'] <= 40), 'Age'] = 5
            dataset.loc[(dataset['Age'] > 40) & (dataset['Age'] <= 66), 'Age'] = 6
            dataset.loc[dataset['Age'] > 66, 'Age'] = 6

            # Converting Fare to integer
            dataset['Fare'] = dataset['Fare'].fillna(0).astype(int)

        return {'X_train': X,
                'X_test': X_test,
                'y_train': y}



