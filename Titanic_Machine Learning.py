import pandas as pd

train_data = pd.read_csv(r'D:\Kaggle\titanic\train.csv')
test_data = pd.read_csv(r'D:\Kaggle\titanic\test.csv')

train_data.head()  # View the first few rows
train_data.info()  # Check for missing values and data types
train_data.describe()  # Summary statistics

import seaborn as sns
import matplotlib.pyplot as plt

# # Plot survival count
# sns.countplot(x='Survived', data=train_data)
# plt.show()
#
# # Plot survival by gender
# sns.countplot(x='Survived', hue='Sex', data=train_data)
# plt.show()

train_data.isnull().sum()

train_data['Age'].fillna(train_data['Age'].median(), inplace=True)

train_data['Embarked'].fillna(train_data['Embarked'].mode()[0], inplace=True)

train_data['FamilySize'] = train_data['SibSp'] + train_data['Parch']

# Convert 'Sex' to numerical values
train_data['Sex'] = train_data['Sex'].map({'male': 0, 'female': 1})

# Convert 'Embarked' to numerical values
train_data['Embarked'] = train_data['Embarked'].map({'S': 0, 'C': 1, 'Q': 2})

features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked', 'FamilySize']
X = train_data[features]
y = train_data['Survived']


from sklearn.model_selection import train_test_split

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)


from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)


from sklearn.metrics import accuracy_score

y_pred = model.predict(X_val)
accuracy = accuracy_score(y_val, y_pred)
print(f'Accuracy: {accuracy:.2f}')

test_data['Age'].fillna(test_data['Age'].median(), inplace=True)
test_data['Fare'].fillna(test_data['Fare'].median(), inplace=True)
test_data['Sex'] = test_data['Sex'].map({'male': 0, 'female': 1})
test_data['Embarked'] = test_data['Embarked'].map({'S': 0, 'C': 1, 'Q': 2})
test_data['FamilySize'] = test_data['SibSp'] + test_data['Parch']

X_test = test_data[features]
test_predictions = model.predict(X_test)


submission = pd.DataFrame({
    'PassengerId': test_data['PassengerId'],
    'Survived': test_predictions
})
submission.to_csv(r'D:\Kaggle\titanic\submission.csv', index=False)
