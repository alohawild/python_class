#!/usr/bin/env python
"""
    Copyright 2017 by Michael Wild (alohawild)

    Licensed under the Apache License, Version 2.0 (the "License");
    you may not use this file except in compliance with the License.
    You may obtain a copy of the License at
        http://www.apache.org/licenses/LICENSE-2.0

    Unless required by applicable law or agreed to in writing, software
    distributed under the License is distributed on an "AS IS" BASIS,
    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
    See the License for the specific language governing permissions and
    limitations under the License.

==============================================================================
This program reads in the Titanic training and test sets. It aligns the data to some basic forms and creates a column
for each value. It then runs various models and generates a file for each model



"""
__author__ = 'michaelwild'

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, VotingClassifier, ExtraTreesClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier

class AlignData:
    """
    Take the supplied "Titanic" data and align it into a new dictionary
    """

#    def __init__(self):

    def aligntitle(self,title):
        """
        :param title:
        :return:
        """
        if title in ['Mlle', 'Ms', 'Mme']:
            return 'Miss'
        elif title in ['Mr', 'Miss', 'Mrs', 'Master']:
            return title
        elif title in ['Capt', 'Col', 'Don', 'Jonkheer', 'Major', 'Rev', 'Sir']:
            return 'Mr'
        elif title in ['the Countess', 'Mme', 'Lady', 'Dona', 'Countess']:
            return 'Mrs'
        elif title == 'Dr':
            return 'Mr'  # It is 1912
        else:
            return 'Mr'  # It is 1912

    def alignnow(self,df):
        """

        :param df: Titanic dataframe
        :return:
        """
        df = pd.concat([df, pd.get_dummies(df['Sex'], prefix='Sex')], axis=1)

        df['Title'] = df['Name'].str.split(".").str[0]
        df['Title'] = df['Title'].str.split(" ").str[-1]
        df['Title'] = df['Title'].apply(lambda x: self.aligntitle(x))

        df['Incomplete'] = 0
        df['Incomplete'] = df[['Cabin', 'Embarked', 'Ags']].apply(lambda x:
                                                                    1 if (
                                                                    pd.isnull(x['Cabin']) or pd.isnull(x['Embarked']) or
                                                                    pd.isnull(x['Ags']))
                                                                    else 0, axis=1)

        df['Embarked'] = df['Embarked'].fillna('S')
        df = pd.concat([df, pd.get_dummies(df['Embarked'], prefix='Embarked')], axis=1)

        # The cool kid stuff
        age_means = df.pivot_table('Ags', index='Title', aggfunc='mean')
        age_means.reset_index()

        df['Ags'] = df[['Ags', 'Title']].apply(lambda x: age_means.loc[x['Title'], 'Ags']
                                                    if (pd.isnull(x['Ags'])) else x['Ags'], axis=1)

        print(df)

        df = pd.concat([df, pd.get_dummies(df['Title'], prefix='Title')], axis=1)

        df['FreeFare'] = 0
        df['FreeFare'] = df[['FreeFare', 'Fare']].apply(lambda x:
                                          1 if pd.isnull(x['Fare']) or x['Fare'] <= 0.0
                                          else 0, axis=1)

        df['Fare'] = df[['Fare']].apply(lambda x:
                                          df['Fare'].mean() if pd.isnull(x['Fare']) or x['Fare'] <= 0.0
                                          else x['Fare'], axis=1)

        df = pd.concat([df, pd.get_dummies(df['Pclass'], prefix='Pclass')], axis=1)

        # This seems to be the same thing
        df['FamilySize'] = df['SibSp'] + df['Parxg']

        sc = StandardScaler()
        scale_columns = ['Ags', 'Fare', 'FamilySize']
        df_s = sc.fit_transform(df[scale_columns])
        df_s = pd.DataFrame((df_s), columns=scale_columns, index=df.index.get_values())

        # add the scaled columns back into the dataframe
        df[scale_columns] = df_s

        df = df.drop(['Name', 'Ticket', 'Cabin', 'Sex', 'Embarked', 'SibSp', 'Parxg', 'Pclass', 'Title'], axis=1)

        return df

def movesurvived(df):
    """
    Move the second column to first
    :param df: Dataframe of training data
    :return:
    """
    # move Survived to front
    cols = df.columns.tolist()
    cols = [cols[1]] + cols[0:1] + cols[2:]
    df = df[cols]

    return df


# =============================================================


version = "1.01"
program = "Great Scott!"

testmode = False

# =============================================================
# Main program begins here

print(program, " Version ", version)

print("Run various models and create files for each. Great Scott! That is a lot of data")

print("Attempting to load train CSV file...")

df_train_data = pd.read_csv('train fixed.csv')

df_align_train = df_train_data

print("...Aligning data....")

align = AlignData()

df_align_train = align.alignnow(df_align_train)

#print (df_align_train)

#move Survived to front
df_align_train = movesurvived(df_align_train)

train_data = df_align_train.values

print("Attempting to load test CSV file...")

df_test_data = pd.read_csv('test fixed.csv')

df_align_test = df_test_data

df_align_test = align.alignnow(df_align_test)

test_data = df_align_test.values

print("...Modeling data...")

model = RandomForestClassifier(random_state=29,
                                  bootstrap=False, criterion='entropy', max_depth=None, max_features=5,
                                  min_samples_leaf=3, min_samples_split=2, n_estimators=500
                                 )

model = model.fit(train_data[0:,2:], train_data[0:,0])

feature_imp = pd.DataFrame(data=model.feature_importances_)

feature_imp.columns = ['Value']
feature_imp = feature_imp.assign(Feature=df_align_train.columns[2:])

feature_imp.sort_values(['Value'], ascending=False, inplace=True)
feature_imp.reset_index(level=0, inplace=True)

prediction = model.predict(test_data[:,1:])

result = np.c_[test_data[:,0].astype(int), prediction.astype(int)]
df_result = pd.DataFrame(result[:,0:2], columns=['PassengerId', 'Survived'])

df_result.to_csv('Random Forest.csv', index=False)

print("Random Forest")
print(feature_imp)
print("...")

model = ExtraTreesClassifier(random_state=29,
                                  bootstrap=False, criterion='gini', max_depth=None, max_features=10,
                                  min_samples_leaf=3, min_samples_split=2, n_estimators=500
                                 )
model = model.fit(train_data[0:,2:], train_data[0:,0])

feature_imp = pd.DataFrame(data=model.feature_importances_)

feature_imp.columns = ['Value']
feature_imp = feature_imp.assign(Feature=df_align_train.columns[2:])

feature_imp.sort_values(['Value'], ascending=False, inplace=True)
feature_imp.reset_index(level=0, inplace=True)

prediction = model.predict(test_data[:,1:])

result = np.c_[test_data[:,0].astype(int), prediction.astype(int)]
df_result = pd.DataFrame(result[:,0:2], columns=['PassengerId', 'Survived'])

df_result.to_csv('Extratrees.csv', index=False)

print("ExtraTrees")
print(feature_imp)
print("...")

model = AdaBoostClassifier(base_estimator=DecisionTreeClassifier(criterion='entropy', max_depth=1, max_leaf_nodes=None,
                                                                     min_samples_leaf=5, min_samples_split=2),
                                learning_rate=0.1, n_estimators=500, random_state=29)

model = model.fit(train_data[0:,2:], train_data[0:,0])

feature_imp = pd.DataFrame(data=model.feature_importances_)

feature_imp.columns = ['Value']
feature_imp = feature_imp.assign(Feature=df_align_train.columns[2:])

feature_imp.sort_values(['Value'], ascending=False, inplace=True)
feature_imp.reset_index(level=0, inplace=True)

prediction = model.predict(test_data[:,1:])

result = np.c_[test_data[:,0].astype(int), prediction.astype(int)]
df_result = pd.DataFrame(result[:,0:2], columns=['PassengerId', 'Survived'])

df_result.to_csv('AdaBoost.csv', index=False)

print("AdaBoost")
print(feature_imp)
print("...")

model = VotingClassifier(estimators=[('lr', LogisticRegression(random_state=29, penalty='l1', C=1)),
                            ('nb', GaussianNB()),
                            ('knn', KNeighborsClassifier(n_neighbors=5, weights='uniform')),
                            ('rf', RandomForestClassifier(random_state=29,
                                  bootstrap=False, criterion='entropy', max_depth=None, max_features=5,
                                  min_samples_leaf=3, min_samples_split=2, n_estimators=500))],
                           voting='soft'
                          )

model = model.fit(train_data[0:,2:], train_data[0:,0])

prediction = model.predict(test_data[:,1:])

result = np.c_[test_data[:,0].astype(int), prediction.astype(int)]
df_result = pd.DataFrame(result[:,0:2], columns=['PassengerId', 'Survived'])

df_result.to_csv('Voting.csv', index=False)

print("Voting")
print("No features")
print("...")

print("...End of Line....")
