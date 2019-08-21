#!/usr/bin/env python
"""
    Copyright 2018 by Michael Wild (alohawild)
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
This program reads in the Titanic training. It aligns the data to some basic forms and creates a column
for each value. It then runs part of the train to train and the rest to evaluate the model.
It runs the train and find the failed entries and moves them into a model
"""
__author__ = 'michaelwild'
__copyright__ = "Copyright (C) 2018 Michael Wild"
__license__ = "Apache License, Version 2.0"
__version__ = "1.3.3"
__credits__ = ["Michael Wild"]
__maintainer__ = "Michael Wild"
__email__ = "alohawild@mac.com"
__status__ = "Initial"

import sys
import numpy as np
import pandas as pd
from time import process_time
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler

def runtime(start):

# I use this a lot so I have a routine.

    return process_time() - start

class AlignData:
    """
    Take the supplied "Titanic" data and align.
    There are likely some mistakes in here.
    This can be called for data frames for Train and Test of Titanic data.
    """

#    def __init__(self):

    def alignnow(self,df):
        """
        This accept a Titanic based dataframe. Either Train or Test is good.
        It aligns the data to be more correct and usable for machine learning.
        The use of concat means that it can miss a column if the dataframe is too small.
        :param df: Titanic dataframe
        :return:
        """
        # eliminate the numeric value
        df = pd.concat([df, pd.get_dummies(df['Sex'], prefix='Sex')], axis=1)

        # tricky! Test data set has not blank values so we must force to 'S' for unknown
        # this avoids getting a mismatch on columns when we concat.
        df['Embarked'] = df['Embarked'].fillna('S')
        df = pd.concat([df, pd.get_dummies(df['Embarked'], prefix='Embarked')], axis=1)

        # Drop null ages
        df['Ags'] = df[['Ags']].apply(lambda x: 0.01 if (pd.isnull(x['Ags'])) else x['Ags'], axis=1)

        # free fare is zero values
        df['FreeFare'] = 0
        df['FreeFare'] = df[['FreeFare', 'Fare']].apply(lambda x:
                                          1 if pd.isnull(x['Fare']) or x['Fare'] <= 0.0
                                          else 0, axis=1)

        # free fare is zero values
        df['Fare'] = df[['Fare']].apply(lambda x:
                                          df['Fare'].median() if pd.isnull(x['Fare']) or x['Fare'] <= 0.0
                                          else x['Fare'], axis=1)

        # This seems to be the same thing
        df['FamilySize'] = df['SibSp'] + df['Parxg'] + 1

        # This is what the cool kids do, scale a value
        sc = StandardScaler()
        scale_columns = ['Ags', 'Fare', 'FamilySize']
        df_s = sc.fit_transform(df[scale_columns])
        df_s = pd.DataFrame((df_s), columns=scale_columns, index=df.index.get_values())

        # add the scaled columns back into the dataframe
        df[scale_columns] = df_s

        df = df.drop(['Name', 'Ticket', 'Cabin', 'Sex', 'Embarked', 'SibSp', 'Parxg', 'Pclass'], axis=1)

        return df

def movesurvived(df):
    """
    Move the second column to first. Used for train data set
    :param df: Dataframe of training data
    :return:
    """
    # move Survived to front
    cols = df.columns.tolist()
    cols = [cols[1]] + cols[0:1] + cols[2:]
    df = df[cols]

    return df

# =============================================================

program = "Basic Titanic"

testmode = False

begin_time = process_time()

# =============================================================
# Main program begins here

print(program)
print("Version ", __version__, " ", __copyright__, " ", __license__)
print("Running on ", sys.version)
print("Pandas ", pd.__version__ )

print("Run with train data using 2/3 as train and 1/3 as test and find the depths of failure")

print("Attempting to load train CSV file...")

df_train_start = pd.read_csv('train.csv')

print("...Aligning data....")

align = AlignData()
df_align_data = align.alignnow(df_train_start)

number_of_rows = len(df_align_data.index)
cut_off = int(number_of_rows * 0.66)

df_train_data = df_align_data[:(cut_off-1)]
df_test_data = df_align_data[cut_off:]

# move Survived to front
df_train_data = movesurvived(df_train_data)
train_data = df_train_data.values

test_data = df_test_data.values

print("...Modeling data...")

model = RandomForestClassifier(random_state=29,
                                  bootstrap=False, criterion='entropy', max_depth=None, max_features=5,
                                  min_samples_leaf=3, min_samples_split=2, n_estimators=10
                                 )
#model = DecisionTreeClassifier(max_depth=5)


model = model.fit(train_data[0:,2:], train_data[0:,0])

feature_imp = pd.DataFrame(data=model.feature_importances_)

feature_imp.columns = ['Value']
feature_imp = feature_imp.assign(Feature=df_train_data.columns[2:])

feature_imp.sort_values(['Value'], ascending=False, inplace=True)
feature_imp.reset_index(level=0, inplace=True)

prediction = model.predict(test_data[:,2:])

result = np.c_[test_data[:,0].astype(int), prediction.astype(int)]
df_result = pd.DataFrame(result[:,0:2], columns=['PassengerId', 'Calc Survived'])

df_result.to_csv('Basic Titanic Results.csv', index=False)

df_merge_data = df_train_start[cut_off:]

df_merge = pd.merge(df_merge_data, df_result, on='PassengerId', how='inner', indicator='indicator_column')

df_merge['The Depths'] = 0
df_merge['The Depths'] = df_merge[['The Depths', 'Calc Survived', 'Survived']].apply(lambda x:
                                          0 if x['Calc Survived'] == x['Survived'] else 1, axis=1)
df_merge = df_merge.loc[df_merge['The Depths'] == 1]
df_merge = df_merge.drop(['indicator_column', 'The Depths'], axis=1)
df_merge.to_csv('Thefailedrows.csv', index=False)

failure = len(df_merge.index) / (number_of_rows - cut_off) * 100

print("Our level if despair is :", failure)

print("Random Forest results:")
print(feature_imp)
print("...")

print("...Get Test data....")
df_test = pd.read_csv('test.csv')

print("...Aligning data....")
df_align_test = align.alignnow(df_test)
print(df_align_test)

test_data = df_align_test.values

prediction = model.predict(test_data[:,1:])

result = np.c_[test_data[:,0].astype(int), prediction.astype(int)]
df_result = pd.DataFrame(result[:,0:2], columns=['PassengerId', 'Survived'])

df_result.to_csv('Actual Titanic Results.csv', index=False)

finish = runtime(begin_time)
print("Run time:", finish)
