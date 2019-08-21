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
This program reads in the Titanic training and test and makes a prediction and write it out


"""
import sys
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from xgboost.sklearn import XGBClassifier


class GetTitanicData:
    """
    Load Titanic data into data frame
    Align data and test it
    :return:
    """

    def __init__(self, train_file='train fixed.csv', test_file='test fixed.csv',
                 error_file='error file.csv'):
        """
        Read in data as we start up
        """
        self.df_train = pd.read_csv(train_file)
        self.df_test = pd.read_csv(test_file)

        self.df_Titanic = self.align_data()
        self.df_Titanic = self.prepare_data(self.df_Titanic)

        self.model = RandomForestClassifier(random_state=29,
                                   bootstrap=True, criterion='entropy', max_depth=None,
                                   max_features=20,
                                   min_samples_leaf=3, min_samples_split=2, n_estimators=500
                                   )


        df_errors = self.train_data_and_validate(self.df_Titanic, self.model, verbose=True)

        if (error_file != ''):
            df_errors.to_csv(error_file, index=False)

        self.df_result = self.train_run(self.df_Titanic, self.model, verbose=True)

    def align_data(self):
        """
        Combine all the data into one data frame with 'Test' set to 1 for test data
        :return:
        """
        # Make the data sets match in structure

        self.df_test['Survived'] = 0
        # move Survived to second place
        cols = self.df_test.columns.tolist()
        cols = [cols[len(cols)-1]] + cols[:(len(cols)-1)]
        self.df_test = self.df_test[cols]

        self.df_test['Test'] = 1

        # move Survived to front
        cols = self.df_test.columns.tolist()
        cols = [cols[1]] + cols[0:1] + cols[2:]
        self.df_test = self.df_test[cols]

        self.df_train['Test'] = 0

        frames = [self.df_train, self.df_test]
        df_merge = pd.concat(frames)

        return df_merge

    def align_title(self,title):
        """
        This is a 1912 based title alignment. This accept in a title and returns a generalized title.
        We have mixed doctors in...there is one Female doctor that this is wrong for but it has no impact
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
        else:
            return 'Mr'  # We get one doctor wrong...it still works

    def age_force(self, title):
        """
        Well the pivot table I was using stopped working and had a strange error after the last update...
        so these values never change so why bother anyway...
        :param title:
        :return:
        """

        if title in ['Miss']:
            return 21.83

        elif title in ['Mr']:
            return 32.81
        elif title in ['Mrs']:
            return 37.05

        elif title in ['Master']:
            return 5.48
        else:
            return 29.29

    def prepare_data(self, df):
        """
        prepare the data to be used for training and testing
        :param df: Titanic Data Frame
        :return:
        """
        #  Class needs to be broken out
        df = pd.concat([df, pd.get_dummies(df['Pclass'], prefix='Class')], axis=1)

        #  Sex is now matched as a value and not a number
        df = pd.concat([df, pd.get_dummies(df['Sex'], prefix='Sex')], axis=1)

        #  align ticket to be more normalized
        df['Ticket'] = df['Ticket'].str.replace('.','')
        df['Ticket'] = df['Ticket'].str.replace('/', '')
        df['Ticket'] = df['Ticket'].str.replace(' ', '')

        #  Tickets are now replaced with a single value
        df['Ticket'] = df['Ticket'].str[0:1]
        df = pd.concat([df, pd.get_dummies(df['Ticket'], prefix='Ticket')], axis=1)

        #  Cabin also broken out
        df['Cabin'] = df['Cabin'].fillna('Unknown')
        df['Cabin'] = df['Cabin'].str[0:1]
        df = pd.concat([df, pd.get_dummies(df['Cabin'], prefix='Cabin')], axis=1)

        df['Title'] = df['Name'].str.split(".").str[0]
        df['Title'] = df['Title'].str.split(" ").str[-1]
        df['Title'] = df['Title'].apply(lambda x: self.align_title(x))
        df = pd.concat([df, pd.get_dummies(df['Title'], prefix='Title')], axis=1)

        #  pivot table broke in version of Pandas
        # age_mean = pd.DataFrame()
        #age_mean = df.pivot_table('Ags', index=['Title'], aggfunc=np.mean)
        #print(age_mean)

        df['Ags'] = df[['Ags', 'Title']].apply(lambda x:
                                                 self.age_force([x['Title']]) if pd.isnull(x['Ags'])
                                                 else x['Ags'], axis=1)

        # free fare is set to median
        df['Fare'] = df[['Fare']].apply(lambda x:
                                        df['Fare'].median() if pd.isnull(x['Fare']) or x['Fare'] <= 0.0
        #                                0.0 if pd.isnull(x['Fare']) or x['Fare']
                                        else x['Fare'], axis=1)

        # This seems to be the same thing
        df['FamilySize'] = df['SibSp'] + df['Parxg']

        # This is what the cool kids do, scale a value
        sc = StandardScaler()
        scale_columns = ['Ags', 'Fare', 'FamilySize']
        df_s = sc.fit_transform(df[scale_columns])
        df_s = pd.DataFrame((df_s), columns=scale_columns, index=df.index.get_values())

        # add the scaled columns back into the dataframe
        df[scale_columns] = df_s

        df = df.drop(['Name', 'Ticket', 'Cabin', 'Sex', 'Embarked', 'SibSp', 'Parxg', 'Pclass', 'Title'], axis=1)


        return df

    def train_data_and_validate(self, df, model, verbose=True):
        """

        :param df: Titanic combined prepared data
        :param model: random forest
        :param verbose: True to trace
        :return: data frame of all failed values in original format with extra two columns
        """
        number_of_rows = len(df.index)
        cut_off = int(number_of_rows * 0.66)

        df_train = df.loc[df['Test'] == 0]
        df_train = df_train.drop(['Test'], axis=1)

        number_of_rows = len(df_train.index)
        cut_off = int(number_of_rows * 0.66)

        df_train_data = df_train[:cut_off - 1]
        df_validate_data = df_train[cut_off:]

        #  Data frames are not used by the trees routines
        train_data = df_train_data.values

        model = model.fit(train_data[0:, 2:], train_data[0:, 1])

        feature_imp = pd.DataFrame(data=model.feature_importances_)

        feature_imp.columns = ['Value']
        feature_imp = feature_imp.assign(Feature=df_train_data.columns[2:])

        feature_imp.sort_values(['Value'], ascending=False, inplace=True)
        feature_imp.reset_index(level=0, inplace=True)

        if verbose:
            print("Features...")
            print(feature_imp)
            print("...")

        #  Data frames are not used by the trees routines
        validate_data = df_validate_data.values

        prediction = model.predict(validate_data[:, 2:])

        result = np.c_[validate_data[:, 0].astype(int), prediction.astype(int)]
        df_result = pd.DataFrame(result[:, 0:2], columns=['PassengerId', 'Calc Survived'])

        proba = model.predict_proba(validate_data[:, 2:])

        result = np.c_[validate_data[:, 0].astype(float), proba.astype(float)]
        df_proba = pd.DataFrame(result[:, 0:2], columns=['PassengerId', 'Prob'])

        df_merge = pd.merge(df_validate_data, df_result, on='PassengerId', how='inner')
        df_merge = pd.merge(df_merge, df_proba, on='PassengerId', how='inner')

        df_merge['The Depths'] = 0
        df_merge['The Depths'] = df_merge[['The Depths', 'Calc Survived', 'Survived']].apply(lambda x:
                                                                                             0 if x['Calc Survived'] ==
                                                                                                  x['Survived'] else 1,
                                                                                             axis=1)
        df_merge = df_merge.loc[df_merge['The Depths'] == 1]

        failure = len(df_merge.index) / len(df_validate_data.index) * 100

        if verbose:
            print("Our level if despair is :", failure)
            print("Count of miss:", len(df_merge.index))
            print("...")

        cols = ['PassengerId', 'Prob', 'Calc Survived']
        df_final = df_merge[cols]
        df_final = pd.merge(df_final, self.df_train, on='PassengerId', how='inner')
        df_final = df_final.drop(['Test'], axis=1)

        return df_final

    def train_run(self, df, model, verbose=True):
        """

        :param df: Titanic prepared data
        :param model: random tree
        :param verbose: True for trace
        :return: prediction in data frame
        """
        #  Train data extracted
        df_train_data = df.loc[df['Test'] == 0]
        df_train_data = df_train_data.drop(['Test'], axis=1)

        #  Data frames are not used by the trees routines
        train_data = df_train_data.values

        model = model.fit(train_data[0:, 2:], train_data[0:, 1])

        feature_imp = pd.DataFrame(data=model.feature_importances_)

        feature_imp.columns = ['Value']
        feature_imp = feature_imp.assign(Feature=df_train_data.columns[2:])

        feature_imp.sort_values(['Value'], ascending=False, inplace=True)
        feature_imp.reset_index(level=0, inplace=True)

        if verbose:
            print("Features...")
            print(feature_imp)
            print("...")

        #  Test data in DF is extracted
        df_test_data = df.loc[df['Test'] == 1]
        df_test_data = df_test_data.drop(['Test'], axis=1)
        #  Data frames are not used by the trees routines

        test_data = df_test_data.values

        prediction = model.predict(test_data[:, 2:])

        result = np.c_[test_data[:, 0].astype(int), prediction.astype(int)]
        df_result = pd.DataFrame(result[:, 0:2], columns=['PassengerId', 'Survived'])

        return df_result

    def get_results(self):

        return self.df_result

    def get_prepared_data(self):

        return self.df_Titanic

class TitanicBoost(GetTitanicData):
    """
    Titanic data with Boost option and reduced data
    """
    def __init__(self, train_file='train fixed.csv', test_file='test fixed.csv',
                 error_file='boost error file.csv'):
        """
        Read in data as we start up
        """
        self.df_train = pd.read_csv(train_file)
        self.df_test = pd.read_csv(test_file)

        self.df_Titanic = self.align_data()
        self.df_Titanic = self.prepare_data(self.df_Titanic)

        self.model = XGBClassifier(learning_rate=0.01,
                                    n_estimators=5000,
                                    max_depth=7,
                                    min_child_weight=5,
                                    gamma=0.0,
                                    subsample=0.8,
                                    colsample_bytree=0.8,
                                    reg_alpha=1e-05,
                                    objective='binary:logistic',
                                    nthread=4,
                                    scale_pos_weight=1.0,
                                    seed=29)

        df_errors = self.train_data_and_validate(self.df_Titanic, self.model, verbose=True)

        if (error_file != ''):
            df_errors.to_csv(error_file, index=False)

        self.df_result = self.train_run(self.df_Titanic, self.model, verbose=True)

    def prepare_data(self, df):
        """
        prepare the data to be used for training and testing
        :param df: Titanic Data Frame
        :return:
        """
        df['SexValue'] = 0
        df['SexValue'] = df[['SexValue', 'Sex']].apply(lambda x: 1 if x['Sex'] == 'male' else 0, axis=1)

        df['Title'] = df['Name'].str.split(".").str[0]
        df['Title'] = df['Title'].str.split(" ").str[-1]
        df['Title'] = df['Title'].apply(lambda x: self.align_title(x))
        df = pd.concat([df, pd.get_dummies(df['Title'], prefix='Title')], axis=1)

        df['Ags'] = df[['Ags', 'Title']].apply(lambda x:
                                                 self.age_force([x['Title']]) if pd.isnull(x['Ags'])
                                                 else x['Ags'], axis=1)
        #  Class needs to be broken out
        df = pd.concat([df, pd.get_dummies(df['Pclass'], prefix='Class')], axis=1)

        # free fare is set to median
        df['Fare'] = df[['Fare']].apply(lambda x:
                                        df['Fare'].median() if pd.isnull(x['Fare']) or x['Fare'] <= 0.0
        #                                0.0 if pd.isnull(x['Fare']) or x['Fare']
                                        else x['Fare'], axis=1)

        # This seems to be the same thing
        df['FamilySize'] = df['SibSp'] + df['Parxg']

        # This is what the cool kids do, scale a value
        sc = StandardScaler()
        scale_columns = ['Ags', 'Fare', 'FamilySize','SexValue']
        df_s = sc.fit_transform(df[scale_columns])
        df_s = pd.DataFrame((df_s), columns=scale_columns, index=df.index.get_values())

        # add the scaled columns back into the dataframe
        df[scale_columns] = df_s

        df = df.drop(['Name', 'Ticket', 'Cabin', 'Embarked', 'SibSp', 'Parxg', 'Pclass', 'Title', "Sex"], axis=1)

        return df

def to_xml(df, filename=None, mode='w'):
    def row_to_xml(row):
        xml = ['<item>']
        for i, col_name in enumerate(row.index):
            xml.append('  <field name="{0}">{1}</field>'.format(col_name, row.iloc[i]))
        xml.append('</item>')
        return '\n'.join(xml)

    res = '\n'.join(df.apply(row_to_xml, axis=1))

    if filename is None:
        return res
    with open(filename, mode) as f:
        f.write(res)

# =============================================================

__author__ = 'Michael Wild'

version = "1.01"
program = "Final Titanic"

# =============================================================
# Main program begins here

pd.DataFrame.to_xml = to_xml

print(program, " Version ", version)

the_data = GetTitanicData()

results = the_data.get_results()

#results.to_csv("Final Titanic data.csv", index=False) #  .77 in Kaggle
#results.to_xml("Final Titanic data.xml")
print("Try the boost model")

df_data = the_data.get_prepared_data()
#df_data.to_csv("Final prepared Titanic data.csv", index=False)
df_data.to_xml("Final prepared Titanic data.xml")

df_copy = the_data.get_prepared_data()
df_copy_train = df_copy.loc[df_copy['Test'] == 0]
df_copy_train = df_copy_train.drop(['Test'], axis=1)
df_copy_train.to_csv("Titanic train data.csv", index=False)

df_copy_test = df_copy.loc[df_copy['Test'] == 1]
df_copy_test = df_copy_test.drop(['Test', 'Survived'], axis=1)
df_copy_test.to_csv("Titanic test data.csv", index=False)

boost_model = XGBClassifier(learning_rate =0.01,
                      n_estimators=5000,
                      max_depth=2,
                      min_child_weight=5,
                      gamma=0.0,
                      subsample=0.8,
                      colsample_bytree=0.8,
                      reg_alpha=1e-05,
                      objective= 'binary:logistic',
                      nthread=4,
                      scale_pos_weight=1.0,
                      seed=29)

the_data.train_data_and_validate(df_data, boost_model)

more_data = TitanicBoost(error_file="boost error file.csv")

print("End of line....")
