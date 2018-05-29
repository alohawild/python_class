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
This program reads in the Titanic train set and tries to create a function
to determine survivors, read in the test set, and create a final version



"""
__author__ = 'michaelwild'

import sys
import csv
import numpy as np

coefText = ['Age', 'Class', 'Fare', 'Sex', 'Cabin', 'Loc']

def randomChange():
    # This routine creates a random number between -1 and 1.
    # Uniform is used to get a even spread of values
    return (np.random.uniform() * 2.0) - 1


class LoadCSV:
    """
    A simple load of dictionaries from CSV labelled files.

    Fails hard if file not there.
    """

    def __init__(self, fileName='train.csv'):
        # Override the start up routine to save the file name and data.
        self.ourList = []
        self.fileName = fileName
        self.loadFile()

    def loadFile(self):
        # Gets the CSV file, with a hard fail, and loads into a list

        try:
            csvFile = open(self.fileName, encoding='utf-8')

        except IOError:
            print("Can't open file")
            sys.exit(1)

        reader = csv.DictReader(csvFile)

        for row in reader:
            self.ourList.append(row)
        csvFile.close()

class AlignData:
    """
    Take the supplied "Titanic" data and align it into a new dictionary
    """

#    def __init__(self):

    def alignAge(self,row):
        # Force age to a number and impute as needed.
        try:
            ageValue = int(row['Ags']) / 100.0
        except:
            ageValue = 0.0

        if ageValue<=0.0 or ageValue>=0.95:
            if row['Sex'] == "male":
                ageValue = 0.35
            else:
                ageValue = 0.30

        return ageValue

    def alignClass(self,row):
        # Force class value to number
        if ((row['Pclass'] == '1')):
            classValue = 1.0
        else:
            if ((row['Pclass'] == '2')):
                classValue = 0.50
            else:
                classValue = 0.0

        return classValue

    def alignFare(self, row):
        # align fare to value. Do not impute
        try:
            fareString = row['Fare']
            fareValue = float(fareString) / 1000.0
        except:
            fareValue = 0.0

        return fareValue

    def alignSex(self, row):
        # Force to a number (why zero for male?)
        if row['Sex'] == "male":
            sexValue = 0.0
        else:
            sexValue = 1.0

        return sexValue

    def alignSurvived(self, row):
        # Change test value to be number
        if row['Survived'] == "0":
            survivedValue = 0.0
        else:
            survivedValue = 1.0

        return survivedValue

    def alignCabin(self, row):
        # Is there a cabin filled in? Force to number
        if (row['Cabin']):
            cabinValue = 0.0
        else:
            cabinValue = 1.0

        return cabinValue

    def loadData(self,TitanicData, alignedData, cabinNot, cabinSurvive, train=True):
        # Load up and align the data, try to find matching cabins.
        for row in TitanicData:

            if train:  # Training data

                if row['Survived'] == "0":
                    if not row['Cabin'] in cabinNot:
                        cabinNot.append(row['Cabin'])
                else:
                    if not row['Cabin'] in cabinSurvive:
                        cabinSurvive.append(row['Cabin'])

                alignedLine =   {
                        'PassengerId'   : row['PassengerId'],
                        coefText[0]     : self.alignAge(row),
                        coefText[1]     : self.alignClass(row),
                        coefText[2]     : self.alignFare(row),
                        coefText[3]     : self.alignSex(row),
                        coefText[4]     : self.alignCabin(row),
                        coefText[5]     : 0.0,
                        'Survived'      : self.alignSurvived(row),
                        'Best'          : 0.0,
                        'Now'           : 0.0
                }
            else:
               # Check if cabin was in lists of previous values where we know what happened

               if row['Cabin'] in cabinNot and not row['Cabin'] in cabinSurvive:
                    cabinGood = 0.0
               else:
                    if row['Cabin'] in cabinSurvive and not row['Cabin'] in cabinNot:
                        cabinGood = 1.0
                    else:
                        cabinGood = 0.5

               alignedLine = {
                    'PassengerId': row['PassengerId'],
                    coefText[0]: self.alignAge(row),
                    coefText[1]: self.alignClass(row),
                    coefText[2]: self.alignFare(row),
                    coefText[3]: self.alignSex(row),
                    coefText[4]: self.alignCabin(row),
                    coefText[5]: cabinGood,
                }


            alignedData.append(alignedLine)

        if train:
            for row in alignedData:
                if row['Cabin'] in cabinNot and not row['Cabin'] in cabinSurvive:
                        cabinGood = 0.0
                else:
                    if row['Cabin'] in cabinSurvive and not row['Cabin'] in cabinNot:
                        cabinGood = 1.0
                    else:
                        cabinGood = 0.5
                row[coefText[5]] = cabinGood

    def acceptNow(self, alignedData):
        # Copy values
        for row in alignedData:
            row['Best'] = row['Now']

    def evaluateSurive(self, checked, survived):
        # How far away are we from the correct answer
        return abs(checked - survived)

    def calcNow(self, alignedData, c):
        # Calculate the results of applying a coefficient to all the aligned training data.
        # this is done by matching the matching columns

        rightNow = 0
        rightBest = 0
        totalNow = 0.0
        totalBest = 0.0

        for row in alignedData:
            newSurvivedValue = 0.0
            for line in coefText:
                newSurvivedValue = row[line] * c[line] + newSurvivedValue

            row['Now'] = newSurvivedValue
            totalNow = totalNow + self.evaluateSurive(row['Now'], row['Survived'])
            totalBest = totalBest + self.evaluateSurive(row['Best'], row['Survived'])
            if abs(row['Best'] - row['Survived']) < 0.5:
                rightBest = rightBest + 1
            if abs(row['Now'] - row['Survived']) < 0.5:
                rightNow = rightNow + 1
        return totalNow<=totalBest and rightNow>rightBest

    def howGood(self, alignedData):
        # Was this better calc
        right = 0
        for row in alignedData:
            if abs(row['Best'] - row['Survived']) < 0.5:
                right = right + 1

        return right

    def leMorte(self, alignedData):
        # How did we do with our prediction based on our data.

        rightAlive = 0
        rightDead = 0
        aliveCount = 0
        deadCount = 0
        for row in alignedData:

            if row['Survived'] == 0 and row['Best']<0.5:
                rightDead = rightDead + 1
            else:
                if row['Survived'] == 1 and row['Best']>= 0.5:
                    rightAlive = rightAlive + 1
                #else:
                #    print("ERROR:", row)
            if row['Survived'] == 1:
                aliveCount = aliveCount + 1
            else:
                deadCount = deadCount + 1

        print("Right Alive:", rightAlive," /", aliveCount)
        print("Right Dead:", rightDead, " / ", deadCount)
        print("Total:", len(alignedData))

class CoefData:
    """
    Take the supplied "Titanic" data and align it into a new dictionary
    """
    
    randomSelect = 0
    coefUndo = 0.0

    def __init__(self):
        self.coefList = {

            coefText[0]: randomChange(),
            coefText[1]: randomChange(),
            coefText[2]: randomChange(),
            coefText[3]: randomChange(),
            coefText[4]: randomChange(),
            coefText[5]: randomChange()
        }

    def randomCoef(self):

        self.randomSelect = np.random.randint(5)
        self.coefUndo = self.coefList[coefText[self.randomSelect]]
        self.coefList[coefText[self.randomSelect]] = randomChange()

    def unDoCoef(self):

        self.coefList[coefText[self.randomSelect]] = self.coefUndo

    def slamCoef(self, slamIt, slamValue):

        self.coefUndo = self.coefList[coefText[slamIt]]
        self.coefList[coefText[slamIt]] = slamValue

    def randomAll(self):

        self.coefList = {

            coefText[0]: randomChange(),
            coefText[1]: randomChange(),
            coefText[2]: randomChange(),
            coefText[3]: randomChange(),
            coefText[4]: randomChange(),
            coefText[5]: randomChange()
        }

class FinalAnswer:
    """
    Write final answer to file
    """

    def __init__(self, finalList, coefList, fileName='finalnewanswer.csv'):

        try:
            finalFile = open(fileName, encoding='utf-8', mode='w')

        except IOError:
            print("Can't open file")
            sys.exit(1)

        finalFile.write('PassengerId,Survived\n')

        for row in finalList:
            newSurvivedValue = 0.0
            for i in range(0, len(coefText)):
                newSurvivedValue = row[coefText[i]] * coefList.coefList[coefText[i]] + newSurvivedValue
            if newSurvivedValue > 0.5:
                lineValue = row['PassengerId'] + ',' + '1' + '\n'
            else:
                lineValue = row['PassengerId'] + ',' + '0' + '\n'
            finalFile.write(lineValue)

        finalFile.close()

def runMonte(coefList, alignedTrain, handleData):
    """
    Run The Monte Carlo plan
    """

    print("Training...")

    handleData.calcNow(alignedTrain, coefList.coefList)
    handleData.acceptNow(alignedTrain)

    epochCount = 1001 #loop -1 this value

    print("Running Epochs" )

    right = 0.0
    # Run process to get initial value
    for i in range(1, epochCount):
        coefList.randomCoef()
        if handleData.calcNow(alignedTrain, coefList.coefList):
            handleData.acceptNow(alignedTrain)
            right = handleData.howGood(alignedTrain)

        else:
            # Set it back
            coefList.unDoCoef()

    handleData.leMorte(alignedTrain)

    return handleData.howGood(alignedTrain)

# =============================================================


version = "2.01"
program = "Monte"

testMode = False

# =============================================================
# Main program begins here



print(program, " Version ", version)

print("Attempting to load CSV files....")

trainData = LoadCSV('train.csv')

testData = LoadCSV('test.csv')

print("Data loaded...aligning data")

coefList = CoefData()
lastCoef = CoefData()

alignedTest = []
alignedTrain = []
handleData = AlignData()
cabinNot = []
cabinSurvive = []

handleData.loadData(trainData.ourList, alignedTrain, cabinNot, cabinSurvive)
handleData.loadData(testData.ourList, alignedTest, cabinNot, cabinSurvive, False)

LastRunRight = -1.0
right = 0.0

for runs in range(0, 1):
    coefList.randomAll()
    right = runMonte(coefList, alignedTrain, handleData)
    if right > LastRunRight:
        for i in range(0, len(coefText)):
            lastCoef.slamCoef(i, coefList.coefList[coefText[i]])
        LastRunRight = right

percentRight = right / 891 * 100
print("Number Right:", right, "Percent:", percentRight)
print(lastCoef.coefList)


print("End of Training")

print("Final answer writing out...")

finalRun = FinalAnswer(alignedTest, lastCoef, 'final new answer.csv')

print("End of Line...")
