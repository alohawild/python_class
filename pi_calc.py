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

============================================================================================================
This is a pi calculation process using Monte Carlo or simple random process.

Imagine a circle in a box. The edges just touch the box. Lets make it a circle of radius 1.

Thus any point on or within the cirlce are 1 unit or less from the center.

It should be possible to randomly select points in the box and determine if the point is in the circle or not.
The ratio of points in the box over the number of randomly selected should be 1/4*Pi

"""
__author__ = 'michaelwild'

from mpmath import mp,mpf
mp.dps = 50

import os
import sys
import numpy as np
import math
#import random
from time import process_time

def howFar(x,y):
    
    distance = mpf((x * x) + (y * y)) 
    return mpf(np.sqrt(distance))

def runtime(start):

    return process_time() - start

piLoop = 1000000
inCircle = 0

begin_time = process_time()

for i in range(1, piLoop):
    x = mpf(np.random.uniform()* 2) -1
    y = mpf(np.random.uniform() * 2) -1

    if (howFar(x,y)<1.0) :
        inCircle = inCircle + 1
piGuess = mpf(4.0* mpf(inCircle / piLoop))

piError = math.pi - piGuess

print("Loops:", piLoop)
print("Calculated value: ",piGuess, "Error: ", piError)

finish = runtime(begin_time)
print("Run time:", finish)
