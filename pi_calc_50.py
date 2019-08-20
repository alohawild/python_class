#!/usr/bin/env python
"""
    Copyright 2019 by Michael Wild (alohawild)
    
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

We are using the numpy library to get some more exact results. Also numpy is not slow.ArithmeticError

We also added the mpmath library to use 50 digits of floating point accuracy. The calls to mpf force the results into 
50 digits of floating point accuracy.

"""
__author__ = 'michaelwild'

from mpmath import mp,mpf  # This is the floating point accuracy set to 50 for this example
mp.dps = 50

import numpy as np
import math

from time import process_time

def howFar(x,y):

# The distance is always from the center at 0,0 and thus we can dispense with the subtractions of points.
    
    distance = mpf((x * x) + (y * y))  # Orginally I used an exponent as this more appealing to me, but it was less accurate!
    return mpf(np.sqrt(distance))  # From what I have read this is more accurate and faster version

def runtime(start):

# I use this a lot so I have a routine.

    return process_time() - start

print("Pi Example program")

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
