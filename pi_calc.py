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
__copyright__ = "Copyright (C) 2018 Michael Wild"
__license__ = "Apache License, Version 2.0"
__version__ = "0.0.3"
__credits__ = ["Michael Wild"]
__maintainer__ = "Michael Wild"
__email__ = "alohawild@mac.com"
__status__ = "Initial"


import os
import sys
import numpy as np
import math

from time import process_time

def howFar(x,y):
    
    distance = (x * x) + (y * y)
    return np.sqrt(distance)

def runtime(start):

    return process_time() - start
    
# =============================================================

program = "Pi Calc"

piLoop = 1000000
inCircle = 0

begin_time = process_time()

# =============================================================
# Main program begins here

print(program)
print("Version ", __version__, " ", __copyright__, " ", __license__)
print("Running on ", sys.version)


for i in range(1, piLoop):
    x = (np.random.uniform()* 2) -1
    y = (np.random.uniform() * 2) -1

    if (howFar(x,y)<1.0) :
        inCircle = inCircle + 1
piGuess = 4.0* (inCircle / piLoop)

piError = math.pi - piGuess

print("Loops:", piLoop)
print("Calculated value: ",piGuess, "Error: ", piError)

finish = runtime(begin_time)
print("Run time:", finish)
