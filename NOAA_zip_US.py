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
This is an example program of reading data from the Internet without any access control.

This program using the NOAA SDK library to get the forcast from NOAA.


"""
__author__ = 'michaelwild'

from time import process_time
import datetime
from noaa_sdk import noaa

def runtime(start):

# I use this a lot so I have a routine.

    return process_time() - start

print("NOAA Weather example")

begin_time = process_time()

zipcode = '97005'  # Nike WHQ

n = noaa.NOAA()

print("Getting forcast for zipcode:", zipcode)
noaa_time = process_time()
res = n.get_forecasts(zipcode, 'US', True)

print("NOAA response time: ", runtime(noaa_time) )

# Print out just time range and basic forecast
for i in res:
    starttime   = datetime.datetime.strptime(i.get("startTime"), "%Y-%m-%dT%H:%M:%S%z")
    endtime     = datetime.datetime.strptime(i.get("endTime"), "%Y-%m-%dT%H:%M:%S%z")
    forecast     = i.get("shortForecast")

    print(starttime.strftime("%Y-%m-%d %H:%M:%S"),"-",endtime.strftime("%Y-%m-%d %H:%M:%S %Z"),": ", forecast)

finish = runtime(begin_time)
print("Run time:", finish)
