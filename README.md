# Python Class
This is a small set of programs and set-up for a class on Python. Please feel free to use them.
## AWS setup
The class uses a Cloud9 AWS system. Instructions for this is available on AWS websites. Only a basic small system is needed.
Please see [AWS](https://aws.amazon.com). Please make sure that your root AWS account is not used when running Cloud9! Just setup a new Cloud9 small system. 
## Use Python3!
Once you have the system built, use preference option for the new Cloud9 system to set Python to use Python3. Do not try to use Python2 for these examples.
## Update Cloud9
Once you have the enviroment running, some code is needed. First run `git clone https://github.com/alohawild/python_class/` to get a copy of the programs.
### Get Conda
Conda is loaded from the terminal under the window in Cloud9. Remember Cloud9 is Linux.
```
wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh
chmod a+x Miniconda3-latest-Linux-x86_64.sh
./Miniconda3-latest-Linux-x86_64.sh
```
#### Enter and yes and y
When running the last command it will install `conda` and it has many prompts. The answer is always positive. Please ensure you do this right. If the command needs to run again then use a `-u` to update.
#### Close everything running!
It is critical to close all the running windows so that he changes will take affect. The `conda` command will run if it is installed and all running instance are stopped.
### The install
Just run these commands in a new window, one at a time.
```
conda install numpy
conda install pandas
conda install scikit-learn
conda install -c conda-forge matplotlib
```
At this point the pi_calc.py program should run. 
### Install float point addition to allow us to use 50 digit float.
Use `pip install mpmath` to install support for arbitrary-precision floating-point arithmetic. This is from the mpmath.org and has been in use for years. 
### Install NOAA weather Python SDK
Use `pip install noaa-sdk` to install the NOAA SDK.

# Example programs
There are a set of programs that are available to learn Python. Two are easy. Two are amazingly hard. One is given as an example of a possible improvement.
## Pi
The first program `py_calc.py` uses the numpy library for random number. It uses a Monte Carlo process to estimate the mathamatical value of pi. The comments explain how the program works. To outline, one can draw a circe in a box and randomly pick points and check if the point is in the circle. The ratio of success and failures is a form of pi. It is interesting that the method produces a weak result. It also shows you how fast your server is!
### Exercise
Run the program and notice its answer. Change the program to run for more less interations. The program will not improve its results much. The conclusion is not that the program is faulty, but that it relies on Python math and that is not that percise. 

For you to see more I have built an additional program `py_calc_50.py` that is much slower and uses huge fifty digit floating point. It is an example of how you can make something more complex, but not make it run better! It also let me get out an other library!
## Weather
The second program `NOAA_zip_US.py` uses a wrapper for a call to NOAA to get the weather for a US zipcode. This is from https://pypi.org/project/noaa-sdk/.
The program has to deal with translating dates with timezone. Notice that there is only UTC+/- and not an actual timezone. 
From all my reading the suggesting is to get the timezone of the user from a browser and then covert to it. I thus left it alone.
### Exercise
Run the program. You can change it to other zip codes and see the results. Notice that there is no secuirty and one is directly hitting NOAA for the weather.
Putting this in a loop and repeating it over and over will likely get one in some trouble with NOAA.
# Kaggle
The next examples were used in the practise Kaggle contest for the Titanic. Data science may join contests to earn real money by solving real machine leraning problems described in Kaggle. This website provides all the tools to run contests. Usually Python is used to solve the problems. A Titanic practice is recommended. A leaderboard is kept running to generate excitement.
## Titanic
This is a better and more correct program, `BasicTitanic.py`. It uses `Pandas` to change values and is a very good example of using data frame and fixing issues in the data set. Dispair is how bad the error rate is--zero would be perfect.
### Exercise
The code is better but still a bit linear (what can I say--I use to code RPG2). It can be easily revised to create a file and improved. This program is a starting point for would be Python machine learning coding. Please enjoy it and maybe even submit to Kaggle! The lamda code is obscure but so Python. Again, enjoy. 
#### Issue with Cloud9
I was having trouble running it from the IDE and so I use a command to run it from a terminal. Remember to save the code first!
```
cd python_class
python3 BasicTitanic.py
```
## Scripting!
The true power of Python is now displayed in `GoodTitanic.py`. This is a more complete program for the Kaggle exercise on Titanic. You can see more complex use of data clean-up in the code. I also use classes to build everything like a true modern Python coder (remember I am over 50 and wrote BASIC on a Sinclair and so that is saying a lot!). The program reads the files, applies the alignment to the data, trains, and then predicts. All in the class. 
### The Power
I added an additional method to the class to run every classifier I know about and installed so we could see which one was the best. I can also change the Verbose parm to see all the data as it flows. This is where you see why Python is used for Machine Learning. It can even run different code in a loop. 
### Harvest
The code can easily be harvested and applied to other problems and the loop through all the classifiers used to determine the best choice. A more powerful program, and there are examples out in the Internet for this, could vary all the settings for each classifier and then use the best settings to produce the best results.
```
cd python_class
python3 GoodTitanic.py
```
# MicroPython and Express
The `amazing_fez.py' is for MicroPython based controller and specifically Adafruit Circuit Playground Express. This code is for a silly wearable of a MicroPython that lights up based on sound, uses a SparkFun command controllable display, and a wearable GPS module from Adafruit. I built this for the 2018 Makerfaire in Portland.
## The code
The Express has limited memory and I had to remove the comments (!) and most of the tracing prints to serial (!) to keep it running. It fits within the limits of MicroPython and the Express, just.
