# Python Class
This is a small set of programs and set-up for a class on Python.
## AWS setup
The class uses a Cloud9 AWS system. Instructions for this is available on AWS websites. Only a basic small system is needed.
Please see [AWS](https://aws.amazon.com). Please make sure that your root AWS account is not used when running Cloud9!
## Update Cloud9
Once you have the enviroment running, code is needed. First run `git clone https://github.com/alohawild/python_class/` to get a copy of the programs.
### Get Conda
Conda is loaded from the terminal under the window in Cloud9.
```
wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh
chmod a+x Miniconda3-latest-Linux-x86_64.sh
./Miniconda3-latest-Linux-x86_64.sh
```
It is critical to close all the running windows so that he changes will take affect. The `conda` command will run if it is installed and all running instance are stopped.
```
conda install numpy
conda install pandas
conda install scikit-learn
```
At this point the pi_calc.py program should run. 
### Install Google Finances
Use 'pip install googlefinance.client' to install Google Finance


