# Race Track 

## Install Python:
This project is developed in `python 3.8`. 
To install the python version, run:  
> sudo apt install python3.8



## Install Pip:
To install `Pip`, run:
> sudo apt install python3-pip

To check the installed version of pip, run:
> pip --version

If you see an error from above command saying `pip not found`, try:
> pip3 --version



## Create a new environment:
It is always a good practice to create a new environment to build and  
run projects. This installs and keeps all the packages in an isolated  
place that make it more secure, efficient, and reliable.     
`virutalenv` is a popular tool for creating isolated virtual python  
environments.  
To install `virtualenv`, run:
> sudo pip install virtualenv

If you see an error from above command saying `pip not found`, try:
> sudo pip3 install virtualenv

To create a virtual environment named `test` with python version 3.8,  
run:
> virtualenv test --python=python3.8

This will create a directory named `test` in your current directory.  
To activate the environment, run:  
> source test/bin/activate

To deactivate the environment, run:
> deactivate



## Install required dependencies:
There is a file named `requirements.txt` that contains all the  
dependencies required for this project. To install them, first activate  
the environment and then run:
> pip install -r requirements.txt

This will take some time depending upon the network connectivity and   
system configuration, but will install all the dependencies required
for this project.  

If you come across the following error:  
`pip._vendor.urllib3.exceptions.ReadTimeoutError: HTTPSConnectionPool(host='files.pythonhosted.org', port=443): Read timed out.`  
Use `--default-timeout=100` with the installation:  
> pip install -r requirements.txt --default-timeout=100  

You can check the list of all the installed dependencies by running:
> pip list

This will list all the dependencies installed with their versions.
