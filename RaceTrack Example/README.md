# Race Track 
This code solves the racetrack programming assignment given to us in  
Reinforcement Learning lectures at IIITB. You can read the problem  
statement in `docs/Problem Statement.pdf`.

## Code Structure:
The `data` directory contains two subdirectories and a file called   
`Race_track.xlsx`. A brief overview about each:
- **initialisation/** - This subdirectory contains the matrices (values)
    that are used to initialise various variables such as policy, data,
    returns, and rewards, for both on policy and off policy monte carlo 
    methods.
- **saved/** - This subdirectory will contain two another subdirectories
    named - `on` and `off`, where each will contain the trained policy,
    and other results after training the RL agent.
- **Race_track.xlsx** - This contains a sample visualization of the 
    racetrack given to us along with the problem statement in an excel
    format.

During the execution of RL training, logs will be generated and saved in  
the `log/` directory. It will contain information like, where are we at  
current timestamp, on which episode we are on, etc. The naming convention  
of a log file is chosen as a `timestamp` to be able to distinguish it easily. 

The `notebooks/` directory contains various notebooks that I used during the   
development of this project to do some exploratory analysis, write adhoc  
code and so on.

The `run_results/` directory contains the sample visualization of  
racetrack along with the rewards vs number of episodes plot for both  
on policy and off policy monte carlo methods.

The entire project is confirable from `config.py` file wherein you can  
set for how many episodes you want to run the code, where do you want to  
save the logs, and where do you want to save the results.  

The `main.py`, `race_track.py`, and `utility.py` contains the code  
developed as a part of this project. 

The `requirements.txt` contains all the dependencies required for running  
this project. You can install them at once using the following command:
> pip install -r requirements.txt

And now, you can run the code. 

## How to Run the Code?
To run the code, you need to run the `main.py` file. There are several  
arguments that can be passed while running the code. To list all the  
arguments, run:
> python main.py -h


This supports three arguments:
1. `run_on_policy`: To run On Policy Monte Carlo Control code
2. `run_off_policy`: To run Off Policy Monte Carlo Control code
3. `run_on_and_off_policy`: To run On and Off Policy Monte Carlo code

For example, to run off policy monte carlo control code, execute the  
following command:
> python main.py --run_off_policy=True

Similarly, you can execute the following commands:
> python main.py --run_on_policy=True  
> python main.py --run_on_and_off_policy=True


## Things to look out for 
- The running time for the code depends on the length of initial  
sample episode. Sometimes this comes out to be huge that affects   
the running time i.e. you feel nothing's happening and the log  
file is not updating. Well, it's not the case. If you encounter  
any such issue, terminate and run the program again. It will work  
with less than 5 tries.

- The length of episodes samples during the on policy monte carlo  
method is very huge. Hence, this will take a lot of time (couple of  
hours). Hence, it is advisable to run the off policy monte carlo code  
to quickly see the results and to understand the results.
