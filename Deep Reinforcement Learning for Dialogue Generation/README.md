# Deep Reinforcement Learning for Dialogue Generation
This repository implements and improves the implementation of the paper  
[Deep Reinforcement Learning for Dialogue Generation](https://arxiv.org/abs/1606.01541). Compared to the  
original paper, the dataset is changed to [Daily Dialogue Dataset](https://paperswithcode.com/dataset/dailydialog). The   
baseline code is taken from [this](https://github.com/Ls-Dai/Deep-Reinforcement-Learning-for-Dialogue-Generation-in-PyTorch) repository.

Here we first run the baseline model, and then try to solve two high level  
objectives given in the [problem statement](docs/Chatbot%20Project.pdf).
  
### Create a virtual environment:
To run this project, we require `python 3.8`.  <br>  
It is always a better idea to create a virtual environment and install  
all the dependencies in the same. To create the virtual environment,  
we will use `virtualenv` which can be installed via running:
> sudo apt install python3-virtualenv

Once `virtualenv` is installed you can install all the dependencies  
via running:
> pip install -r requirements.txt


### Implement Baseline model
Once we have the virtual environment setup, we can start implementing  
the baseline model as described as it is in the paper but with a  
different dataset.  <br>  
As a first step, let's train the Seq2Seq model via running:
> python train.py

This will run for `1,000 iterations` and save the results in `data/save/cb_model`   
directory.

Once the `Seq2Seq` model is trained, let's train the deep reinforcement  
learning model via running:
> python rl.py

This will run for `10,000 iterations` and save the results in `data/save/RL_model_seq`  
directory.
