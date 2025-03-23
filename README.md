
This repo provides creates a reinforcement learning based agent for https://gym.openai.com/envs/LunarLander-v2/.


This was my project for CSEP 573A class in University of Washington. 
## Setup
- create a new conda environment.
- mac
  - install brew on macbook
  - brew install swig

- linux

    sudo apt update
    sudo apt install swig

- install dependeics

      pip install gym[box2d]==0.23.0 torch numpy matplotlib pygame

# Run heuristic model

    python lunar_lander.py

# Run reinforcement learning models 

    python trainLander.py
