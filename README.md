# The Utility of Explainable AI in Ad Hoc Human-Machine Teaming

This repository contains code for our Minecraft environment within Microsoft Project Malmo. 
Our original repository is based on the [Malmo Collaborative AI Challenge](https://github.com/microsoft/malmo-challenge/blob/master/ai_challenge/pig_chase/).
We have extended the Microsoft Malmo Collaborative AI Challenge environment to allow for 
- Allow human users to play with standard Minecraft controls
- Robot reasoning within continuous state and action space


### File Descriptions
Within this repo, we present 6 key files.
- experiment_main.py - Runfile that will launch 
  threads for the robot agent and another thread for
  monitoring human behavior.
- agent.py - contains macro policy and image display dynamics
- malmo.py - contains low-level control for macro actions and resource tracking
- human_agent.py - tracking for human resources
- experiment_experiment.py - Contains PigChaseEnvironment class
- experiment_world.xml - contains the world and world dynamics

- We also attach two folders that contain the images that represent
the cobot's hierarchical policy.


### Run Instructions 

Prior to running the python programs to run the simulation, Minecraft instances with the 
Malmo mod must be launched:

```shell
cd ~/Downloads/Malmo-0.37..../Minecraft
./launchClient.sh
$ Launch new terminal
./launchClient.sh
```
### Experiment Launch 
```python experiment_main.py```


### Solo build
```python solo_build_main.py```


### Notes
Please note that the environment can be easily modified by generating a new .xml file. The current pygame display
is fitted for a Dell 27 inch wide screen monitor. Different displays may require modifying the size of the display.
Feel free to email at rpaleja3@gatech.edu or post an issue if you have trouble.

![Sample Gameplay Gif](https://github.com/CORE-Robotics-Lab/Utility-of-Explainable-AI-NeurIPS2021/blob/main/sample_gameplay.gif)