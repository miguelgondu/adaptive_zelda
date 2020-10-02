# Adaptive Zelda Experiment

Hello, everyone.

Could I ask you to run this piece of code for an experiment?

## Prerequisites

1. Having Java installed.
2. A way of creating Python environments (e.g. Anaconda).

## Setting up the environment

Create a new conda environment of python:
```
conda create -n "zelda" python=3.7
```

start it and install some of the prerequisites
```
conda activate zelda
pip install scikit-learn scipy numpy matplotlib pandas
```

Also, create the folders for saving the data by running
```
chmod +x ./create_data_folders.sh
./create_data_folders.sh
```

## Playing the game

You will play 23 levels.

- The first 3 levels will allow you to familiarize yourself with the mechanics of the game. You can use the **arrow keys** to move around, and press **spacebar** to use your sword and to start the levels. **The goal is to grab the key and proceed to the door**
- The next 10 levels will correspond to *one* way of adjusting difficulty, and the remainder 10 are *another* way. We are comparing our approach with a baseline, and you'll be served both algorithms at random.

More details on the mechanics:
- The enemies will only walk if you walk.
- You **don't** have to kill all enemies in order to complete the level.
- If you die in a level, you will play it again until a maximum amount of in-game steps has been registered.

Use the arrow keys to move around and the spacebar to use your sword. You can also start the levels by pressin Run

```
python run_experiment.py
```

## After the experiment

Could you please zip the `data` folder and send it to me via email to `migd@itu.dk`? Feel free to take a look at `./data/plots` to see some cool plots.
