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

## Playing the game

You will play 23 levels. Use the arrow keys to move around and the spacebar to use your sword. Run

```
python run_experiment.py
