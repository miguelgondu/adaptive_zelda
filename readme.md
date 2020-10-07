# Adaptive Zelda Experiment

In this experiment, we are testing a system for adjusting the difficulty of games automatically. You will be presented 23 levels in a Zelda-like game (compiled using the [GVGAI framework](http://www.gvgai.net/index.php)). We will try to find a level that, for you, takes 150 in-game steps to solve. If you want to know more about the technique and about the experiment, [click here](). 

Do you want to participate?, follow these instructions:

## Prerequisites

### Having Java installed

Make sure you have Java 13 (or older) installed. You can try this by running `java --version` (or `java -version`, depending on your OS).

```
> java --version
java 13 2019-09-17
Java(TM) SE Runtime Environment (build 13+33)
Java HotSpot(TM) 64-Bit Server VM (build 13+33, mixed mode, sharing)
```

**Anything above 13 should work just fine**. Here are a couple tutorials for installing openJDK 13 in case you don't have it:
- [on Windows](https://java.tutorials24x7.com/blog/how-to-install-openjdk-13-on-windows)
- [on Ubuntu](http://techoral.com/blog/java/install-openjdk-13-ubuntu.html)
- [on Mac](http://techoral.com/blog/java/install-openjdk-13-on-mac.html)

### Having conda (or a way to create Python environments)

If you have conda, you can create a small environment dedicated to this project. Download and install Anaconda from [this link](https://www.anaconda.com/products/individual).

Otherwise, **you should be good as long as you are running Python >=3.6 and run the `pip install ...` command below**.


## Setting up the environment

Start by cloning this repository and going into that folder:
```
git clone https://github.itu.dk/migd/adaptive_zelda_simple.git
cd adaptive_zelda_simple
```

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

To start the experiment, run:

```
python run_experiment.py
```

## After the experiment

Could you please zip the `data` folder and send it to me via email to `migd@itu.dk`? Feel free to take a look at `./data/plots` to see some cool plots.
