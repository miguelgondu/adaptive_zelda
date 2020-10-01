from pathlib import Path
import numpy as np
import pandas as pd
import random
import time
import subprocess
import os
import json
import multiprocessing as mp

from .pcg import print_to_text, create_random_level
from .pcg import expand, shrink
from .pcg import add_walls, remove_walls
from .pcg import add_enemies, remove_enemies
from .pcg import compute_features


def random_solution():
    """
    This function is supposed to create a random genotype.

    Right now: it's returning as genotype a dict containing
    a summary of the level.

    The future: returning a string of the level. (Or something that's storable
    in jsons, maybe a matrix).
    """

    width = np.random.randint(3, 26)
    height = np.random.randint(3, 26)
    indicator = min(width, height)
    placeable_positions = (width - 2)*(height - 2)

    enemies = np.random.randint(indicator // 2, indicator)
    if indicator > 3:
        walls = np.random.randint(indicator // 2, indicator)
    else:
        walls = 0

    while placeable_positions < 3 + enemies + walls:
        if np.random.randint(0, 11) % 2 == 0:
            width += 1
        else:
            height += 1
        placeable_positions = (width - 2)*(height - 2)

    for i in range(5):
        try:
            level = create_random_level(
                width,
                height,
                0,
                enemies,
                walls
            )
        except ValueError as e:
            specs = (width,height,0,enemies,walls)
            print(f"Couldn't create level with specifications {specs}.")
            print(f"Got exception {e}.")
            print(f"Attempt {i+1}/5.")
            level = None

    if level is None:
        print("Trying again with new random speficifications")
        return random_solution()

    return level.tolist() # Will this work?

def binary_decision():
    nonce = random.random()
    return nonce <= 0.5

def random_variation(x):
    """
    Assuming x is a level.

    We're going to implement the following modifications:
    1. expanding width or height.
    2. adding or removing enemies or walls.
    """

    level = np.array(x)

    decider_height = np.random.randint(0,3)
    decider_width = np.random.randint(0,3)

    if decider_height == 0:
        level = expand(level, axis=0)
    elif decider_height == 1:
        level = shrink(level, axis=0)

    if decider_width == 0:
        level = expand(level, axis=1)
    elif decider_width == 1:
        level = shrink(level, axis=1)
    
    enemy_mod = np.random.randint(-2, 3)
    wall_mod = np.random.randint(-2, 3)

    if enemy_mod > 0:
        add_enemies(level, enemy_mod)
    elif enemy_mod < 0:
        remove_enemies(level, -enemy_mod)

    if wall_mod > 0:
        add_walls(level, wall_mod)
    elif wall_mod < 0:
        remove_walls(level, -wall_mod)

    return level.tolist()

def random_selection(X):
    return random.choice(X)

def compute_performance_human(result):
    """
    This function computes the performance
    (i.e. amount of steps) for the human 
    experiment, if win == 0, then it returns
    1000.
    """
    if result["win"] == 1:
        return np.log(result["steps"])
    elif result["win"] == 0:
        return np.log(2000)
    else:
        raise ValueError(f"Weird value in result at win: {result}")

def compute_performance(results, schema=None):
    '''
    This function computes how many steps it takes
    for the agent to solve the level on average.
    '''
    steps = np.array(results["steps"])
    wins = np.array(results["wins"])

    # Override the ones in which the avatar lost
    if schema == "CAP":
        steps[wins == 0] = 2000
    elif schema == "IGNORE":
        steps = steps[wins == 1]
    else:
        # We will just compute the mean
        # without discriminating
        pass

    if schema == "IGNORE":
        if len(steps) == 0:
            return None

    print(f"results: {results}")
    print(f"steps: {steps}")
    return np.log(np.mean(steps))

def run_level(path_to_gvgai, path_to_vgdl, path_to_level, agent, record_path, seed, results):
    pid = os.getpid()
    os.chdir(path_to_gvgai)

    java = subprocess.Popen([
        "java",
        "tracks.singlePlayer.Simulate",
        path_to_vgdl,
        path_to_level,
        agent,
        record_path.replace(".txt", f"_seed_{seed}.txt"),
        str(seed)
    ], stdout=subprocess.PIPE)
    try:
        results_ = java.stdout.readline().decode("utf8")
        results_ = json.loads(results_)
        results[(pid, seed)] = results_
        java.kill()    
    except json.decoder.JSONDecodeError as e:
        print(f"Couldn't decode the results. {e}")
        java.kill()

def aggregate_results(experiment_id, x, agent, game, original_seed, results):
    agg_results = {
        "experiment_id": experiment_id,
        "agent": agent,
        "game": game,
        "original_seed": original_seed,
        "level": print_to_text(x),
        "seeds": [],
        "scores": [],
        "wins": [],
        "steps": []
    }
    for key, _results in results.items():
        _, seed = key
        agg_results["seeds"].append(seed)
        agg_results["scores"].append(_results["score"])
        agg_results["wins"].append(_results["win"])
        agg_results["steps"].append(_results["steps"])

    return agg_results

def load_df_from_generation(path):
    """
    This function loads the results of a
    MAP-Elites generation into a dataframe
    that can be used for the simpler B.O.
    implementation that can be found in
    zelda_experiment.py
    """
    with open(path) as fp:
        gen = json.load(fp)

    vals = [
        v for v in gen.values() if v["solution"] is not None
    ]

    rows = []
    for v in vals:
        rows.append({
            "leniency": v["centroid"][0],
            "reachability": v["centroid"][1],
            "space coverage": v["centroid"][2],
            "performance": v["performance"],
            "level": print_to_text(v["solution"])
        })

    df = pd.DataFrame(rows, columns=rows[-1].keys())
    return df

def features_to_array(feat_dict):
    keys = list(feat_dict.keys())
    keys.sort()
    return [feat_dict[f] for f in keys]
