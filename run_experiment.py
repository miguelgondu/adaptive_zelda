"""
This script 
1. presents the player with three levels that serve as
   onboarding for the player.
2. Randomly chooses to serve the bayesian or baseline
   experiments. 
"""
import numpy as np
import uuid
import time
import json
from utils.gvgai import deploy_human
from simpler_itae_experiment import itae_experiment
from baseline_experiment import baseline_experiment
from visualize_experiment import plot_exp_id

def onboarding(exp_id):
    """
    Plays three simple levels.
    """
    level_1 = [
        ["w", "w", "w", "w", "w", "w", "w"],
        ["w", "A", ".", "+", ".", "g", "w"],
        ["w", "w", "w", "w", "w", "w", "w"]
    ]

    level_2 = [
        ["w", "w", "w", "w", "w"],
        ["w", "w", ".", "g", "w"],
        ["w", ".", "3", ".", "w"],
        ["w", ".", "w", "w", "w"],
        ["w", ".", ".", ".", "w"],
        ["w", "w", "+", ".", "w"],
        ["w", "w", "w", ".", "w"],
        ["w", "A", ".", ".", "w"],
        ["w", "w", "w", "w", "w"]
    ]

    level_3 = [
        ["w", "w", "w", "w", "w", "w", "w"],
        ["w", "2", ".", ".", ".", "g", "w"],
        ["w", ".", "w", ".", ".", "w", "w"],
        ["w", ".", "A", "w", ".", ".", "w"],
        ["w", ".", ".", "w", ".", ".", "w"],
        ["w", ".", ".", "w", "1", ".", "w"],
        ["w", ".", ".", "w", ".", ".", "w"],
        ["w", "+", ".", "3", ".", ".", "w"],
        ["w", ".", ".", ".", ".", ".", "w"],
        ["w", "w", "w", "w", "w", "w", "w"]
    ]

    for i, level in enumerate([level_1, level_2, level_3]):
        # play level.
        deploy_human(level, f"{exp_id}_onboarding_{i}")

if __name__ == "__main__":
    path = "./data/generations/custom_posterior.json"
    max_iterations = 10
    goal = 200
    exp_id = str(uuid.uuid4()) + f"_goal_{goal}_iterations_{max_iterations}"
    # distance_to_goal = 10

    experiment_metadata = {
        "prior_path": path,
        "max_iterations": max_iterations,
        "goal": goal,
        "time": time.time()
    }

    print("Onboarding:")
    # _ = input("Press enter to start with the experiment")
    onboarding(exp_id)
    if np.random.randint(0, 2) == 0:
        experiment_metadata["order"] = "baseline->bayesian"
        
        print("Running the first experiment")
        # _ = input("Press enter to start with the first 10 levels")
        baseline_experiment(path, max_iterations, goal, exp_id)

        print("Running the second experiment")
        # _ = input("Press enter to start with the second 10 levels")
        itae_experiment(path, max_iterations, goal, exp_id)
    else:
        experiment_metadata["order"] = "bayesian->baseline"
        
        print("Running the first experiment")
        # _ = input("Press enter to start with the first 10 levels")
        itae_experiment(path, max_iterations, goal, exp_id)

        print("Running the second experiment")
        # _ = input("Press enter to start with the second 10 levels")
        baseline_experiment(path, max_iterations, goal, exp_id)

    print(experiment_metadata)
    with open(f"./data/experiment_results/{exp_id}_metadata.json", "w") as fp:
        json.dump(experiment_metadata, fp)
    
    # Plotting the comparison between baseline and itae.
    plot_exp_id(exp_id)