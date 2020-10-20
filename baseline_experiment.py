"""
This script implements a baseline experiment
in which we present the user with levels noisly
selected from the prior.
"""
import json
import numpy as np
from scipy.spatial import cKDTree
from sklearn.preprocessing import MinMaxScaler

from utils.pcg import level_from_text
from utils.gvgai import deploy_human

from zelda_experiment import ZeldaExperiment

def baseline_experiment(path, max_iterations, goal, exp_id, verbose=False):
    # TODO: change this search to be in 2D space.
    experiment = {
        "path": path,
        "goal": goal,
        "exp_id": exp_id,
        "iterations": []
    }

    ze = ZeldaExperiment(
        path,
        goal,
        projection=["leniency", "reachability"]
    )
    prior = ze.prior

    centroids = prior.loc[:, ["leniency", "reachability"]].values

    scaler = MinMaxScaler()
    scaled_cs = scaler.fit_transform(centroids)

    # Construct a KDTree with these centroids
    kdtree = cKDTree(scaled_cs)

    # Add noise, and then query the point closest
    # and iterate.
    center = [0.5] * len(centroids[0])
    index = kdtree.query(center)[1]
    c_current = centroids[index]
    x_current = level_from_text(prior.loc[index, "level"])

    # print("Deploying level: ")
    # print(x_current)
    print(f"Deploying level 1/{max_iterations}", end="\r", flush=True)
    p_current, _ = deploy_human(x_current, exp_id + f"_baseline_0")
    p_current = round(np.exp(p_current)) # since deploy_human models in log
    o_current = - np.abs(p_current - goal)

    if verbose:
        print(f"First performance: {p_current}")

    # Saving the 0th iteration:
    experiment["iterations"].append(
        {
            "centroid": tuple(c_current.tolist()),
            "level": x_current,
            "performance": p_current,
            "objective": int(o_current),
            "became_new": None
        }
    )
    for i in range(max_iterations - 1):
        scaled_c = scaler.transform([c_current])
        scaled_c = scaled_c[0]

        # Compute new point
        new_point = scaled_c + np.random.normal(scale=0.05, size=scaled_c.size)

        # Convert it to the closest one in the grid
        index = kdtree.query(new_point)[1]
        new_point = centroids[index]

        # Deploy it and record
        x_new = level_from_text(prior.loc[index, "level"])
        print(f"Deploying level {i+2}/{max_iterations}", end="\r", flush=True)
        p_new, _ = deploy_human(x_new, exp_id + f"_baseline_{i+1}")
        p_new = round(np.exp(p_new)) # since deploy human models in log
        o_new = - np.abs(p_new - goal)

        if verbose:
            print(f"Performance: {p_new}")

        experiment["iterations"].append(
            {
                "centroid": tuple(new_point.tolist()),
                "level": x_new,
                "performance": p_new,
                "objective": int(o_new),
                "became_new": False # overwritten if true.
            }
        )

        # Compare and reassign
        escape_flag = np.random.random()
        if escape_flag < 0.95:
            # compute objective functions:
            if o_new > o_current:
                # new becomes current
                if verbose:
                    print("New center of exploration!")
                c_current = new_point
                p_current = p_new
                o_current = -np.abs(p_current - goal)
                x_current = x_new
                experiment["iterations"][-1]["became_new"] = True
        else:
            # We escape, and new becomes current
            if verbose:
                print("New center of exploration!")
                print("(but because of escaping)")
            c_current = new_point
            p_current = p_new
            o_current = -np.abs(p_current - goal)
            x_current = x_new
            experiment["iterations"][-1]["became_new"] = True

            print("-"*50)
    
    if verbose:
        print("Best level: ")
        print(np.array(x_current))

        print("Best performance:")
        print(p_current)

        print("Experiment:")
        print(experiment)

    with open(f"./data/experiment_results/{exp_id}_baseline.json", "w") as fp:
        json.dump(experiment, fp)

if __name__ == "__main__":
    baseline_experiment(
        "./data/generations/custom_posterior.json",
        10,
        200,
        "one_test"
    )
