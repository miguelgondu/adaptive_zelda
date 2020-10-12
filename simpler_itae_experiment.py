import json
import numpy as np
from zelda_experiment import ZeldaExperiment
from utils.gvgai import deploy_human, features_to_array

def itae_experiment(path, max_iterations, goal, exp_id, verbose=False):
    behaviors = []
    times = []
    data = []

    for it in range(max_iterations):
        if verbose:
            print("="*30 + f" Iteration {it} " + "="*30)
            print("Fitting with:")
            print(f"Behaviors: {behaviors}")
            print(f"times: {times}")
        ze = ZeldaExperiment(
            path,
            goal,
            behaviors=behaviors,
            times=times, # takes time, not log(time)
            projection=["leniency", "reachability"],
            verbose=verbose
        )

        ze.plot_projected(f"./data/plots/{exp_id}_iteration_{it}.jpg")
        ze.save_3D_plot(f"./data/plots/plot_3D_{exp_id}_iteration_{it}")
        ze.save_3D_plot(f"./data/plots/plot_3D_no_sigma_{exp_id}_iteration_{it}", plot_sigma=False)

        level = ze.next_level()
        if verbose:
            print("Deploying level: ")
            print(level)
        print(f"Deploying level {it+1}/{max_iterations}", end="\r", flush=True)
        p, beh = deploy_human(level, exp_id + f"_itae_{it}") # returns log(steps)

        # convert beh to a list
        beh = features_to_array(beh)
        time = round(np.exp(p))

        if verbose:
            print(f"Time it took: {time}. Behavior: {beh}")

        behaviors.append(beh)
        times.append(time)

        # Saving the data for json dumping
        data.append({
            "iteration": it,
            "level": level,
            "behavior": beh,
            "time": time
        })

    if verbose:    
        print("Saving the data")

    with open(f"./data/experiment_results/{exp_id}_itae.json", "w") as fp:
        json.dump(data, fp)

if __name__ == "__main__":
    itae_experiment(
        "./data/generations/custom_posterior.json",
        10,
        200,
        "one_test"
    )
