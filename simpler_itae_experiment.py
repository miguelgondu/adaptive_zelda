import json
import numpy as np
from zelda_experiment import ZeldaExperiment
from utils.gvgai import deploy_human, features_to_array

def itae_experiment(path, max_iterations, goal, exp_id, verbose=False, model_parameters=None):
    behaviors = []
    times = []
    data = []

    if model_parameters is None:
        mp = {
            "linear": True,
            "exp": False,
            "rbf": True,
            "noise": True,
            "acquisition": "ucb",
            "kappa": 0.03
        }
    else:
        mp = model_parameters

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
            verbose=verbose,
            model_parameters=mp
        )

        try:
            ze.plot_projected(f"./data/plots/{exp_id}_iteration_{it}.jpg")
            ze.save_3D_plot(f"./data/plots/plot_3D_{exp_id}_iteration_{it}")
            ze.save_3D_plot(f"./data/plots/plot_3D_no_sigma_{exp_id}_iteration_{it}", plot_sigma=False)
        except Exception as e:
            if verbose:
                print(f"Couldn't plot. Got this exception: {e} ({type(e)})")

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
        print("Plotting last image:")

    ze = ZeldaExperiment(
            path,
            goal,
            behaviors=behaviors,
            times=times, # takes time, not log(time)
            projection=["leniency", "reachability"],
            verbose=verbose,
            model_parameters=mp
    )

    try:
        ze.plot_projected(f"./data/plots/{exp_id}_iteration_{max_iterations}.jpg")
        ze.save_3D_plot(f"./data/plots/plot_3D_{exp_id}_iteration_{max_iterations}")
        ze.save_3D_plot(f"./data/plots/plot_3D_no_sigma_{exp_id}_iteration_{max_iterations}", plot_sigma=False)
    except Exception as e:
        print(f"Couldn't plot. Got this exception: {e} ({type(e)})")
    
    if verbose:    
        print("Saving the data")

    with open(f"./data/experiment_results/{exp_id}_itae.json", "w") as fp:
        json.dump(data, fp)

if __name__ == "__main__":
    itae_experiment(
        "./data/generations/custom_posterior.json",
        10,
        150,
        "testing_model_parameters"
    )
