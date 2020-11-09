import json
import glob
import numpy as np
import matplotlib.pyplot as plt

from zelda_experiment import ZeldaExperiment
# from utils.visualize import plot_generations

def plot_exp_id(exp_id, goal=150):
    # Gathering the baseline
    with open(f"./data/experiment_results/{exp_id}_baseline.json") as fp:
        baseline = json.load(fp)

    goal = baseline["goal"]
    iterations = baseline["iterations"]

    performances_baseline = np.array([
        res["performance"] for res in iterations
    ])

    # Gathering the ITAE experiments
    ## Everything is in the metadatas.

    with open(f"./data/experiment_results/{exp_id}_itae.json") as fp:
        itae_exp = json.load(fp)
    
    # print(itae_exp)
    performances_itae = [
        v["time"] for v in itae_exp
    ]
    performances_itae = np.array(performances_itae)

    # print(performances_baseline)
    # print(performances_itae)

    _, ax = plt.subplots(1, 1, figsize=(6, 6))
    ax.plot(performances_baseline, "or", label="Baseline")
    ax.plot(performances_itae, "xb", label="itae")
    ax.set_ylim((0, 400))
    ax.set_title("Time to solve per level")
    ax.set_ylabel("Time [in-game steps]")
    ax.set_xlabel("Iteration")
    ax.axhline(y=goal, c="k")
    ax.axhline(y=goal-50, c="b", linestyle="--")
    ax.axhline(y=goal+50, c="b", linestyle="--")
    ax.set_title(exp_id)
    ax.legend(loc=1)

    max_it = len(performances_itae)
    in_range_itae = performances_itae[performances_itae >= goal-50]
    in_range_itae = in_range_itae[in_range_itae <= goal+50]
    in_range_itae = len(in_range_itae)
    in_range_baseline = performances_baseline[performances_baseline >= goal-50]
    in_range_baseline = in_range_baseline[in_range_baseline <= goal+50]
    in_range_baseline = len(in_range_baseline)
    ax.text(0.05, 350, f"Itae: {in_range_itae}/{max_it}" + "\n" + f"Base: {in_range_baseline}/{max_it}")

    plt.savefig(f"./data/plots/{exp_id}.jpg")

def get_all_exp_ids():
    exp_files = glob.glob("./data/experiment_results/*_baseline.json")
    return [
        f.replace("_baseline.json", "").replace("./data/experiment_results/", "") for f in exp_files
    ]

def aggregate_by(ze, f1, f2):
    df = ze.prior
    mu, _ = ze._compute_mu_and_sigma()
    df["new_mean"] = mu
    grouped_df = df.groupby([f1, f2]).mean()
    points = np.array([
        list(p) for p in grouped_df.index
    ])
    colors = grouped_df["new_mean"]
    black_dot = ze.prior.loc[ze.indices_tested[-1], [f1, f2]].to_list()
    return points, colors, black_dot

def plot_scatter(ze, f1, f2, save_path, vmin=0, vmax=np.log(2000), features=None):
    points, colors, black_dot = aggregate_by(ze, f1, f2)
    goal = ze.goal
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(3*6, 6))
    fig.suptitle(f"t = {ze.times}")
    plot = ax1.scatter(
        points[:, 0],
        points[:, 1],
        c=colors,
        vmin=vmin,
        vmax=vmax,
        marker="s",
        s=20
    )
    ax1.plot(black_dot[0], black_dot[1], "ok", markersize=12)
    ax1.set_title("Time steps to solve")
    if features is not None:
        ax1.set_xlabel(features[0])
        ax1.set_ylabel(features[1])
    plt.colorbar(plot, ax=ax1)
    
    plot2 = ax2.scatter(
        points[:, 0],
        points[:, 1],
        vmin=-vmax,
        vmax=0,
        c=-np.abs(colors-np.log(goal)),
        marker="s",
        s=20
    )
    ax2.plot(black_dot[0], black_dot[1], "ok", markersize=12)
    ax2.set_title(f"Distance to {goal} time steps")
    if features is not None:
        ax2.set_xlabel(features[0])
        ax2.set_ylabel(features[1])
    plt.colorbar(plot2, ax=ax2)
    ax2.set_xlabel(f"{ze.gpr.kernel_}")

    mu, sigma = ze.gpr.predict(ze.domain, return_std=True)
    df = ze.prior.copy()
    df["pure_mu"] = mu
    colors_pure_mu = df.groupby([f1, f2]).mean()["pure_mu"]
    plot3 = ax3.scatter(
        points[:, 0],
        points[:, 1],
        c=colors_pure_mu
    )
    ax3.set_title("\"impact\" on prior")
    plt.colorbar(plot3, ax=ax3)

    plt.savefig(save_path)
    plt.close()

def plot_updates(exp_id, f1, f2, vmax=2000):
    """
    This script loads up the itae results
    and plots the updates.

    See the mockup at local_visualize.ipynb.
    """
    # a bunch of stuff needs to be loaded from the metadata
    metadata_path = f"./data/experiment_results/{exp_id}_metadata.json"
    with open(metadata_path) as fp:
        metadata = json.load(fp)

    prior_path = metadata["prior_path"]
    goal = metadata["goal"]

    with open(f"./data/experiment_results/{exp_id}_itae.json") as fp:
        data = json.load(fp)
    
    behaviors = [v["behavior"] for v in data]
    times = [v["time"] for v in data]

    for i in range(len(times)):
        current_b = behaviors[:i+1]
        current_t = times[:i+1]

        ze = ZeldaExperiment(
            prior_path,
            goal,
            behaviors=current_b,
            times=current_t,
            projection=[f1, f2],
            verbose=False
        )

        save_path = f"./data/plots/{exp_id}_iteration_{i}.jpg"
        save_path_3D = f"./data/plots/plot_3D_with_sigma_{exp_id}_iteration_{i}.jpg"
        plot_scatter(ze, f1, f2, save_path, vmax=vmax)

        ze_3D = ZeldaExperiment(
            prior_path,
            goal,
            behaviors=current_b,
            times=current_t,
            projection=[f1, f2],
            verbose=False
        )

        # ze_3D.view_3D_plot()
        ze_3D.save_3D_plot(save_path_3D, plot_sigma=True)
        plt.close("all")

if __name__ == "__main__":
    # exp_ids = get_all_exp_ids()
    # for i, exp_id in enumerate(exp_ids):
    #     print(f"Processing {exp_id} ({i+1}/{len(exp_ids)})")
    #     plot_exp_id(exp_id, goal=200)
    
    # exp_ids = [
    #     "082a3a57-473d-4c6b-8f02-b30c9bfee20b",
    #     "9ae0cdc2-a2ac-44dd-a19a-bf569b5f993e",
    #     "6c073e84-99d2-4c1c-b48f-ac425bbe5cc8"
    # ]

    exp_ids = [
        "cf3d1f77-27b8-488c-93cd-454a7307800e_goal_200_iterations_10"
    ]
    for i, exp_id in enumerate(exp_ids):
        print(f"Processing {exp_id} ({i+1}/{len(exp_ids)})")
        plot_exp_id(exp_id, goal=200)
        # plot_updates(exp_id, "leniency", "reachability", vmax=np.log(500))
