'''
This script contains auxiliary functions that
allow you to plot a txt as a level using the
sprites on the sprites folder.
'''
import matplotlib.pyplot as plt
import PIL
import numpy as np
from operator import itemgetter

# from pymelites.visualizing_generations import plot_generations

def load_level_to_matrix(level_txt):
    temp_level = level_txt.split("\n")
    level = []
    for row in temp_level:
        level.append(list(row))
    # print(level)
    return level

def plot_level_from_array(ax, level, title=None):
    all_chars = set([])
    for row in level:
        all_chars = all_chars.union(set(row))
    image_paths = {char: f"sprites/{char}.png" for char in all_chars}
    image_paths["f"] = f"sprites/f.png"

    image = []
    for row in level:
        image_row = []
        for char in row:
            if char == ".":
                char = "f"
            image_row.append(PIL.Image.open(image_paths[char]).convert("RGB"))
        image.append(image_row)

    image = [
        np.hstack([np.asarray(img) for img in row]) for row in image
    ]
    image = np.vstack([np.asarray(img) for img in image])
    ax.imshow(image)
    ax.axis("off")
    if title is not None:
        ax.set_title(title)
    

def save_level_from_array(level, image_path, title=None, dpi=100):
    all_chars = set([])
    for row in level:
        all_chars = all_chars.union(set(row))
    image_paths = {char: f"sprites/{char}.png" for char in all_chars}
    image_paths["f"] = f"sprites/f.png"

    image = []
    for row in level:
        image_row = []
        for char in row:
            if char == ".":
                char = "f"
            image_row.append(PIL.Image.open(image_paths[char]).convert("RGB"))
        image.append(image_row)

    image = [
        np.hstack([np.asarray(img) for img in row]) for row in image
    ]
    image = np.vstack([np.asarray(img) for img in image])
    plt.imshow(image)
    plt.axis("off")
    if title is not None:
        plt.title(title)
    plt.savefig(image_path, bbox_inches="tight", dpi=dpi)
    plt.close()

def save_level_from_txt(level_txt, image_path, title=None):
    level = load_level_to_matrix(level_txt)
    all_chars = set([])
    for row in level_txt.split("\n"):
        # print(set(list(row)))
        all_chars = all_chars.union(set(list(row)))
    # print(all_chars)
    image_paths = {char: f"sprites/{char}.png" for char in all_chars}
    image_paths["f"] = f"sprites/f.png"

    image = []
    for row in level:
        image_row = []
        for char in row:
            if char == ".":
                char = "f"
            image_row.append(PIL.Image.open(image_paths[char]).convert("RGB"))
        image.append(image_row)

    image = [
        np.hstack([np.asarray(img) for img in row]) for row in image
    ]
    image = np.vstack([np.asarray(img) for img in image])
    plt.imshow(image)
    plt.axis("off")
    if title:
        plt.title(title)
    plt.savefig(image_path, bbox_inches="tight")
    plt.close()

def save_level(level_path, image_path):
    with open(level_path) as fp:
        level_txt = fp.read()

    save_level_from_txt(level_txt, image_path)

def aggregate_performance(generation, key_x, key_y, by="average"):
    # TODO: implement by="max" as well.
    aggregated_performance = {}
    for doc in generation.values():
        if doc["performance"] is None:
            continue

        keys = list(doc["features"].keys())
        keys.sort()
        index_x, index_y = keys.index(key_x), keys.index(key_y)

        proj_centroid = (doc["centroid"][index_x], doc["centroid"][index_y])
        if proj_centroid not in aggregated_performance:
            aggregated_performance[proj_centroid] = []

        aggregated_performance[proj_centroid].append(doc["performance"])

    aggregated_performance = {
        k: sum(v)/len(v) for k, v in aggregated_performance.items()
    }

    return aggregated_performance

def aggregate_winrate(generation, key_x, key_y, by="average"):
    aggregated_winrate = {}
    for doc in generation.values():
        if doc["performance"] is None:
            continue

        keys = list(doc["features"].keys())
        keys.sort()
        index_x, index_y = keys.index(key_x), keys.index(key_y)

        proj_centroid = (doc["centroid"][index_x], doc["centroid"][index_y])
        if proj_centroid not in aggregated_winrate:
            aggregated_winrate[proj_centroid] = []

        wins = doc["metadata"]["wins"]
        winrate = sum(wins)/len(wins)
        aggregated_winrate[proj_centroid].append(winrate)

    aggregated_winrate = {
        k: sum(v)/len(v) for k, v in aggregated_winrate.items()
    }

    return aggregated_winrate

def plot_mean_performance(ax, generation, partition, vmin=0, vmax=1, size=500, plot_labels=True, plot_colorbar=True):
    partition_items = list(partition.items())
    partition_items.sort(key=itemgetter(0))

    key_x = partition_items[0][0]
    key_y = partition_items[1][0]
    xlims = partition_items[0][1][:2]
    ylims = partition_items[1][1][:2]

    aggregated_performance = aggregate_performance(generation, key_x, key_y)

    points = []
    performances = []
    for centroid, mean_perf in aggregated_performance.items():
        points.append(centroid)
        performances.append(mean_perf)
    
    points = np.array(points)
    performances = np.array(performances)

    scatter = ax.scatter(points[:, 0], points[:, 1], c=performances, vmin=vmin, vmax=vmax, s=size, marker="s", cmap="inferno")
    ax.set_xlim(xlims)
    ax.set_ylim(ylims)
    if plot_labels:
        ax.set_xlabel(key_x)
        ax.set_ylabel(key_y)
    if plot_colorbar:
        plt.colorbar(scatter, ax=ax, ticks=[0, 0.5, 1])

def plot_mean_winrate(ax, generation, partition, vmin=0, vmax=1, size=500, plot_labels=True, plot_colorbar=True, return_scatter=False):
    partition_items = list(partition.items())
    partition_items.sort(key=itemgetter(0))

    key_x = partition_items[0][0]
    key_y = partition_items[1][0]
    xlims = partition_items[0][1][:2]
    ylims = partition_items[1][1][:2]

    aggregated_winrate = aggregate_winrate(generation, key_x, key_y)

    points = []
    winrates = []
    for centroid, mean_winrate in aggregated_winrate.items():
        points.append(centroid)
        winrates.append(mean_winrate)

    points = np.array(points)
    winrates = np.array(winrates)

    scatter = ax.scatter(points[:, 0], points[:, 1], c=winrates, vmin=vmin, vmax=vmax, s=size, marker="s", cmap="inferno")
    ax.set_xlim(xlims)
    ax.set_ylim(ylims)
    if plot_labels:
        ax.set_xlabel(key_x)
        ax.set_ylabel(key_y)

    if plot_colorbar:
        plt.colorbar(scatter, ax=ax)

    if return_scatter:
        return scatter

def plot_best_level(ax, generation, winrate_as_title=True):
    best_level, its_performance, its_winrate, its_centroid = None, -np.Inf, None, None
    for doc in generation.values():
        if doc["performance"] is not None:
            if best_level is None:
                best_level = doc["solution"]
                wins = doc["metadata"]["wins"]
                its_performance = doc["performance"]
                its_winrate = sum(wins)/len(wins)
                its_centroid = doc["centroid"]
            
            if doc["performance"] > its_performance:
                best_level = doc["solution"]
                wins = doc["metadata"]["wins"]
                its_performance = doc["performance"]
                its_winrate = sum(wins)/len(wins)
                its_centroid = doc["centroid"]

    print(f"best centroid = {its_centroid}")
    if winrate_as_title:
        plot_level_from_array(ax, best_level, title=f"Winrate: {its_winrate}")
    else:
        plot_level_from_array(ax, best_level)

# def visualize_experiment(paths, partition):
#     """
#     This function plots the generation-like
#     objects that are in the paths iterable
#     using the partition object.
#     """
#     plot_generations(paths, partitions=partition)

def plot_3D(generation):
    points = []
    colors = []
    for v in generation.values():
        if v["performance"] is not None:
            points.append(list(v["centroid"]))
            colors.append(v["performance"])
    
    points = np.array(points)
    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    ax.scatter(points[:, 0], points[:, 1], points[:, 2], marker="s", c=colors, s=100)
    plt.show()
