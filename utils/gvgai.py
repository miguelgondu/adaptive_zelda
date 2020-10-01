import json
import os
import subprocess
import numpy as np
from pathlib import Path

from .pcg import compute_features
from .pcg import EMPTY

def print_to_text(level, path=None):
    rows = []
    for i, row in enumerate(level):
        row_string = ""
        for obj in row:
            if obj == EMPTY and EMPTY == "":
                row_string += " "
            else:
                row_string += obj

        if i < len(level) - 1:
            row_string += "\n"

        rows.append(row_string)

    text = "".join(rows)
    if path:
        with open(path, "w") as f:
            f.write(text)
    
    return text

def features_to_array(feat_dict):
    keys = list(feat_dict.keys())
    keys.sort()
    return [feat_dict[f] for f in keys]

def deploy_human(x, level_name, path_to_gvgai=None):
    """
    Plays level x with a human.
    """
    current_dir = os.getcwd()
    cwd = Path(current_dir)
    path_to_gvgai = cwd / "compiled_gvgai_w_fbf"

    path_to_data = cwd / "data"
    path_to_vgdl = path_to_data / f"zelda_vgdl.txt"
    path_to_level = path_to_data / "levels" / f"{level_name}.txt"
    record_path = path_to_data / "playtraces" / f"{level_name}.txt"

    seed = 17

    # Call run-game with the human agent.
    print_to_text(
        x,
        path_to_level
    )

    os.chdir(path_to_gvgai)
    total_time = 0
    while total_time < 2000:
        results = None
        java = subprocess.Popen([
            "java",
            "tracks.singlePlayer.Play",
            str(path_to_vgdl),
            str(path_to_level),
            str(record_path),
            str(seed),
            "true"
        ], stdout=subprocess.PIPE)

        try:
            results = java.stdout.readline().decode("utf8")
            results = json.loads(results)
        except json.decoder.JSONDecodeError as e:
            print(f"Couldn't decode the results. Got this exception: {e}.")
            results = None
        finally:
            java.kill()
        
        if results is not None:
            total_time += results["steps"]
            if results["win"] == 1:
                break

    # At this point, we're sure results isn't none.
    os.chdir(current_dir)
    # performance = compute_performance_human(results)

    performance = np.log(total_time)
    features = compute_features(x)
    return performance, features