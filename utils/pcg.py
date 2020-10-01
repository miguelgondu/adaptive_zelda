"""
This script generates Zelda's dungeon levels at
random. It also contains compute_features, which is
used both locally and by the MAP-Elites and
ITAE experiments.
"""

import numpy as np
import random
from queue import LifoQueue, PriorityQueue

# According to zelda.txt in the GVGAI-framework
WALL = "w"
AVATAR = "A"
EMPTY = "."
GOAL = "g"
ENEMY = "2"
ENEMY1 = "1"
ENEMY2 = "2"
ENEMY3 = "3"
PATH = "x"
KEY = "+"

# seed = 2
# np.random.seed(seed)
# random.seed(seed)
def binary_decision():
    nonce = random.random()
    return nonce <= 0.5

def create_empty_level(width, height):
    level = np.zeros((width, height), dtype=str)
    level[level == ""] = EMPTY
    return level

def place_edge_walls(level):
    height, width = level.shape
    level[0, :] = WALL
    level[:, 0] = WALL
    level[height - 1, :] = WALL
    level[:, width - 1] = WALL

def get_placeable_positions(level):
    height, width = level.shape
    placeable_positions = set([])
    for i in range(height):
        for j in range(width):
            if level[i, j] == EMPTY:
                placeable_positions.add((i, j))

    return placeable_positions

def position_object(level, object_string, placeable_positions=None, ideal_position=None):
    """
    This function takes a level (a numpy array) and places the object at random.
    It also takes an optional parameter called ideal_position to force the algorithm
    to place said object on said place if it is empty. If that place isn't empty, the
    algorithm will select an empty position at random.

    This algorithm modifies the level IN PLACE.
    """
    if ideal_position:
        if level[ideal_position] == EMPTY:
            level[ideal_position] = object_string
            return

    if placeable_positions == set([]):
        raise ValueError(f"There are no placeable positions for object {object_string} in {level}")

    if placeable_positions is None:
        placeable_positions = get_placeable_positions(level)
        if not placeable_positions:
            raise ValueError(f"The level has no placeable positions for the object {object_string}: {level}")

    obj_position = random.choice(list(placeable_positions))
    placeable_positions.remove(obj_position)
    level[obj_position] = object_string

def get_neighbors(level, position):
    height, width = level.shape
    x, y = position[0], position[1]

    if x < 0 or x >= height:
        raise ValueError(f"Position is out of bounds in x: {position}")

    if y < 0 or y >= width:
        raise ValueError(f"Position is out of bounds in x: {position}")
    
    neighbors = []

    if x - 1 > 0:
        neighbors.append((x - 1, y))
    
    if x + 1 < height:
        neighbors.append((x + 1, y))
    
    if y - 1 > 0:
        neighbors.append((x, y - 1))
    
    if y + 1 < width:
        neighbors.append((x, y + 1))

    random.shuffle(neighbors)
    return neighbors

def get_neighbor_objs(level, node, obj):
    neighbors = get_neighbors(level, node)
    return set([(position, level[position]) for position in neighbors])

def get_positions(level, obj):
    positions = np.where(level == obj)
    return [(x, y) for x, y in zip(positions[0], positions[1])]

def find_path(level, obj_1=AVATAR, obj_2=GOAL):
    parents = {}
    position_obj_1 = get_positions(level, obj_1)
    position_obj_2 = get_positions(level, obj_2)
    
    if len(position_obj_1) > 1:
        raise ValueError(f"Objects should appear only once, and {obj_1} appears {len(position_obj_1)} times")
    
    if len(position_obj_2) > 1:
        raise ValueError(f"Objects should appear only once, and {obj_2} appears {len(position_obj_2)} times")

    position_obj_1 = position_obj_1[0]
    position_obj_2 = position_obj_2[0]

    stack = LifoQueue()

    # Initializing the stack.
    stack.put(position_obj_1)
    visited_positions = set([position_obj_1])
    parents[position_obj_1] = None

    while not stack.empty():
        position = stack.get()
        neighbors = get_neighbors(level, position)
        for neighbor in neighbors:
            if neighbor in visited_positions:
                continue

            visited_positions.add(neighbor)
            parents[neighbor] = position

            if level[neighbor] == obj_2:
                # We found the goal.
                parent = position
                son = neighbor
                path = {son: parent}
                while parent is not None:
                    son, parent = parent, parents[parent]
                    path[son] = parent
                return path

            if level[neighbor] != WALL:
                stack.put(neighbor)
    
    raise ValueError(f"Object {obj_1} and {obj_2} are not connected in the level.")

def euclidean_distance(point_1, point_2):
    p1 = np.array(point_1)
    p2 = np.array(point_2)
    return np.sum((p1 - p2) ** 2)

def _reconstruct_path(came_from, root, goal):
    path = []
    current = goal
    while current:
        path.append(current)
        current = came_from[current]
    return path[::-1]

def a_star_path(level, root_text=AVATAR, goal_text=KEY):
    parents = {}
    positions_root = get_positions(level, root_text)
    positions_goal = get_positions(level, goal_text)

    if len(positions_root) > 1:
        raise AssertionError(f"Objects should appear only once, and {root_text} appears {len(positions_root)} times")

    if len(positions_goal) > 1:
        raise AssertionError(f"Objects should appear only once, and {goal_text} appears {len(positions_goal)} times")

    root = positions_root[0]
    goal = positions_goal[0]
    h = lambda x: euclidean_distance(x, goal)
    g = {root: 0}

    came_from = {root: None}
    visited = set([])

    # Initializing the heap
    heap = PriorityQueue()
    heap.put((g[root] + h(root), root))
    visited.add(root)
    while not heap.empty():
        cost, node = heap.get()
        if node == goal:
            return _reconstruct_path(came_from, root, goal)

        neighbors = get_neighbors(level, node)
        for neighbor in neighbors:
            if level[neighbor] == WALL:
                # Ignore walls
                continue

            if level[neighbor] == GOAL and goal_text != GOAL:
                # Ignore the door if we are not going for it
                continue

            if neighbor not in g:
                g[neighbor] = np.Inf
            tentative_g = g[node] + 1
            if tentative_g < g[neighbor]:
                came_from[neighbor] = node
                g[neighbor] = tentative_g
                if neighbor not in visited:
                    heap.put((g[neighbor] + h(neighbor), neighbor))
                    visited.add(neighbor)

    raise ValueError(f"Objects {root_text} and {goal_text} are disconnected in the level.")

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

def create_random_level(width, height, loot, enemies, walls, draw_path=False):
    level = create_empty_level(width, height)
    place_edge_walls(level)
    placeable_positions = get_placeable_positions(level)

    # Positioning player and goals.
    position_object(level, AVATAR, placeable_positions=placeable_positions)

    position_object(level, KEY, placeable_positions=placeable_positions)
    path_to_key = find_path(level, AVATAR, KEY)

    position_object(level, GOAL, placeable_positions=placeable_positions - set(path_to_key.keys()))
    path_to_goal = find_path(level, KEY, GOAL)
    nodes_path = set(path_to_key.keys()).union(set(path_to_goal.keys()))

    assert GOAL in level and AVATAR in level and KEY in level

    if draw_path:
        for node in nodes_path:
            level[node] += PATH

    placeable_positions = get_placeable_positions(level)
    # Positioning enemies.
    for _ in range(enemies):
        enemy = random.choice([ENEMY1, ENEMY2, ENEMY3])
        position_object(level, enemy, placeable_positions=placeable_positions)

    # Positioning walls. (The new way)
    placeable_positions = list(get_placeable_positions_inc_path(level))
    random.shuffle(placeable_positions)
    for _ in range(min(len(placeable_positions), walls)):
        pos = placeable_positions.pop()
        level[pos] = WALL

    return level

def scramble_prompt(level, prompt, scrambles=1):
    """
    Right now, it's only implemented for scrambling
    enemies.

    TODO: what if the prompt is a wall, what if it is the avatar,
    the key, the goal?
    """
    current_prompt_positions = get_positions(prompt)
    placeable_positions = get_placeable_positions(level)

    old_pos = random.choice(current_prompt_positions)
    new_pos = random.choice(list(placeable_positions))

    level[old_pos] = EMPTY
    level[new_pos] = prompt

# New variations

def get_path_positions(level):
    path_to_key = a_star_path(level, root_text=AVATAR, goal_text=KEY)
    path_to_goal = a_star_path(level, root_text=KEY, goal_text=GOAL)
    return set(path_to_key).union(set(path_to_goal))

def get_placeable_positions_inc_path(level):
    floor_tiles = get_positions(level, ".")
    non_placeable_pos = get_path_positions(level)
    placeable_pos = set(floor_tiles) - non_placeable_pos
    return placeable_pos

def expand(level, axis=0):
    height, width = level.shape
    if axis == 0:
        index = np.random.randint(low=1, high=height-1)
    elif axis == 1:
        index = np.random.randint(low=1, high=width-1)
    if axis == 0:
        new_row = np.array([WALL] + [EMPTY]*(width-2) + [WALL])
        level = np.vstack((level[:index, :], new_row, level[index:, :]))
    elif axis == 1:
        new_column = np.array([[WALL] + [EMPTY]*(height-2) + [WALL]])
        level = np.hstack((level[:, :index], new_column.T, level[:, index:]))
    return level

def is_solvable(level):
    try:
        path_to_key = a_star_path(level, root_text=AVATAR, goal_text=KEY)
        path_to_goal = a_star_path(level, root_text=KEY, goal_text=GOAL)
        return True
    except ValueError:
        return False

def remove_index(level, index, axis=0):
    if axis == 0:
        new_level = np.vstack((level[:index, :], level[index+1:, :]))
    elif axis == 1:
        new_level = np.hstack((level[:, :index], level[:, index+1:]))
    else:
        raise ValueError(f"Unexpected axis: {axis}. Was expecting 0 or 1.")
    return new_level

def breaks_connectivity(level, index, axis=0):
    """Checks if removing col/row index breaks connectivity"""
    new_level = remove_index(level, index, axis=axis)
    return not is_solvable(new_level)

def shrink(level, axis=0):
    height, width = level.shape
    if axis == 0:
        removable_indices = set(range(1, height-1))
    elif axis == 1:
        removable_indices = set(range(1, width-1))
    avatar_pos = get_positions(level, AVATAR)[0]
    key_pos = get_positions(level, KEY)[0]
    goal_pos = get_positions(level, GOAL)[0]

    removable_indices -= set([pos[axis] for pos in [avatar_pos, key_pos, goal_pos]])
    while len(removable_indices) > 0:
        selected_index = random.choice(list(removable_indices))
        if not breaks_connectivity(level, selected_index, axis=axis):
            return remove_index(level, selected_index, axis=axis)
        else:
            removable_indices.remove(selected_index)
    
    print(f"All indices along axis {axis} seem to be important.")
    return level

def add_enemies(level, amount):
    """
    This function adds an enemy at random in any of the floor tiles
    in place.
    """
    floor_tiles = get_positions(level, EMPTY)
    random.shuffle(floor_tiles)
    for _ in range(min(len(floor_tiles), amount)):
        pos = floor_tiles.pop()
        enemy = random.choice([ENEMY1, ENEMY2, ENEMY3])
        level[pos] = enemy

def remove_enemies(level, amount):
    """
    This function removes an enemy at random replacing them for
    floor tiles in place.
    """
    enemy_pos = get_positions(level, ENEMY1)
    enemy_pos += get_positions(level, ENEMY2)
    enemy_pos += get_positions(level, ENEMY3)

    random.shuffle(enemy_pos)
    for _ in range(min(len(enemy_pos), amount)):
        pos = enemy_pos.pop()
        level[pos] = EMPTY

def add_walls(level, amount):
    """
    This function takes a level and adds walls in a
    row or in a column if mazelike is True, or at
    random if mazelike is False.

    TODO: implement the mazelike part.
    """
    placeable_pos = list(get_placeable_positions_inc_path(level))
    random.shuffle(placeable_pos)
    for _ in range(min(len(placeable_pos), amount)):
        pos = placeable_pos.pop()
        level[pos] = WALL
        assert is_solvable(level), "Level isn't solvable. There's a bug in add_walls"

def remove_walls(level, amount):
    """
    This function removes {amount} walls in the inner part of the level.
    """
    height, width = level.shape
    inner_level = level[1:height-1, 1:width-1]
    wall_pos = [(x+1, y+1) for (x,y) in get_positions(inner_level, WALL)]
    random.shuffle(wall_pos)
    for _ in range(min(len(wall_pos), amount)):
        pos = wall_pos.pop()
        level[pos] = EMPTY

def compute_features(x):
    '''
    New features:
        - space coverage: opposite of sparseness.
        - inverse leniency: amount of enemies over total.
        - inverse reachability: length of A* paths.
    '''
    features = {}
    if not isinstance(x, np.ndarray):
        level = np.array(x)
    else:
        level = x

    # Feature 1: space coverage.
    level_txt = print_to_text(level)
    total_chars = len(level_txt.replace("\n", ""))
    coverage = (total_chars - level_txt.count(EMPTY)) / total_chars
    features["space coverage"] = coverage

    # Feature 2: inverse leniency.
    total_enemies = level_txt.count(ENEMY1)
    total_enemies += level_txt.count(ENEMY2)
    total_enemies += level_txt.count(ENEMY3)
    features["leniency"] = total_enemies

    # Feature 3: inverse reachability.
    path_to_key = a_star_path(level, root_text=AVATAR, goal_text=KEY)
    path_to_goal = a_star_path(level, root_text=KEY, goal_text=GOAL)
    inv_reachability = len(path_to_key) + len(path_to_goal)
    features["reachability"] = inv_reachability

    return features

def level_from_text(text):
    rows = text.split("\n")
    rows = [
        list(r) for r in rows
    ]
    return rows
