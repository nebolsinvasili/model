import random
import numpy as np

def random_polar():
    theta = random() * 2 * np.pi
    r = random()
    return r * np.cos(theta), r * np.sin(theta)

def rand_point(radius: float,
               show: bool = False):
    theta = random.random() * 2 * np.pi
    r = random.random() * radius
    new_xyz = np.array([r * np.cos(theta), r * np.sin(theta), 0])
    
    if show:
        print(f"Случайная точка: {new_xyz} мм")
    return new_xyz


def rand_angle(limit: list, 
               show: bool = False):
    angle = np.zeros(3)
    angle[-1] = random.uniform(*limit)
    return angle

def target(move, radius=500, limit=[0, 180], R=100, r=25):
    unique_coords = []
    unique_angles = []
    while True:
        coord, angle = rand_point(radius=radius), rand_angle(limit=limit)
        if is_unique(unique_coords, coord) and is_unique(unique_angles, angle) and singularity(*coord[:2], fi=angle[2], R=R, r=r):
            continue
        if move(coord, angle):
            unique_coords.append(coord)
            unique_angles.append(angle)
            yield coord, angle
        else:
            continue

def is_unique(array_list, new_array):
    return any(np.array_equal(existing, new_array) for existing in array_list)

def singularity(x, y, fi, R, r):
        return all(
            [
                np.sin(fi) == 0,
                np.power(x, 2) + np.power(y, 2)
                == np.power(R, 2) - 2 * R * r * np.cos(fi) + np.power(r, 2),
            ]
        )


if __name__ == "__main__":
    pass