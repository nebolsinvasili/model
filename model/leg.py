import math

import numpy as np

from .ground import Ground
from .platform import Platform


class Leg:
    def __init__(
        self,
        ground: Ground,
        platform: Platform,
        dtype: type = np.float16,
    ) -> None:
        self.ground: Ground = ground
        self.platform: Platform = platform

        self.leghts = Leg.get_distance_legs(self.ground.joints, self.platform.joints, dtype=dtype)
        self.angles = Leg.get_angle_leg(self.ground.joints, self.platform.joints, dtype=dtype)
    
    def log(self):
        data = {}
        for i, distance in enumerate(self.leghts):
            data.update({f'Ld_{i+1}': distance})

        for i, angle in enumerate(self.angles):
            data.update({f'La_{i+1}': angle})
            
        return data

    @staticmethod
    def get_distance_legs(
        ground_joints,
        platform_joints,
        dtype: type = np.float16,
    ):
        return np.array(
            [np.linalg.norm(diff) for diff in ground_joints - platform_joints]
        ).astype(dtype)

    @staticmethod
    def get_angle_leg(
        ground_joints,
        platform_joints,
        dtype: type = np.float16,
    ):
        return np.array(
            [
                np.rad2deg(np.arctan2(*np.flip(diff[:2], axis=0)))
                for diff in platform_joints - ground_joints
            ]
        ).astype(dtype)

    @staticmethod
    def get_coords_leg(coord, angle, length):
        x2 = coord[0] + length * math.cos(np.deg2rad(angle))
        y2 = coord[1] + length * math.sin(np.deg2rad(angle))
        return (coord[0], coord[1]), (x2, y2)


if __name__ == "__main__":
    pass
