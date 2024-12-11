import os
import pandas as pd
from typing import Union
from loguru import logger
import numpy as np

from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.ticker import MaxNLocator

from utils.target import target

from model.ground import Ground
from model.platform import Platform
from model.leg import Leg


class RPR:
    def __init__(self,
                 ground_joints: float = 100,
                 platform_coord = np.array([0, 0, 0]),
                 platform_angle = np.array([0, 0, 0]),
                 platform_joints: float = 25,
                 Lmin: float = 10,
                 Lmax: float = 190,
                 dtype: type = np.float16,
                 scale: float = 1.0,
                 fig: Figure = None,
                 axis: Axes = None,
                 name: str = "RPR",
                 filename: str='test.csv'   
    ):  
        self.dtype = dtype
        self.ground_joints = ground_joints
        self.ground: Ground = Ground(
            coord=np.array([0, 0, 0]),
            angle=np.array([0, 0, 90]),
            joints=self.ground_joints,
            dtype=self.dtype,
            scale=scale,
        )
        self.platform_joints = platform_joints
        self.platform = Platform(
            coord=platform_coord,
            angle=platform_angle,
            joints=platform_joints,
            dtype=self.dtype,
            scale=scale,
        )

        self.leg = Leg(self.ground, self.platform, dtype=self.dtype)
        
        self.Lmin = Lmin
        self.Lmax = Lmax

        self.moving = False
        self.queue = []

        self.fig, self.axis = fig, axis

        self.name = name
        self.filename = filename
        
        self.log()
    
    def move(
        self,
        in_coord: Union[np.ndarray, list] = None,
        in_angle: Union[np.ndarray, list] = None,
    ):
        coord, angle, offset, joints, dj, aj = self.platform.move(in_coord, in_angle)
        ld, la = Leg.get_distance_legs(self.ground.joints, joints, dtype=self.dtype), Leg.get_angle_leg(self.ground.joints, joints, dtype=self.dtype)
        if all((ld >= self.Lmin) & (ld <= self.Lmax)):
            if self.moving:
                self.queue.append((coord, angle))
                return False

            self.moving = True
            self.platform.update(coord, angle, offset, joints, dj, aj)
            self.leg.leghts = ld
            self.leg.angles = la
            self.log()
            self.moving = False

            return True
        else:
            return False
        
    
    def log(self):
        data = self.ground.log()
        data.update(self.platform.log())
        data.update(self.leg.log())

        if not os.path.exists(self.filename):
            keys_df = pd.DataFrame([data.keys()], columns=data.keys())
            keys_df.to_csv(self.filename, header=False, index=False)
        pd.DataFrame([data]).to_csv(self.filename, header=False, index=False, mode='a')

        logger.info(
                f"{self.name} | MOVE "
                f" | Coord: {self.platform.coord} "
                f" | Angle: {self.platform.angle} "
                f" | Offsets: ({self.platform.offset[0]}, {self.platform.offset[1]})"
                f" | DJ: {self.platform.dj}"
                f" | AJ: {self.platform.aj}"
                f" | L: {self.leg.leghts, self.leg.angles}"
            )

    def plot(self, axis: Axes):
        # axis.plot(*self.platform.coord[:2], "ro")  # Plot center
        axis.plot(
            *zip(
                *np.concatenate(
                    (self.platform.joints, self.platform.joints[0][None, :]), axis=0
                )[:, :2].tolist()
            ),
            color="blue",
        )

        r = np.stack((self.ground.joints, self.platform.joints), axis=1)[
            :, :, :2
        ].tolist()
        for i in r:
            axis.plot(*zip(*i), color="blue")  # Plot legs
    

    @staticmethod
    def singularity(x, y, fi, R, r):
        return all(
            [
                np.sin(fi) != 0,
                np.power(x, 2) + np.power(y, 2)
                != np.power(R, 2) - 2 * R * r * np.cos(fi) + np.power(r, 2),
            ]
        )

if __name__ == "__main__":
    import time
    from matplotlib import pyplot as plt
    
    
    R = 100
    r = 25
    Lmin, Lmax = 25, 175
    radius = 50
    limit = [10, 170]
    n = 1

    rpr = RPR(
        ground_joints=R,
        platform_joints=r,
        Lmin=Lmin,
        Lmax=Lmax,
        name="RPR",
    )
    gen = target(radius=radius, limit=limit, R=R, r=r, move=rpr.move)

    for idx in range(n):
        coord, angle = next(gen)

        # Plot
        if idx % 5 == 0:
            fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(4, 4))
            rpr.plot(axis=ax)

            ax.margins(0.25)
            # Настройка количества тиков на оси X
            ax.xaxis.set_major_locator(MaxNLocator(10))  # Настройка для 5 основных тиков на оси X

            # Настройка количества тиков на оси Y
            ax.yaxis.set_major_locator(MaxNLocator(10))  # Настройка для 5 основных тиков на оси Y

            plt.show(block=False)
            plt.pause(0.1)
            time.sleep(1)
            plt.close()
