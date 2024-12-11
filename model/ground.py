from typing import Union

import numpy as np

from .detail import Detail


class Ground(Detail):
    def __init__(
        self,
        coord: Union[np.ndarray, list] = np.array([0, 0, 0]),
        angle: Union[np.ndarray, list, float] = np.array([0, 0, 0]),
        joints: Union[np.ndarray, list, float] = 50,
        scale: float = 1.0,
        dtype: type = np.float16,
    ):
        super().__init__(coord, angle, joints, dtype, scale)
        self.data = self.log()
    
    def log(self):
        data = {}
        for i, xyz in enumerate(self.joints):
            for axis, value in zip("xyz", xyz):
                data.update({f"A{i+1}_{axis}": value})
        return data


if __name__ == "__main__":
    ground = Ground(
        coord=np.array([0, 0, 0]),
        angle=np.array([0, 0, 90]),
        joints=100,
    )
    print(ground.data)
