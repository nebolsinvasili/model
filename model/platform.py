from typing import Union

import numpy as np
from numpy.linalg import multi_dot

from model.detail import Detail

np.set_printoptions(precision=3, floatmode="fixed")


class Rotate:
    def __init__(self,
                 offset_angle: np.ndarray,
                 joints: np.ndarray,
                 dtype: type = np.float32):
        self.offset_angle = offset_angle
        self.joints = joints
        self.dtype = dtype

        self.r_joints = self.rotate_joints()

    @staticmethod
    def Rx(theta):
        return np.array(
            [
                [1, 0, 0],
                [0, np.cos(theta), np.sin(theta)],
                [0, -np.sin(theta), np.cos(theta)],
            ]
        )

    @staticmethod
    def Ry(phi):
        return np.array(
            [
                [np.cos(phi), 0, -np.sin(phi)],
                [0, 1, 0],
                [np.sin(phi), 0, np.cos(phi)],
            ]
        )

    @staticmethod
    def Rz(psi):
        return np.array(
            [
                [np.cos(psi), np.sin(psi), 0],
                [-np.sin(psi), np.cos(psi), 0],
                [0, 0, 1],
            ]
        )

    def rotate_joints(self) -> np.ndarray:
        off_x, off_y, off_z = np.deg2rad(self.offset_angle)
        return multi_dot(
            [
                self.joints,
                Rotate.Rx(theta=off_x),
                Rotate.Ry(phi=off_y),
                Rotate.Rz(psi=off_z),
            ]
        ).astype(self.dtype)


class Platform(Detail):
    def __init__(
        self,
        coord: Union[np.ndarray, list] = np.array([0, 0, 0]),
        angle: Union[np.ndarray, list] = np.array([0, 0, 0]),
        joints: Union[np.ndarray, list, float] = 50,
        dtype: type = np.float16,
        scale: float = 1.0,
        speed: float = 1.0,
        name: str = "platform",
    ):
        super().__init__(coord, angle, joints, dtype, scale, speed, name)

        self.dj = Platform.distance_attact_joints(self.coord, self.joints, self.dtype)
        self.aj = Platform.angle_attact_joints(self.coord, self.joints, self.dtype)
        self.offset = np.array([[0, 0, 0], [0, 0, 0]])
        self.data = self.log()
    
    def log(self):
        data = {}
        for i, xyz in enumerate(self.joints):
            for axis, value in zip("xyz", xyz):
                data.update({f"B{i+1}_{axis}": value})
        
        for i, value in enumerate(self.dj):
            data.update({f"B{i+1}_dj": value})

        for i, value in enumerate(self.aj):
            data.update({f"B{i+1}_aj": value})
        
        for axis, value in zip('xyz', self.coord):
            data.update({f'{axis}': value})
        data.update({'fi': self.angle[2]})

        return data

    def move(
        self,
        coord: Union[np.ndarray, list] = None,
        angle: Union[np.ndarray, list] = None,
    ):
        coord = coord if coord is not None else self.coord
        angle = angle if angle is not None else self.angle

        offset = np.array(
            [np.subtract(coord, self.coord), np.subtract(angle, self.angle)]
        )
        if offset.any() != 0:
            joints = self.joints - self.coord_old
            joints = Rotate(offset_angle=offset[1], joints=joints, dtype=self.dtype).r_joints
            joints = joints + coord

            dj = Platform.distance_attact_joints(coord, joints, dtype=self.dtype)
            aj = Platform.angle_attact_joints(coord, joints, dtype=self.dtype)
        return coord, angle, offset, joints, dj, aj

    def update(
        self,
        coord: Union[np.ndarray, list],
        angle: Union[np.ndarray, list],
        offset: np.ndarray,
        joints: np.ndarray,
        dj, 
        aj,
    ):
        self.coord = coord
        self.angle = angle
        self.offset = offset
        self.joints = joints

        self.dj = dj
        self.aj = aj

        self.data = self.log()
            

    @classmethod
    def distance_joint(cls, a, b):
        """
        distances from coord to joint
        """
        return np.linalg.norm(a - b)

    @classmethod
    def angle_joint(cls, a, b):
        """
        angle from coord to joint
        """
        return np.rad2deg(np.arctan2(*np.flip(a - b, axis=0)))

    @classmethod
    def distance_attact_joints(cls, coord: np.ndarray, joints: np.ndarray, dtype):
        return np.apply_along_axis(
            lambda joint: Platform.distance_joint(coord, joint), axis=1, arr=joints
        ).astype(dtype)

    @classmethod
    def angle_attact_joints(cls, coord: np.ndarray, joints: np.ndarray, dtype):
        return np.apply_along_axis(
            lambda joint: Platform.angle_joint(joint[:2], coord[:2]), axis=1, arr=joints
        ).astype(dtype)


if __name__ == "__main__":
    pass
