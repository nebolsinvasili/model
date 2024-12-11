import inspect
from typing import Union

import numpy as np
from loguru import logger


class Detail:
    @classmethod
    def verific_coord(cls, value: Union[np.ndarray, list]) -> np.ndarray:
        try:
            value = value if isinstance(value, np.ndarray) else np.array(value)
            if value.shape[0] != 3 or value.ndim > 1:  # if value.shape != (3,):
                raise ValueError(
                    f"Shape of coord must be (3,). Get shape {value.shape}"
                )
            return value
        except ValueError as e:
            logger.error(e)
            raise

    @classmethod
    def verific_angle(cls, value: Union[np.ndarray, list]) -> np.ndarray:
        try:
            value = value if isinstance(value, np.ndarray) else np.array([value])
            if value.shape[0] != 3 or value.ndim > 1:
                raise ValueError(
                    f"Shape of angle must be (3,). Get shape {value.shape}"
                )
            return value
        except ValueError as e:
            logger.error(e)
            raise

    @classmethod
    def verific_joints(cls, value: Union[np.ndarray, list]) -> np.ndarray:
        try:
            value = value if isinstance(value, np.ndarray) else np.array([value])
            if value.ndim != 2 or value.shape[1] != 3:
                raise ValueError(
                    f"Shape of joint must be (None, 3). Get shape {value.shape}"
                )
            return value
        except ValueError as e:
            logger.error(e)
            raise

    def __init__(
        self,
        coord: Union[np.ndarray, list],
        angle: Union[np.ndarray, list],
        joints: Union[np.ndarray, list, float],
        dtype: type = np.float16,
        scale: float = 1.0,
        speed: float = 1.0,
        name: str = "detail",
    ) -> None:
        
        self.dtype = dtype
        # TODO: current values Detail
        self._coord = Detail.verific_coord(value=coord).astype(self.dtype)
        self._angle = Detail.verific_angle(value=angle).astype(self.dtype)
        if isinstance(joints, (list, np.ndarray)):
            self._joints = Detail.verific_joints(joints).astype(self.dtype)
        else:
            self._joints = self.set_joints(
                coord=self.coord, radius=joints, init_angle=self.angle[2]
            ).astype(self.dtype)

        # TODO: old values Detail
        self.joints_old = self._joints
        self.coord_old = self._coord
        self.angle_old = self._angle

        # TODO: move values Detail
        self.speed = speed
        self.moving = False
        self.queue = []

        # TODO: other values Detail
        self.scale = scale
        self.name = name

    @property
    def coord(self) -> np.ndarray:
        return self._coord

    @coord.setter
    def coord(self, value: Union[np.ndarray, list]):
        self.coord_old = self._coord
        self._coord = Detail.verific_coord(value=value).astype(self.dtype)

    @property
    def angle(self) -> np.ndarray:
        return self._angle

    @angle.setter
    def angle(self, value: Union[np.ndarray, list]):
        self.angle_old = self._angle
        self._angle = Detail.verific_angle(value=value).astype(self.dtype)

    @property
    def joints(self) -> np.ndarray:
        return self._joints

    @joints.setter
    def joints(self, value: Union[np.ndarray, list, float]):
        self.joints_old = self._joints
        self._joints = Detail.verific_joints(value=value).astype(self.dtype)

    @staticmethod
    def set_joints(
        coord: Union[np.ndarray, list],
        radius: float,
        init_angle: float,
        dtype = np.float16,
    ):
        angles = np.deg2rad(np.array([0, 120, 240]) + init_angle)
        return np.column_stack(
            (
                coord[0] + radius * np.cos(angles),
                coord[1] + radius * np.sin(angles),
                np.zeros(angles.shape),
            )
        ).astype(dtype)


    def in_init(self):
        # Получаем текущий стек вызовов
        stack = inspect.stack()
        # Проверяем, находится ли метод '__init__' среди активных вызовов
        for frame_info in stack:
            if frame_info.function == '__init__':
                return True
        return False


if __name__ == "__main__":
    detail = Detail(
        coord=np.array([0, 0, 0]),
        angle=np.array([0, 0, 0]),
        joints=100,
    )
    print(detail.coord)
    print(detail.angle)
    print(detail.joints)
