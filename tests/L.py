import math
import numpy as np


R = 100 # Радиус точек крепления манипулятора к поверхности
r = 25 # Радиус платформы

center, fi = (22.042575859020857, -25.476000346875743), 105.3253516618673 #  Координаты TCP (англ. Tool Center Point — центральная точка инструмента) 

xq = math.sqrt(3) * R / 2 # При R = 100: xq = 86.60254037844386
yq = R / 2 # При R = 100: yq = 50.0


A = {'A1': (0, R, 0),
     'A2': (-xq, -yq, 0),
     'A3': (xq, -yq, 0)
     }

gamma = {'gamma_1': 90, 
         'gamma_2': -30, 
         'gamma_3': -150
         }

def L_matlab(A, r, gamma, x, y, fi):
    return math.sqrt(math.pow((x + r * math.cos(fi + gamma) - A[0]), 2) + math.pow((y + r * math.sin(fi + gamma) - A[1]), 2))

print(A)

for idx, (A, gamma) in enumerate(zip(A.values(),
                                 gamma.values())):
        print(f'Штанга № {idx+1}: {L_matlab(A=A, r=r, gamma=gamma, x=center[0], y=center[1], fi=fi)}')