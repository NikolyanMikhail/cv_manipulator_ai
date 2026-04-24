import numpy as np
import csv
import random

def translation(x, y, z): return np.array([[1,0,0,x], [0,1,0,y], [0,0,1,z], [0,0,0,1]])
def rot_y(theta): c, s = np.cos(theta), np.sin(theta); return np.array([[c,0,s,0], [0,1,0,0], [-s,0,c,0], [0,0,0,1]])
def rot_z(theta): c, s = np.cos(theta), np.sin(theta); return np.array([[c,-s,0,0], [s,c,0,0], [0,0,1,0], [0,0,0,1]])

def forward_kinematics(t1, t2, t3, t4):
    T1 = translation(0.012, 0, 0) @ rot_z(t1)
    T2 = translation(0, 0, 0.0595) @ rot_y(t2)
    T3 = translation(0.024, 0, 0.128) @ rot_y(t3)
    T4 = translation(0.124, 0, 0) @ rot_y(t4)
    EE = translation(0.126, 0, 0)
    T_final = T1 @ T2 @ T3 @ T4 @ EE
    return T_final[0, 3], T_final[1, 3], T_final[2, 3]

NUM_SAMPLES = 20000
output_file = 'dataset.csv'
valid_samples = 0

print("Генерация умных данных (камера смотрит вниз)...")
with open(output_file, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['x', 'y', 'z', 't1', 't2', 't3', 't4'])
    
    while valid_samples < NUM_SAMPLES:
        t1 = random.uniform(-1.0, 1.0)
        t2 = random.uniform(-1.2, 1.2)
        t3 = random.uniform(-1.2, 1.2)
        
        # ЭЛЕГАНТНЫЙ ХАК: Заставляем клешню смотреть вниз (Pitch = ~90 градусов)
        t4 = 1.57 - t2 - t3 
        
        # Проверяем, что t4 не вырвало сустав
        if -1.5 <= t4 <= 1.5:
            x, y, z = forward_kinematics(t1, t2, t3, t4)
            # Берем только точки над столом в рабочей зоне
            if z > 0.02 and 0.1 < x < 0.4:
                writer.writerow([x, y, z, t1, t2, t3, t4])
                valid_samples += 1

print(f"Готово! Сохранено в {output_file}")
