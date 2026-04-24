import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Float64
from cv_bridge import CvBridge
import cv2
import numpy as np
import torch
import torch.nn as nn
import os
import math
from ultralytics import YOLO

class IKNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(3, 256), nn.ReLU(), 
            nn.Linear(256, 256), nn.ReLU(), 
            nn.Linear(256, 4)
        )
    def forward(self, x): return self.net(x)

class AIBrainNode(Node):
    def __init__(self):
        super().__init__('ai_brain_node')
        self.bridge = CvBridge()
        
        self.ik = IKNet()
        self.ik.load_state_dict(torch.load(os.path.expanduser('~/robot_ws/src/cv_manipulator_task/neural_network/ik_model.pth'), weights_only=True))
        self.ik.eval()
        self.yolo = YOLO(os.path.expanduser('~/robot_ws/src/cv_manipulator_task/neural_network/best.pt'))

        self.pubs = {f'j{i}': self.create_publisher(Float64, f'/joint{i}_cmd', 10) for i in range(1, 5)}
        self.pub_gl = self.create_publisher(Float64, '/gripper_left_joint_cmd', 10)
        self.pub_gr = self.create_publisher(Float64, '/gripper_right_joint_cmd', 10)
        
        self.create_subscription(Image, '/camera/image_raw', self.vision_callback, 10)
        self.pub_vision = self.create_publisher(Image, '/robot_eye', 10)
        
        self.transit_z = 0.22 
        self.safe_z = 0.16    
        
        self.target_local = [0.18, 0.0, self.safe_z] 
        self.y_dir = 1
        self.radar_angle = 0.0 
        self.radar_dir = 1
        
        self.cur_angles = np.array([0.0, 0.5, -0.5, 1.0]) 
        self.state = "SCAN"
        self.cube_err_x = 0.0
        self.cube_err_y = 0.0
        self.sees_cube = False
        self.timer = 0
        
        self.BASE_TARGET_X = 320 
        self.BASE_TARGET_Y = 250 
        self.OPEN_WIDTH = 0.040  
        self.PRESS_CRUSH = -0.050 
        self.target_grip = self.OPEN_WIDTH
        
        self.create_timer(0.03, self.control_loop)
        self.get_logger().info("Поехали")

    def move_gripper(self, pos):
        self.target_grip = pos
        self.pub_gl.publish(Float64(data=float(pos)))
        self.pub_gr.publish(Float64(data=float(pos)))

    def vision_callback(self, msg):
        frame = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        
        dynamic_offset = (0.22 - self.target_local[2]) * 80
        current_target_y = int(self.BASE_TARGET_Y + dynamic_offset)
        
        # === БЕСКОНЕЧНЫЙ ЩИТ КОРЗИНЫ ===
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        lower_yellow = np.array([20, 100, 100])
        upper_yellow = np.array([40, 255, 255])
        mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
        
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            if cv2.contourArea(largest_contour) > 800:
                x, y, w, h = cv2.boundingRect(largest_contour)
                pad = 25 
                
                x1 = max(0, x - pad)
                x2 = min(640, x + w + pad)
                y2 = min(480, y + h + pad)
                
                cv2.rectangle(frame, (x1, 0), (x2, y2), (0, 0, 0), -1)
                cv2.rectangle(frame, (x1, 0), (x2, y2), (0, 255, 255), 2)
                cv2.putText(frame, "INFINITY BIN SHIELD", (x1+5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        self.sees_cube = False
        res = self.yolo(frame, verbose=False)
        best_dist = float('inf') 
        best_target = None
        
        for r in res:
            for b in r.boxes:
                if b.conf[0] > 0.55: 
                    x1, y1, x2, y2 = map(int, b.xyxy[0])
                    err_x = ((x1+x2)/2 - self.BASE_TARGET_X) / 320.0
                    err_y = ((y1+y2)/2 - current_target_y) / 240.0
                    
                    dist = math.sqrt(err_x**2 + err_y**2)
                    
                    if dist < best_dist:
                        best_dist = dist
                        best_target = (err_x, err_y, x1, y1, x2, y2)
        
        if best_target is not None:
            self.cube_err_x, self.cube_err_y, x1, y1, x2, y2 = best_target
            self.sees_cube = True
            
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, "TARGET LOCKED", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
        cv2.circle(frame, (self.BASE_TARGET_X, current_target_y), 6, (0, 0, 255), -1)
        cv2.putText(frame, f"STATE: {self.state}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        self.pub_vision.publish(self.bridge.cv2_to_imgmsg(frame, "bgr8"))

    def control_loop(self):
        if self.state == "SCAN":
            self.move_gripper(self.OPEN_WIDTH) 
            self.radar_angle += 0.015 * self.radar_dir
            if self.radar_angle > 1.57: 
                self.radar_angle = 1.57
                self.radar_dir = -1
            if self.radar_angle < -4.71: 
                self.radar_angle = -4.71
                self.radar_dir = 1
            
            self.target_local[1] += 0.002 * self.y_dir
            if self.target_local[1] > 0.06 or self.target_local[1] < -0.06:
                self.y_dir *= -1
                
            self.target_local[0] = 0.18 
            self.target_local[2] = self.safe_z 
            
            if self.sees_cube: 
                self.state = "ALIGN"
                self.timer = 0

        elif self.state == "ALIGN":
            self.timer += 1 
            self.radar_angle -= np.clip(self.cube_err_x * 0.020, -0.005, 0.005) 
            self.target_local[0] -= np.clip(self.cube_err_y * 0.010, -0.004, 0.004) 
            self.target_local[1] *= 0.50 
            
            if not self.sees_cube or self.timer > 200:
                self.state = "SCAN"
            
            if abs(self.cube_err_x) < 0.05 and abs(self.cube_err_y) < 0.05: 
                self.state = "PLUNGE_SETUP"
                self.timer = 0

        elif self.state == "PLUNGE_SETUP":
            self.target_local[2] -= 0.003 
            if self.sees_cube:
                self.radar_angle -= np.clip(self.cube_err_x * 0.010, -0.003, 0.003)
                self.target_local[0] -= np.clip(self.cube_err_y * 0.005, -0.002, 0.002)
                self.target_local[1] *= 0.40
            if self.target_local[2] <= 0.080: 
                self.state = "FINE_ALIGN"
                self.timer = 0

        elif self.state == "FINE_ALIGN":
            if not self.sees_cube: 
                self.state = "SCAN"
            else:
                self.timer += 1
                self.radar_angle -= np.clip(self.cube_err_x * 0.005, -0.001, 0.001)
                self.target_local[0] -= np.clip(self.cube_err_y * 0.003, -0.001, 0.001)
                self.target_local[1] *= 0.20 
                
                # Идеальная, чистая логика наведения (без хаков ближнего боя)
                if abs(self.cube_err_x) < 0.015 and abs(self.cube_err_y) < 0.015:
                    self.state = "BLIND_DROP"
                    self.timer = 0
                elif self.timer > 150: 
                    self.state = "BLIND_DROP"
                    self.timer = 0

        elif self.state == "BLIND_DROP":
            self.target_local[2] -= 0.003 
            if self.target_local[2] <= 0.010: 
                self.state = "SETTLE"
                self.timer = 0

        elif self.state == "SETTLE":
            self.timer += 1
            if self.timer > 15: 
                self.state = "HYDRAULIC_PRESS"
                self.timer = 0

        elif self.state == "HYDRAULIC_PRESS":
            self.timer += 1
            self.target_grip -= 0.0015 
            self.target_grip = max(self.target_grip, self.PRESS_CRUSH)
            self.move_gripper(self.target_grip)
            if self.target_grip <= self.PRESS_CRUSH:
                self.state = "HOLD_AND_WAIT"
                self.timer = 0

        elif self.state == "HOLD_AND_WAIT":
            self.timer += 1
            self.move_gripper(self.PRESS_CRUSH)
            if self.timer > 40: 
                self.state = "LIFT_HIGH" 

        elif self.state == "LIFT_HIGH":
            self.target_local[2] += 0.005
            if self.target_local[2] >= self.transit_z: 
                self.state = "MOVE_TO_BIN" 

        elif self.state == "MOVE_TO_BIN":
            self.move_gripper(self.PRESS_CRUSH)
            bin_angle = 1.37
            
            self.radar_angle += (bin_angle - self.radar_angle) * 0.06
            self.target_local[0] += (0.24 - self.target_local[0]) * 0.06
            self.target_local[1] *= 0.50 
            
            if abs(self.radar_angle - bin_angle) < 0.05 and abs(self.target_local[0] - 0.24) < 0.01:
                self.state = "LOWER_TO_BIN"

        elif self.state == "LOWER_TO_BIN":
            self.target_local[2] -= 0.003 
            if self.target_local[2] <= 0.160: 
                self.state = "BOMB_DROP"
                self.timer = 0

        elif self.state == "BOMB_DROP":
            self.timer += 1
            self.move_gripper(self.OPEN_WIDTH) 
            if self.timer > 20: 
                self.state = "LIFT_AWAY"

        elif self.state == "LIFT_AWAY":
            self.target_local[2] += 0.006 
            if self.target_local[2] >= self.transit_z:
                self.state = "SCAN"

        if self.radar_angle > 1.57: self.radar_angle = 1.57
        if self.radar_angle < -4.71: self.radar_angle = -4.71

        self.target_local[0] = np.clip(self.target_local[0], 0.05, 0.35)
        self.target_local[1] = np.clip(self.target_local[1], -0.15, 0.15)

        with torch.no_grad():
            raw = self.ik(torch.tensor([self.target_local], dtype=torch.float32)).numpy()[0]
            target_j1 = raw[0] + self.radar_angle
            
            alpha = 0.35
            self.cur_angles[0] = (1.0 - alpha) * self.cur_angles[0] + alpha * target_j1
            self.cur_angles[1] = (1.0 - alpha) * self.cur_angles[1] + alpha * raw[1]
            self.cur_angles[2] = (1.0 - alpha) * self.cur_angles[2] + alpha * raw[2]
            self.cur_angles[3] = (1.0 - alpha) * self.cur_angles[3] + alpha * raw[3]
            
            for i in range(4): self.pubs[f'j{i+1}'].publish(Float64(data=float(self.cur_angles[i])))

def main():
    rclpy.init()
    rclpy.spin(AIBrainNode())
    rclpy.shutdown()

if __name__ == '__main__': main()