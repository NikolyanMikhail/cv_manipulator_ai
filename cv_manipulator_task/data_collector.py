import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from rclpy.qos import qos_profile_sensor_data
import cv2
import os
import time

class DataCollector(Node):
    def __init__(self):
        super().__init__('data_collector')
        self.bridge = CvBridge()
        self.save_dir = os.path.expanduser('~/robot_ws/dataset/images')
        os.makedirs(self.save_dir, exist_ok=True)
        self.create_subscription(Image, '/camera/image_raw', self.image_callback, qos_profile_sensor_data)
        
        self.img_counter = 0
        self.last_save_time = time.time()
        self.save_interval = 1.5 
        
        self.get_logger().info(f"📸 Фотоаппарат запущен! Сохраняю фото в папку: {self.save_dir}")
        self.get_logger().info("Перейди в Gazebo и двигай кубик. Для остановки съемки нажми Ctrl+C здесь.")

    def image_callback(self, msg):
        current_time = time.time()
        frame = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        display_frame = frame.copy()
        if (int(current_time * 2) % 2) == 0:
            cv2.circle(display_frame, (20, 20), 10, (0, 0, 255), -1)
            
        cv2.imshow("Data Collector - REC", display_frame)
        cv2.waitKey(1)
        if current_time - self.last_save_time >= self.save_interval:
            filename = os.path.join(self.save_dir, f"cube_frame_{self.img_counter:04d}.jpg")
            cv2.imwrite(filename, frame)
            self.get_logger().info(f"💾 Снято: {filename}")
            self.img_counter += 1
            self.last_save_time = current_time

def main(args=None):
    rclpy.init(args=args)
    node = DataCollector()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Съемка завершена!")
    finally:
        cv2.destroyAllWindows()
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()