import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2

class CameraReader(Node):
    def __init__(self):
        super().__init__('camera_reader')
        self.bridge = CvBridge()
        self.create_subscription(Image, '/robot_eye', self.image_callback, 10)

    def image_callback(self, msg):
        cv2.imshow("Robot Eyes Monitor", self.bridge.imgmsg_to_cv2(msg, "bgr8"))
        cv2.waitKey(1)

def main(args=None):
    rclpy.init(args=args)
    rclpy.spin(CameraReader())
    cv2.destroyAllWindows()
    rclpy.shutdown()

if __name__ == '__main__':
    main()