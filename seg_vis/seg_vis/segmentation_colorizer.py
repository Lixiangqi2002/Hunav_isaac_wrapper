import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import numpy as np


class SegmentationColorizer(Node):
    def __init__(self):
        super().__init__('segmentation_colorizer')

        self.bridge = CvBridge()

        # -------------------------------
        # Topics (hardcoded dual mapping)
        # -------------------------------
        self.instance_input = '/segmentation_instance'
        self.instance_output = '/instance_segmentation_color'

        self.semantic_input = '/segmentation_semantic'
        self.semantic_output = '/semantic_segmentation_color'

        # -------------------------------
        # Subscribers
        # -------------------------------
        self.sub_instance = self.create_subscription(
            Image,
            self.instance_input,
            self.instance_callback,
            10
        )

        self.sub_semantic = self.create_subscription(
            Image,
            self.semantic_input,
            self.semantic_callback,
            10
        )

        # -------------------------------
        # Publishers
        # -------------------------------
        self.pub_instance = self.create_publisher(
            Image,
            self.instance_output,
            10
        )

        self.pub_semantic = self.create_publisher(
            Image,
            self.semantic_output,
            10
        )

        self.get_logger().info(f'[Instance] {self.instance_input} -> {self.instance_output}')
        self.get_logger().info(f'[Semantic] {self.semantic_input} -> {self.semantic_output}')

    # ============================================================
    # Color mapping
    # ============================================================
    def id_to_color(self, seg: np.ndarray) -> np.ndarray:
        """
        Convert segmentation ID map to RGB image.

        Args:
            seg (np.ndarray): HxW int32 segmentation map

        Returns:
            np.ndarray: HxWx3 uint8 RGB image
        """
        h, w = seg.shape
        color = np.zeros((h, w, 3), dtype=np.uint8)

        unique_ids = np.unique(seg)

        for obj_id in unique_ids:
            if obj_id <= 0:
                rgb = (0, 0, 0)
            else:
                # Deterministic color mapping (stable across frames)
                r = (obj_id * 53) % 256
                g = (obj_id * 97) % 256
                b = (obj_id * 193) % 256
                rgb = (r, g, b)

            color[seg == obj_id] = rgb

        return color

    # ============================================================
    # Callbacks
    # ============================================================
    def instance_callback(self, msg: Image):
        self.process_image(msg, self.pub_instance, tag="Instance")

    def semantic_callback(self, msg: Image):
        self.process_image(msg, self.pub_semantic, tag="Semantic")

    def process_image(self, msg: Image, publisher, tag=""):
        """
        Convert segmentation image to colored image and publish.
        """
        try:
            seg = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
            seg = np.array(seg)

            if seg.ndim != 2:
                self.get_logger().warning(f'[{tag}] Expected 2D image, got {seg.shape}')
                return

            color = self.id_to_color(seg)

            out_msg = self.bridge.cv2_to_imgmsg(color, encoding='rgb8')
            out_msg.header = msg.header
            publisher.publish(out_msg)

        except Exception as e:
            self.get_logger().error(f'[{tag}] Failed: {e}')


def main(args=None):
    rclpy.init(args=args)
    node = SegmentationColorizer()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
