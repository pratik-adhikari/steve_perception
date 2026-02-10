#!/usr/bin/env python3
"""
SAM3 Detector Node for ROS2
Subscribes to camera topics and performs text-prompted object detection using SAM3 Docker.

Usage:
    ros2 run steve_perception sam3_detector --ros-args \\
        -p camera_topic:=/head_camera/color/image_raw \\
        -p query:="bottle"
"""

import base64
import io
from typing import List, Optional

import cv2
import numpy as np
import rclpy
import requests
from cv_bridge import CvBridge
from geometry_msgs.msg import Point
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Header
from visualization_msgs.msg import Marker, MarkerArray


class SAM3DetectorNode(Node):
    """ROS2 node for SAM3-based object detection from camera feeds."""
    
    def __init__(self):
        super().__init__('sam3_detector')
        
        # Parameters
        self.declare_parameter('camera_topic', '/head_camera/color/image_raw')
        self.declare_parameter('sam3_url', 'http://localhost:5005')
        self.declare_parameter('query', 'bottle')
        self.declare_parameter('publish_rate', 1.0)  # Hz
        self.declare_parameter('visualize', True)
        
        self.camera_topic = self.get_parameter('camera_topic').value
        self.sam3_url = self.get_parameter('sam3_url').value
        self.query = self.get_parameter('query').value
        self.publish_rate = self.get_parameter('publish_rate').value
        self.visualize = self.get_parameter('visualize').value
        
        # CV Bridge
        self.bridge = CvBridge()
        
        # Latest image
        self.latest_image: Optional[np.ndarray] = None
        self.latest_header: Optional[Header] = None
        
        # Subscribers
        self.image_sub = self.create_subscription(
            Image,
            self.camera_topic,
            self.image_callback,
            10
        )
        
        # Publishers
        self.viz_pub = self.create_publisher(Image, '/sam3/visualization', 10)
        self.marker_pub = self.create_publisher(MarkerArray, '/sam3/markers', 10)
        
        # Timer for detection
        timer_period = 1.0 / self.publish_rate
        self.timer = self.create_timer(timer_period, self.detection_callback)
        
        # Check SAM3 availability
        if not self.check_sam3():
            self.get_logger().error('SAM3 server not available!')
            self.get_logger().error('Start with: cd ~/steve_ros2_ws/src/steve_perception/docker/sam3 && docker compose up -d')
        else:
            self.get_logger().info(f'SAM3 Detector initialized')
            self.get_logger().info(f'  Camera: {self.camera_topic}')
            self.get_logger().info(f'  Query: "{self.query}"')
            self.get_logger().info(f'  Rate: {self.publish_rate} Hz')
    
    def check_sam3(self) -> bool:
        """Check if SAM3 server is available."""
        try:
            response = requests.get(f'{self.sam3_url}/health', timeout=2)
            return response.status_code == 200
        except:
            return False
    
    def image_callback(self, msg: Image):
        """Store latest camera image."""
        try:
            self.latest_image = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
            self.latest_header = msg.header
        except Exception as e:
            self.get_logger().error(f'Image conversion error: {e}')
    
    def encode_image(self, image: np.ndarray) -> str:
        """Encode image to base64."""
        _, buffer = cv2.imencode('.jpg', image, [cv2.IMWRITE_JPEG_QUALITY, 90])
        return base64.b64encode(buffer).decode('utf-8')
    
    def detection_callback(self):
        """Periodic detection callback."""
        if self.latest_image is None:
            self.get_logger().warn(f'No image received on {self.camera_topic}', throttle_duration_sec=5.0)
            return
        
        try:
            # Call SAM3 API
            img_b64 = self.encode_image(self.latest_image)
            response = requests.post(
                f'{self.sam3_url}/segment_text',
                json={'image': img_b64, 'prompt': self.query},
                timeout=30
            )
            
            if response.status_code != 200:
                self.get_logger().error(f'SAM3 API error: {response.text}')
                return
            
            data = response.json()
            masks = data.get('masks', [])
            
            self.get_logger().info(f'Detected {len(masks)} objects for query "{self.query}"')
            
            if self.visualize and len(masks) > 0:
                self.publish_visualization(masks)
                self.publish_markers(masks)
        
        except Exception as e:
            self.get_logger().error(f'Detection error: {e}')
    
    
    def publish_visualization(self, masks: List[dict]):
        """Publish annotated image."""
        if self.latest_image is None:
            return
        
        # Create visualization
        vis_image = self.latest_image.copy().astype(np.float32)
        h, w = vis_image.shape[:2]
        
        # Generate colors
        np.random.seed(42)
        colors = np.random.randint(0, 255, (len(masks), 3))
        
        for i, mask_data in enumerate(masks):
            # Decode mask
            mask = self.decode_mask(mask_data['mask'])
            
            # Get mask size and scaling factors
            mask_h, mask_w = mask.shape
            scale_x = w / mask_w
            scale_y = h / mask_h
            
            # Resize mask to match image
            if mask.shape != (h, w):
                mask = cv2.resize(mask.astype(np.uint8), (w, h)) > 0
            
            # Apply color overlay
            color = colors[i % len(colors)]
            for c in range(3):
                vis_image[:, :, c] = np.where(
                    mask,
                    vis_image[:, :, c] * 0.5 + color[c] * 0.5,
                    vis_image[:, :, c]
                )
            
            # Draw bbox - scale from mask coordinates to image coordinates
            bbox = mask_data['bbox']
            if len(bbox) == 4:
                x1 = int(bbox[0] * scale_x)
                y1 = int(bbox[1] * scale_y)
                x2 = int(bbox[2] * scale_x)
                y2 = int(bbox[3] * scale_y)
                
                cv2.rectangle(vis_image, (x1, y1), (x2, y2), color.tolist(), 3)
                
                # Label
                score = mask_data.get('score', 0)
                label = f"#{i+1} {score:.2f}"
                cv2.putText(vis_image, label, (x1, max(y1 - 10, 20)),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, color.tolist(), 2)
        
        # Publish
        vis_msg = self.bridge.cv2_to_imgmsg(vis_image.astype(np.uint8), 'bgr8')
        vis_msg.header = self.latest_header
        self.viz_pub.publish(vis_msg)
    
    def decode_mask(self, mask_b64: str) -> np.ndarray:
        """Decode base64 mask."""
        from PIL import Image as PILImage
        mask_bytes = base64.b64decode(mask_b64)
        mask_img = PILImage.open(io.BytesIO(mask_bytes))
        return np.array(mask_img) > 0
    
    def publish_markers(self, masks: List[dict]):
        """Publish RViz markers for detections."""
        marker_array = MarkerArray()
        h, w = self.latest_image.shape[:2]
        
        for i, mask_data in enumerate(masks):
            # Get mask size for scaling
            mask = self.decode_mask(mask_data['mask'])
            mask_h, mask_w = mask.shape
            scale_x = w / mask_w
            scale_y = h / mask_h
            
            # Scale bbox
            bbox = mask_data['bbox']
            x1 = int(bbox[0] * scale_x)
            y1 = int(bbox[1] * scale_y)
            x2 = int(bbox[2] * scale_x)
            y2 = int(bbox[3] * scale_y)
            
            # Create text marker
            marker = Marker()
            marker.header = self.latest_header
            marker.ns = 'sam3_detections'
            marker.id = i
            marker.type = Marker.TEXT_VIEW_FACING
            marker.action = Marker.ADD
            
            # Position at bbox center (in image coordinates)
            marker.pose.position.x = float((x1 + x2) / 2)
            marker.pose.position.y = float((y1 + y2) / 2)
            marker.pose.position.z = 0.0
            
            # Scale
            marker.scale.z = 0.1
            
            # Color
            marker.color.r = 1.0
            marker.color.g = 1.0
            marker.color.b = 1.0
            marker.color.a = 1.0
            
            # Text
            score = mask_data.get('score', 0)
            marker.text = f"{self.query} {score:.2f}"
            
            marker_array.markers.append(marker)
        
        self.marker_pub.publish(marker_array)


def main(args=None):
    rclpy.init(args=args)
    node = SAM3DetectorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
