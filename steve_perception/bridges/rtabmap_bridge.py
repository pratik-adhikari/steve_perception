from typing import Dict, Optional
from rclpy.node import Node
from rtabmap_msgs.msg import RGBDImage
from sensor_msgs.msg import CameraInfo
from rclpy.qos import qos_profile_sensor_data

# Import our frame types
from steve_perception.core.frame_types import RGBDFrame
from cv_bridge import CvBridge
from sensor_msgs.msg import CompressedImage
import numpy as np

class RtabmapBridge:
    """
    Adapter that listens to CameraAgent frames and publishes RTAB-Map compatible RGBDImage messages.
    """

    def __init__(self, node: Node, cameras_cfg: Dict[str, dict]):
        """
        Initialize the bridge with publishers for each camera.
        
        Args:
            node: The ROS 2 node instance (PerceptionNode)
            cameras_cfg: Dictionary of camera configurations
        """
        self.node = node
        self.log = node.get_logger()
        self.pubs = {}
        self.frame_id_to_name = {}

        for cam_name, cam_cfg in cameras_cfg.items():
            # Determine output topic
            out_topic = cam_cfg.get("rgbd_topic")
            if not out_topic:
                out_topic = f"/steve_perception/{cam_name}/rgbd_image"
            
            # Create publisher
            self.pubs[cam_name] = node.create_publisher(RGBDImage, out_topic, 10)
            self.log.info(f"[RtabmapBridge] Publishing {cam_name} -> {out_topic}")

            # Map frame_id to camera name for routing
            frame_id = str(cam_cfg.get("frame_id"))
            self.frame_id_to_name[frame_id] = cam_name

        self._cv_bridge = CvBridge()

    def on_frame(self, frame: RGBDFrame) -> None:
        """
        Callback invoked by CameraAgent when a new synchronized frame is ready.
        """
        cam_name = self.frame_id_to_name.get(frame.frame_id)
        if not cam_name:
            # Check if we can find it by other means or log warning
            # Some setups might have matching frame_ids, or we search by prefix?
            # For now strict mapping:
            return

        publisher = self.pubs.get(cam_name)
        if not publisher:
            return

        # Prepare RGBDImage message
        # frame.rgb_msg, frame.depth_msg, frame.camera_info_msg are cached from source
        if frame.rgb_msg is None or frame.depth_msg is None:
            return

        msg = RGBDImage()
        msg.header = frame.rgb_msg.header
        # Ensure frame_id matches the optical frame expected by RTAB-Map
        # Ensure frame_id matches the optical frame expected by RTAB-Map
        msg.header.frame_id = frame.frame_id
        
        # JPEG Compress RGB to reduce message size (prevent DDS drops)
        if frame.rgb is not None:
            try:
                # fram.rgb is BGR8 (from CameraAgent)
                msg.rgb_compressed = self._cv_bridge.cv2_to_compressed_imgmsg(frame.rgb, dst_format='jpg')
                msg.rgb_compressed.header = msg.header
                msg.rgb_compressed.format = "jpeg"
            except Exception as e:
                self.log.warn(f"Failed to compress RGB: {e}")
                # Fallback to raw if compression fails
                msg.rgb = frame.rgb_msg
        else:
             msg.rgb = frame.rgb_msg

        # Switch to Raw 16UC1 (Millimeters)
        # This is roughly 600KB (640*480*2 bytes), which is safe for UDP.
        # It avoids the overhead and potential decoding bugs of PNG compression.
        if frame.depth_m is not None:
             depth_mm = (frame.depth_m * 1000.0).astype(np.uint16)
             # Use standard cvt to create a sensor_msgs/Image
             msg.depth = self._cv_bridge.cv2_to_imgmsg(depth_mm, encoding="16UC1")
             msg.depth.header = msg.header
        else:
             msg.depth = frame.depth_msg

        # Use cached info but stamp consistently
        info = frame.camera_info_msg
        if info is not None:
            # We must copy or modify the info to match the RGB timestamp
            # Ideally we don't mutate the cached object in the frame, 
            # but CameraInfo is a simple message.
            # Let's create a shallow copy logic by assigning fields if needed, 
            # but here assigning the object reference + headers is standard for Python msg.
            # To be safe, we assign headers.
            info.header = msg.header
            info.header.frame_id = frame.frame_id

            msg.rgb_camera_info = info
            msg.depth_camera_info = info

        publisher.publish(msg)
