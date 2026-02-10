# SAM3 Detection Results
# Contains all objects detected from a single text query

std_msgs/Header header        # Timestamp and frame_id
string query                  # Text query used ("bottle", "cup", etc.)
Sam3Object[] objects          # List of detected objects
sensor_msgs/Image visualization  # Annotated image with overlays
