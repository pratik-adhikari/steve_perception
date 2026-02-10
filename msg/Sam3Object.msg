# SAM3 Object Detection Result
# Represents a single detected object from SAM3

int32 id                      # Unique ID for this detection
float32 score                 # Confidence score (0.0 - 1.0)
geometry_msgs/Point centroid  # 2D centroid (x, y, 0)
int32[4] bbox                 # Bounding box [x1, y1, x2, y2]
sensor_msgs/Image mask        # Binary segmentation mask
