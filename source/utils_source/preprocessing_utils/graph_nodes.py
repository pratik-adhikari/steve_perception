import numpy as np
from scipy.spatial import KDTree
import open3d as o3d
from typing import Optional

class ObjectNode:
    """
    Represents a generic object within a 3D scene, storing properties such as geometry, 
    semantic information, and state attributes.

    The ObjectNode class serves as a base for other scene objects, providing properties 
    like centroid calculation, color, confidence, and a convex hull tree for spatial queries.
    This class can be extended to add specific functionalities for different types of objects.

    Attributes:
        object_id (int): Unique identifier for the object.
        color (tuple): RGB color representation of the object.
        sem_label (str): Semantic label categorizing the object (e.g., "drawer," "table").
        centroid (np.ndarray): The centroid of the object, computed from the provided 3D points.
        points (np.ndarray): Array of 3D points defining the object's geometry.
        mesh_mask (np.ndarray): Binary mask representing the object's mesh.
        confidence (float, optional): Confidence score associated with the detection.
        movable (bool): Indicates if the object is movable. Defaults to True.
        hull_tree (spatial.KDTree): Spatial KD-tree structure for the object's convex hull.
        pose (np.ndarray): Estimated pose matrix for the object based on its points and centroid.
        bb (np.ndarray): Axis-aligned or oriented bounding box depending on which is smaller.
        dimensions (np.ndarray): Dimensions of the object, derived from its bounding box (width, depth, height).
    """

    def __init__(self, object_id: int, color: tuple, sem_label: str, points: np.ndarray, mesh_mask: np.ndarray, confidence: float = 1.0, movable: bool = True, yolo_box: Optional[dict] = None, source_imgs: list = None):
        """
        Initializes an ObjectNode with specified attributes.

        :param object_id: Unique identifier for the object.
        :param color: RGB color tuple associated with the object.
        :param sem_label: Semantic label of the object.
        :param points: Point cloud data for the object.
        :param mesh_mask: Boolean mask indicating the object's presence in the mesh.
        :param confidence: Confidence score of the detection. Defaults to 1.0.
        :param movable: Boolean indicating if the object is movable. Defaults to True.
        :param yolo_box: Optional dictionary containing 'center', 'extent', 'rotation' from YOLO 3D detection.
        :param source_imgs: List of source image filenames.
        """
        self.object_id = object_id
        self.color = color
        self.sem_label = sem_label
        self.points = points
        self.mesh_mask = mesh_mask
        self.confidence = confidence
        self.movable = movable
        self.centroid = np.mean(points, axis=0) if len(points) > 0 else np.zeros(3)
        self.bbox_pose = None # Will be set if yolo_box is provided
        self.source_imgs = source_imgs # Track source images
        
        self.update_hull_tree()
        self.compute_pose(self.points, self.centroid)
        self.get_dimensions()
        
        # [Fix for 2D Square Objects]
        if yolo_box:
            center = np.array(yolo_box['center'])
            extent = np.array(yolo_box['extent'])
            rotation = np.array(yolo_box['rotation'])
            
            # Create OBB from YOLO data
            self.bb = o3d.geometry.OrientedBoundingBox(center, rotation, extent)
            self.bb.color = (0, 0, 1) # Blue
            self.dimensions = extent
            # Note: ObjectNode uses self.bb, DrawerNode uses self.box. A bit inconsistent but keeping legacy.
            
            # [Fix for Visualization] Sync self.pose with OBB pose so downstream visualizers use the correct rotation
            self.pose = np.eye(4)
            self.pose[:3, :3] = rotation
            self.pose[:3, 3] = center
        
    
    def update_hull_tree(self) -> None:
        """
        Updates the convex hull tree for the object in place using a KD-tree.
        
        This method constructs or updates a KD-tree based on the object's 3D points, allowing for 
        efficient spatial queries within the object's convex hull.
        """
        self.hull_tree = KDTree(self.points)
        
    def compute_pose(self, points: np.ndarray, centroid: np.ndarray) -> None:
        """
        Computes the pose of an object given its 3D points and centroid.
        
        This function calculates the orientation and position of an object in 3D space by aligning 
        the principal axes of its points to the global axes using PCA.

        :param points: Array of 3D points representing the object's geometry.
        :param centroid: The centroid of the object as a 3D point.
        """
        points_centered = points - centroid
        covariance_matrix = np.cov(points_centered, rowvar=False)
        eigenvalues, eigenvectors = np.linalg.eigh(covariance_matrix)
        sorted_idx = np.argsort(eigenvalues)[::-1]
        eigenvectors = eigenvectors[:, sorted_idx]
        R = eigenvectors
        if np.linalg.det(R) < 0:
            R[:, -1] *= -1
        object_pose = np.eye(4)
        object_pose[:3, :3] = R
        object_pose[:3, 3] = centroid
        self.pose = object_pose
        
    # def get_z_aligned_obb(self) -> None:
    #     xy_points = self.points[:, :2]
    #     cov_matrix = np.cov(xy_points, rowvar=False)
    #     eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
        
    #     rotation_matrix = np.eye(3)
    #     rotation_matrix[:2, :2] = eigenvectors 
    #     rotated_points = self.points @ rotation_matrix.T

    #     min_bound = rotated_points.min(axis=0)
    #     max_bound = rotated_points.max(axis=0)

    #     center = (min_bound + max_bound) / 2
    #     center[2] = (self.points[:, 2].min() + self.points[:, 2].max()) / 2  # Keep original Z center
    #     extents = max_bound - min_bound
        
    #     self.bb = o3d.geometry.OrientedBoundingBox(center, rotation_matrix, extents)
    #     self.dimensions = extents     
        
    def get_dimensions(self) -> None:
        """
        Computes the dimensions of the object based on its bounding box.
        
        This function calculates the axis aligned and oriented bounding box and takes the smaller one.
        """
        point_cloud = o3d.geometry.PointCloud()
        point_cloud.points = o3d.utility.Vector3dVector(self.points)

        # obb = point_cloud.get_minimal_oriented_bounding_box()
        # [User Request] Changed to get_oriented_bounding_box because get_minimal_oriented_bounding_box was removed in Open3D 0.16.0
        obb = point_cloud.get_oriented_bounding_box()
        height_idx = np.argmax(np.abs(obb.R.T @ [0, 0, 1]))
        width_depth = sorted([i for i in range(3) if i != height_idx], key=lambda i: obb.extent[i], reverse=True)
        order =  width_depth + [height_idx]
        obb_extent = obb.extent[order]
        
        self.bb = obb
        self.dimensions = obb_extent
    
    def transform(self, transformation: np.ndarray, force: bool = False) -> None:
        """
        Applies a transformation to the node in place. 
        
        If the `force` flag is set to True, the transformation is 
        applied even if certain conditions might normally prevent it (e.g., immovable objects).

        :param transformation: A translation, a 3x3 rotation or a 4x4 homogeneous transformation matrix to apply to the node's points.
        :param force: Flag to force the transformation, regardless of the node's movable status. Defaults to False.
        """
        if isinstance(transformation, np.ndarray):
            if transformation.shape == (3,):
                self.centroid += transformation
                self.points += transformation
                self.pose[:3, 3] += transformation
            elif transformation.shape == (3, 3):
                self.points = np.dot(transformation, self.points.T).T
                self.centroid = np.dot(transformation, self.centroid)
                self.pose = np.dot(transformation, self.pose[:3, :3])
            elif transformation.shape == (4, 4):
                self.points = np.dot(transformation, np.vstack((self.points.T, np.ones(self.points.shape[0])))).T[:, :3]
                self.centroid = np.dot(transformation, np.append(self.centroid, 1))[:3]
                self.pose = np.dot(transformation, self.pose)  
            else:
                raise ValueError("Invalid argument shape. Expected (3,) for translation, (3,3) for rotation, or (4,4) for homogeneous transformation.")
        else:
            raise TypeError("Invalid argument type. Expected numpy.ndarray.")
        
        self.update_hull_tree()
        self.get_dimensions()


class DrawerNode(ObjectNode):
    """
    Represents a drawer in the 3D scene, inheriting properties and methods from ObjectNode.

    The DrawerNode class includes additional properties and methods specific to drawers, such as 
    plane segmentation, containment relationships, and sign checking. The plane segmentation helps 
    define the drawer's orientation and position in the scene, and `contains` tracks items related 
    to the drawer.

    Attributes:
        equation (tuple): Plane equation of the drawer derived via RANSAC segmentation.
        box (optional): 3D bounding box of the drawer, if applicable.
        belongs_to (optional): Parent object or relationship attribute.
        contains (list): List of objects contained within the drawer.
    """

    def __init__(self, object_id: int, color: tuple, sem_label: str, points: np.ndarray, mesh_mask: np.ndarray, confidence: float = 1.0, movable: bool = True, yolo_box: Optional[dict] = None, source_imgs: list = None):
        """
        Initializes a DrawerNode with specified attributes and performs plane segmentation.

        :param yolo_box: Optional dictionary containing 'center', 'extent', 'rotation' from YOLO 3D detection.
                         If provided, forces the box dimensions instead of heuristic calculation.
        :param source_imgs: List of source image filenames used for detection.
        """
        super().__init__(object_id, color, sem_label, points, mesh_mask, confidence, movable, yolo_box=yolo_box, source_imgs=source_imgs)
        pcd = o3d.geometry.PointCloud()
        if len(points) > 0:
            pcd.points = o3d.utility.Vector3dVector(points)
            self.equation, _ = pcd.segment_plane(distance_threshold=0.02, ransac_n=3, num_iterations=1000)
        else:
            self.equation = np.array([0, 0, 1, 0]) # Default up vector if no points

        self.box = None
        self.belongs_to = None
        self.contains = []
        
        # [Fix for 2D Square Drawers]
        if yolo_box:
            center = np.array(yolo_box['center'])
            extent = np.array(yolo_box['extent'])
            rotation = np.array(yolo_box['rotation'])
            
            # Create OBB from YOLO data
            self.box = o3d.geometry.OrientedBoundingBox(center, rotation, extent)
            self.box.color = (0, 0, 1) # Blue
            self.dimensions = extent
            # Also update centroid to match box center if desired? 
            # self.centroid = center # Actually, let's keep centroid as point average, but box as "True" extent
            
            # [Fix for Visualization] Sync self.pose with OBB pose
            self.pose = np.eye(4)
            self.pose[:3, :3] = rotation
            self.pose[:3, 3] = center
    
    def sign_check(self, point: np.ndarray) -> bool:
        """
        Determines whether a given point lies on the positive side of the object's plane.
        
        This method checks if the provided 3D point is located on the positive side of the plane 
        represented by the object's plane equation. It uses the dot product of the plane normal 
        with the point coordinates and an offset to make this determination.

        :param point: A 3D point as a numpy array to be checked against the plane equation.
        :return: True if the point lies on the positive side of the plane, False otherwise.
        """
        return np.dot(self.equation[:3], point) + self.equation[3] > 0
    
    def add_box(self, shelf_centroid: np.ndarray) -> None:
        """
        Adds a bounding box to indicate the drawer's spatial appearance.
        
        Adds the box attribute of the DrawerNode based on a heuristic by computing the intersection with a parallel
        plane (to the initially estimated Drawer plane) anchored at the shelf centroid.

        :param shelf_centroid: A 3D numpy array representing the centroid of the shelf, used as a reference.
        """
        intersection = self.compute_intersection(shelf_centroid)
        
        bbox_points = []
        for point in self.points:
            bbox_points.append(point)
            bbox_points.append(point + 2* (shelf_centroid - intersection))

        points = np.array(bbox_points)

        tmp_pcd = o3d.geometry.PointCloud()
        tmp_pcd.points = o3d.utility.Vector3dVector(points)
          
        # self.box = tmp_pcd.get_minimal_oriented_bounding_box()
        # [User Request] Changed to get_oriented_bounding_box because get_minimal_oriented_bounding_box was removed in Open3D 0.16.0
        self.box = tmp_pcd.get_oriented_bounding_box()
    
    def compute_intersection(self, ray_start: np.ndarray) -> Optional[np.ndarray]:
        """
        This method calculates the intersection point of a ray, starting from `ray_start`, with the plane defined by the object's equation.

        :param ray_start: A 3D numpy array representing the starting point of the ray.
        :return: A 3D numpy array representing the intersection point with the plane, or None if the ray is parallel to the plane.
        """
        signed_distance = (np.dot(self.equation[:3], ray_start) + self.equation[3]) / np.linalg.norm(self.equation[:3])
        
        if signed_distance > 0:
            direction = -self.equation[:3]
        else:
            direction = self.equation[:3]

        numerator = - (np.dot(self.equation[:3], ray_start) + self.equation[3])
        denominator = np.dot(self.equation[:3], direction)

        if denominator == 0:
            print("The ray is parallel to the plane and does not intersect it.")
            return
        
        t = numerator / denominator
        intersection_point = ray_start + t * direction

        return intersection_point
    
    def transform(self, transformation: np.ndarray, force: bool = False) -> None:
        """
        Applies a transformation to the drawer node in place. 
        
        The DrawerNode is restricted in a way that it can only be moved along the plane normal.
        Compared to the ObjectNode, the DrawerNode's box is also updated based on the transformation.
        If the `force` flag is set to True, arbitrary transformations are allowed.

        :param transformation: A translation, a 3x3 rotation or a 4x4 homogeneous transformation matrix to apply to the node's points.
        :param force: Flag to force the transformation, regardless of the node's movable status. Defaults to False.
        """
        if force:
            super().transform(transformation)
            if isinstance(transformation, np.ndarray):
                if transformation.shape == (3,):
                    self.box.translate(transformation)
                elif transformation.shape == (4, 4):
                    translation = transformation[:3, 3]
                    rotation = transformation[:3, :3]
                    self.box = self.box.rotate(rotation, center=np.array([0, 0, 0]))
                    self.box.translate(translation)
        else:
            if isinstance(transformation, np.ndarray) and (transformation.shape == (3,) or transformation.shape == (4, 4)):
                normal = self.equation[:3]
                normal /= np.linalg.norm(normal)
                new_location = np.dot(transformation, np.append(self.centroid, 1))[:3] - self.centroid
                translation = np.dot(new_location, normal) * normal
                self.centroid += translation
                self.points += translation
                self.tracking_points += translation
                self.pose[:3, 3] += translation
                self.box.translate(translation)
                self.update_hull_tree()
                for node in self.contains:
                    node.transform(translation)
            else:
                raise TypeError("Invalid argument type. Expected numpy.ndarray of shape (3,) or (4,4).")
