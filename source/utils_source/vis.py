"""
Utils for visualization.
"""

from __future__ import annotations

import colorsys
import copy

import numpy as np

import cv2
import open3d as o3d
from matplotlib import pyplot as plt
from utils_source.importer import PointCloud, Vector3dVector
from utils_source.object_detetion import Detection


def normalize_image(image: np.ndarray) -> np.ndarray:
    """
    Normalize image prior to plotting.
    """
    return cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)


def show_image(image: np.ndarray, title: str = "Image"):
    """
    Show an RGB image.
    """
    normalized_image = normalize_image(image)
    cv2.imshow(title, normalized_image)
    while True:
        if cv2.waitKey(1) & 0xFF == 27:  # ESC key is pressed
            break
    cv2.destroyAllWindows()
    return normalized_image


def show_depth_image(depth_image: np.ndarray, title: str = "Depth Image"):
    """
    Show a colormapped depth image.
    """
    min_val = np.min(depth_image)
    max_val = np.max(depth_image)
    depth_range = max_val - min_val
    depth8 = (255.0 / depth_range * (depth_image - min_val)).astype("uint8")
    depth8_rgb = cv2.cvtColor(depth8, cv2.COLOR_GRAY2RGB)
    colored_image = cv2.applyColorMap(depth8_rgb, cv2.COLORMAP_JET)
    return show_image(colored_image, title)


def show_two_geometries_colored(geometry1, geometry2, color1=(1, 0, 0), color2=(0, 1, 0)) -> None:
    """
    Given two open3d geometries, color them and visualize them.
    :param geometry1:
    :param geometry2:
    :param color1: color for first geometry
    :param color2: color for second geometry
    """
    geometry1_colored = copy.deepcopy(geometry1)
    geometry2_colored = copy.deepcopy(geometry2)
    geometry1_colored.paint_uniform_color(color1)
    geometry2_colored.paint_uniform_color(color2)
    o3d.visualization.draw_geometries([geometry1_colored, geometry2_colored])


def show_point_cloud_in_out(points: np.ndarray, in_mask: np.ndarray) -> None:
    pcd = PointCloud()
    pcd.points = Vector3dVector(points)
    pcd_in = pcd.select_by_index(np.where(in_mask)[0])
    pcd_out = pcd.select_by_index(np.where(~in_mask)[0])
    show_two_geometries_colored(pcd_out, pcd_in)


def generate_distinct_colors(n: int) -> list[tuple[float, float, float]]:
    """
    Generate n visually distinct RGB colors.

    Args:
    - n (int): The number of distinct colors to generate.

    Returns:
    - List[tuple[int, int, int]]: A list of tuples representing RGB colors.
    """
    colors = []
    for i in range(n):
        # Divide the hue space into equal parts
        hue = i / n
        # Fixed saturation and lightness for high contrast and brightness
        saturation = 0.7
        lightness = 0.5
        # Convert HSL color to RGB
        r, g, b = colorsys.hls_to_rgb(hue, lightness, saturation)
        colors.append((r, g, b))

    return colors


def draw_boxes(image: np.ndarray, detections: list[Detection]) -> None:
    vis_image = image.copy()
    names = sorted(list(set([det.name for det in detections])))
    names_dict = {name: i for i, name in enumerate(names)}
    colors = generate_distinct_colors(len(names_dict))
    for name, conf, (xmin, ymin, xmax, ymax) in detections:
        color = colors[names_dict[name]]
        xmin, ymin, xmax, ymax = map(int, [xmin, ymin, xmax, ymax])
        cv2.rectangle(vis_image, (xmin, ymin), (xmax, ymax), color, thickness=2)
        label = f"{name}: {conf:.2f}"
        cv2.putText(vis_image, label, (xmin, max(0, ymin - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    cv2.imshow("Detections", vis_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows() 


def get_bbox_lines(centroid, dimensions, pose=None):
    """
    Generate lines for a 3D bounding box given centroid, dimensions, and optional pose.
    Returns x, y, z lists for 3D plotting including None for line breaks.
    """
    # Dimensions in Open3D OBB are usually full extents.
    # Local corners centered at 0
    dx, dy, dz = dimensions[0]/2, dimensions[1]/2, dimensions[2]/2
    
    local_corners = np.array([
        [-dx, -dy, -dz], [dx, -dy, -dz], [dx, dy, -dz], [-dx, dy, -dz], # Bottom 0-3
        [-dx, -dy, dz], [dx, -dy, dz], [dx, dy, dz], [-dx, dy, dz]      # Top 4-7
    ])
    
    # Rotate if pose is provided
    # Pose is 4x4 matrix. Rotation is 3x3 top-left.
    if pose is not None:
        rotation = pose[:3, :3]
        # Although centroid is usually pose[:3, 3], we use the provided centroid argument 
        # which comes from node.centroid, to be consistent with point cloud mean.
        # But OBB center might differ from Point Cloud centroid slightly. 
        # Usually OBB is created from points.
        # Let's rotate corners and add to centroid.
        rotated_corners = np.dot(local_corners, rotation.T)
        corners = rotated_corners + centroid
    else:
        corners = local_corners + centroid
    
    # 12 edges
    lines = [
        (0, 1), (1, 2), (2, 3), (3, 0), # Bottom loop
        (4, 5), (5, 6), (6, 7), (7, 4), # Top loop
        (0, 4), (1, 5), (2, 6), (3, 7)  # Pillars
    ]
    
    x, y, z = [], [], []
    for start, end in lines:
        x.extend([corners[start, 0], corners[end, 0], None])
        y.extend([corners[start, 1], corners[end, 1], None])
        z.extend([corners[start, 2], corners[end, 2], None])
        
    return x, y, z


def visualize_scene_graph_interactive(scene_graph, output_path: str):
    """
    Generate an interactive 3D visualization of the scene graph using Plotly.
    Saved as an HTML file.
    """
    try:
        import plotly.graph_objects as go
    except ImportError:
        print("Plotly is not installed. Skipping interactive visualization.")
        return

    from utils_source.preprocessing_utils.graph_nodes import DrawerNode

    fig = go.Figure()
    pos = {}

    def add_object_trace(oid, label, centroid, dims, color, symbol='circle', pose=None):
        pos[oid] = centroid
        
        # Centroid
        fig.add_trace(go.Scatter3d(
            x=[centroid[0]], y=[centroid[1]], z=[centroid[2]],
            mode='markers+text',
            marker=dict(size=5, color=color, symbol=symbol),
            text=[f"{oid}:{label}"],
            textposition="top center",
            name=f"{oid}:{label}",
            hoverinfo='text'
        ))
        
        # Bounding Box
        if dims is not None:
             # handle dimensions usually being (l, w, h) or similar
             # Ensure dims is numpy array or list
             bx, by, bz = get_bbox_lines(centroid, dims, pose)
             fig.add_trace(go.Scatter3d(
                x=bx, y=by, z=bz,
                mode='lines',
                line=dict(color=color, width=2),
                showlegend=False,
                hoverinfo='skip'
            ))

    # Iterate nodes
    for node in scene_graph.nodes.values():
        label = scene_graph.label_mapping.get(node.sem_label, "ID not found")
        
        # Get color (0-1 float or 0-255 int depending on source)
        # Using rgb string for plotly
        c = node.color
        if np.max(c) <= 1.0:
            color_str = f'rgb({int(c[0]*255)},{int(c[1]*255)},{int(c[2]*255)})'
        else:
            color_str = f'rgb({int(c[0])},{int(c[1])},{int(c[2])})'

        # Helper to get pose safely
        pose = getattr(node, 'bbox_pose', getattr(node, 'pose', None))
        # If node has 'bb' (oriented bbox), we might want its center/rotation.
        # node.dimensions usually comes from get_dimensions() which sets self.bb = obb
        # and self.dimensions = obb.extent.
        # But obb has its own center/rotation.
        # If we use node.centroid and node.pose, it should align if node.pose is the object pose.
        # For DrawerNode, check constraints.
        
        # Determine format/color
        if isinstance(node, DrawerNode):
             # Drawers
             add_object_trace(node.object_id, label, node.centroid, node.dimensions, color_str, 'diamond', pose=pose)
        else:
             # Movable or Furniture
             add_object_trace(node.object_id, label, node.centroid, getattr(node, 'dimensions', None), color_str, 'circle', pose=pose)

    # Connections
    cx, cy, cz = [], [], []
    for src, target in scene_graph.outgoing.items():
        if src in pos and target in pos:
            p1 = pos[src]
            p2 = pos[target]
            cx.extend([p1[0], p2[0], None])
            cy.extend([p1[1], p2[1], None])
            cz.extend([p1[2], p2[2], None])

    fig.add_trace(go.Scatter3d(
        x=cx, y=cy, z=cz,
        mode='lines',
        line=dict(color='black', width=3, dash='dash'),
        name='Connections'
    ))

    fig.update_layout(
        title="Interactive 3D Scene Graph",
        scene=dict(
            xaxis_title='X',
            yaxis_title='Y',
            zaxis_title='Z',
            aspectmode='data'
        ),
        margin=dict(r=0, l=0, b=0, t=40)
    )

    print(f"[Visualization] Saving interactive graph to {output_path}")
    fig.write_html(output_path)
