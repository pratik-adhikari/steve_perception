
import open3d as o3d
import numpy as np
import os
import sys

def analyze_cloud(path, label):
    print(f"--- Analyzing {label} ---")
    print(f"Path: {path}")
    
    if not os.path.exists(path):
        print("FILE NOT FOUND!")
        return None

    # Load
    try:
        if path.endswith('.obj'):
            mesh = o3d.io.read_triangle_mesh(path)
            if mesh.has_vertices():
                pcd = o3d.geometry.PointCloud()
                pcd.points = mesh.vertices
                if mesh.has_vertex_colors():
                    pcd.colors = mesh.vertex_colors
            else:
                print("Empty OBJ")
                return None
        else:
            pcd = o3d.io.read_point_cloud(path)
    except Exception as e:
        print(f"Load Error: {e}")
        return None
        
    if not pcd.has_points():
        print("No points found.")
        return None
        
    # Metrics
    points = np.asarray(pcd.points)
    count = points.shape[0]
    
    # Extent
    min_bound = pcd.get_min_bound()
    max_bound = pcd.get_max_bound()
    extent = max_bound - min_bound
    volume = extent[0] * extent[1] * extent[2]
    
    # Density (approximate via Nearest Neighbor)
    # Sampling 1000 points for speed
    if count > 1000:
        sample_indices = np.random.choice(count, 1000, replace=False)
        sample_pcd = pcd.select_by_index(sample_indices)
    else:
        sample_pcd = pcd
        
    pcd_tree = o3d.geometry.KDTreeFlann(pcd)
    distances = []
    for p in np.asarray(sample_pcd.points):
        [k, idx, _] = pcd_tree.search_knn_vector_3d(p, 2) # 2 because 1st is itself
        if k > 1:
            p2 = points[idx[1]]
            dist = np.linalg.norm(p - p2)
            distances.append(dist)
            
    mean_nn_dist = np.mean(distances) if distances else 0.0
    
    # Color
    has_color = pcd.has_colors()
    color_stats = "None"
    if has_color:
        colors = np.asarray(pcd.colors)
        mean_color = np.mean(colors, axis=0)
        std_color = np.std(colors, axis=0)
        # Check if monochrome
        is_gray = np.allclose(colors[:,0], colors[:,1], atol=0.01) and np.allclose(colors[:,1], colors[:,2], atol=0.01)
        color_stats = f"Mean: {mean_color}, Std: {std_color}, IsGray: {is_gray}"

    print(f"Point Count: {count}")
    print(f"Extent: {extent}")
    print(f"Volume: {volume:.4f}")
    print(f"Mean Nearest Neighbor Dist (Resolution): {mean_nn_dist:.6f}")
    print(f"Has Color: {has_color}")
    print(f"Color Stats: {color_stats}")
    print("\n")
    
    return {
        "label": label,
        "count": count,
        "volume": volume,
        "resolution": mean_nn_dist,
        "has_color": has_color,
        "is_gray": is_gray if has_color else False
    }

# Files to Compare
files = [
    ("/home/user/adhikarip1/steve_ws/steve_ws2/src/stretch-compose/data/data/autowalk_scans/old/2025_08_21/pointcloud_20250821_143231.ply", "Legacy Scan (Sample)"),
    ("data/test_data/export/scene.ply", "New Optimized Export (scene.ply)"),
    ("data/test_data/export/visualization/mesh.ply", "New Colored Mesh (mesh.ply)")
]

results = []
for p, l in files:
    res = analyze_cloud(p, l)
    if res: results.append(res)
    
