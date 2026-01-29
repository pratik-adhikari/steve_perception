#!/usr/bin/env python3
"""
Clean Point Cloud using Statistical Outlier Removal.
(Alternative to PointCleanNet when external weights are unavailable)
"""

import open3d as o3d
import argparse
import os
import sys
import numpy as np

def clean_cloud(input_path, output_path, neighbors=50, std_ratio=2.0, display=False):
    print(f"Loading {input_path}...")
    pcd = o3d.io.read_point_cloud(input_path)
    
    if not pcd.has_points():
        print("Error: Empty point cloud")
        return

    print(f"Original points: {len(pcd.points):,}")
    
    # Statistical Outlier Removal
    print(f"Applying Statistical Outlier Removal (buffers={neighbors}, std={std_ratio})...")
    cl, ind = pcd.remove_statistical_outlier(nb_neighbors=neighbors, std_ratio=std_ratio)
    
    # Select inliers
    pcd_clean = pcd.select_by_index(ind)
    print(f"Cleaned points:  {len(pcd_clean.points):,}")
    print(f"Removed:         {len(pcd.points) - len(pcd_clean.points):,}")
    
    # Save
    o3d.io.write_point_cloud(output_path, pcd_clean)
    print(f"Saved to {output_path}")
    
    if display:
        print("Visualizing...")
        o3d.visualization.draw_geometries([pcd_clean], window_name="Cleaned Cloud")

def main():
    parser = argparse.ArgumentParser(description="Clean Point Cloud")
    parser.add_argument("input", help="Input PLY file")
    parser.add_argument("--output", help="Output PLY file (default: input_clean.ply)")
    parser.add_argument("--neighbors", type=int, default=50, help="Number of neighbors for stats")
    parser.add_argument("--std", type=float, default=2.0, help="Std dev ratio (lower = more aggressive)")
    parser.add_argument("--replace", action="store_true", help="Overwrite input file")
    
    args = parser.parse_args()
    
    input_path = args.input
    if args.output:
        output_path = args.output
    elif args.replace:
        output_path = input_path
    else:
        base, ext = os.path.splitext(input_path)
        output_path = f"{base}_clean{ext}"
        
    clean_cloud(input_path, output_path, args.neighbors, args.std)

if __name__ == "__main__":
    main()
