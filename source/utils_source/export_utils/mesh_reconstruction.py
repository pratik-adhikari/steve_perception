#!/usr/bin/env python3
"""
Mesh Reconstruction using Screened Poisson.
Generates a watertight, colored mesh from a dense point cloud.
"""

import open3d as o3d
import numpy as np
import argparse
import sys
import copy

def reconstruct_mesh(input_ply, output_mesh, depth=9, scale=1.1, density_threshold=None):
    print(f"Loading cloud: {input_ply}")
    pcd = o3d.io.read_point_cloud(input_ply)
    
    if not pcd.has_points():
        print("Error: Empty point cloud")
        return

    print(f"Points: {len(pcd.points):,}")
    
    # 1. Estimate Normals (Critical for Poisson)
    print("Estimating normals...")
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
    # orient normals? usually needed. Assume outside for now.
    pcd.orient_normals_consistent_tangent_plane(100)

    # 2. Poisson Reconstruction
    print(f"Running Poisson Reconstruction (depth={depth})...")
    with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Debug) as cm:
        mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
            pcd, depth=depth, scale=scale, linear_fit=True)
    
    print(f"Generated Mesh: {len(mesh.vertices):,} verts, {len(mesh.triangles):,} ind")

    # 3. Clean Garbage (Poisson creates a 'bubble', remove low density parts)
    print("Cleaning low density vertices...")
    vertices_to_remove = densities < np.quantile(densities, 0.05) if density_threshold is None else densities < density_threshold
    mesh.remove_vertices_by_mask(vertices_to_remove)
    print(f"Cleaned Mesh: {len(mesh.vertices):,} verts")
    
    # 4. Color Transfer (Project color from Cloud -> Mesh)
    print("Transferring colors from point cloud to mesh...")
    if pcd.has_colors():
        # Build KDTree for cloud
        pcd_tree = o3d.geometry.KDTreeFlann(pcd)
        points = np.asarray(pcd.points)
        colors = np.asarray(pcd.colors)
        
        mesh_colors = []
        mesh_verts = np.asarray(mesh.vertices)
        
        # For each mesh vertex, find nearest point in cloud
        for v in mesh_verts:
            [k, idx, _] = pcd_tree.search_knn_vector_3d(v, 1)
            if k > 0:
                mesh_colors.append(colors[idx[0]])
            else:
                mesh_colors.append([0.5, 0.5, 0.5]) # Gray fallback
        
        mesh.vertex_colors = o3d.utility.Vector3dVector(np.array(mesh_colors))
        print("Color transfer complete.")
    else:
        print("Warning: Input cloud has no colors! Mesh will be gray.")

    # 5. Save
    o3d.io.write_triangle_mesh(output_mesh, mesh)
    print(f"Saved to {output_mesh}")

def main():
    parser = argparse.ArgumentParser(description="Poisson Mesh Reconstruction")
    parser.add_argument("input", help="Input PointCloud (PLY)")
    parser.add_argument("output", help="Output Mesh (PLY)")
    parser.add_argument("--depth", type=int, default=9, help="Octree depth (higher=more detail/noise)")
    
    args = parser.parse_args()
    
    reconstruct_mesh(args.input, args.output, args.depth)

if __name__ == "__main__":
    main()
