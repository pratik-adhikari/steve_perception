import json
import os
import numpy as np

def diagnose():
    path = "data/pipeline_output/generated_graph/furniture.json"
    if not os.path.exists(path):
        print(f"File not found: {path}")
        return

    with open(path, 'r') as f:
        data = json.load(f)
    
    furniture = data.get("furniture", {})
    print(f"Found {len(furniture)} furniture items.")
    
    suspicious_count = 0
    for fid, item in furniture.items():
        dims = np.array(item["dimensions"])
        label = item["label"]
        max_dim = np.max(dims)
        
        # Heuristic: Chairs/Tables shouldn't be > 3 meters usually
        # Beds might be 2-3m.
        # Anything > 5m is definitely suspicious for indoor scenes.
        
        if max_dim > 3.0:
            print(f"[SUSPICIOUS] ID {fid} ({label}): Dimensions {dims} (Max: {max_dim:.2f}m)")
            suspicious_count += 1
        else:
             print(f"[OK] ID {fid} ({label}): Max dim {max_dim:.2f}m")
             
    if suspicious_count > 0:
        print(f"\nDiagnosis: {suspicious_count} objects have suspiciously large dimensions.")
        print("This suggests segmentation masks contain outliers (noise points far from the object).")
        print("Recommendation: Implement outlier removal (DBSCAN/Statistical) in SceneGraph construction.")
    else:
        print("\nDiagnosis: Dimensions look reasonable. Issue might be elsewhere.")

if __name__ == "__main__":
    diagnose()
