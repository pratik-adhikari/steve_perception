import os
import shutil
import glob

export_dir = "data/pipeline_output/export"

# 1. Clean up old symlinks created manually or by previous scripts
# We expect 'color', 'depth', 'poses', 'intrinsic' might be symlinks or dirs
cleanup_targets = ["color", "depth", "poses", "intrinsic", "intrinsics.txt"]
for t in cleanup_targets:
    path = os.path.join(export_dir, t)
    if os.path.islink(path):
        os.unlink(path)
        print(f"Removed symlink: {t}")
    elif os.path.isdir(path) and t in ["color", "depth", "poses"]: 
        # CAREFUL: If it's a real dir but not frames based?
        # Check if it is the frames dir (unlikely unless I failed symlinking before)
        # We'll create fresh dirs later
        pass

# 2. Source is 'frames/'
frames_dir = os.path.join(export_dir, "frames")
if not os.path.exists(frames_dir):
    print("Error: frames/ directory not found. Cannot migrate.")
    exit(1)

# 3. Create new flat directories
new_dirs = ["color", "depth", "poses"]
for d in new_dirs:
    os.makedirs(os.path.join(export_dir, d), exist_ok=True)

# 4. Move and Rename
# Source: frames/color/000000.jpg -> Target: color/0.jpg

def migrate_folder(src_name, dst_name, ext):
    src_path = os.path.join(frames_dir, src_name)
    dst_path = os.path.join(export_dir, dst_name)
    
    files = sorted(glob.glob(os.path.join(src_path, f"*{ext}")))
    print(f"Migrating {len(files)} files from {src_name} to {dst_name}...")
    
    for f in files:
        fname = os.path.basename(f)
        # Handle 000000.jpg or 0.jpg (if my script created symlinks inside frames/color)
        # We only want to move the actual file content if possible, or copy
        if os.path.islink(f):
            continue # Skip symlinks inside source
            
        try:
            # Parse index
            idx = int(fname.split('.')[0])
            new_fname = f"{idx}{ext}"
            shutil.copy2(f, os.path.join(dst_path, new_fname))
        except ValueError:
            print(f"Skipping {fname}, not an integer index")

migrate_folder("color", "color", ".jpg")
migrate_folder("depth", "depth", ".png")
migrate_folder("pose", "poses", ".txt") # Note rename: pose -> poses

# 5. Intrinsics
# frames/intrinsic/000000.txt -> intrinsics.txt
intrinsics_src = glob.glob(os.path.join(frames_dir, "intrinsic", "*.txt"))
if intrinsics_src:
    src = intrinsics_src[0] # Take first
    shutil.copy2(src, os.path.join(export_dir, "intrinsics.txt"))
    print("Created intrinsics.txt")

# 6. Cleanup frames dir?
# shutil.rmtree(frames_dir)
print("Migration done. You can delete 'frames/' folder manually if verified.")
