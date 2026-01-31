# Segmentation & Scene Graph: Technical Walkthrough

Audience: developers extending perception for home environments (drawers, containers, general objects).

## 1) End-to-End Data Flow
```mermaid
flowchart LR
    A[RTAB-Map DB (rtabmap.db)] --> B[export_data.py\nRGB/Depth/Poses/scene.ply]
    B -->|scene.ply + frames| C[run_segmentation.py]
    C -->|Mask3D baseline + OpenYOLO3D open-vocab| D[OutputManager\nmasks + objects + mesh_labeled.ply]
    D --> E[Scene Graph Builder\nscenegraph_generator.py]
    E --> F[graph.json / scene.json\n(spatial + semantic relations)]
```

Key artifacts:
- `export/scene.ply` – colored point cloud for inference.
- `export/visualization/raw_mesh.ply` – high-quality mesh used to paint labels.
- `<model>_output/` – masks, per-object PLYs, label map, labeled mesh.
- `graph.json` – nodes (objects) + edges (support, proximity, containment).

## 2) What run_segmentation.py Actually Does
1. Load config (`source/configs/segmentation.yaml`), apply CLI overrides.
2. Validate config with Pydantic → guards bad params early.
3. Select adapter:
   - **OpenYOLO3DAdapter**: Open-vocabulary; text prompts from `vocab` list; uses point cloud; supports frame subsampling (`frame_step`).
   - **Mask3DAdapter**: Closed-set 200 classes (ScanNet); voxelizes mesh; faster on dense scenes.
4. Adapter outputs `SegmentationResult` (masks, classes, scores, points, colors, mesh_path).
5. `OutputManager.save_all`:
   - Writes label map CSV (for scene graph).
   - Writes SceneGraph-compatible `mask3d_output/predictions.txt` + masks.
   - Writes `objects/*.ply` with metadata (centroid, bbox dims, confidence).
   - Paints `mesh_labeled.ply` by transferring mask colors onto `raw_mesh.ply` (KD-tree).

## 2.1) Mask3D vs OpenYOLO3D: what “Mask3D first, then OpenYOLO3D” means
- **Dedicated Mask3D pass**: We explicitly run `--model mask3d` to get closed-set, geometry-stable 3D instances. It ingests a *colored* point cloud/mesh (colors improve features, but it does not use the original RGB images). Labels come from the ScanNet200 head.
- **OpenYOLO3D pass**: Internally combines (a) a 2D open-vocab detector (YOLO-World) on the RGB frames with the provided vocabulary, and (b) a class-agnostic 3D proposal network akin to Mask3D. The 2D detector supplies semantics; the 3D head refines masks in point-cloud space.
- **Why order matters, even though OpenYOLO3D has a Mask3D-like head**:
  - OpenYOLO3D’s 3D refinement is steered by 2D detections; if 2D boxes are loose or colors are weak, 3D masks can bloat.
  - A dedicated Mask3D pass provides a high-confidence, closed-set geometric prior that is independent of 2D detection noise.
  - Fusion rule: keep all Mask3D instances; add OpenYOLO3D instances only when (i) they don’t overlap Mask3D above IoU<0.7, or (ii) they bring in novel/open-vocab classes. This yields stability + openness.

### Point-cloud quality knobs (affects both)
- Density: Too sparse → missing thin drawer fronts; too dense → memory/NN noise. Keep voxel_size 0.02–0.05.
- Outliers: Floating points create bloated masks; run statistical outlier removal if needed.
- Colors: OpenYOLO3D needs good color; Mask3D less so. Ensure `scene.ply` has colors.
- Intrinsics: Needed for 2D→3D projection in OpenYOLO3D; mismatched intrinsics give oversized boxes/masks.

## 3) Model Matrix (when to use what)

| Model | Vocabulary | Geometry Input | Strengths | Weak Spots | Recommended Use |
|-------|------------|----------------|-----------|------------|-----------------|
| OpenYOLO3D | Open (text prompts) | point cloud (scene.ply) + RGB frames | Finds novel classes; 2D open-vocab + 3D refinement | Sensitive to intrinsics/colors; loose 2D boxes can bloat 3D masks; frame_step too high → misses | Add open-vocab objects after Mask3D baseline |
| Mask3D | Closed (ScanNet200) | colored mesh/point cloud | Stable geometry-aligned masks; doesn’t rely on original RGB frames at inference (but benefits from point colors) | Misses OOV classes; voxel_size too big misses thin parts | Baseline segmentation and scene-graph inputs |
| OpenMask3D | Open-vocab retrieval | colored mesh + CLIP | Retrieves by text; good for “find red mug” | Heavy server step; slow CPU fallback | Retrieval/grounding; augment scene graph with embeddings |
| OneFormer3D* | Open/closed hybrid | needs multi-modal support | Panoptic consistency | Emerging; integration pending | Future upgrade for unified masks |
| SAM2/3* | Promptable | RGB images | Crisp 2D masks, good for drawer fronts | Needs lifting to 3D; scale to all frames costly | Few-shot prompts on hard objects |
| YOLO-Drawer (2D) | Drawer detection | RGB images | Specialized drawer cues | 2D only; needs re-projection | Seed drawer priors for 3D fusion |

\*planned/experimental — not wired in current repo.

## 4) Drawer / Cabinet Strategy
1. **2D detection** (YOLO-Drawer or SAM prompts) on `export/color/*.jpg`.
2. **Lift to 3D**:
   - Use corresponding depth + pose to back-project detections into 3D boxes.
   - Merge across frames with ICP or overlap IoU in 3D.
3. **Refine with Mask3D/OpenYOLO3D masks**:
   - Intersect 3D masks with lifted boxes; keep high-precision regions.
4. **Scene Graph edges**:
   - `contains`: drawer ↔ objects inside (based on point inclusion).
   - `supports`: cabinet ↔ drawer; drawer ↔ contents if z-gap < threshold.
5. Store drawer nodes with pose, AABB/OBB, openness state (if observed trajectory exists).

## 5) Scene Graph Construction (current)
- Inputs: `mask3d_output/predictions.txt`, masks, mesh/point cloud.
- Relations:
  - **Proximity**: centroid distance < r.
  - **Support**: vertical alignment and small z-gap; supporting surface normal ~ (0,0,1).
  - **Containment**: points of object inside another’s OBB.
- Outputs: `graph.json` (full), `scene.json` (furniture), `objects/*.json`.

## 6) How to Add/Swap Models
**Mask3D only:**
- `python3 source/scripts/run_segmentation.py --data data/pipeline_output --model mask3d --conf-threshold 0.45`

**OpenYOLO3D with custom vocab:**
- `python3 source/scripts/run_segmentation.py --data data/pipeline_output --model openyolo3d --vocab custom --config my_config.yaml`
- In `my_config.yaml`, set `vocabulary: { mode: custom, custom_classes: ["drawer", "cabinet", "microwave"] }`

### What “vocabulary” means (and presets)
- The 2D detector in OpenYOLO3D/OpenMask3D uses text prompts. `--vocab` selects a preset list:
  - `furniture`: compact list focused on indoor furniture (includes “drawer”, “cabinet”, “handle”).
  - `coco`, `lvis`: broader, standard datasets; more classes but can add noise.
  - `custom`: you supply `custom_classes` (e.g., `["drawer", "cabinet drawer", "wardrobe handle"]`).
- For drawer-heavy tasks, prefer `custom` or `furniture` and explicitly include drawer/handle variants.

**OpenMask3D retrieval (after Mask3D):**
- Run pipeline in `README_HANDOVER_2026-01-11.txt` (`run_openmask_pipeline.py`, then `query_openmask.py "red mug"`).
- Cache CLIP features under `data/openmask_features/<scan>/`.

**SAM2/SAM3 (2D prompts) planned hook:**
- Add a pre-seg step over RGB frames → save 2D masks → lift via depth/pose → merge into `objects/` before scene graph.

### YOLO-Drawer vs OpenYOLO3D
- YOLO-Drawer is a specialized 2D detector (trained for drawer fronts/handles). You can run it on `export/color/*.jpg`, lift boxes to 3D, and fuse with Mask3D/OpenYOLO3D masks. It’s not a drop-in replacement for OpenYOLO3D; use it as a *prior* to tighten drawers/handles when open-vocab boxes are loose.

## 7) Quality Levers
- **Export**
  - Ensure `scene.ply` has colors; if missing, OpenYOLO3D confidence collapses.
  - Keep voxel_size modest (0.02–0.05) to avoid over-thinning small objects.
- **OpenYOLO3D**
  - `frame_step`: lower (2–5) for clutter; higher (8–12) for speed.
  - `conf_threshold`: start 0.25 for open-vocab; raise to 0.4 to reduce noise.
- **Mask3D**
  - `voxel_size`: 0.02 default; reduce to 0.01 for thin structures (drawers), but memory↑.
  - `threshold`: 0.45–0.6 depending on desired precision.
- **OutputManager**
  - Downsample before KDTree if point count >2M to avoid OOM.

## 8) Failure → Diagnosis
- No masks: check segmentation log for “NO INSTANCES DETECTED”; verify `scene.ply` existence and color channels.
- Wrong labels: confirm vocab mode; for drawers, add custom class names that match model text encoder (“drawer”, “cabinet drawer”).
- Broken mesh_labeled: ensure `export/visualization/raw_mesh.ply` exists; else it falls back to point-cloud coloring.
- Bloated boxes/masks (typically OpenYOLO3D): lower frame_step (denser frames), tighten 2D detector NMS/conf in OpenYOLO3D config, and rely on Mask3D baseline with overlap suppression.

## 9) Roadmap (implementation order)
1) Enforce input validation (scene.ply + counts) and per-frame intrinsics saving.
2) Add drawer pipeline (2D detector → 3D lift → merge with 3D masks).
3) Add OneFormer3D/SAM2 hooks as optional pre/post steps.
4) Enhance scene graph with AABB/OBB and containment edges for drawers/containers.
5) Benchmark configs (frame_step, thresholds) on a small home scan and publish defaults.
