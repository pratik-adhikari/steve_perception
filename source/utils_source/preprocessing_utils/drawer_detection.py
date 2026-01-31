from __future__ import annotations

import os.path
from logging import Logger
from typing import Optional
import shutil

import numpy as np

import cv2
from matplotlib import pyplot as plt
import colorsys
from utils_source.docker_communication import save_files, send_request
from collections import namedtuple

from utils_source.recursive_config import Config

BBox = namedtuple("BBox", ["xmin", "ymin", "xmax", "ymax"])
Detection = namedtuple("Detection", ["file", "name", "conf", "bbox"])

config = Config("drawer")

COLORS = {
    "door": (0.651, 0.243, 0.957),
    "handle": (0.522, 0.596, 0.561),
    "cabinet door": (0.549, 0.047, 0.169),
    "refrigerator door": (0.082, 0.475, 0.627),
}

CATEGORIES = {"0": "door", "1": "handle", "2": "cabinet door", "3": "refrigerator door"}

def generate_distinct_colors(n: int) -> list[tuple[float, float, float]]:
    colors = []
    for i in range(n):
        hue = i / n
        saturation = 0.7
        lightness = 0.5
        r, g, b = colorsys.hls_to_rgb(hue, lightness, saturation)
        colors.append((r, g, b))

    return colors

def draw_boxes(image: np.ndarray, detections: list[Detection], output_path: str) -> None:
    plt.figure(figsize=(16, 10))
    plt.imshow(image)
    ax = plt.gca()
    names = sorted(list(set([det.name for det in detections])))
    names_dict = {name: i for i, name in enumerate(names)}
    colors = generate_distinct_colors(len(names_dict))

    for _, name, conf, (xmin, ymin, xmax, ymax) in detections:
        w, h = xmax - xmin, ymax - ymin
        color = colors[names_dict[name]]
        ax.add_patch(plt.Rectangle((xmin, ymin), w, h, fill=False, color=color, linewidth=6))
        text = f"{name}: {conf:0.2f}"
        ax.text(xmin, ymin, text, fontsize=15, bbox=dict(facecolor="yellow", alpha=0.5))
    plt.axis("off")
    if detections != []:
        plt.savefig(output_path)
    plt.close()

def predict_yolodrawer(
    image: np.ndarray,
    image_name: str,
    logger: Optional[Logger] = None,
    timeout: int = 90,
    input_format: str = "rgb",
    vis_block: bool = False,
    debug_output_dir: Optional[str] = None
) -> list[Detection] | None:
    assert image.shape[-1] == 3
    if input_format == "bgr":
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    address_details = {'ip': "127.0.0.1", 'port': 5004, 'route': "yolodrawer/predict"}
    address = f"http://{address_details['ip']}:{address_details['port']}/{address_details['route']}"
    
    os.makedirs("tmp", exist_ok=True)

    image_prefix = os.path.basename(image_name)
    save_data = [(f"{os.path.splitext(image_prefix)[0]}.npy", np.save, image)]
    image_path, *_ = save_files(save_data, "tmp")

    paths_dict = {"image": image_path}
    if logger:
        logger.info(f"Sending request to {address}!")
    contents = send_request(address, paths_dict, {}, timeout, "tmp")
    if logger:
        logger.info("Received response!")

    # no detections
    if len(contents) == 0:
        if vis_block:
             if debug_output_dir:
                 os.makedirs(debug_output_dir, exist_ok=True)
                 out_path = os.path.join(debug_output_dir, f"{image_prefix}_no_detections.png")
                 draw_boxes(image, [], out_path)
             # Legacy fallback omitted for clarity/safety if not needed, or we can keep it inside an else
        return [], 0

    classes = contents["classes"]
    confidences = contents["confidences"]
    bboxes = contents["bboxes"]

    vis_thresh = config["drawer_model"].get("vis_threshold", 0.5)
    return_thresh = config["drawer_model"].get("conf_threshold", 0.7) # Lowered to 0.7 to allow consolidation

    vis_detections = []
    return_detections = []

    for cls, conf, bbox in zip(classes, confidences, bboxes):
        name = CATEGORIES[str(int(cls))]
        if name != "handle":
            # Detection object
            det = Detection(image_name, name, conf, BBox(*bbox))
            
            # Add to visualization list if above low threshold
            if conf > vis_thresh:
                vis_detections.append(det)
            
            # Add to return list if above high threshold
            if conf > return_thresh:
                return_detections.append(det)

    if vis_block:
        if debug_output_dir:
            os.makedirs(debug_output_dir, exist_ok=True)
            out_path = os.path.join(debug_output_dir, f"{image_prefix}_detections.png")
            draw_boxes(image, vis_detections, out_path)
        else:
             # Legacy path
            path = config.get_subpath("ipad_scans")
            ending = config["pre_scanned_graphs"]["high_res"]
            draw_boxes(image, vis_detections, os.path.join(path, ending, "detections", f"{image_prefix}_detections.png"))
    
    shutil.rmtree("tmp")
    
    return return_detections, len(return_detections)
