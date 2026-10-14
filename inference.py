import os
import json
from pathlib import Path

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import geopandas as gpd
import numpy as np
import rasterio
import torch
from PIL import Image, ImageDraw, ImageFont
from rasterio.plot import reshape_as_image
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction
from shapely.geometry import Polygon

from config import CONFIG
from model import build_model


PROJECT_ROOT = Path(__file__).resolve().parent

"""
Georeferenced object detection inference for GeoTIFF rasters.

This module runs a trained torchvision detection model (wrapped for SAHI
sliced inference) over a GeoTIFF, then writes two kinds of outputs:

1. A PNG with bounding boxes and labels drawn on the raster.
2. A shapefile of georeferenced polygons in the raster's CRS

SAHI (Slicing Aided Hyper Inference) is used to slice the image into
overlapping tiles so that small objects can be detected more reliably.
"""


def resolve_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else PROJECT_ROOT / path


def read_raster(raster_path: str):
    """
    Read a GeoTIFF and return its pixel data, CRS, and affine transform.
    """
    with rasterio.open(raster_path) as src:
        img = src.read()          # (C, H, W)
        crs = src.crs
        transform = src.transform

    img = reshape_as_image(img)   # (H, W, C)

    # Keep RGB bands or replicate a grayscale band into RGB.
    if img.ndim == 2:
        img = np.stack([img, img, img], axis=-1)
    elif img.shape[2] == 1:
        img = np.repeat(img, 3, axis=2)
    elif img.shape[2] >= 3:
        img = img[:, :, :3]
    else:
        raise ValueError(f"Expected a single-band or RGB raster, got {img.shape[2]} bands.")

    # Normalise to uint8
    if img.dtype != np.uint8:
        img_min, img_max = img.min(), img.max()
        if img_max > img_min:
            img = ((img - img_min) / (img_max - img_min) * 255).astype(np.uint8)
        else:
            img = np.zeros_like(img, dtype=np.uint8)

    return img, crs, transform


def pixel_bbox_to_world(bbox_pixel, transform):
    """
    Convert all four pixel bounding-box corners to a world-coordinate polygon.
    """
    x_min, y_min, x_max, y_max = bbox_pixel
    corners = [
        transform * (x_min, y_min),
        transform * (x_max, y_min),
        transform * (x_max, y_max),
        transform * (x_min, y_max),
    ]
    return Polygon(corners)


def save_result_with_bbox(img_array: np.ndarray,
                          result,
                          output_path: str | Path,
                          line_width: int = 2,
                          font_size: int = 14) -> None:
    """
    Draw bounding boxes and labels on an image and save to detection_result directory.
    """
    pil_img = Image.fromarray(img_array).convert("RGB")
    draw = ImageDraw.Draw(pil_img)

    try:
        font = ImageFont.truetype("arial.ttf", font_size)
    except IOError:
        font = ImageFont.load_default()

    for pred in result.object_prediction_list:
        bbox = pred.bbox.to_voc_bbox()
        label = pred.category.name
        score = pred.score.value
        color = (255, 165, 0)

        # Draw rectangle
        draw.rectangle(
            [bbox[0], bbox[1], bbox[2], bbox[3]],
            outline=color,
            width=line_width
        )

        # Build label text
        text = f"{label}: {score:.2f}"

        try:
            bbox_text = draw.textbbox((bbox[0], bbox[1] - font_size - 2), text, font=font)
        except AttributeError:
            tw, th = draw.textsize(text, font=font)
            bbox_text = (bbox[0], bbox[1] - th - 2,
                         bbox[0] + tw, bbox[1])

        draw.rectangle(bbox_text, fill=color)
        draw.text(
            (bbox_text[0], bbox_text[1]),
            text,
            fill=(255, 255, 255),
            font=font
        )

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pil_img.save(output_path)
    print(f"Image saved: {output_path}")


def save_bbox_to_shp(result, output_shp_path: str, crs, transform) -> None:
    """
    Convert SAHI detections to a georeferenced polygon vector file (.shp).

    Each detection's pixel-space bounding box is transformed to world
    coordinates, and the resulting polygons are written to a GeoDataFrame
    with three attribute columns:

        class: the predicted class name.
        confidence: the detection confidence score.
    """
    class_names = []
    confidences = []
    geometries = []

    for pred in result.object_prediction_list:
        class_names.append(pred.category.name)
        confidences.append(f"{float(pred.score.value):.4f}")
        bbox_pixel = pred.bbox.to_voc_bbox()
        geometries.append(pixel_bbox_to_world(bbox_pixel, transform))

    gdf = gpd.GeoDataFrame({
        "class":      class_names,
        "confidence": confidences,
        "geometry":   geometries,
    }, crs=crs)

    output_shp_path = Path(output_shp_path)
    output_shp_path.parent.mkdir(parents=True, exist_ok=True)
    gdf.to_file(output_shp_path)
    print(f"Shapefile saved: {output_shp_path}  ({len(gdf)} features)")


def build_detection_model(cfg: dict):
    """Build the Torchvision model and wrap it for SAHI inference."""
    inf_cfg = cfg["inference"]
    dataset_cfg = cfg["dataset"]
    device = inf_cfg.get("device", "cuda:0" if torch.cuda.is_available() else "cpu")

    annotation_path = resolve_path(dataset_cfg["train"]) / dataset_cfg["ann_file"]
    with annotation_path.open("r", encoding="utf-8") as file:
        categories = json.load(file)["categories"]

    model = build_model(
        num_classes=len(categories) + 1,
        architecture="frcnn_mobilenet",
    )
    model_path = resolve_path(inf_cfg["model_path"])
    if not model_path.is_file():
        raise FileNotFoundError(f"Model checkpoint not found: {model_path}")

    checkpoint = torch.load(model_path, map_location=device)
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(state_dict, strict=True)

    model.to(device)
    model.eval()

    category_names = inf_cfg.get("category_names") or [
        category["name"] for category in categories
    ]
    category_mapping = {
        str(index + 1): name
        for index, name in enumerate(category_names)
    }

    return AutoDetectionModel.from_pretrained(
        model_type=inf_cfg.get("model_type", "torchvision"),
        model=model,
        confidence_threshold=inf_cfg["conf_threshold"],
        image_size=inf_cfg.get("image_size"),
        device=device,
        category_mapping=category_mapping,
    )


def main() -> None:
    inference_cfg = CONFIG["inference"]
    image_path = resolve_path(inference_cfg["image_path"])
    output_dir = resolve_path(inference_cfg["save_result_dir"])
    slice_size = inference_cfg["tile_size"]
    overlap = inference_cfg["overlap"]
    postprocess_type = inference_cfg["postprocess"]

    print(f"Reading raster: {image_path}")
    img_array, crs, transform = read_raster(image_path)

    if crs is None:
        raise ValueError(f"GeoTIFF has no CRS, cannot georeference detections: {image_path}")
    
    print(f"Image shape: {img_array.shape}  |  CRS: {crs}")

    print(f"Loading configured frcnn_mobilenet model")
    detection_model = build_detection_model(CONFIG)

    result = get_sliced_prediction(
        img_array,
        detection_model,
        slice_height=slice_size,
        slice_width=slice_size,
        overlap_height_ratio=overlap,
        overlap_width_ratio=overlap,
        postprocess_type=postprocess_type,
        confidence_threshold=inference_cfg["conf_threshold"],
    )

    n = len(result.object_prediction_list)
    print(f"Detected {n} object(s)")

    output_dir.mkdir(parents=True, exist_ok=True)

    base_name = image_path.stem
    image_output = output_dir / f"{base_name}_detections.png"
    save_result_with_bbox(img_array, result, image_output)

    shapefile_output = output_dir / f"{base_name}_detections.shp"
    save_bbox_to_shp(result, shapefile_output, crs, transform)

    print(f"\nAll results saved to: {output_dir}")

if __name__ == "__main__":
    main()