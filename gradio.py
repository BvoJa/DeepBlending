import os
import html
import inspect
import shutil
import sys
import urllib.request
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from my_run import first_pass_blend, prepare_source_object


def import_gradio_package():
    script_dir = Path(__file__).resolve().parent
    removed_paths = []
    for entry in list(sys.path):
        candidate = Path(entry or ".").resolve()
        if candidate == script_dir:
            sys.path.remove(entry)
            removed_paths.append(entry)
    try:
        import gradio as gr
    finally:
        for entry in reversed(removed_paths):
            sys.path.insert(0, entry)
    return gr


gr = import_gradio_package()

DEFAULT_OUTPUT_DIR = "results/gradio"

INSTALL_LOADING_JS = """
() => {
  if (window.deepBlendingLoadingInstalled) {
    return;
  }
  window.deepBlendingLoadingInstalled = true;

  window.deepBlendingSetLoading = (message) => {
    const panel = document.getElementById("blend-loading-panel");
    const label = document.getElementById("blend-loading-label");
    if (!panel || !label) {
      return;
    }
    label.textContent = message || "Working...";
    panel.classList.add("is-active");
  };

  window.deepBlendingHideLoading = () => {
    const panel = document.getElementById("blend-loading-panel");
    if (panel) {
      panel.classList.remove("is-active");
    }
  };

  document.addEventListener(
    "change",
    (event) => {
      const input = event.target;
      if (!input || input.type !== "file" || !input.files || input.files.length === 0) {
        return;
      }
      const count = input.files.length;
      const noun = count === 1 ? "image" : "images";
      window.deepBlendingSetLoading(`Uploading ${count} ${noun}...`);
    },
    true
  );
}
"""

SHOW_UPLOAD_LOADING_JS = "() => { window.deepBlendingSetLoading?.('Preparing uploaded image...'); return []; }"
SHOW_PREVIEW_LOADING_JS = "() => { window.deepBlendingSetLoading?.('Creating placement preview...'); return []; }"
SHOW_EDIT_LOADING_JS = "() => { window.deepBlendingSetLoading?.('Running first-pass blending...'); return []; }"
SHOW_SAM_LOADING_JS = "() => { window.deepBlendingSetLoading?.('Extracting mask with SAM...'); return []; }"
SAM_MODEL_CACHE = {}
SAM_CHECKPOINT_NAME = "efficient_sam_vits.pt"
SAM_CHECKPOINT_URL = "https://huggingface.co/Adapter/DragonDiffusion/resolve/main/model/efficient_sam_vits.pt"


class SamSetupError(Exception):
    pass

DESCRIPTION = """
# Deep Image Blending
Gradio demo for first-pass object blending. Upload a source image, brush the object mask, upload a target image, choose placement and loss weights, then click `Edit`.
"""

BLEND_DESCRIPTION = """
## First-Pass Object Blending
Usage:
- Upload a source image and brush directly over the object.
- Or upload a mask image in the mask box.
- Upload a target image using the same plain image-upload style as DragonDiffusion.
- Optionally paste Kaggle/local file paths to avoid slow browser upload.
- Adjust target size and object center.
- Adjust the loss weights that contribute to the first-pass total loss.
- Click `Preview Placement` to inspect the mask and location.
- Click `Edit` to run first-pass image blending.
"""

examples_blend = [
    ["data/1_source.png", "data/1_mask.png", "data/1_target.png", 512, 256, 256],
    ["data/2_source.png", "data/2_mask.png", "data/2_target.png", 512, 256, 256],
    ["data/3_source.png", "data/3_mask.png", "data/3_target.png", 512, 256, 256],
    ["data/4_source.png", "data/4_mask.png", "data/4_target.png", 512, 256, 256],
    ["data/5_source.png", "data/5_mask.png", "data/5_target.png", 512, 256, 256],
]


def loading_panel_html(message="", active=False):
    active_class = " is-active" if active else ""
    text = html.escape(message or "Working...")
    return f"""
<div id="blend-loading-panel" class="blend-loading{active_class}">
  <div class="blend-loading-spinner" aria-hidden="true"></div>
  <div id="blend-loading-label" class="blend-loading-label">{text}</div>
</div>
"""


def show_loading(message):
    return gr.update(value=loading_panel_html(message, active=True))


def show_preview_loading():
    return show_loading("Creating placement preview...")


def show_edit_loading():
    return show_loading("Running first-pass blending...")


def show_upload_loading():
    return show_loading("Preparing uploaded image...")


def hide_loading():
    return gr.update(value=loading_panel_html())


def candidate_sam_checkpoint_paths(extra_path=None):
    candidates = [
        extra_path,
        os.environ.get("EFFICIENT_SAM_CHECKPOINT"),
        Path(__file__).resolve().parent / "models" / SAM_CHECKPOINT_NAME,
        Path("/kaggle/working/DeepBlending/models") / SAM_CHECKPOINT_NAME,
        Path("/kaggle/working/DeepImageBlending/models") / SAM_CHECKPOINT_NAME,
        Path("/home/bvoja/Documents/DragonDiffusion/models") / SAM_CHECKPOINT_NAME,
        Path("/kaggle/input/efficient-sam") / SAM_CHECKPOINT_NAME,
    ]
    kaggle_input = Path("/kaggle/input")
    if kaggle_input.exists():
        candidates.extend(kaggle_input.glob(f"*/{SAM_CHECKPOINT_NAME}"))
        candidates.extend(kaggle_input.glob(f"*/*/{SAM_CHECKPOINT_NAME}"))
    return candidates


def default_sam_checkpoint_path():
    for candidate in candidate_sam_checkpoint_paths():
        if not candidate:
            continue
        path = Path(candidate).expanduser()
        if path.exists():
            return str(path)
    return str(Path(__file__).resolve().parent / "models" / SAM_CHECKPOINT_NAME)


def download_sam_checkpoint(destination):
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        with urllib.request.urlopen(SAM_CHECKPOINT_URL, timeout=30) as response:
            with open(destination, "wb") as output_file:
                shutil.copyfileobj(response, output_file)
    except Exception as exc:
        if destination.exists():
            try:
                destination.unlink()
            except OSError:
                pass
        raise SamSetupError(
            "EfficientSAM checkpoint was not found, and automatic download failed. "
            "On Kaggle, either enable internet or add/upload efficient_sam_vits.pt "
            "and paste its path in the EfficientSAM checkpoint box."
        ) from exc
    return str(destination.resolve())


def resolve_sam_checkpoint(checkpoint_path):
    raw_path = (checkpoint_path or "").strip()
    candidates = candidate_sam_checkpoint_paths(raw_path or None)
    for candidate in candidates:
        if not candidate:
            continue
        path = Path(candidate).expanduser()
        if not path.is_absolute():
            path = Path(__file__).resolve().parent / path
        if path.exists():
            return str(path.resolve())

    if raw_path:
        missing_path = Path(raw_path).expanduser()
        if not missing_path.is_absolute():
            missing_path = Path(__file__).resolve().parent / missing_path
        if missing_path.name == SAM_CHECKPOINT_NAME:
            return download_sam_checkpoint(missing_path)
        raise SamSetupError(
            f"EfficientSAM checkpoint was not found at {missing_path}. "
            "Paste the correct efficient_sam_vits.pt path, or leave this box blank "
            "so the demo can search Kaggle paths and try automatic download."
        )

    destination = Path(__file__).resolve().parent / "models" / SAM_CHECKPOINT_NAME
    return download_sam_checkpoint(destination)


def resolve_sam_device(gpu_id):
    import torch

    value = str(gpu_id or "auto").strip().lower()
    if value == "auto":
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if value == "cpu":
        return torch.device("cpu")
    if value.startswith("cuda"):
        if not torch.cuda.is_available():
            raise SamSetupError("CUDA was selected for SAM, but CUDA is not available. Use `auto` or `cpu`.")
        return torch.device(value)
    if value.isdigit():
        return torch.device(f"cuda:{value}" if torch.cuda.is_available() else "cpu")
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def get_sam_model(checkpoint_path, gpu_id):
    import torch
    from sam.efficient_sam.efficient_sam import build_efficient_sam

    checkpoint = resolve_sam_checkpoint(checkpoint_path)
    device = resolve_sam_device(gpu_id)
    cache_key = (checkpoint, str(device))
    if cache_key not in SAM_MODEL_CACHE:
        model = build_efficient_sam(
            encoder_patch_embed_dim=384,
            encoder_num_heads=6,
            checkpoint=checkpoint,
        )
        model.eval().to(device)
        SAM_MODEL_CACHE[cache_key] = model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    return SAM_MODEL_CACHE[cache_key], device


def load_css():
    css_path = Path(__file__).with_name("style.css")
    if css_path.exists():
        return css_path.read_text()
    return ""


def make_upload_image(label, image_mode="RGB"):
    kwargs = {
        "label": label,
        "interactive": True,
        "type": "numpy",
        "image_mode": image_mode,
    }
    params = inspect.signature(gr.Image).parameters
    if "source" in params:
        kwargs["source"] = "upload"
    else:
        kwargs["sources"] = ["upload"]
    return gr.Image(**kwargs)


def make_source_draw_image(label):
    if hasattr(gr, "ImageEditor"):
        return gr.ImageEditor(
            label=label,
            sources=["upload"],
            type="numpy",
            image_mode="RGBA",
            brush=gr.Brush(
                default_size=35,
                colors=[("#ff7828", 0.35)],
                default_color=("#ff7828", 0.35),
                color_mode="fixed",
            ),
            eraser=gr.Eraser(default_size=35),
            layers=True,
            transforms=[],
        )

    if hasattr(gr, "ImageMask"):
        return gr.ImageMask(
            label=label,
            sources=["upload"],
            type="numpy",
            image_mode="RGBA",
            brush=gr.Brush(
                default_size=35,
                colors=[("#ff7828", 0.35)],
                default_color=("#ff7828", 0.35),
                color_mode="fixed",
            ),
            eraser=gr.Eraser(default_size=35),
            transforms=[],
        )

    image_params = inspect.signature(gr.Image).parameters
    kwargs = {
        "label": label,
        "interactive": True,
        "type": "numpy",
    }
    if "tool" in image_params:
        kwargs["tool"] = "sketch"
    if "source" in image_params:
        kwargs["source"] = "upload"
    else:
        kwargs["sources"] = ["upload"]
    return gr.Image(**kwargs)


def center_position(ts):
    center = int(ts // 2)
    return center, center


def fit_placement(x, y, ss, ts):
    half = ss * 0.5
    if ss > ts:
        raise gr.Error("Source size (--ss) must be less than or equal to target size (--ts).")

    min_center = int(np.ceil(half))
    max_center = int(np.floor(ts - half))
    fitted_x = int(np.clip(int(x), min_center, max_center))
    fitted_y = int(np.clip(int(y), min_center, max_center))
    return fitted_x, fitted_y


def fit_source_placement(x, y, source_shape, ts):
    source_h, source_w = source_shape[:2]
    if source_h > ts or source_w > ts:
        raise gr.Error(
            f"Source image size {source_h}x{source_w} is larger than target size {ts}x{ts}. "
            "Use a larger target size or a smaller source image."
        )

    min_x = int(np.ceil(source_h * 0.5))
    max_x = int(np.floor(ts - source_h * 0.5))
    min_y = int(np.ceil(source_w * 0.5))
    max_y = int(np.floor(ts - source_w * 0.5))
    fitted_x = int(np.clip(int(x), min_x, max_x))
    fitted_y = int(np.clip(int(y), min_y, max_y))
    return fitted_x, fitted_y


def to_rgb_array(image):
    if image is None:
        return None
    if isinstance(image, np.ndarray):
        return np.array(Image.fromarray(image.astype(np.uint8)).convert("RGB"))
    return np.array(Image.open(image).convert("RGB"))


def to_rgba_array(image):
    if image is None:
        return None
    if isinstance(image, np.ndarray):
        return np.array(Image.fromarray(image.astype(np.uint8)).convert("RGBA"))
    return np.array(Image.open(image).convert("RGBA"))


def to_mask_array(image):
    if image is None:
        return None
    if isinstance(image, np.ndarray):
        mask = np.array(Image.fromarray(image.astype(np.uint8)).convert("L"))
    else:
        mask = np.array(Image.open(image).convert("L"))
    mask[mask > 0] = 255
    return mask.astype(np.uint8)


def resolve_target(target_image, target_path):
    path = (target_path or "").strip()
    if path:
        if not os.path.exists(path):
            raise gr.Error(f"Target path does not exist: {path}")
        return path
    if target_image is None:
        raise gr.Error("Upload a target image or enter a Kaggle/local target path.")
    return target_image


def load_image_path(path, mode):
    path = (path or "").strip()
    if not path:
        return None
    if not os.path.exists(path):
        raise gr.Error(f"Image path does not exist: {path}")
    return np.array(Image.open(path).convert(mode))


def load_images_from_paths(source_path, mask_path, target_path):
    has_source = bool((source_path or "").strip())
    source = load_image_path(source_path, "RGB") if has_source else gr.update()
    has_mask = bool((mask_path or "").strip())
    mask = load_image_path(mask_path, "L") if has_mask else gr.update()
    target = load_image_path(target_path, "RGB") if (target_path or "").strip() else gr.update()
    source_original = source if has_source else gr.update()
    sam_points = [] if has_source else gr.update()
    sam_point_labels = [] if has_source else gr.update()
    active_mask = to_mask_array(mask) if has_mask else (None if has_source else gr.update())
    return source, mask, target, source_original, sam_points, sam_point_labels, active_mask


def store_active_mask(mask_image):
    return to_mask_array(mask_image)


def activate_drawn_source_mask(source_image, source_path="", source_original_image=None):
    drawn_source, drawn_mask = extract_drawn_source_and_mask(source_image)
    source = load_image_path(source_path, "RGB")
    if source is None:
        source = drawn_source
    if source is None:
        source = to_rgb_array(source_original_image)
    if source is None:
        raise gr.Error("Upload a source image before drawing the mask.")
    if drawn_mask is None or drawn_mask.max() == 0:
        return None, None, source, "Source image ready. Brush over the object to create a mask."
    return drawn_mask, drawn_mask, source, "Brush mask ready."


def extract_drawn_source_and_mask(source_image):
    if source_image is None:
        return None, None

    if not isinstance(source_image, dict):
        return to_rgb_array(source_image), None

    if "image" in source_image and "mask" in source_image:
        source = to_rgb_array(source_image.get("image"))
        mask = to_mask_array(source_image.get("mask"))
        return source, mask

    background = source_image.get("background")
    composite = source_image.get("composite")
    source = to_rgb_array(background if background is not None else composite)

    mask = None
    if source is not None:
        mask = np.zeros(source.shape[:2], dtype=np.uint8)
        for layer in source_image.get("layers") or []:
            layer = to_rgba_array(layer)
            if layer is None:
                continue
            if layer.ndim == 3 and layer.shape[2] == 4:
                layer_mask = layer[:, :, 3] > 0
            elif layer.ndim == 3:
                layer_mask = np.any(layer[:, :, :3] > 0, axis=2)
            else:
                layer_mask = layer > 0
            if layer_mask.shape != mask.shape:
                layer_mask = np.array(Image.fromarray(layer_mask.astype(np.uint8) * 255).resize(mask.shape[::-1])) > 0
            mask[layer_mask] = 255
        if mask.max() == 0:
            if background is not None and composite is not None:
                background_rgb = to_rgb_array(background)
                composite_rgb = to_rgb_array(composite)
                if composite_rgb.shape[:2] != background_rgb.shape[:2]:
                    composite_rgb = np.array(Image.fromarray(composite_rgb).resize(background_rgb.shape[1::-1]))
                diff = np.max(
                    np.abs(composite_rgb.astype(np.int16) - background_rgb.astype(np.int16)),
                    axis=2,
                )
                mask = (diff > 8).astype(np.uint8) * 255
            if mask is not None and mask.max() == 0:
                mask = None

    return source, mask


def resolve_source_and_mask(source_image, mask_image, source_path="", source_original_image=None, mask_path="", active_mask=None):
    source_from_path = load_image_path(source_path, "RGB")
    source_from_original = to_rgb_array(source_original_image)
    drawn_source, drawn_mask = extract_drawn_source_and_mask(source_image)
    source = source_from_path if source_from_path is not None else source_from_original
    if source is None:
        source = drawn_source
    if source is None:
        source = to_rgb_array(source_image)

    mask_from_path = load_image_path(mask_path, "L")
    uploaded_mask = to_mask_array(mask_from_path)
    if uploaded_mask is not None:
        if source is None:
            raise gr.Error("Upload a source image before using a mask image.")
        return source, uploaded_mask

    if drawn_mask is not None:
        if source is None:
            raise gr.Error("Upload a source image before drawing the mask.")
        return source, drawn_mask

    uploaded_mask = to_mask_array(active_mask)
    if uploaded_mask is None:
        uploaded_mask = to_mask_array(mask_image)
    if uploaded_mask is not None:
        if source is None:
            raise gr.Error("Upload a source image before using a mask image.")
        return source, uploaded_mask

    if source is None:
        raise gr.Error("Upload a source image first.")
    raise gr.Error("Draw over the object in the source image, or upload a mask image.")


def resolve_source_for_sam(source_image, source_path="", fallback_image=None):
    source_from_path = load_image_path(source_path, "RGB")
    if source_from_path is not None:
        return source_from_path.astype(np.uint8)

    drawn_source, _ = extract_drawn_source_and_mask(source_image)
    if drawn_source is not None:
        return drawn_source.astype(np.uint8)

    source = to_rgb_array(fallback_image)
    if source is not None:
        return source.astype(np.uint8)

    raise gr.Error("Upload a source image before using SAM mask extraction.")


def store_source_original(source_image, source_path=""):
    source = resolve_source_for_sam(source_image, source_path)
    return source, None, [], [], "Source image ready. Brush over the object to create a mask."


def normalize_box_points(points):
    if len(points) < 2:
        return points
    x1, y1 = points[0]
    x2, y2 = points[1]
    left = min(int(x1), int(x2))
    right = max(int(x1), int(x2))
    top = min(int(y1), int(y2))
    bottom = max(int(y1), int(y2))
    return [[left, top], [right, bottom]]


def overlay_mask(image, mask, color=(255, 120, 40), alpha=0.42):
    output = image.astype(np.float32).copy()
    mask_bool = mask > 0
    if np.any(mask_bool):
        output[mask_bool] = output[mask_bool] * (1 - alpha) + np.array(color, dtype=np.float32) * alpha
    return np.clip(output, 0, 255).astype(np.uint8)


def draw_sam_prompts(image, points, mask=None):
    canvas = image.astype(np.uint8).copy()
    if mask is not None:
        canvas = overlay_mask(canvas, mask)

    pil_image = Image.fromarray(canvas).convert("RGB")
    draw = ImageDraw.Draw(pil_image)
    for point in points[:2]:
        x, y = int(point[0]), int(point[1])
        radius = 8
        draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=(255, 0, 0), outline=(255, 255, 255), width=2)

    if len(points) >= 2:
        (x1, y1), (x2, y2) = normalize_box_points(points)
        draw.rectangle((x1, y1, x2, y2), outline=(255, 0, 0), width=3)

    return np.array(pil_image)


def sam_error_message(exc):
    message = str(exc).strip() or exc.__class__.__name__
    return f"SAM mask extraction failed: {message}"


def clear_sam_selection(source_original_image, source_image, source_path):
    source = to_rgb_array(source_original_image)
    if source is None:
        source = resolve_source_for_sam(source_image, source_path)
    return source, source, [], [], "SAM box cleared. Click two opposite corners on the source image."


def segment_source_with_sam(
    source_image,
    source_path,
    source_original_image,
    sam_points,
    sam_point_labels,
    sam_checkpoint_path,
    gpu_id,
    evt: gr.SelectData,
):
    original = to_rgb_array(source_original_image)
    if original is None:
        original = resolve_source_for_sam(source_image, source_path)
    original = original.astype(np.uint8)

    points = [list(point) for point in (sam_points or [])]
    point_labels = list(sam_point_labels or [])
    x, y = int(evt.index[0]), int(evt.index[1])
    if len(points) >= 2:
        points = []
        point_labels = []

    points.append([x, y])
    point_labels.append(2 if len(points) == 1 else 3)

    if len(points) == 1:
        annotated = draw_sam_prompts(original, points)
        return annotated, original, gr.update(), gr.update(), gr.update(), points, point_labels, "First SAM corner selected. Click the opposite corner."

    points = normalize_box_points(points)
    point_labels = [2, 3]
    annotated = draw_sam_prompts(original, points)
    try:
        import torch
        from torchvision import transforms

        model, device = get_sam_model(sam_checkpoint_path, gpu_id)
        input_points = np.array(points, dtype=np.float32)
        input_labels = np.array(point_labels, dtype=np.int64)
        pts_sampled = torch.reshape(torch.tensor(input_points, device=device), [1, 1, -1, 2])
        pts_labels = torch.reshape(torch.tensor(input_labels, device=device), [1, 1, -1])
        img_tensor = transforms.ToTensor()(original).to(device)

        with torch.no_grad():
            predicted_logits, predicted_iou = model(
                img_tensor[None, ...],
                pts_sampled,
                pts_labels,
            )
    except Exception as exc:
        return annotated, original, gr.update(), gr.update(), gr.update(), points, point_labels, sam_error_message(exc)

    mask = torch.ge(predicted_logits[0, 0, 0, :, :], 0).float().cpu().numpy()
    mask_image = (mask * 255).astype(np.uint8)
    annotated = draw_sam_prompts(original, points, mask_image)
    best_iou = float(predicted_iou[0, 0, 0].detach().cpu())
    status = f"SAM mask extracted. Estimated IoU: {best_iou:.3f}"
    return annotated, original, mask_image, mask_image, mask_image, points, point_labels, status


def segment_source_image_clicks_with_sam(
    sam_click_mode,
    source_image,
    source_path,
    source_original_image,
    sam_points,
    sam_point_labels,
    sam_checkpoint_path,
    gpu_id,
    evt: gr.SelectData,
):
    if not sam_click_mode:
        return (
            gr.update(),
            gr.update(),
            gr.update(),
            gr.update(),
            gr.update(),
            sam_points or [],
            sam_point_labels or [],
            gr.update(),
        )

    return segment_source_with_sam(
        source_image,
        source_path,
        source_original_image,
        sam_points,
        sam_point_labels,
        sam_checkpoint_path,
        gpu_id,
        evt,
    )


def placement_preview(source_image, source_path, source_original_image, mask_image, mask_path, active_mask, target_image, target_path, ss, ts, mask_scale, x, y):
    ss = int(ss)
    ts = int(ts)

    target_image = resolve_target(target_image, target_path)
    source_input, mask_input = resolve_source_and_mask(source_image, mask_image, source_path, source_original_image, mask_path, active_mask)
    source_image, prepared_mask = prepare_source_object(source_input, mask_input, ss, mask_scale)
    source_h, source_w = source_image.shape[:2]
    x, y = fit_source_placement(x, y, source_image.shape, ts)
    source = Image.fromarray(source_image.astype(np.uint8)).convert("RGB")
    target = Image.fromarray(to_rgb_array(target_image).astype(np.uint8)).convert("RGB").resize((ts, ts))
    mask = Image.fromarray((prepared_mask * 255).astype(np.uint8)).convert("L")
    mask_display = (prepared_mask * 255).astype(np.uint8)

    source_np = np.array(source).astype(np.float32)
    target_np = np.array(target).astype(np.float32)
    mask_np = (np.array(mask) > 0).astype(np.float32)

    top = int(x - source_h * 0.5)
    left = int(y - source_w * 0.5)
    preview = target_np.copy()
    region = preview[top: top + source_h, left: left + source_w]

    alpha = mask_np[..., None] * 0.65
    region[:] = region * (1 - alpha) + source_np * alpha

    outline = mask_np > 0
    preview_region = preview[top: top + source_h, left: left + source_w]
    preview_region[outline] = preview_region[outline] * 0.65 + np.array([255, 80, 40]) * 0.35

    return np.clip(preview, 0, 255).astype(np.uint8), mask_display, x, y


def run_first_pass(
    source_image,
    source_path,
    source_original_image,
    mask_image,
    mask_path,
    active_mask,
    target_image,
    target_path,
    ss,
    ts,
    mask_scale,
    x,
    y,
    gpu_id,
    num_steps,
    grad_weight,
    style_weight,
    content_weight,
    tv_weight,
    seed,
    output_dir,
):
    target_image = resolve_target(target_image, target_path)
    source_image, mask_image = resolve_source_and_mask(source_image, mask_image, source_path, source_original_image, mask_path, active_mask)
    ss = int(ss)
    ts = int(ts)
    prepared_source, _ = prepare_source_object(source_image, mask_image, ss, mask_scale)
    x, y = fit_source_placement(x, y, prepared_source.shape, ts)
    seed_value = None if seed is None or int(seed) < 0 else int(seed)

    image, output_path, history = first_pass_blend(
        source_image=source_image,
        mask_image=mask_image,
        target_image=target_image,
        output_dir=output_dir or DEFAULT_OUTPUT_DIR,
        ss=ss,
        ts=ts,
        x=x,
        y=y,
        gpu_id=gpu_id,
        num_steps=int(num_steps),
        grad_weight=float(grad_weight),
        style_weight=float(style_weight),
        content_weight=float(content_weight),
        tv_weight=float(tv_weight),
        mask_scale=float(mask_scale),
        seed=seed_value,
        progress_interval=max(1, int(num_steps) // 20),
    )

    losses = history[-1] if history else {}
    status = f"Saved first-pass image to {Path(output_path).resolve()}"
    return [image], output_path, losses, status, x, y


def clear_demo():
    return None, "", None, "", None, None, "", None, [], [], None, None, None, None, {}, "", loading_panel_html()


def create_demo_blend(runner):
    with gr.Blocks() as demo:
        source_original_image = gr.State(value=None)
        sam_points = gr.State([])
        sam_point_labels = gr.State([])
        active_mask_state = gr.State(value=None)
        ss = gr.State(value=512)
        mask_scale = gr.State(value=1.0)

        gr.Markdown(BLEND_DESCRIPTION)
        with gr.Row():
            with gr.Column():
                with gr.Group():
                    gr.Markdown("# INPUT")
                    gr.Markdown("## 1. Upload source image and draw object mask")
                    source_image = make_source_draw_image("Source image")
                    source_path = gr.Textbox(
                        label="Fast source path on Kaggle/local machine",
                        placeholder="/kaggle/input/your-dataset/source.jpg",
                    )

                    gr.Markdown("## 2. Optional mask upload")
                    mask_image = make_upload_image("Mask image", image_mode="L")
                    mask_path = gr.Textbox(
                        label="Fast mask path on Kaggle/local machine",
                        placeholder="/kaggle/input/your-dataset/mask.png",
                    )
                    gr.Markdown("## 3. Upload target image")
                    target_image = make_upload_image("Target image")
                    target_path = gr.Textbox(
                        label="Fast target path on Kaggle/local machine",
                        placeholder="/kaggle/input/your-dataset/target.jpg",
                    )
                    load_paths_button = gr.Button("Load Images From Paths")

                    gr.Markdown("## 4. Position and size")
                    ts = gr.Slider(128, 1024, value=512, step=1, label="Target size (--ts)")
                    with gr.Row():
                        x = gr.Slider(0, 1024, value=256, step=1, label="Vertical center (--x)")
                        y = gr.Slider(0, 1024, value=256, step=1, label="Horizontal center (--y)")

                    with gr.Row():
                        center_button = gr.Button("Use Target Center")
                        preview_button = gr.Button("Preview Placement")
                    with gr.Row():
                        run_button = gr.Button("Edit", variant="primary")
                        clear_button = gr.Button("Clear")

                    with gr.Group():
                        gr.Markdown("## 5. Optimization")
                        with gr.Row():
                            gpu_id = gr.Dropdown(
                                ["auto", "cpu", "cuda:0", "cuda:1"],
                                value="auto",
                                label="Device (--gpu_id)",
                                allow_custom_value=True,
                            )
                            num_steps = gr.Slider(1, 3000, value=100, step=1, label="Steps (--num_steps)")
                        gr.Markdown("## Loss weights")
                        with gr.Row():
                            grad_weight = gr.Slider(0, 50000, value=1e4, step=100, label="Gradient loss weight")
                            style_weight = gr.Slider(0, 50000, value=1e4, step=100, label="Style loss weight")
                        with gr.Row():
                            content_weight = gr.Slider(0, 10, value=1.0, step=0.1, label="Content loss weight")
                            tv_weight = gr.Number(value=1e-6, label="TV loss weight")
                        with gr.Accordion("Advanced options", open=False):
                            seed = gr.Number(value=0, precision=0, label="Seed, use -1 for random")
                            output_dir = gr.Textbox(value=DEFAULT_OUTPUT_DIR, label="Output directory (--output_dir)")

            with gr.Column():
                with gr.Group():
                    gr.Markdown("# OUTPUT")
                    with gr.Row():
                        mask_preview = gr.Image(label="Active mask", type="numpy", image_mode="L")
                        preview_image = gr.Image(label="Placement preview", type="numpy")

                    gr.Markdown("<h5><center>Results</center></h5>")
                    output = gr.Gallery(label="Results", columns=1, height="auto")

                    with gr.Row():
                        output_file = gr.File(label="Saved first_pass.png")
                        losses = gr.JSON(label="Latest logged losses")
                    status = gr.Textbox(label="Status", interactive=False)
                    loading_panel = gr.HTML(loading_panel_html())

        with gr.Column():
            gr.Markdown("Try some of the examples below ⬇️")
            gr.Examples(
                examples=examples_blend,
                inputs=[source_image, mask_image, target_image, ts, x, y],
            )

        demo.load(
            fn=None,
            inputs=[],
            outputs=[],
            js=INSTALL_LOADING_JS,
            queue=False,
        )

        load_paths_event = load_paths_button.click(
            show_upload_loading,
            inputs=[],
            outputs=[loading_panel],
            queue=False,
            show_progress="hidden",
            js=SHOW_UPLOAD_LOADING_JS,
        ).then(
            load_images_from_paths,
            inputs=[source_path, mask_path, target_path],
            outputs=[source_image, mask_image, target_image, source_original_image, sam_points, sam_point_labels, active_mask_state],
            show_progress="full",
        )
        load_paths_event.success(hide_loading, inputs=[], outputs=[loading_panel], queue=False, show_progress="hidden")
        load_paths_event.failure(hide_loading, inputs=[], outputs=[loading_panel], queue=False, show_progress="hidden")

        source_upload_event = source_image.upload(
            show_upload_loading,
            inputs=[],
            outputs=[loading_panel],
            queue=False,
            show_progress="hidden",
            js=SHOW_UPLOAD_LOADING_JS,
        ).then(
            store_source_original,
            inputs=[source_image, source_path],
            outputs=[source_original_image, active_mask_state, sam_points, sam_point_labels, status],
            queue=False,
            show_progress="hidden",
        )
        source_upload_event.success(hide_loading, inputs=[], outputs=[loading_panel], queue=False, show_progress="hidden")
        source_upload_event.failure(hide_loading, inputs=[], outputs=[loading_panel], queue=False, show_progress="hidden")

        mask_upload_event = mask_image.upload(
            show_upload_loading,
            inputs=[],
            outputs=[loading_panel],
            queue=False,
            show_progress="hidden",
            js=SHOW_UPLOAD_LOADING_JS,
        ).then(
            store_active_mask,
            inputs=[mask_image],
            outputs=[active_mask_state],
            queue=False,
            show_progress="hidden",
        )
        mask_upload_event.success(hide_loading, inputs=[], outputs=[loading_panel], queue=False, show_progress="hidden")
        mask_upload_event.failure(hide_loading, inputs=[], outputs=[loading_panel], queue=False, show_progress="hidden")

        target_upload_event = target_image.upload(
                show_upload_loading,
                inputs=[],
                outputs=[loading_panel],
                queue=False,
                show_progress="hidden",
                js=SHOW_UPLOAD_LOADING_JS,
        )
        target_upload_event.success(hide_loading, inputs=[], outputs=[loading_panel], queue=False, show_progress="hidden")
        target_upload_event.failure(hide_loading, inputs=[], outputs=[loading_panel], queue=False, show_progress="hidden")

        source_image.change(
            activate_drawn_source_mask,
            inputs=[source_image, source_path, source_original_image],
            outputs=[mask_image, mask_preview, source_original_image, status],
            queue=False,
            show_progress="hidden",
        )
        center_button.click(center_position, inputs=[ts], outputs=[x, y])

        preview_event = preview_button.click(
            show_preview_loading,
            inputs=[],
            outputs=[loading_panel],
            queue=False,
            show_progress="hidden",
            js=SHOW_PREVIEW_LOADING_JS,
        ).then(
            placement_preview,
            inputs=[source_image, source_path, source_original_image, mask_image, mask_path, active_mask_state, target_image, target_path, ss, ts, mask_scale, x, y],
            outputs=[preview_image, mask_preview, x, y],
            show_progress="full",
            show_progress_on=[preview_image, mask_preview],
        )
        preview_event.success(hide_loading, inputs=[], outputs=[loading_panel], queue=False, show_progress="hidden")
        preview_event.failure(hide_loading, inputs=[], outputs=[loading_panel], queue=False, show_progress="hidden")

        run_event = run_button.click(
            show_edit_loading,
            inputs=[],
            outputs=[loading_panel],
            queue=False,
            show_progress="hidden",
            js=SHOW_EDIT_LOADING_JS,
        ).then(
            runner,
            inputs=[
                source_image,
                source_path,
                source_original_image,
                mask_image,
                mask_path,
                active_mask_state,
                target_image,
                target_path,
                ss,
                ts,
                mask_scale,
                x,
                y,
                gpu_id,
                num_steps,
                grad_weight,
                style_weight,
                content_weight,
                tv_weight,
                seed,
                output_dir,
            ],
            outputs=[output, output_file, losses, status, x, y],
            show_progress="full",
            show_progress_on=[output, status],
        )
        run_event.success(hide_loading, inputs=[], outputs=[loading_panel], queue=False, show_progress="hidden")
        run_event.failure(hide_loading, inputs=[], outputs=[loading_panel], queue=False, show_progress="hidden")

        clear_button.click(
            clear_demo,
            inputs=[],
            outputs=[
                source_image,
                source_path,
                mask_image,
                mask_path,
                active_mask_state,
                target_image,
                target_path,
                source_original_image,
                sam_points,
                sam_point_labels,
                mask_preview,
                preview_image,
                output,
                output_file,
                losses,
                status,
                loading_panel,
            ],
        )
    return demo


if __name__ == "__main__":
    os.makedirs(DEFAULT_OUTPUT_DIR, exist_ok=True)
    with gr.Blocks(css=load_css()) as demo:
        gr.Markdown(DESCRIPTION)
        with gr.Tabs():
            with gr.TabItem("First-Pass Blending"):
                create_demo_blend(run_first_pass)

    demo.queue(max_size=20, default_concurrency_limit=3)
    demo.launch(server_name="0.0.0.0", share=True)
