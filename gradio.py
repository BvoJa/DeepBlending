import os
import inspect
import sys
from pathlib import Path

import numpy as np
from PIL import Image

from my_run import first_pass_blend


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

DESCRIPTION = """
# Deep Image Blending
Gradio demo for first-pass object blending. Upload source and target images, upload or draw the object mask, choose placement and loss weights, then click `Edit`.
"""

BLEND_DESCRIPTION = """
## First-Pass Object Blending
Usage:
- Upload source and target images using the same plain image-upload style as DragonDiffusion.
- Upload a mask image, or draw over the object to create the mask.
- Upload a target image, or paste a Kaggle/local file path to avoid slow browser upload.
- Adjust source size, target size, and object center.
- Adjust the loss weights that contribute to the first-pass total loss.
- Click `Preview Placement` to inspect the mask and location.
- Click `Edit` to run first-pass image blending.
"""

examples_blend = [
    ["data/1_source.png", "data/1_target.png", 300, 512, 200, 235],
    ["data/2_source.png", "data/2_target.png", 300, 512, 200, 235],
    ["data/3_source.png", "data/3_target.png", 300, 512, 200, 235],
    ["data/4_source.png", "data/4_target.png", 300, 512, 200, 235],
    ["data/5_source.png", "data/5_target.png", 300, 512, 200, 235],
]


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


def make_source_mask_input(label):
    image_params = inspect.signature(gr.Image).parameters
    if "tool" in image_params:
        kwargs = {
            "label": label,
            "interactive": True,
            "type": "numpy",
            "tool": "sketch",
        }
        if "source" in image_params:
            kwargs["source"] = "upload"
        else:
            kwargs["sources"] = ["upload"]
        return gr.Image(**kwargs)

    return gr.ImageEditor(
        label=label,
        sources=["upload"],
        type="numpy",
        image_mode="RGBA",
        brush=gr.Brush(default_size=35, colors=["rgba(255, 120, 40, 0.75)"], color_mode="fixed"),
        eraser=gr.Eraser(default_size=35),
        layers=True,
        transforms=[],
    )


def load_source_for_mask_editor(source_image):
    if source_image is None:
        return None
    return to_rgb_array(source_image)


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


def resolve_source_and_mask(source_image, mask_image, source_editor):
    source = to_rgb_array(source_image)

    uploaded_mask = to_mask_array(mask_image)
    if uploaded_mask is not None:
        if source is None:
            source, _ = editor_to_source_and_mask(source_editor)
        if source is None:
            raise gr.Error("Upload a source image before using a mask image.")
        return source, uploaded_mask

    drawn_source, drawn_mask = editor_to_source_and_mask(source_editor)
    if source is not None:
        drawn_source = source
    return drawn_source, drawn_mask


def editor_to_source_and_mask(source_editor):
    if source_editor is None:
        raise gr.Error("Upload a source image and draw the object mask first.")

    if not isinstance(source_editor, dict):
        raise gr.Error("Use the source editor so the demo can read your drawn mask.")

    if "image" in source_editor and "mask" in source_editor:
        source = to_rgb_array(source_editor.get("image"))
        raw_mask = source_editor.get("mask")
        if source is None or raw_mask is None:
            raise gr.Error("Upload a source image and draw the object mask first.")

        raw_mask = np.asarray(raw_mask)
        if raw_mask.ndim == 3 and raw_mask.shape[2] == 4:
            mask = (raw_mask[:, :, 3] > 0).astype(np.uint8) * 255
        elif raw_mask.ndim == 3:
            mask = (np.any(raw_mask[:, :, :3] > 0, axis=2)).astype(np.uint8) * 255
        else:
            mask = (raw_mask > 0).astype(np.uint8) * 255

        if mask.shape != source.shape[:2]:
            mask = np.array(Image.fromarray(mask).resize(source.shape[:2][::-1]))
        if mask.max() == 0:
            raise gr.Error("Draw over the object in the source image before previewing or running.")
        return source, mask

    source = source_editor.get("background")
    if source is None:
        source = source_editor.get("composite")
    source = to_rgb_array(source)
    if source is None:
        raise gr.Error("Upload a source image before drawing the mask.")

    mask = np.zeros(source.shape[:2], dtype=np.uint8)
    for layer in source_editor.get("layers") or []:
        if layer is None:
            continue
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
        raise gr.Error("Draw over the object in the source image before previewing or running.")

    return source, mask


def placement_preview(source_image, source_editor, mask_image, target_image, target_path, ss, ts, x, y):
    ss = int(ss)
    ts = int(ts)
    x, y = fit_placement(x, y, ss, ts)

    target_image = resolve_target(target_image, target_path)
    source_image, mask_image = resolve_source_and_mask(source_image, mask_image, source_editor)
    source = Image.fromarray(source_image.astype(np.uint8)).convert("RGB").resize((ss, ss))
    target = Image.fromarray(to_rgb_array(target_image).astype(np.uint8)).convert("RGB").resize((ts, ts))
    mask = Image.fromarray(mask_image.astype(np.uint8)).convert("L").resize((ss, ss))

    source_np = np.array(source).astype(np.float32)
    target_np = np.array(target).astype(np.float32)
    mask_np = (np.array(mask) > 0).astype(np.float32)

    top = int(x - ss * 0.5)
    left = int(y - ss * 0.5)
    preview = target_np.copy()
    region = preview[top: top + ss, left: left + ss]

    alpha = mask_np[..., None] * 0.65
    region[:] = region * (1 - alpha) + source_np * alpha

    outline = mask_np > 0
    preview_region = preview[top: top + ss, left: left + ss]
    preview_region[outline] = preview_region[outline] * 0.65 + np.array([255, 80, 40]) * 0.35

    return np.clip(preview, 0, 255).astype(np.uint8), (mask_np * 255).astype(np.uint8), x, y


def run_first_pass(
    source_image,
    source_editor,
    mask_image,
    target_image,
    target_path,
    ss,
    ts,
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
    source_image, mask_image = resolve_source_and_mask(source_image, mask_image, source_editor)
    ss = int(ss)
    ts = int(ts)
    x, y = fit_placement(x, y, ss, ts)
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
        seed=seed_value,
        progress_interval=max(1, int(num_steps) // 20),
    )

    losses = history[-1] if history else {}
    status = f"Saved first-pass image to {Path(output_path).resolve()}"
    return [image], output_path, losses, status, x, y


def clear_demo():
    return None, None, None, None, "", None, None, None, None, {}, ""


def create_demo_blend(runner):
    with gr.Blocks() as demo:
        gr.Markdown(BLEND_DESCRIPTION)
        with gr.Row():
            with gr.Column():
                with gr.Group():
                    gr.Markdown("# INPUT")
                    gr.Markdown("## 1. Upload source image")
                    source_image = make_upload_image("Source image")

                    gr.Markdown("## 2. Upload or draw object mask")
                    mask_image = make_upload_image("Mask image", image_mode="L")
                    source_editor = make_source_mask_input("Draw mask on source image")

                    gr.Markdown("## 3. Upload target image")
                    target_image = make_upload_image("Target image")
                    target_path = gr.Textbox(
                        label="Fast target path on Kaggle/local machine",
                        placeholder="/kaggle/input/your-dataset/target.jpg",
                    )

                    gr.Markdown("## 4. Position and size")
                    with gr.Row():
                        ss = gr.Slider(64, 768, value=300, step=1, label="Source size (--ss)")
                        ts = gr.Slider(128, 1024, value=512, step=1, label="Target size (--ts)")
                    with gr.Row():
                        x = gr.Slider(0, 1024, value=200, step=1, label="Vertical center (--x)")
                        y = gr.Slider(0, 1024, value=235, step=1, label="Horizontal center (--y)")

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
                        mask_preview = gr.Image(label="Drawn mask", type="numpy", image_mode="L")
                        preview_image = gr.Image(label="Placement preview", type="numpy")

                    gr.Markdown("<h5><center>Results</center></h5>")
                    output = gr.Gallery(label="Results", columns=1, height="auto")

                    with gr.Row():
                        output_file = gr.File(label="Saved first_pass.png")
                        losses = gr.JSON(label="Latest logged losses")
                    status = gr.Textbox(label="Status", interactive=False)

        with gr.Column():
            gr.Markdown("Try some of the examples below ⬇️")
            gr.Examples(
                examples=examples_blend,
                inputs=[source_image, target_image, ss, ts, x, y],
            )

        source_image.change(load_source_for_mask_editor, inputs=[source_image], outputs=[source_editor])
        center_button.click(center_position, inputs=[ts], outputs=[x, y])
        preview_button.click(
            placement_preview,
            inputs=[source_image, source_editor, mask_image, target_image, target_path, ss, ts, x, y],
            outputs=[preview_image, mask_preview, x, y],
        )
        run_button.click(
            runner,
            inputs=[
                source_image,
                source_editor,
                mask_image,
                target_image,
                target_path,
                ss,
                ts,
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
        )
        clear_button.click(
            clear_demo,
            inputs=[],
            outputs=[source_image, source_editor, mask_image, target_image, target_path, mask_preview, preview_image, output, output_file, losses, status],
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
