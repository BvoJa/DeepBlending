import os
import sys
from pathlib import Path

import numpy as np
from PIL import Image

from my_run import first_pass_blend, validate_placement


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
EXAMPLES = [
    ["data/1_source.png", "data/1_target.png", 300, 512, 200, 235],
    ["data/2_source.png", "data/2_target.png", 300, 512, 200, 235],
    ["data/3_source.png", "data/3_target.png", 300, 512, 200, 235],
    ["data/4_source.png", "data/4_target.png", 300, 512, 200, 235],
    ["data/5_source.png", "data/5_target.png", 300, 512, 200, 235],
]


def center_position(ts):
    center = int(ts // 2)
    return center, center


def to_rgb_array(image):
    if image is None:
        return None
    if isinstance(image, np.ndarray):
        return np.array(Image.fromarray(image.astype(np.uint8)).convert("RGB"))
    return np.array(Image.open(image).convert("RGB"))


def editor_to_source_and_mask(source_editor):
    if source_editor is None:
        raise gr.Error("Upload a source image and draw the object mask first.")

    if not isinstance(source_editor, dict):
        raise gr.Error("Use the source editor so the demo can read your drawn mask.")

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
        layer = np.asarray(layer)
        if layer.ndim == 3 and layer.shape[2] == 4:
            layer_mask = layer[:, :, 3] > 0
        elif layer.ndim == 3:
            layer_mask = np.any(layer[:, :, :3] > 0, axis=2)
        else:
            layer_mask = layer > 0
        mask[layer_mask] = 255

    if mask.max() == 0:
        raise gr.Error("Draw over the object in the source image before previewing or running.")

    return source, mask


def placement_preview(source_editor, target_image, ss, ts, x, y):
    if target_image is None:
        raise gr.Error("Upload a target image first.")

    ss = int(ss)
    ts = int(ts)
    x = int(x)
    y = int(y)
    validate_placement(x, y, ss, ts)

    source_image, mask_image = editor_to_source_and_mask(source_editor)
    source = Image.fromarray(source_image.astype(np.uint8)).convert("RGB").resize((ss, ss))
    target = Image.fromarray(target_image.astype(np.uint8)).convert("RGB").resize((ts, ts))
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

    return np.clip(preview, 0, 255).astype(np.uint8), (mask_np * 255).astype(np.uint8)


def run_first_pass(
    source_editor,
    target_image,
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
    if target_image is None:
        raise gr.Error("Upload a target image first.")

    source_image, mask_image = editor_to_source_and_mask(source_editor)
    seed_value = None if seed is None or int(seed) < 0 else int(seed)
    image, output_path, history = first_pass_blend(
        source_image=source_image,
        mask_image=mask_image,
        target_image=target_image,
        output_dir=output_dir or DEFAULT_OUTPUT_DIR,
        ss=int(ss),
        ts=int(ts),
        x=int(x),
        y=int(y),
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
    return image, output_path, losses, status


def build_demo():
    with gr.Blocks(title="Deep Image Blending") as demo:
        gr.Markdown("# Deep Image Blending")

        with gr.Row():
            source_editor = gr.ImageEditor(
                label="Source image: draw over the object to create the mask",
                sources=["upload", "clipboard"],
                type="numpy",
                image_mode="RGBA",
                brush=gr.Brush(default_size=35, colors=["rgba(255, 120, 40, 0.75)"], color_mode="fixed"),
                eraser=gr.Eraser(default_size=35),
                layers=True,
                transforms=[],
            )
            target_image = gr.Image(
                label="Target image",
                sources=["upload", "clipboard"],
                type="numpy",
                image_mode="RGB",
            )

        with gr.Row():
            ss = gr.Slider(64, 768, value=300, step=1, label="Source size (--ss)")
            ts = gr.Slider(128, 1024, value=512, step=1, label="Target size (--ts)")
            x = gr.Slider(0, 1024, value=200, step=1, label="Vertical center (--x)")
            y = gr.Slider(0, 1024, value=235, step=1, label="Horizontal center (--y)")

        with gr.Row():
            center_button = gr.Button("Use Target Center")
            preview_button = gr.Button("Preview Placement")
            run_button = gr.Button("Run First Pass", variant="primary")

        with gr.Accordion("Optimization", open=True):
            with gr.Row():
                gpu_id = gr.Dropdown(
                    ["auto", "cpu", "cuda:0", "cuda:1"],
                    value="auto",
                    label="Device (--gpu_id)",
                    allow_custom_value=True,
                )
                num_steps = gr.Slider(1, 3000, value=100, step=1, label="Steps (--num_steps)")
                seed = gr.Number(value=0, precision=0, label="Seed, use -1 for random")
                output_dir = gr.Textbox(value=DEFAULT_OUTPUT_DIR, label="Output directory (--output_dir)")

            with gr.Row():
                grad_weight = gr.Number(value=1e4, label="Gradient weight")
                style_weight = gr.Number(value=1e4, label="Style weight")
                content_weight = gr.Number(value=1.0, label="Content weight")
                tv_weight = gr.Number(value=1e-6, label="TV weight")

        with gr.Row():
            preview_image = gr.Image(label="Placement preview", type="numpy")
            mask_preview = gr.Image(label="Drawn mask", type="numpy", image_mode="L")
            result_image = gr.Image(label="First-pass result", type="numpy")

        with gr.Row():
            output_file = gr.File(label="Saved first_pass.png")
            losses = gr.JSON(label="Latest logged losses")

        status = gr.Textbox(label="Status", interactive=False)

        center_button.click(center_position, inputs=[ts], outputs=[x, y])
        preview_button.click(
            placement_preview,
            inputs=[source_editor, target_image, ss, ts, x, y],
            outputs=[preview_image, mask_preview],
        )
        run_button.click(
            run_first_pass,
            inputs=[
                source_editor,
                target_image,
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
            outputs=[result_image, output_file, losses, status],
        )

        gr.Examples(
            examples=EXAMPLES,
            inputs=[source_editor, target_image, ss, ts, x, y],
        )

    return demo


if __name__ == "__main__":
    os.makedirs(DEFAULT_OUTPUT_DIR, exist_ok=True)
    build_demo().launch()
