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
    ["data/1_source.png", "data/1_mask.png", "data/1_target.png", 300, 512, 200, 235],
    ["data/2_source.png", "data/2_mask.png", "data/2_target.png", 300, 512, 200, 235],
    ["data/3_source.png", "data/3_mask.png", "data/3_target.png", 300, 512, 200, 235],
    ["data/4_source.png", "data/4_mask.png", "data/4_target.png", 300, 512, 200, 235],
    ["data/5_source.png", "data/5_mask.png", "data/5_target.png", 300, 512, 200, 235],
]


def center_position(ts):
    center = int(ts // 2)
    return center, center


def file_to_numpy(file_path, mode):
    if file_path is None:
        return None
    path = file_path.name if hasattr(file_path, "name") else file_path
    return np.array(Image.open(path).convert(mode))


def load_uploaded_files(source_file, mask_file, target_file):
    return (
        file_to_numpy(source_file, "RGB"),
        file_to_numpy(mask_file, "L"),
        file_to_numpy(target_file, "RGB"),
    )


def placement_preview(source_image, mask_image, target_image, ss, ts, x, y):
    if source_image is None or mask_image is None or target_image is None:
        raise gr.Error("Upload source, mask, and target images first.")

    ss = int(ss)
    ts = int(ts)
    x = int(x)
    y = int(y)
    validate_placement(x, y, ss, ts)

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

    return np.clip(preview, 0, 255).astype(np.uint8)


def run_first_pass(
    source_image,
    mask_image,
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
    if source_image is None or mask_image is None or target_image is None:
        raise gr.Error("Upload source, mask, and target images first.")

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
            source_file = gr.File(label="Source file", file_types=["image"], type="filepath")
            mask_file = gr.File(label="Mask file", file_types=["image"], type="filepath")
            target_file = gr.File(label="Target file", file_types=["image"], type="filepath")

        load_files_button = gr.Button("Load Selected Files")

        with gr.Row():
            source_image = gr.Image(
                label="Source object image",
                sources=["upload", "clipboard"],
                type="numpy",
                image_mode="RGB",
            )
            mask_image = gr.Image(
                label="Source object mask",
                sources=["upload", "clipboard"],
                type="numpy",
                image_mode="L",
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
            result_image = gr.Image(label="First-pass result", type="numpy")

        with gr.Row():
            output_file = gr.File(label="Saved first_pass.png")
            losses = gr.JSON(label="Latest logged losses")

        status = gr.Textbox(label="Status", interactive=False)

        load_files_button.click(
            load_uploaded_files,
            inputs=[source_file, mask_file, target_file],
            outputs=[source_image, mask_image, target_image],
        )
        center_button.click(center_position, inputs=[ts], outputs=[x, y])
        preview_button.click(
            placement_preview,
            inputs=[source_image, mask_image, target_image, ss, ts, x, y],
            outputs=[preview_image],
        )
        run_button.click(
            run_first_pass,
            inputs=[
                source_image,
                mask_image,
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
            inputs=[source_image, mask_image, target_image, ss, ts, x, y],
        )

    return demo


if __name__ == "__main__":
    os.makedirs(DEFAULT_OUTPUT_DIR, exist_ok=True)
    build_demo().launch()
