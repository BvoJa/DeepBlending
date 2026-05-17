import os
import html
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

DESCRIPTION = """
# Deep Image Blending
Gradio demo for first-pass object blending. Upload a source image, draw or upload its mask, upload a target image, choose placement and loss weights, then click `Edit`.
"""

BLEND_DESCRIPTION = """
## First-Pass Object Blending
Usage:
- Upload a source image and draw over the object to create the mask.
- Or upload a mask image in the mask box.
- Upload a target image using the same plain image-upload style as DragonDiffusion.
- Optionally paste Kaggle/local file paths to avoid slow browser upload.
- Adjust source size, target size, and object center.
- Adjust the loss weights that contribute to the first-pass total loss.
- Click `Preview Placement` to inspect the mask and location.
- Click `Edit` to run first-pass image blending.
"""

examples_blend = [
    ["data/1_source.png", "data/1_mask.png", "data/1_target.png", 300, 512, 200, 235],
    ["data/2_source.png", "data/2_mask.png", "data/2_target.png", 300, 512, 200, 235],
    ["data/3_source.png", "data/3_mask.png", "data/3_target.png", 300, 512, 200, 235],
    ["data/4_source.png", "data/4_mask.png", "data/4_target.png", 300, 512, 200, 235],
    ["data/5_source.png", "data/5_mask.png", "data/5_target.png", 300, 512, 200, 235],
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


def load_image_path(path, mode):
    path = (path or "").strip()
    if not path:
        return None
    if not os.path.exists(path):
        raise gr.Error(f"Image path does not exist: {path}")
    return np.array(Image.open(path).convert(mode))


def load_images_from_paths(source_path, mask_path, target_path):
    source = load_image_path(source_path, "RGB") if (source_path or "").strip() else gr.update()
    mask = load_image_path(mask_path, "L") if (mask_path or "").strip() else gr.update()
    target = load_image_path(target_path, "RGB") if (target_path or "").strip() else gr.update()
    return source, mask, target


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


def resolve_source_and_mask(source_image, mask_image, source_path="", mask_path=""):
    source_from_path = load_image_path(source_path, "RGB")
    drawn_source, drawn_mask = extract_drawn_source_and_mask(source_image)
    source = source_from_path if source_from_path is not None else drawn_source
    if source is None:
        source = to_rgb_array(source_image)

    mask_from_path = load_image_path(mask_path, "L")
    uploaded_mask = to_mask_array(mask_from_path if mask_from_path is not None else mask_image)
    if uploaded_mask is not None:
        if source is None:
            raise gr.Error("Upload a source image before using a mask image.")
        return source, uploaded_mask

    if drawn_mask is not None:
        if source is None:
            raise gr.Error("Upload a source image before drawing the mask.")
        return source, drawn_mask

    if source is None:
        raise gr.Error("Upload a source image first.")
    raise gr.Error("Draw over the object in the source image, or upload a mask image.")


def placement_preview(source_image, source_path, mask_image, mask_path, target_image, target_path, ss, ts, x, y):
    ss = int(ss)
    ts = int(ts)
    x, y = fit_placement(x, y, ss, ts)

    target_image = resolve_target(target_image, target_path)
    source_image, mask_image = resolve_source_and_mask(source_image, mask_image, source_path, mask_path)
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
    source_path,
    mask_image,
    mask_path,
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
    source_image, mask_image = resolve_source_and_mask(source_image, mask_image, source_path, mask_path)
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
    return None, "", None, "", None, "", None, None, None, None, {}, "", loading_panel_html()


def create_demo_blend(runner):
    with gr.Blocks() as demo:
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
                inputs=[source_image, mask_image, target_image, ss, ts, x, y],
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
            outputs=[source_image, mask_image, target_image],
            show_progress="full",
        )
        load_paths_event.success(hide_loading, inputs=[], outputs=[loading_panel], queue=False, show_progress="hidden")
        load_paths_event.failure(hide_loading, inputs=[], outputs=[loading_panel], queue=False, show_progress="hidden")

        for upload_component in (source_image, mask_image, target_image):
            upload_event = upload_component.upload(
                show_upload_loading,
                inputs=[],
                outputs=[loading_panel],
                queue=False,
                show_progress="hidden",
                js=SHOW_UPLOAD_LOADING_JS,
            )
            upload_event.success(hide_loading, inputs=[], outputs=[loading_panel], queue=False, show_progress="hidden")
            upload_event.failure(hide_loading, inputs=[], outputs=[loading_panel], queue=False, show_progress="hidden")

        source_image.change(
            hide_loading,
            inputs=[],
            outputs=[loading_panel],
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
            inputs=[source_image, source_path, mask_image, mask_path, target_image, target_path, ss, ts, x, y],
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
                mask_image,
                mask_path,
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
                target_image,
                target_path,
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
