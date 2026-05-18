import argparse
import os
from pathlib import Path

import numpy as np
import torch
import torch.optim as optim
from PIL import Image
from skimage.io import imsave

from utils import (
    MeanShift,
    Vgg16,
    compute_gt_gradient,
    gram_matrix,
    laplacian_filter_tensor,
    numpy2tensor,
)


def resolve_device(gpu_id="auto"):
    if isinstance(gpu_id, str):
        value = gpu_id.strip().lower()
        if value == "auto":
            return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        if value == "cpu":
            return torch.device("cpu")
        if value.startswith("cuda"):
            return torch.device(value)
        if value.isdigit():
            return torch.device(f"cuda:{value}" if torch.cuda.is_available() else "cpu")
    if isinstance(gpu_id, int):
        return torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu")
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def load_rgb_image(image, size):
    if isinstance(image, np.ndarray):
        pil_image = Image.fromarray(image.astype(np.uint8))
    else:
        pil_image = Image.open(image)
    return np.array(pil_image.convert("RGB").resize((size, size)))


def load_rgb_array(image):
    if isinstance(image, np.ndarray):
        pil_image = Image.fromarray(image.astype(np.uint8))
    else:
        pil_image = Image.open(image)
    return np.array(pil_image.convert("RGB"))


def load_mask_array(image, size=None):
    if isinstance(image, np.ndarray):
        if image.ndim == 3:
            pil_image = Image.fromarray(image.astype(np.uint8)).convert("L")
        else:
            pil_image = Image.fromarray(image.astype(np.uint8))
    else:
        pil_image = Image.open(image)
    pil_image = pil_image.convert("L")
    if size is not None:
        pil_image = pil_image.resize(size, Image.NEAREST)
    mask = np.array(pil_image)
    mask[mask > 0] = 1
    return mask.astype(np.uint8)


def load_mask_image(image, size):
    return load_mask_array(image, (size, size))


def mask_bbox(mask):
    ys, xs = np.where(mask > 0)
    if len(xs) == 0 or len(ys) == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def expand_bbox(box, width, height, padding_ratio=0.04):
    left, top, right, bottom = box
    pad = int(round(max(right - left, bottom - top) * padding_ratio))
    return (
        max(0, left - pad),
        max(0, top - pad),
        min(width, right + pad),
        min(height, bottom + pad),
    )


def paste_center(canvas, image):
    canvas_h, canvas_w = canvas.shape[:2]
    image_h, image_w = image.shape[:2]
    top = (canvas_h - image_h) // 2
    left = (canvas_w - image_w) // 2
    dst_top = max(0, top)
    dst_left = max(0, left)
    dst_bottom = min(canvas_h, top + image_h)
    dst_right = min(canvas_w, left + image_w)
    src_top = max(0, -top)
    src_left = max(0, -left)
    src_bottom = src_top + (dst_bottom - dst_top)
    src_right = src_left + (dst_right - dst_left)
    if dst_bottom > dst_top and dst_right > dst_left:
        canvas[dst_top:dst_bottom, dst_left:dst_right] = image[src_top:src_bottom, src_left:src_right]
    return canvas


def prepare_source_object(source_image, mask_image, size, mask_scale=1.0):
    source = load_rgb_array(source_image)
    mask = load_mask_array(mask_image)
    if mask.shape[:2] != source.shape[:2]:
        raise ValueError(
            "The mask must have the same height and width as the source image. "
            "Use the SAM mask generated from this source image, or upload a matching mask."
        )

    box = mask_bbox(mask)
    if box is None:
        raise ValueError("The mask is empty. Draw, extract, or upload a non-empty object mask.")

    left, top, right, bottom = box
    return source[top:bottom, left:right], mask[top:bottom, left:right]


def paste_source_mask_to_target(x_start, y_start, target_img, mask):
    canvas_mask = np.zeros(target_img.shape[:2], dtype=mask.dtype)
    top = int(x_start - mask.shape[0] * 0.5)
    left = int(y_start - mask.shape[1] * 0.5)
    canvas_mask[top:top + mask.shape[0], left:left + mask.shape[1]] = mask
    return canvas_mask


def prepare_full_source_and_mask(source_image, mask_image, size):
    source = load_rgb_image(source_image, size)
    mask = load_mask_image(mask_image, size)
    return source, mask


def tensor_to_image(tensor):
    image = tensor.transpose(1, 3).transpose(1, 2).detach().cpu().numpy()[0]
    return np.clip(image, 0, 255).astype(np.uint8)


def validate_placement(x, y, ss, ts):
    half = ss * 0.5
    if x - half < 0 or y - half < 0 or x + half > ts or y + half > ts:
        raise ValueError(
            "The source window must fit inside the target canvas. "
            f"Use x/y between {int(half)} and {int(ts - half)} for ss={ss}, ts={ts}."
        )


def validate_source_placement(x, y, source_shape, target_shape):
    source_h, source_w = source_shape[:2]
    target_h, target_w = target_shape[:2]
    half_h = source_h * 0.5
    half_w = source_w * 0.5
    if x - half_h < 0 or y - half_w < 0 or x + half_h > target_h or y + half_w > target_w:
        raise ValueError(
            "The source image must fit inside the target canvas. "
            f"Use x between {int(half_h)} and {int(target_h - half_h)}, "
            f"and y between {int(half_w)} and {int(target_w - half_w)} "
            f"for source size {source_h}x{source_w} and target size {target_h}x{target_w}."
        )


def first_pass_blend(
    source_image,
    mask_image,
    target_image,
    style_image=None,
    output_dir="results/gradio",
    ss=512,
    ts=512,
    x=256,
    y=256,
    gpu_id="auto",
    num_steps=1000,
    grad_weight=1e4,
    style_weight=1e4,
    content_weight=1.0,
    tv_weight=1e-6,
    mask_scale=1.0,
    seed=None,
    progress_interval=10,
    save_output=True,
):
    device = resolve_device(gpu_id)
    if seed is not None:
        torch.manual_seed(int(seed))
        if device.type == "cuda":
            torch.cuda.manual_seed_all(int(seed))

    os.makedirs(output_dir, exist_ok=True)

    style_optimizes_target = style_image is not None
    source_img, mask_img = prepare_source_object(source_image, mask_image, ss, mask_scale)
    target_img = load_rgb_image(target_image, ts)
    style_img = load_rgb_image(style_image if style_optimizes_target else target_image, ts)
    validate_source_placement(x, y, source_img.shape, target_img.shape)

    canvas_mask = paste_source_mask_to_target(x, y, target_img, mask_img)
    canvas_mask = numpy2tensor(canvas_mask, device)
    canvas_mask = canvas_mask.squeeze(0).repeat(3, 1).view(3, ts, ts).unsqueeze(0)
    outside_mask = (canvas_mask - 1) * (-1)

    gt_gradient = compute_gt_gradient(x, y, source_img, target_img, mask_img, device)

    source_img = torch.from_numpy(source_img).unsqueeze(0).transpose(1, 3).transpose(2, 3).float().to(device)
    target_img = torch.from_numpy(target_img).unsqueeze(0).transpose(1, 3).transpose(2, 3).float().to(device)
    style_img = torch.from_numpy(style_img).unsqueeze(0).transpose(1, 3).transpose(2, 3).float().to(device)
    input_img = torch.randn(target_img.shape, device=device)
    input_img.requires_grad_()
    if style_optimizes_target:
        target_style_img = target_img.clone().detach().requires_grad_()
        optimized_tensors = [input_img, target_style_img]
    else:
        target_style_img = target_img
        optimized_tensors = [input_img]

    mask_img = numpy2tensor(mask_img, device)
    source_h, source_w = source_img.shape[2], source_img.shape[3]
    mask_img = mask_img.squeeze(0).repeat(3, 1).view(3, source_h, source_w).unsqueeze(0)

    optimizer = optim.LBFGS(optimized_tensors)
    mse = torch.nn.MSELoss()
    mean_shift = MeanShift(device)
    vgg = Vgg16().to(device).eval()
    with torch.no_grad():
        style_reference_features = vgg(mean_shift(style_img))
        style_reference_gram = [gram_matrix(feature).detach() for feature in style_reference_features]

    history = []
    run = [0]

    def compose_blend():
        target_side = target_style_img if style_optimizes_target else target_img
        return input_img * canvas_mask + target_side * outside_mask

    while run[0] <= num_steps:

        def closure():
            blend_img = compose_blend()

            pred_gradient = laplacian_filter_tensor(blend_img, device)
            grad_loss = 0
            for c in range(len(pred_gradient)):
                grad_loss += mse(pred_gradient[c], gt_gradient[c])
            grad_loss /= len(pred_gradient)
            grad_loss *= grad_weight

            blend_features_style = vgg(mean_shift(blend_img))
            blend_gram_style = [gram_matrix(feature) for feature in blend_features_style]

            style_loss = 0
            for layer in range(len(blend_gram_style)):
                style_loss += mse(blend_gram_style[layer], style_reference_gram[layer])
            style_loss /= len(blend_gram_style)
            style_loss *= style_weight

            source_h, source_w = source_img.shape[2], source_img.shape[3]
            top = int(x - source_h * 0.5)
            left = int(y - source_w * 0.5)
            blend_obj = blend_img[:, :, top:top + source_h, left:left + source_w]
            source_object_features = vgg(mean_shift(source_img * mask_img))
            blend_object_features = vgg(mean_shift(blend_obj * mask_img))
            content_loss = content_weight * mse(blend_object_features.relu2_2, source_object_features.relu2_2)
            content_loss *= content_weight

            tv_loss = torch.sum(torch.abs(blend_img[:, :, :, :-1] - blend_img[:, :, :, 1:])) + torch.sum(
                torch.abs(blend_img[:, :, :-1, :] - blend_img[:, :, 1:, :])
            )
            tv_loss *= tv_weight

            loss = grad_loss + style_loss + content_loss + tv_loss
            optimizer.zero_grad()
            loss.backward()

            if progress_interval > 0 and run[0] % progress_interval == 0:
                history.append(
                    {
                        "step": run[0],
                        "grad": float(grad_loss.detach().cpu()),
                        "style": float(style_loss.detach().cpu()),
                        "content": float(content_loss.detach().cpu()),
                        "tv": float(tv_loss.detach().cpu()),
                        "total": float(loss.detach().cpu()),
                    }
                )
            run[0] += 1
            return loss

        optimizer.step(closure)

    input_img.data.clamp_(0, 255)
    if style_optimizes_target:
        target_style_img.data.clamp_(0, 255)
    blend_img = compose_blend()
    blend_img_np = tensor_to_image(blend_img)

    output_path = os.path.join(output_dir, "first_pass.png")
    if save_output:
        imsave(output_path, blend_img_np)

    return blend_img_np, output_path, history


def parse_args():
    parser = argparse.ArgumentParser(description="Run first-pass Deep Image Blending only.")
    parser.add_argument("--source_file", type=str, default="data/1_source.png", help="path to the source image")
    parser.add_argument("--mask_file", type=str, default="data/1_mask.png", help="path to the mask image")
    parser.add_argument("--target_file", type=str, default="data/1_target.png", help="path to the target image")
    parser.add_argument("--style_file", type=str, default=None, help="optional style-reference image for style loss")
    parser.add_argument("--output_dir", type=str, default="results/my_run", help="path to output")
    parser.add_argument("--ss", type=int, default=512, help="kept for compatibility; source and mask are not resized")
    parser.add_argument("--ts", type=int, default=512, help="target image size")
    parser.add_argument("--x", type=int, default=256, help="vertical location center")
    parser.add_argument("--y", type=int, default=256, help="horizontal location center")
    parser.add_argument("--gpu_id", type=str, default="auto", help="auto, cpu, cuda:0, or GPU index")
    parser.add_argument("--num_steps", type=int, default=1000, help="number of first-pass iterations")
    parser.add_argument("--grad_weight", type=float, default=1e4, help="gradient loss weight")
    parser.add_argument("--style_weight", type=float, default=1e4, help="style loss weight")
    parser.add_argument("--content_weight", type=float, default=1.0, help="content loss weight")
    parser.add_argument("--tv_weight", type=float, default=1e-6, help="total variation loss weight")
    parser.add_argument("--mask_scale", type=float, default=1.0, help="kept for compatibility; source and mask are not scaled")
    parser.add_argument("--seed", type=int, default=None, help="optional random seed")
    return parser.parse_args()


def main():
    args = parse_args()
    image, output_path, history = first_pass_blend(
        source_image=args.source_file,
        mask_image=args.mask_file,
        target_image=args.target_file,
        style_image=args.style_file,
        output_dir=args.output_dir,
        ss=args.ss,
        ts=args.ts,
        x=args.x,
        y=args.y,
        gpu_id=args.gpu_id,
        num_steps=args.num_steps,
        grad_weight=args.grad_weight,
        style_weight=args.style_weight,
        content_weight=args.content_weight,
        tv_weight=args.tv_weight,
        mask_scale=args.mask_scale,
        seed=args.seed,
    )
    print(f"Saved first-pass blend to {Path(output_path).resolve()}")
    if history:
        print("Last logged losses:", history[-1])
    return image


if __name__ == "__main__":
    main()
