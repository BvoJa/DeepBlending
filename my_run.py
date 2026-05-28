import argparse
import os
from pathlib import Path

import numpy as np
import torch
import torch.optim as optim
from PIL import Image
from skimage.io import imsave
from torchvision import models

from utils import (
    compute_gt_gradient,
    laplacian_filter_tensor,
)


STYLE_LAYERS = ["r11", "r21", "r31", "r41", "r51"]
CONTENT_LAYERS = ["r42"]
STYLE_CHANNELS = [64, 128, 256, 512, 512]
NEURAL_STYLE_BGR_MEAN = (0.40760392, 0.45795686, 0.48501961)
TORCHVISION_RGB_MEAN = (0.485, 0.456, 0.406)
TORCHVISION_RGB_STD = (0.229, 0.224, 0.225)
TORCHVISION_RELU_LAYERS = {
    1: "r11",
    3: "r12",
    6: "r21",
    8: "r22",
    11: "r31",
    13: "r32",
    15: "r33",
    17: "r34",
    20: "r41",
    22: "r42",
    24: "r43",
    26: "r44",
    29: "r51",
    31: "r52",
    33: "r53",
    35: "r54",
}


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


def image_to_nchw_tensor(image, device):
    image = np.ascontiguousarray(image)
    return torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0).float().contiguous().to(device)


def image_to_pil_rgb(image):
    if isinstance(image, np.ndarray):
        return Image.fromarray(image.astype(np.uint8)).convert("RGB")
    return Image.open(image).convert("RGB")


def aspect_scale_image(pil_image, size):
    width, height = pil_image.size
    if (width <= height and width == size) or (height <= width and height == size):
        return pil_image
    if width < height:
        new_width = size
        new_height = int(size * height / width)
    else:
        new_height = size
        new_width = int(size * width / height)
    return pil_image.resize((new_width, new_height), Image.BILINEAR)


def torchvision_preprocess_image(image, size, device, keep_aspect=False):
    pil_image = image_to_pil_rgb(image)
    if keep_aspect:
        pil_image = aspect_scale_image(pil_image, size)
    else:
        pil_image = pil_image.resize((size, size), Image.BILINEAR)
    array = np.asarray(pil_image, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(array).permute(2, 0, 1)
    mean = tensor.new_tensor(TORCHVISION_RGB_MEAN).view(3, 1, 1)
    std = tensor.new_tensor(TORCHVISION_RGB_STD).view(3, 1, 1)
    tensor = (tensor - mean) / std
    return tensor.unsqueeze(0).contiguous().to(device)


def torchvision_preprocess_tensor(image_tensor):
    tensor = image_tensor / 255.0
    mean = tensor.new_tensor(TORCHVISION_RGB_MEAN).view(1, 3, 1, 1)
    std = tensor.new_tensor(TORCHVISION_RGB_STD).view(1, 3, 1, 1)
    return (tensor - mean) / std


def neural_style_preprocess_image(image, size, device, keep_aspect=False):
    pil_image = image_to_pil_rgb(image)
    if keep_aspect:
        pil_image = aspect_scale_image(pil_image, size)
    else:
        pil_image = pil_image.resize((size, size), Image.BILINEAR)
    array = np.asarray(pil_image, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(array).permute(2, 0, 1)
    tensor = tensor[torch.LongTensor([2, 1, 0])]
    mean = tensor.new_tensor(NEURAL_STYLE_BGR_MEAN).view(3, 1, 1)
    tensor = (tensor - mean) * 255.0
    return tensor.unsqueeze(0).contiguous().to(device)


def neural_style_preprocess_tensor(image_tensor):
    tensor = image_tensor / 255.0
    tensor = tensor[:, [2, 1, 0], :, :]
    mean = tensor.new_tensor(NEURAL_STYLE_BGR_MEAN).view(1, 3, 1, 1)
    return (tensor - mean) * 255.0


def torchvision_postprocess_image(tensor):
    if tensor.dim() == 4:
        tensor = tensor[0]
    image = tensor.detach().cpu().clone()
    mean = image.new_tensor(TORCHVISION_RGB_MEAN).view(3, 1, 1)
    std = image.new_tensor(TORCHVISION_RGB_STD).view(3, 1, 1)
    image = image * std + mean
    image.clamp_(0, 1)
    image = image.permute(1, 2, 0).numpy() * 255.0
    return image.astype(np.uint8)


class StyleGramMatrix(torch.nn.Module):
    def forward(self, input):
        batch, channels, height, width = input.size()
        features = input.reshape(batch, channels, height * width)
        gram = torch.bmm(features, features.transpose(1, 2))
        return gram / (height * width)


class StyleGramMSELoss(torch.nn.Module):
    def forward(self, input, target):
        return torch.nn.MSELoss()(StyleGramMatrix()(input), target)


class TorchvisionVGG19(torch.nn.Module):
    def __init__(self):
        super().__init__()
        try:
            weights = models.VGG19_Weights.DEFAULT
            features = models.vgg19(weights=weights).features
        except (AttributeError, TypeError):
            features = models.vgg19(pretrained=True).features
        for module in features.modules():
            if isinstance(module, torch.nn.ReLU):
                module.inplace = False
        for parameter in features.parameters():
            parameter.requires_grad = False
        self.features = features

    def forward(self, image, out_keys):
        outputs = {}
        wanted = set(out_keys)
        for index, layer in enumerate(self.features):
            image = layer(image)
            key = TORCHVISION_RELU_LAYERS.get(index)
            if key in wanted:
                outputs[key] = image
                if len(outputs) == len(wanted):
                    break
        return [outputs[key] for key in out_keys]


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


def make_content_reference_image(x_start, y_start, source_img, target_img, mask):
    content_reference = target_img.copy()
    source_h, source_w = source_img.shape[:2]
    top = int(x_start - source_h * 0.5)
    left = int(y_start - source_w * 0.5)
    region = content_reference[top:top + source_h, left:left + source_w]
    object_pixels = mask > 0
    region[object_pixels] = source_img[object_pixels]
    return content_reference


def prepare_full_source_and_mask(source_image, mask_image, size):
    source = load_rgb_image(source_image, size)
    mask = load_mask_image(mask_image, size)
    return source, mask


def tensor_to_image(tensor):
    image = tensor.transpose(1, 3).transpose(1, 2).detach().cpu().numpy()[0]
    return np.clip(image, 0, 255).astype(np.uint8)


def make_grads_contiguous(tensors):
    for tensor in tensors:
        if tensor.grad is not None and not tensor.grad.is_contiguous():
            tensor.grad = tensor.grad.contiguous()


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
    style_image,
    output_dir="results/gradio",
    ss=512,
    ts=512,
    x=256,
    y=256,
    gpu_id="auto",
    num_steps=1000,
    grad_weight=1e4,
    style_weight=10.0,
    content_weight=1.0,
    tv_weight=1e-6,
    mask_scale=1.0,
    seed=None,
    progress_interval=10,
    save_output=True,
):
    if style_image is None:
        raise ValueError("Upload a style-reference image before running first-pass blending.")

    device = resolve_device(gpu_id)
    if seed is not None:
        torch.manual_seed(int(seed))
        if device.type == "cuda":
            torch.cuda.manual_seed_all(int(seed))

    os.makedirs(output_dir, exist_ok=True)

    source_np, mask_np = prepare_source_object(source_image, mask_image, ss, mask_scale)
    target_np = load_rgb_image(target_image, ts)
    validate_source_placement(x, y, source_np.shape, target_np.shape)

    gt_gradient = compute_gt_gradient(x, y, source_np, target_np, mask_np, device)
    content_reference_np = make_content_reference_image(x, y, source_np, target_np, mask_np)

    content_reference_tensor = image_to_nchw_tensor(content_reference_np, device)
    input_img = content_reference_tensor.detach().clone()

    optimizer = optim.LBFGS([input_img.requires_grad_()])
    mse = torch.nn.MSELoss()
    style_vgg = TorchvisionVGG19().to(device).eval()

    style_layers = STYLE_LAYERS
    content_layers = CONTENT_LAYERS
    loss_layers = style_layers + content_layers
    loss_fns = [StyleGramMSELoss()] * len(style_layers) + [torch.nn.MSELoss()] * len(content_layers)
    loss_fns = [loss_fn.to(device) for loss_fn in loss_fns]

    style_weights = [float(style_weight) * (1e3 / channels ** 2) for channels in STYLE_CHANNELS]
    content_weights = [float(content_weight) * 1e0 for _ in content_layers]
    weights = style_weights + content_weights

    style_reference_img = neural_style_preprocess_image(style_image, ts, device, keep_aspect=True)
    content_reference_img = neural_style_preprocess_tensor(content_reference_tensor)
    style_targets = [StyleGramMatrix()(feature).detach() for feature in style_vgg(style_reference_img, style_layers)]
    content_targets = [feature.detach() for feature in style_vgg(content_reference_img, content_layers)]
    targets = style_targets + content_targets

    history = []
    run = [0]

    while run[0] <= num_steps:

        def closure():
            pred_gradient = laplacian_filter_tensor(input_img, device)
            grad_loss = 0
            for c in range(len(pred_gradient)):
                grad_loss += mse(pred_gradient[c], gt_gradient[c])
            grad_loss /= len(pred_gradient)
            grad_loss *= grad_weight

            outputs = style_vgg(neural_style_preprocess_tensor(input_img), loss_layers)
            style_content_layer_losses = [
                weights[layer_index] * loss_fns[layer_index](activation, targets[layer_index])
                for layer_index, activation in enumerate(outputs)
            ]
            style_loss = sum(style_content_layer_losses[:len(style_layers)])
            content_loss = sum(style_content_layer_losses[len(style_layers):])

            tv_loss = torch.sum(torch.abs(input_img[:, :, :, :-1] - input_img[:, :, :, 1:])) + torch.sum(
                torch.abs(input_img[:, :, :-1, :] - input_img[:, :, 1:, :])
            )
            tv_loss *= tv_weight

            loss = grad_loss + style_loss + content_loss + tv_loss
            optimizer.zero_grad()
            loss.backward()
            make_grads_contiguous([input_img])

            if progress_interval > 0 and run[0] % progress_interval == 0:
                history.append(
                    {
                        "step": run[0],
                        "grad": float(grad_loss.detach().cpu()),
                        "style": float(style_loss.detach().cpu()),
                        "style_weight": float(style_weight),
                        "content": float(content_loss.detach().cpu()),
                        "tv": float(tv_loss.detach().cpu()),
                        "total": float(loss.detach().cpu()),
                    }
                )
            run[0] += 1
            return loss

        optimizer.step(closure)

    input_img.data.clamp_(0, 255)
    input_img_np = tensor_to_image(input_img)

    output_path = os.path.join(output_dir, "first_pass.png")
    if save_output:
        imsave(output_path, input_img_np)

    return input_img_np, output_path, history


def second_pass_blend(
    first_pass_image,
    style_image,
    output_dir="results/gradio",
    ts=512,
    gpu_id="auto",
    num_steps=500,
    style_weight=1e6,
    content_weight=1.0,
    tv_weight=0.0,
    seed=None,
    progress_interval=10,
    save_output=True,
):
    if style_image is None:
        raise ValueError("Upload a style-reference image before running the second pass.")

    device = resolve_device(gpu_id)
    if seed is not None:
        torch.manual_seed(int(seed))
        if device.type == "cuda":
            torch.cuda.manual_seed_all(int(seed))

    os.makedirs(output_dir, exist_ok=True)

    vgg = TorchvisionVGG19().to(device).eval()
    content_img = torchvision_preprocess_image(first_pass_image, ts, device)
    style_img = torchvision_preprocess_image(style_image, ts, device, keep_aspect=True)
    opt_img = content_img.detach().clone().requires_grad_()

    style_layers = STYLE_LAYERS
    content_layers = CONTENT_LAYERS
    loss_layers = style_layers + content_layers
    loss_fns = [StyleGramMSELoss()] * len(style_layers) + [torch.nn.MSELoss()] * len(content_layers)
    loss_fns = [loss_fn.to(device) for loss_fn in loss_fns]

    style_weights = [float(style_weight) * (1e3 / channels ** 2) for channels in STYLE_CHANNELS]
    content_weights = [float(content_weight) * 1e0]
    weights = style_weights + content_weights

    style_targets = [StyleGramMatrix()(feature).detach() for feature in vgg(style_img, style_layers)]
    content_targets = [feature.detach() for feature in vgg(content_img, content_layers)]
    targets = style_targets + content_targets

    history = []
    max_iter = int(num_steps)
    show_iter = max(1, int(progress_interval))
    optimizer = optim.LBFGS([opt_img])
    n_iter = [0]

    while n_iter[0] <= max_iter:

        def closure():
            optimizer.zero_grad()
            outputs = vgg(opt_img, loss_layers)
            layer_losses = [
                weights[layer_index] * loss_fns[layer_index](activation, targets[layer_index])
                for layer_index, activation in enumerate(outputs)
            ]
            loss = sum(layer_losses)
            loss.backward()
            make_grads_contiguous([opt_img])
            n_iter[0] += 1

            if show_iter > 0 and n_iter[0] % show_iter == show_iter - 1:
                style_loss = sum(layer_losses[:len(style_layers)])
                content_loss_value = layer_losses[-1]
                history.append(
                    {
                        "step": n_iter[0] + 1,
                        "style": float(style_loss.detach().cpu()),
                        "content": float(content_loss_value.detach().cpu()),
                        "total": float(loss.detach().cpu()),
                    }
                )
            return loss

        optimizer.step(closure)

    second_pass_np = torchvision_postprocess_image(opt_img.data[0].cpu().squeeze())

    output_path = os.path.join(output_dir, "second_pass.png")
    if save_output:
        imsave(output_path, second_pass_np)

    return second_pass_np, output_path, history


def parse_args():
    parser = argparse.ArgumentParser(description="Run first-pass Deep Image Blending with reference-image style loss.")
    parser.add_argument("--source_file", type=str, default="data/1_source.png", help="path to the source image")
    parser.add_argument("--mask_file", type=str, default="data/1_mask.png", help="path to the mask image")
    parser.add_argument("--target_file", type=str, default="data/1_target.png", help="path to the target image")
    parser.add_argument("--style_file", type=str, required=True, help="style-reference image for first-pass style loss")
    parser.add_argument("--output_dir", type=str, default="results/my_run", help="path to output")
    parser.add_argument("--ss", type=int, default=512, help="kept for compatibility; source and mask are not resized")
    parser.add_argument("--ts", type=int, default=512, help="target image size")
    parser.add_argument("--x", type=int, default=256, help="vertical location center")
    parser.add_argument("--y", type=int, default=256, help="horizontal location center")
    parser.add_argument("--gpu_id", type=str, default="auto", help="auto, cpu, cuda:0, or GPU index")
    parser.add_argument("--num_steps", type=int, default=1000, help="number of first-pass iterations")
    parser.add_argument("--grad_weight", type=float, default=1e4, help="gradient loss weight")
    parser.add_argument("--style_weight", type=float, default=10.0, help="first-pass style-reference multiplier on the NeuralStyleTransfer.ipynb layer weights")
    parser.add_argument("--content_weight", type=float, default=1.0, help="content loss weight")
    parser.add_argument("--tv_weight", type=float, default=1e-6, help="total variation loss weight")
    parser.add_argument("--second_steps", type=int, default=500, help="second-pass iterations, matching NeuralStyleTransfer.ipynb max_iter by default")
    parser.add_argument("--second_style_weight", type=float, default=1e6, help="second-pass torchvision VGG style multiplier")
    parser.add_argument("--second_content_weight", type=float, default=1.0, help="second-pass content loss weight")
    parser.add_argument("--second_tv_weight", type=float, default=0.0, help="kept for compatibility; the second pass does not use TV loss")
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
    print(f"Saved first-pass optimized input image to {Path(output_path).resolve()}")
    if history:
        print("Last logged losses:", history[-1])
    return image


if __name__ == "__main__":
    main()
