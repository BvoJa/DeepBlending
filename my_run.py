import argparse
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
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


NOTEBOOK_BGR_MEAN = (0.40760392, 0.45795686, 0.48501961)
NOTEBOOK_STYLE_LAYERS = ["r11", "r21", "r31", "r41", "r51"]
NOTEBOOK_CONTENT_LAYERS = ["r42"]
NOTEBOOK_LOSS_LAYERS = NOTEBOOK_STYLE_LAYERS + NOTEBOOK_CONTENT_LAYERS
NOTEBOOK_STYLE_CHANNELS = [64, 128, 256, 512, 512]


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


def notebook_scale_image(pil_image, size):
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


def notebook_preprocess_image(image, size, device):
    pil_image = notebook_scale_image(image_to_pil_rgb(image), size)
    array = np.asarray(pil_image, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(array).permute(2, 0, 1)
    tensor = tensor[[2, 1, 0], :, :]
    mean = tensor.new_tensor(NOTEBOOK_BGR_MEAN).view(3, 1, 1)
    tensor = (tensor - mean).mul_(255.0)
    return tensor.unsqueeze(0).contiguous().to(device)


def notebook_postprocess_image(tensor):
    if tensor.dim() == 4:
        tensor = tensor[0]
    image = tensor.detach().cpu().clone()
    mean = image.new_tensor(NOTEBOOK_BGR_MEAN).view(3, 1, 1)
    image = image.mul(1.0 / 255.0)
    image = image + mean
    image = image[[2, 1, 0], :, :]
    image.clamp_(0, 1)
    image = image.permute(1, 2, 0).numpy() * 255.0
    return image.astype(np.uint8)


class NotebookGramMatrix(torch.nn.Module):
    def forward(self, input):
        batch, channels, height, width = input.size()
        features = input.view(batch, channels, height * width)
        gram = torch.bmm(features, features.transpose(1, 2))
        gram.div_(height * width)
        return gram


class NotebookGramMSELoss(torch.nn.Module):
    def forward(self, input, target):
        return torch.nn.MSELoss()(NotebookGramMatrix()(input), target)


def resolve_notebook_vgg_checkpoint(checkpoint_path=None):
    if checkpoint_path:
        checkpoint = Path(checkpoint_path).expanduser()
        if checkpoint.exists():
            return checkpoint
        raise FileNotFoundError(f"NeuralStyleTransfer VGG checkpoint was not found at {checkpoint}.")

    root = Path(__file__).resolve().parent
    candidates = [
        root / "Models" / "vgg_conv.pth",
        root / "models" / "vgg_conv.pth",
        root / "vgg_conv.pth",
        Path("Models") / "vgg_conv.pth",
        Path("models") / "vgg_conv.pth",
        Path("vgg_conv.pth"),
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


class NotebookVGG(torch.nn.Module):
    def __init__(self, pool="max"):
        super(NotebookVGG, self).__init__()
        self.conv1_1 = torch.nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.conv1_2 = torch.nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.conv2_1 = torch.nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv2_2 = torch.nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.conv3_1 = torch.nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.conv3_2 = torch.nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.conv3_3 = torch.nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.conv3_4 = torch.nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.conv4_1 = torch.nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.conv4_2 = torch.nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv4_3 = torch.nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv4_4 = torch.nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv5_1 = torch.nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv5_2 = torch.nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv5_3 = torch.nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv5_4 = torch.nn.Conv2d(512, 512, kernel_size=3, padding=1)
        pool_cls = torch.nn.MaxPool2d if pool == "max" else torch.nn.AvgPool2d
        self.pool1 = pool_cls(kernel_size=2, stride=2)
        self.pool2 = pool_cls(kernel_size=2, stride=2)
        self.pool3 = pool_cls(kernel_size=2, stride=2)
        self.pool4 = pool_cls(kernel_size=2, stride=2)
        self.pool5 = pool_cls(kernel_size=2, stride=2)

    def forward(self, image, out_keys):
        out = {}
        out["r11"] = F.relu(self.conv1_1(image))
        out["r12"] = F.relu(self.conv1_2(out["r11"]))
        out["p1"] = self.pool1(out["r12"])
        out["r21"] = F.relu(self.conv2_1(out["p1"]))
        out["r22"] = F.relu(self.conv2_2(out["r21"]))
        out["p2"] = self.pool2(out["r22"])
        out["r31"] = F.relu(self.conv3_1(out["p2"]))
        out["r32"] = F.relu(self.conv3_2(out["r31"]))
        out["r33"] = F.relu(self.conv3_3(out["r32"]))
        out["r34"] = F.relu(self.conv3_4(out["r33"]))
        out["p3"] = self.pool3(out["r34"])
        out["r41"] = F.relu(self.conv4_1(out["p3"]))
        out["r42"] = F.relu(self.conv4_2(out["r41"]))
        out["r43"] = F.relu(self.conv4_3(out["r42"]))
        out["r44"] = F.relu(self.conv4_4(out["r43"]))
        out["p4"] = self.pool4(out["r44"])
        out["r51"] = F.relu(self.conv5_1(out["p4"]))
        out["r52"] = F.relu(self.conv5_2(out["r51"]))
        out["r53"] = F.relu(self.conv5_3(out["r52"]))
        out["r54"] = F.relu(self.conv5_4(out["r53"]))
        out["p5"] = self.pool5(out["r54"])
        return [out[key] for key in out_keys]


def load_notebook_vgg(device, checkpoint_path=None):
    checkpoint = resolve_notebook_vgg_checkpoint(checkpoint_path)
    if checkpoint is None:
        raise FileNotFoundError(
            "Exact NeuralStyleTransfer.ipynb mode requires the notebook VGG checkpoint "
            "`vgg_conv.pth`. Put it at `Models/vgg_conv.pth`, `models/vgg_conv.pth`, "
            "or the project root."
        )
    vgg = NotebookVGG()
    state = torch.load(checkpoint, map_location="cpu")
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    vgg.load_state_dict(state)
    for parameter in vgg.parameters():
        parameter.requires_grad = False
    return vgg.to(device).eval()


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

    source_img, mask_img = prepare_source_object(source_image, mask_image, ss, mask_scale)
    target_img = load_rgb_image(target_image, ts)
    validate_source_placement(x, y, source_img.shape, target_img.shape)

    canvas_mask = paste_source_mask_to_target(x, y, target_img, mask_img)
    canvas_mask = numpy2tensor(canvas_mask, device)
    canvas_mask = canvas_mask.squeeze(0).repeat(3, 1).view(3, ts, ts).unsqueeze(0)

    gt_gradient = compute_gt_gradient(x, y, source_img, target_img, mask_img, device)

    source_img = image_to_nchw_tensor(source_img, device)
    target_img = image_to_nchw_tensor(target_img, device)
    input_img = torch.randn(target_img.shape, device=device)

    mask_img = numpy2tensor(mask_img, device)
    source_h, source_w = source_img.shape[2], source_img.shape[3]
    mask_img = mask_img.squeeze(0).repeat(3, 1).view(3, source_h, source_w).unsqueeze(0)

    optimizer = optim.LBFGS([input_img.requires_grad_()])
    mse = torch.nn.MSELoss()
    mean_shift = MeanShift(device)
    vgg = Vgg16().to(device).eval()

    history = []
    run = [0]

    while run[0] <= num_steps:

        def closure():
            blend_img = torch.zeros(target_img.shape, device=device)
            blend_img = input_img * canvas_mask + target_img * (canvas_mask - 1) * (-1)

            pred_gradient = laplacian_filter_tensor(blend_img, device)
            grad_loss = 0
            for c in range(len(pred_gradient)):
                grad_loss += mse(pred_gradient[c], gt_gradient[c])
            grad_loss /= len(pred_gradient)
            grad_loss *= grad_weight

            target_features_style = vgg(mean_shift(target_img))
            target_gram_style = [gram_matrix(feature) for feature in target_features_style]

            blend_features_style = vgg(mean_shift(input_img))
            blend_gram_style = [gram_matrix(feature) for feature in blend_features_style]

            style_loss = 0
            for layer in range(len(blend_gram_style)):
                style_loss += mse(blend_gram_style[layer], target_gram_style[layer])
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
            make_grads_contiguous([input_img])

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
    blend_img = torch.zeros(target_img.shape, device=device)
    blend_img = input_img * canvas_mask + target_img * (canvas_mask - 1) * (-1)
    blend_img_np = tensor_to_image(blend_img)

    output_path = os.path.join(output_dir, "first_pass.png")
    if save_output:
        imsave(output_path, blend_img_np)

    return blend_img_np, output_path, history


def second_pass_blend(
    first_pass_image,
    style_image,
    output_dir="results/gradio",
    ts=512,
    gpu_id="auto",
    num_steps=500,
    style_weight=1.0,
    content_weight=1.0,
    tv_weight=0.0,
    vgg_checkpoint=None,
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

    vgg = load_notebook_vgg(device, vgg_checkpoint)
    content_img = notebook_preprocess_image(first_pass_image, ts, device)
    style_img = notebook_preprocess_image(style_image, ts, device)
    opt_img = content_img.detach().clone().requires_grad_()

    style_layers = NOTEBOOK_STYLE_LAYERS
    content_layers = NOTEBOOK_CONTENT_LAYERS
    loss_layers = style_layers + content_layers
    loss_fns = [NotebookGramMSELoss()] * len(style_layers) + [torch.nn.MSELoss()] * len(content_layers)
    loss_fns = [loss_fn.to(device) for loss_fn in loss_fns]

    style_weights = [float(style_weight) * (1e3 / channels ** 2) for channels in NOTEBOOK_STYLE_CHANNELS]
    content_weights = [float(content_weight) * 1e0]
    weights = style_weights + content_weights

    style_targets = [NotebookGramMatrix()(feature).detach() for feature in vgg(style_img, style_layers)]
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

    second_pass_np = notebook_postprocess_image(opt_img.data[0].cpu().squeeze())

    output_path = os.path.join(output_dir, "second_pass.png")
    if save_output:
        imsave(output_path, second_pass_np)

    return second_pass_np, output_path, history


def parse_args():
    parser = argparse.ArgumentParser(description="Run first-pass Deep Image Blending, optionally followed by second-pass style optimization.")
    parser.add_argument("--source_file", type=str, default="data/1_source.png", help="path to the source image")
    parser.add_argument("--mask_file", type=str, default="data/1_mask.png", help="path to the mask image")
    parser.add_argument("--target_file", type=str, default="data/1_target.png", help="path to the target image")
    parser.add_argument("--style_file", type=str, default=None, help="optional style-reference image for the second pass")
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
    parser.add_argument("--second_steps", type=int, default=500, help="second-pass iterations, matching NeuralStyleTransfer.ipynb max_iter by default")
    parser.add_argument("--second_style_weight", type=float, default=1.0, help="second-pass notebook-style loss multiplier")
    parser.add_argument("--second_content_weight", type=float, default=1.0, help="second-pass content loss weight")
    parser.add_argument("--second_tv_weight", type=float, default=0.0, help="kept for compatibility; the exact notebook second pass does not use TV loss")
    parser.add_argument("--vgg_checkpoint", type=str, default=None, help="path to NeuralStyleTransfer.ipynb Models/vgg_conv.pth")
    parser.add_argument("--mask_scale", type=float, default=1.0, help="kept for compatibility; source and mask are not scaled")
    parser.add_argument("--seed", type=int, default=None, help="optional random seed")
    return parser.parse_args()


def main():
    args = parse_args()
    image, output_path, history = first_pass_blend(
        source_image=args.source_file,
        mask_image=args.mask_file,
        target_image=args.target_file,
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
    if args.style_file:
        second_image, second_output_path, second_history = second_pass_blend(
            first_pass_image=image,
            style_image=args.style_file,
            output_dir=args.output_dir,
            ts=args.ts,
            gpu_id=args.gpu_id,
            num_steps=args.second_steps,
            style_weight=args.second_style_weight,
            content_weight=args.second_content_weight,
            vgg_checkpoint=args.vgg_checkpoint,
            seed=args.seed,
        )
        print(f"Saved second-pass blend to {Path(second_output_path).resolve()}")
        if second_history:
            print("Last logged second-pass losses:", second_history[-1])
        return second_image
    return image


if __name__ == "__main__":
    main()
