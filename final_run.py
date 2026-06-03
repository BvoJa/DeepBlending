import argparse
import http.cookiejar
import inspect
import os
import re
import shutil
import sys
import urllib.parse
import urllib.request
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from PIL import Image
from skimage.io import imsave
from torchvision import models

from utils import compute_gt_gradient, laplacian_filter_tensor


STYLE_LAYERS = ["r11", "r21", "r31", "r41", "r51"]
CONTENT_LAYERS = ["r42"]
STYLE_CHANNELS = [64, 128, 256, 512, 512]
NEURAL_STYLE_BGR_MEAN = (0.40760392, 0.45795686, 0.48501961)
DEFAULT_OUTPUT_DIR = "results/final_run"
DEFAULT_VGG_MODEL = "Models/vgg_conv.pth"
VGG_MODEL_FILE_ID = "1lLSi8BXd_9EtudRbIwxvmTQ3Ms-Qh6C8"
VGG_MODEL_URL = f"https://drive.google.com/uc?id={VGG_MODEL_FILE_ID}"
DEFAULT_SOURCE_SIZE = 512
DEFAULT_TARGET_SIZE = 512
DEFAULT_NUM_STEPS = 100


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


class VGG(nn.Module):
    def __init__(self, pool="max"):
        super(VGG, self).__init__()
        self.conv1_1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.conv1_2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.conv2_1 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv2_2 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.conv3_1 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.conv3_2 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.conv3_3 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.conv3_4 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.conv4_1 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.conv4_2 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv4_3 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv4_4 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv5_1 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv5_2 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv5_3 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv5_4 = nn.Conv2d(512, 512, kernel_size=3, padding=1)

        if pool == "max":
            self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
            self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
            self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
            self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)
            self.pool5 = nn.MaxPool2d(kernel_size=2, stride=2)
        elif pool == "avg":
            self.pool1 = nn.AvgPool2d(kernel_size=2, stride=2)
            self.pool2 = nn.AvgPool2d(kernel_size=2, stride=2)
            self.pool3 = nn.AvgPool2d(kernel_size=2, stride=2)
            self.pool4 = nn.AvgPool2d(kernel_size=2, stride=2)
            self.pool5 = nn.AvgPool2d(kernel_size=2, stride=2)
        else:
            raise ValueError("pool must be 'max' or 'avg'")

    def forward(self, x, out_keys):
        out = {}
        out["r11"] = F.relu(self.conv1_1(x))
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


class GramMatrix(nn.Module):
    def forward(self, input):
        batch, channels, height, width = input.size()
        features = input.view(batch, channels, height * width)
        gram = torch.bmm(features, features.transpose(1, 2))
        gram.div_(height * width)
        return gram


class GramMSELoss(nn.Module):
    def forward(self, input, target):
        out = nn.MSELoss()(GramMatrix()(input), target)
        return out


def resolve_device(gpu_id="auto"):
    if isinstance(gpu_id, str):
        value = gpu_id.strip().lower()
        if value == "auto":
            return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        if value == "cpu":
            return torch.device("cpu")
        if value.startswith("cuda"):
            return torch.device(value if torch.cuda.is_available() else "cpu")
        if value.isdigit():
            return torch.device(f"cuda:{value}" if torch.cuda.is_available() else "cpu")
    if isinstance(gpu_id, int):
        return torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu")
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def image_to_pil_rgb(image):
    if isinstance(image, Image.Image):
        return image.convert("RGB")
    if isinstance(image, np.ndarray):
        return Image.fromarray(image.astype(np.uint8)).convert("RGB")
    return Image.open(image).convert("RGB")


def resize_short_edge(pil_image, size):
    width, height = pil_image.size
    if width == size and height == size:
        return pil_image
    if width < height:
        new_width = size
        new_height = int(round(size * height / width))
    else:
        new_height = size
        new_width = int(round(size * width / height))
    return pil_image.resize((new_width, new_height), Image.BILINEAR)


def load_rgb_image(image, size):
    pil_image = image_to_pil_rgb(image)
    return np.array(pil_image.resize((size, size), Image.BILINEAR))


def load_mask_array(image, size=None):
    if isinstance(image, Image.Image):
        pil_image = image.convert("L")
    elif isinstance(image, np.ndarray):
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


def notebook_preprocess_image(image, size, device, keep_aspect=True):
    pil_image = image_to_pil_rgb(image)
    pil_image = resize_short_edge(pil_image, size) if keep_aspect else pil_image.resize((size, size), Image.BILINEAR)
    array = np.asarray(pil_image, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(array).permute(2, 0, 1)
    tensor = tensor[torch.LongTensor([2, 1, 0])]
    mean = tensor.new_tensor(NEURAL_STYLE_BGR_MEAN).view(3, 1, 1)
    tensor = (tensor - mean) * 255.0
    return tensor.unsqueeze(0).contiguous().to(device)


def notebook_preprocess_array(array, size, device):
    return notebook_preprocess_image(Image.fromarray(array.astype(np.uint8)).convert("RGB"), size, device, keep_aspect=True)


def notebook_deprocess_tensor_to_rgb255(tensor):
    if tensor.dim() == 3:
        tensor = tensor.unsqueeze(0)
    mean = tensor.new_tensor(NEURAL_STYLE_BGR_MEAN).view(1, 3, 1, 1)
    bgr = tensor / 255.0 + mean
    return bgr[:, [2, 1, 0], :, :] * 255.0


def notebook_postprocess_image(tensor):
    if tensor.dim() == 4:
        tensor = tensor[0]
    image = tensor.detach().cpu().clone()
    mean = image.new_tensor(NEURAL_STYLE_BGR_MEAN).view(3, 1, 1)
    image = image / 255.0 + mean
    image = image[[2, 1, 0], :, :]
    image.clamp_(0, 1)
    image = image.permute(1, 2, 0).numpy() * 255.0
    return image.astype(np.uint8)


def make_grads_contiguous(tensors):
    for tensor in tensors:
        if tensor.grad is not None and not tensor.grad.is_contiguous():
            tensor.grad = tensor.grad.contiguous()


def validate_source_placement(x, y, source_shape, target_shape):
    source_h, source_w = source_shape[:2]
    target_h, target_w = target_shape[:2]
    half_h = source_h * 0.5
    half_w = source_w * 0.5
    if x - half_h < 0 or y - half_w < 0 or x + half_h > target_h or y + half_w > target_w:
        raise ValueError(
            "The source image must fit inside the target canvas. "
            f"Use x between {int(half_h)} and {int(target_h - half_h)}, "
            f"and y between {int(half_w)} and {int(target_w - half_w)}."
        )


def fit_source_placement(x, y, source_shape, target_size):
    source_h, source_w = source_shape[:2]
    if source_h > target_size or source_w > target_size:
        raise gr.Error(
            f"Source image size {source_h}x{source_w} is larger than target size {target_size}x{target_size}."
        )
    min_x = int(np.ceil(source_h * 0.5))
    max_x = int(np.floor(target_size - source_h * 0.5))
    min_y = int(np.ceil(source_w * 0.5))
    max_y = int(np.floor(target_size - source_w * 0.5))
    fitted_x = int(np.clip(int(x), min_x, max_x))
    fitted_y = int(np.clip(int(y), min_y, max_y))
    return fitted_x, fitted_y


def mask_bbox(mask):
    ys, xs = np.where(mask > 0)
    if len(xs) == 0 or len(ys) == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def prepare_source_object(source_image, mask_image=None, size=None, mask_scale=1.0):
    if mask_image is None:
        source = load_rgb_image(source_image, int(size or DEFAULT_SOURCE_SIZE))
        mask = np.ones(source.shape[:2], dtype=np.uint8)
        return source, mask

    source = np.array(image_to_pil_rgb(source_image))
    mask = load_mask_array(mask_image)
    if mask.shape[:2] != source.shape[:2]:
        raise ValueError(
            "The mask must have the same height and width as the source image. "
            "Use the drawn/SAM mask generated from this source image, or upload a matching mask."
        )

    box = mask_bbox(mask)
    if box is None:
        raise ValueError("The mask is empty. Draw, extract, or upload a non-empty object mask.")

    left, top, right, bottom = box
    return source[top:bottom, left:right], mask[top:bottom, left:right]


def make_content_reference_image(x_start, y_start, source_img, target_img, mask):
    content_reference = target_img.copy()
    source_h, source_w = source_img.shape[:2]
    top = int(x_start - source_h * 0.5)
    left = int(y_start - source_w * 0.5)
    region = content_reference[top:top + source_h, left:left + source_w]
    object_pixels = mask > 0
    region[object_pixels] = source_img[object_pixels]
    return content_reference


def candidate_vgg_model_paths(extra_path=None):
    candidates = [
        extra_path,
        os.environ.get("NST_VGG_MODEL"),
        Path(__file__).resolve().parent / DEFAULT_VGG_MODEL,
        Path("/kaggle/working/PytorchNeuralStyleTransfer/Models/vgg_conv.pth"),
        Path("/kaggle/working/DeepImageBlending/Models/vgg_conv.pth"),
    ]
    kaggle_input = Path("/kaggle/input")
    if kaggle_input.exists():
        candidates.extend(kaggle_input.glob("*/vgg_conv.pth"))
        candidates.extend(kaggle_input.glob("*/*/vgg_conv.pth"))
    return candidates


def resolve_vgg_model_path(model_path=""):
    for candidate in candidate_vgg_model_paths((model_path or "").strip() or None):
        if not candidate:
            continue
        path = Path(candidate).expanduser()
        if not path.is_absolute():
            path = Path(__file__).resolve().parent / path
        if path.exists():
            return path.resolve()
    return None


def default_vgg_download_path(model_path=""):
    raw_path = (model_path or "").strip()
    if raw_path:
        path = Path(raw_path).expanduser()
        if not path.is_absolute():
            path = Path(__file__).resolve().parent / path
        if path.suffix:
            return path
        return path / "vgg_conv.pth"
    return Path(__file__).resolve().parent / DEFAULT_VGG_MODEL


def response_is_download(response):
    content_disposition = response.headers.get("Content-Disposition", "")
    content_type = response.headers.get("Content-Type", "")
    return "attachment" in content_disposition.lower() or "text/html" not in content_type.lower()


def parse_google_drive_confirm(html):
    patterns = [
        r"confirm=([0-9A-Za-z_]+)",
        r'name="confirm"\s+value="([^"]+)"',
        r"confirm=([^;&]+)&amp;id=",
    ]
    for pattern in patterns:
        match = re.search(pattern, html)
        if match:
            return urllib.parse.unquote(match.group(1))
    return None


def parse_google_drive_uuid(html):
    patterns = [
        r"uuid=([0-9A-Fa-f-]+)",
        r'name="uuid"\s+value="([^"]+)"',
    ]
    for pattern in patterns:
        match = re.search(pattern, html)
        if match:
            return urllib.parse.unquote(match.group(1))
    return None


def stream_response_to_file(response, destination):
    with open(destination, "wb") as output_file:
        shutil.copyfileobj(response, output_file)


def download_vgg_model_with_gdown(destination):
    try:
        import gdown
    except ImportError as exc:
        raise RuntimeError("gdown is not installed") from exc

    result = gdown.download(VGG_MODEL_URL, str(destination), quiet=False)
    if not result or not Path(destination).exists() or Path(destination).stat().st_size == 0:
        raise RuntimeError("gdown did not create a valid VGG model file")


def download_vgg_model_with_urllib(destination):
    cookie_jar = http.cookiejar.CookieJar()
    opener = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(cookie_jar))
    opener.addheaders = [("User-Agent", "Mozilla/5.0")]

    first_url = f"https://drive.google.com/uc?export=download&id={VGG_MODEL_FILE_ID}"
    response = opener.open(first_url, timeout=60)
    with response:
        if response_is_download(response):
            stream_response_to_file(response, destination)
            return

        html = response.read().decode("utf-8", errors="ignore")
        confirm = parse_google_drive_confirm(html)
        if not confirm:
            raise RuntimeError("Google Drive did not return a downloadable confirmation token")
        uuid = parse_google_drive_uuid(html)

    confirmed_url = (
        "https://drive.google.com/uc?"
        f"export=download&confirm={urllib.parse.quote(confirm)}&id={VGG_MODEL_FILE_ID}"
    )
    if uuid:
        confirmed_url += f"&uuid={urllib.parse.quote(uuid)}"
    response = opener.open(confirmed_url, timeout=60)
    with response:
        stream_response_to_file(response, destination)


def download_vgg_model(destination):
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = destination.with_name(destination.name + ".part")

    download_errors = []
    for download_fn in (download_vgg_model_with_gdown, download_vgg_model_with_urllib):
        if temporary_path.exists():
            temporary_path.unlink()
        try:
            print(f"Downloading NST VGG model from {VGG_MODEL_URL} to {destination}...")
            download_fn(temporary_path)
            if temporary_path.exists() and temporary_path.stat().st_size > 0:
                temporary_path.replace(destination)
                return destination.resolve()
            download_errors.append(f"{download_fn.__name__}: downloaded file was empty")
        except Exception as exc:
            download_errors.append(f"{download_fn.__name__}: {exc}")

    if temporary_path.exists():
        temporary_path.unlink()
    raise RuntimeError("Automatic VGG model download failed. " + " | ".join(download_errors))


def torch_load_weights(path):
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(path, map_location="cpu")


def normalize_state_dict(state):
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    if isinstance(state, dict):
        return {key.replace("module.", "", 1): value for key, value in state.items()}
    return state


def load_torchvision_weights_into_notebook_vgg(vgg):
    try:
        weights = models.VGG19_Weights.DEFAULT
        features = models.vgg19(weights=weights).features
    except (AttributeError, TypeError):
        features = models.vgg19(pretrained=True).features

    mapping = {
        "conv1_1": 0,
        "conv1_2": 2,
        "conv2_1": 5,
        "conv2_2": 7,
        "conv3_1": 10,
        "conv3_2": 12,
        "conv3_3": 14,
        "conv3_4": 16,
        "conv4_1": 19,
        "conv4_2": 21,
        "conv4_3": 23,
        "conv4_4": 25,
        "conv5_1": 28,
        "conv5_2": 30,
        "conv5_3": 32,
        "conv5_4": 34,
    }
    with torch.no_grad():
        for name, index in mapping.items():
            source_layer = features[index]
            target_layer = getattr(vgg, name)
            target_layer.weight.copy_(source_layer.weight)
            target_layer.bias.copy_(source_layer.bias)


def load_notebook_vgg(device, model_path="", allow_torchvision_fallback=False, download_if_missing=True):
    vgg = VGG()
    resolved_path = resolve_vgg_model_path(model_path)
    download_error = None
    if resolved_path is None and download_if_missing:
        try:
            resolved_path = download_vgg_model(default_vgg_download_path(model_path))
        except Exception as exc:
            download_error = exc

    if resolved_path is not None:
        vgg.load_state_dict(normalize_state_dict(torch_load_weights(resolved_path)))
        weight_source = f"NST VGG weights: {resolved_path}"
    elif allow_torchvision_fallback:
        load_torchvision_weights_into_notebook_vgg(vgg)
        weight_source = "torchvision VGG19 fallback weights"
    else:
        download_message = f" Automatic download failed: {download_error}" if download_error else ""
        raise FileNotFoundError(
            "Could not find Models/vgg_conv.pth from NST.ipynb. "
            "Place it at Models/vgg_conv.pth, set NST_VGG_MODEL, paste its path in the VGG model path box, "
            f"or allow automatic download from {VGG_MODEL_URL}."
            f"{download_message}"
        )

    for parameter in vgg.parameters():
        parameter.requires_grad = False
    vgg.to(device).eval()
    return vgg, weight_source


def gradient_style_transfer(
    source_image,
    target_image,
    style_image,
    mask_image=None,
    output_dir=DEFAULT_OUTPUT_DIR,
    source_size=DEFAULT_SOURCE_SIZE,
    target_size=DEFAULT_TARGET_SIZE,
    x=None,
    y=None,
    gpu_id="auto",
    num_steps=DEFAULT_NUM_STEPS,
    grad_weight=1e4,
    style_weight=1.0,
    content_weight=1.0,
    seed=None,
    vgg_model_path="",
    allow_torchvision_fallback=False,
    download_vgg_model_if_missing=True,
    progress_interval=50,
    save_output=True,
):
    if source_image is None:
        raise ValueError("Upload or provide a source image.")
    if target_image is None:
        raise ValueError("Upload or provide a target image.")
    if style_image is None:
        raise ValueError("Upload or provide a style image.")

    device = resolve_device(gpu_id)
    if seed is not None:
        torch.manual_seed(int(seed))
        if device.type == "cuda":
            torch.cuda.manual_seed_all(int(seed))

    source_size = int(source_size)
    target_size = int(target_size)
    x = int(target_size // 2 if x is None else x)
    y = int(target_size // 2 if y is None else y)
    os.makedirs(output_dir, exist_ok=True)

    source_np, mask_np = prepare_source_object(source_image, mask_image, source_size)
    target_np = load_rgb_image(target_image, target_size)
    validate_source_placement(x, y, source_np.shape, target_np.shape)

    gt_gradient = compute_gt_gradient(x, y, source_np, target_np, mask_np, device)
    content_reference_np = make_content_reference_image(x, y, source_np, target_np, mask_np)

    style_tensor = notebook_preprocess_image(style_image, target_size, device, keep_aspect=True)
    content_tensor = notebook_preprocess_array(content_reference_np, target_size, device)
    opt_img = torch.randn(content_tensor.size(), device=device).type_as(content_tensor.data) * 1e-3
    opt_img.requires_grad_()

    vgg, weight_source = load_notebook_vgg(
        device,
        vgg_model_path,
        allow_torchvision_fallback,
        download_if_missing=download_vgg_model_if_missing,
    )

    style_layers = STYLE_LAYERS
    content_layers = CONTENT_LAYERS
    loss_layers = style_layers + content_layers
    loss_fns = [GramMSELoss()] * len(style_layers) + [nn.MSELoss()] * len(content_layers)
    loss_fns = [loss_fn.to(device) for loss_fn in loss_fns]

    style_weights = [float(style_weight) * (1e3 / channels ** 2) for channels in STYLE_CHANNELS]
    content_weights = [float(content_weight) * 1e0]
    weights = style_weights + content_weights

    with torch.no_grad():
        style_targets = [GramMatrix()(activation).detach() for activation in vgg(style_tensor, style_layers)]
        content_targets = [activation.detach() for activation in vgg(content_tensor, content_layers)]
    targets = style_targets + content_targets

    weights = [float(weight) for weight in weights]
    targets = [target.to(device) for target in targets]

    max_iter = int(num_steps)
    show_iter = max(1, int(progress_interval))
    mse = nn.MSELoss()
    optimizer = optim.LBFGS([opt_img])
    n_iter = [0]
    history = []

    while n_iter[0] <= max_iter:

        def closure():
            optimizer.zero_grad()

            out = vgg(opt_img, loss_layers)
            layer_losses = [
                weights[index] * loss_fns[index](activation, targets[index])
                for index, activation in enumerate(out)
            ]
            style_content_loss = torch.stack(layer_losses).sum()

            rgb_opt_img = notebook_deprocess_tensor_to_rgb255(opt_img)
            pred_gradient = laplacian_filter_tensor(rgb_opt_img, device)
            grad_loss = 0
            for channel_index in range(len(pred_gradient)):
                grad_loss += mse(pred_gradient[channel_index], gt_gradient[channel_index])
            grad_loss /= len(pred_gradient)
            grad_loss *= float(grad_weight)

            loss = style_content_loss + grad_loss
            loss.backward()
            make_grads_contiguous([opt_img])

            n_iter[0] += 1
            if n_iter[0] % show_iter == show_iter - 1:
                style_loss = torch.stack(layer_losses[:len(style_layers)]).sum()
                content_loss = torch.stack(layer_losses[len(style_layers):]).sum()
                history.append(
                    {
                        "step": n_iter[0] + 1,
                        "grad": float(grad_loss.detach().cpu()),
                        "style": float(style_loss.detach().cpu()),
                        "content": float(content_loss.detach().cpu()),
                        "total": float(loss.detach().cpu()),
                    }
                )
                print(f"Iteration: {n_iter[0] + 1}, loss: {loss.item():.6f}")

            return loss

        optimizer.step(closure)

    output_np = notebook_postprocess_image(opt_img.detach()[0].cpu().squeeze())
    output_path = os.path.join(output_dir, "final_pass.png")
    if save_output:
        imsave(output_path, output_np)

    if not history:
        history.append({"step": n_iter[0], "weight_source": weight_source})
    else:
        history[-1]["weight_source"] = weight_source
    return output_np, output_path, history


def resolve_image(image, path, label):
    path = (path or "").strip()
    if path:
        if not os.path.exists(path):
            raise gr.Error(f"{label} path does not exist: {path}")
        return path
    if image is None:
        raise gr.Error(f"Upload a {label.lower()} image or provide a local path.")
    return image


def make_upload_image(label):
    kwargs = {
        "label": label,
        "interactive": True,
        "type": "numpy",
        "image_mode": "RGB",
    }
    params = inspect.signature(gr.Image).parameters
    if "source" in params:
        kwargs["source"] = "upload"
    else:
        kwargs["sources"] = ["upload"]
    return gr.Image(**kwargs)


def center_position(target_size):
    center = int(target_size // 2)
    return center, center


def placement_preview(source_image, source_path, target_image, target_path, source_size, target_size, x, y):
    source_image = resolve_image(source_image, source_path, "Source")
    target_image = resolve_image(target_image, target_path, "Target")
    source_np = load_rgb_image(source_image, int(source_size)).astype(np.float32)
    target_np = load_rgb_image(target_image, int(target_size)).astype(np.float32)
    x, y = fit_source_placement(x, y, source_np.shape, int(target_size))

    source_h, source_w = source_np.shape[:2]
    top = int(x - source_h * 0.5)
    left = int(y - source_w * 0.5)
    preview = target_np.copy()
    region = preview[top:top + source_h, left:left + source_w]
    region[:] = region * 0.35 + source_np * 0.65
    return np.clip(preview, 0, 255).astype(np.uint8), x, y


def run_gradio(
    source_image,
    source_path,
    target_image,
    target_path,
    style_image,
    style_path,
    source_size,
    target_size,
    x,
    y,
    gpu_id,
    num_steps,
    grad_weight,
    style_weight,
    content_weight,
    seed,
    vgg_model_path,
    download_vgg_model_if_missing,
    allow_torchvision_fallback,
    output_dir,
):
    source_image = resolve_image(source_image, source_path, "Source")
    target_image = resolve_image(target_image, target_path, "Target")
    style_image = resolve_image(style_image, style_path, "Style")

    prepared_source = load_rgb_image(source_image, int(source_size))
    x, y = fit_source_placement(x, y, prepared_source.shape, int(target_size))
    seed_value = None if seed is None or int(seed) < 0 else int(seed)

    try:
        image, output_path, history = gradient_style_transfer(
            source_image=source_image,
            target_image=target_image,
            style_image=style_image,
            mask_image=None,
            output_dir=output_dir or DEFAULT_OUTPUT_DIR,
            source_size=int(source_size),
            target_size=int(target_size),
            x=x,
            y=y,
            gpu_id=gpu_id,
            num_steps=int(num_steps),
            grad_weight=float(grad_weight),
            style_weight=float(style_weight),
            content_weight=float(content_weight),
            seed=seed_value,
            vgg_model_path=vgg_model_path,
            download_vgg_model_if_missing=bool(download_vgg_model_if_missing),
            allow_torchvision_fallback=bool(allow_torchvision_fallback),
            progress_interval=max(1, int(num_steps) // 20),
        )
    except Exception as exc:
        raise gr.Error(str(exc))

    losses = {"final_run": history[-1] if history else {}}
    status = f"Saved final image to {Path(output_path).resolve()}"
    return image, output_path, losses, status, x, y


def clear_demo():
    return None, "", None, "", None, "", None, None, {}, "", DEFAULT_TARGET_SIZE // 2, DEFAULT_TARGET_SIZE // 2


def load_css():
    css_path = Path(__file__).with_name("style.css")
    if css_path.exists():
        return css_path.read_text()
    return ""


def create_demo():
    with gr.Blocks(css=load_css()) as demo:
        gr.Markdown("# Final Run")
        gr.Markdown("NST.ipynb style/content optimization with the my_run.py gradient loss and a three-image interface.")

        with gr.Row():
            with gr.Column():
                gr.Markdown("## Input")
                source_image = make_upload_image("Source image")
                source_path = gr.Textbox(label="Source path", placeholder="/kaggle/input/your-dataset/source.jpg")
                target_image = make_upload_image("Target image")
                target_path = gr.Textbox(label="Target path", placeholder="/kaggle/input/your-dataset/target.jpg")
                style_image = make_upload_image("Style image")
                style_path = gr.Textbox(label="Style path", placeholder="/kaggle/input/your-dataset/style.jpg")

                gr.Markdown("## Placement")
                with gr.Row():
                    source_size = gr.Slider(64, 1024, value=DEFAULT_SOURCE_SIZE, step=1, label="Source size")
                    target_size = gr.Slider(128, 1024, value=DEFAULT_TARGET_SIZE, step=1, label="Target size")
                with gr.Row():
                    x = gr.Slider(0, 1024, value=DEFAULT_TARGET_SIZE // 2, step=1, label="Vertical center")
                    y = gr.Slider(0, 1024, value=DEFAULT_TARGET_SIZE // 2, step=1, label="Horizontal center")
                with gr.Row():
                    center_button = gr.Button("Use Target Center")
                    preview_button = gr.Button("Preview")

                gr.Markdown("## Optimization")
                with gr.Row():
                    gpu_id = gr.Dropdown(
                        ["auto", "cpu", "cuda:0", "cuda:1"],
                        value="auto",
                        label="Device",
                        allow_custom_value=True,
                    )
                    num_steps = gr.Slider(1, 3000, value=DEFAULT_NUM_STEPS, step=1, label="LBFGS max iterations")
                grad_weight = gr.Number(value=1e4, label="Gradient loss weight")
                with gr.Accordion("Advanced", open=False):
                    style_weight = gr.Number(value=1.0, label="NST style weight multiplier")
                    content_weight = gr.Number(value=1.0, label="NST content weight multiplier")
                    seed = gr.Number(value=0, precision=0, label="Seed, use -1 for random")
                    vgg_model_path = gr.Textbox(value="", label="VGG model path", placeholder=DEFAULT_VGG_MODEL)
                    download_vgg_model_if_missing = gr.Checkbox(
                        value=True,
                        label="Download NST VGG model if missing",
                    )
                    allow_torchvision_fallback = gr.Checkbox(
                        value=False,
                        label="Allow torchvision VGG19 fallback if vgg_conv.pth is missing",
                    )
                    output_dir = gr.Textbox(value=DEFAULT_OUTPUT_DIR, label="Output directory")

                with gr.Row():
                    run_button = gr.Button("Run", variant="primary")
                    clear_button = gr.Button("Clear")

            with gr.Column():
                gr.Markdown("## Output")
                preview = gr.Image(label="Placement preview", type="numpy")
                output = gr.Image(label="Final image", type="numpy")
                output_file = gr.File(label="Saved output image")
                losses = gr.JSON(label="Latest logged losses")
                status = gr.Textbox(label="Status", interactive=False)

        center_button.click(center_position, inputs=[target_size], outputs=[x, y])
        preview_button.click(
            placement_preview,
            inputs=[source_image, source_path, target_image, target_path, source_size, target_size, x, y],
            outputs=[preview, x, y],
        )
        run_button.click(
            run_gradio,
            inputs=[
                source_image,
                source_path,
                target_image,
                target_path,
                style_image,
                style_path,
                source_size,
                target_size,
                x,
                y,
                gpu_id,
                num_steps,
                grad_weight,
                style_weight,
                content_weight,
                seed,
                vgg_model_path,
                download_vgg_model_if_missing,
                allow_torchvision_fallback,
                output_dir,
            ],
            outputs=[output, output_file, losses, status, x, y],
            show_progress="full",
        )
        clear_button.click(
            clear_demo,
            inputs=[],
            outputs=[source_image, source_path, target_image, target_path, style_image, style_path, preview, output, losses, status, x, y],
        )

    return demo


def parse_args():
    parser = argparse.ArgumentParser(description="NST.ipynb optimization with my_run.py gradient loss.")
    parser.add_argument("--cli", action="store_true", help="run once instead of launching Gradio")
    parser.add_argument("--source_file", type=str, default=None, help="path to the source image")
    parser.add_argument("--mask_file", type=str, default=None, help="optional path to a source object mask")
    parser.add_argument("--target_file", type=str, default=None, help="path to the target image")
    parser.add_argument("--style_file", type=str, default=None, help="path to the style image")
    parser.add_argument("--output_dir", type=str, default=DEFAULT_OUTPUT_DIR, help="path to output")
    parser.add_argument("--source_size", "--ss", type=int, default=DEFAULT_SOURCE_SIZE, help="source image size")
    parser.add_argument("--target_size", "--ts", type=int, default=DEFAULT_TARGET_SIZE, help="target image size")
    parser.add_argument("--x", type=int, default=None, help="vertical location center")
    parser.add_argument("--y", type=int, default=None, help="horizontal location center")
    parser.add_argument("--gpu_id", type=str, default="auto", help="auto, cpu, cuda:0, or GPU index")
    parser.add_argument("--num_steps", type=int, default=DEFAULT_NUM_STEPS, help="NST LBFGS max iterations")
    parser.add_argument("--grad_weight", type=float, default=1e4, help="gradient loss weight")
    parser.add_argument("--style_weight", type=float, default=1.0, help="multiplier on NST style weights")
    parser.add_argument("--content_weight", type=float, default=1.0, help="multiplier on NST content weights")
    parser.add_argument("--seed", type=int, default=0, help="optional random seed, use -1 for random")
    parser.add_argument("--vgg_model_path", type=str, default="", help="path to NST vgg_conv.pth")
    parser.add_argument(
        "--no_download_vgg_model",
        action="store_true",
        help="do not download the NST vgg_conv.pth automatically if it is missing",
    )
    parser.add_argument(
        "--allow_torchvision_fallback",
        action="store_true",
        help="use torchvision VGG19 weights if vgg_conv.pth is missing",
    )
    parser.add_argument("--server_name", type=str, default="0.0.0.0", help="Gradio server name")
    parser.add_argument("--server_port", type=int, default=None, help="Gradio server port")
    parser.add_argument("--share", action="store_true", help="create a public Gradio link")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.cli:
        if not args.source_file or not args.target_file or not args.style_file:
            raise SystemExit("--cli requires --source_file, --target_file, and --style_file")
        seed_value = None if args.seed is None or int(args.seed) < 0 else int(args.seed)
        image, output_path, history = gradient_style_transfer(
            source_image=args.source_file,
            target_image=args.target_file,
            style_image=args.style_file,
            mask_image=args.mask_file,
            output_dir=args.output_dir,
            source_size=args.source_size,
            target_size=args.target_size,
            x=args.x,
            y=args.y,
            gpu_id=args.gpu_id,
            num_steps=args.num_steps,
            grad_weight=args.grad_weight,
            style_weight=args.style_weight,
            content_weight=args.content_weight,
            seed=seed_value,
            vgg_model_path=args.vgg_model_path,
            download_vgg_model_if_missing=not args.no_download_vgg_model,
            allow_torchvision_fallback=args.allow_torchvision_fallback,
        )
        print(f"Saved final image to {Path(output_path).resolve()}")
        if history:
            print("Last logged losses:", history[-1])
        return image

    demo = create_demo()
    demo.queue(max_size=20, default_concurrency_limit=1)
    demo.launch(server_name=args.server_name, server_port=args.server_port, share=True)


if __name__ == "__main__":
    main()
