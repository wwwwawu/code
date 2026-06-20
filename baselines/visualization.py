import re
from pathlib import Path

import numpy as np
from PIL import Image, ImageChops, ImageFilter


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def _safe_stem(image_path, name_hint=None):
    stem = name_hint if name_hint is not None else Path(image_path).stem
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", stem).strip("_") or "image"


def _resize_prediction_to_image(mask, image_size, unpad_from_square=False):
    if not unpad_from_square:
        return mask.resize(image_size, Image.Resampling.NEAREST)

    mask_width, mask_height = mask.size
    image_width, image_height = image_size
    if mask_width != mask_height:
        return mask.resize(image_size, Image.Resampling.NEAREST)

    scale = mask_width / float(max(image_width, image_height))
    resized_width = int(round(image_width * scale))
    resized_height = int(round(image_height * scale))
    left = max(0, (mask_width - resized_width) // 2)
    top = max(0, (mask_height - resized_height) // 2)
    right = min(mask_width, left + resized_width)
    bottom = min(mask_height, top + resized_height)
    return mask.crop((left, top, right, bottom)).resize(image_size, Image.Resampling.NEAREST)


def _pad_to_square(image, fill):
    width, height = image.size
    if width == height:
        return image
    side = max(width, height)
    canvas = Image.new(image.mode, (side, side), fill)
    left = (side - width) // 2
    top = (side - height) // 2
    canvas.paste(image, (left, top))
    return canvas


def _make_square_output(image, fill, output_size=None):
    image = _pad_to_square(image, fill)
    if output_size and image.size != (output_size, output_size):
        resample = Image.Resampling.NEAREST if image.mode == "L" else Image.Resampling.BICUBIC
        image = image.resize((output_size, output_size), resample)
    return image


def _image_tensor_to_pil(image_tensor, mean=IMAGENET_MEAN, std=IMAGENET_STD):
    array = image_tensor.detach().float().cpu().numpy()
    if array.ndim == 4:
        array = array[0]
    if array.shape[0] == 3:
        array = np.transpose(array, (1, 2, 0))
    mean = np.array(mean, dtype=np.float32).reshape(1, 1, 3)
    std = np.array(std, dtype=np.float32).reshape(1, 1, 3)
    array = np.clip(array * std + mean, 0.0, 1.0)
    return Image.fromarray((array * 255.0).round().astype(np.uint8), mode="RGB")


def save_prediction_visualization(
    pred_mask,
    image_path,
    output_root,
    sample_index,
    alpha=0.45,
    name_hint=None,
    unpad_from_square=False,
    square_output=True,
    output_size=None,
    image_tensor=None,
):
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    mask_array = pred_mask.detach().squeeze().cpu().numpy().astype(np.uint8) * 255
    mask = Image.fromarray(mask_array, mode="L")
    if image_tensor is not None:
        image = _image_tensor_to_pil(image_tensor)
        if mask.size != image.size:
            mask = mask.resize(image.size, Image.Resampling.NEAREST)
    else:
        with Image.open(image_path) as source:
            image = source.convert("RGB")
        mask = _resize_prediction_to_image(mask, image.size, unpad_from_square=unpad_from_square)

    base_name = f"{sample_index:05d}_{_safe_stem(image_path, name_hint=name_hint)}"
    mask_path = output_root / f"{base_name}_mask.png"
    overlay_path = output_root / f"{base_name}_overlay.jpg"

    red_layer = Image.new("RGB", image.size, (255, 35, 35))
    highlighted = Image.blend(image, red_layer, alpha)
    overlay = image.copy()
    overlay.paste(highlighted, mask=mask)

    outer = mask.filter(ImageFilter.MaxFilter(5))
    inner = mask.filter(ImageFilter.MinFilter(5))
    boundary = ImageChops.difference(outer, inner)
    overlay.paste(Image.new("RGB", image.size, (255, 230, 0)), mask=boundary)

    if square_output:
        mask = _make_square_output(mask, fill=0, output_size=output_size)
        overlay = _make_square_output(overlay, fill=(0, 0, 0), output_size=output_size)

    mask.save(mask_path)
    overlay.save(overlay_path, quality=95)

    return str(mask_path), str(overlay_path)
