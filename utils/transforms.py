import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
# from torchvision.transforms import Compose, Resize, ToTensor, Normalize, InterpolationMode
from ProtoWD_lib.transform import image_transform, ResizeMaxSize
from ProtoWD_lib.constants import OPENAI_DATASET_MEAN, OPENAI_DATASET_STD

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class ResizePadMask:
    def __init__(self, image_size):
        self.image_size = image_size

    def __call__(self, mask):
        width, height = mask.size
        scale = self.image_size / float(max(width, height))
        new_width = int(round(width * scale))
        new_height = int(round(height * scale))
        mask = TF.resize(
            mask,
            [new_height, new_width],
            interpolation=transforms.InterpolationMode.NEAREST,
        )

        pad_left = (self.image_size - new_width) // 2
        pad_top = (self.image_size - new_height) // 2
        pad_right = self.image_size - new_width - pad_left
        pad_bottom = self.image_size - new_height - pad_top
        return TF.pad(mask, [pad_left, pad_top, pad_right, pad_bottom], fill=0)


def normalize(pred, max_value=None, min_value=None):
    if max_value is None or min_value is None:
        return (pred - pred.min()) / (pred.max() - pred.min())
    else:
        return (pred - min_value) / (max_value - min_value)

def get_transform(args):
    stretch_to_square = getattr(args, "stretch_to_square", False)
    backbone_type = getattr(args, "backbone_type", "clip")

    if backbone_type in {"dinov2", "dinov3"}:
        mean, std = IMAGENET_MEAN, IMAGENET_STD
    elif backbone_type == "sam":
        mean, std = None, None
    else:
        mean, std = OPENAI_DATASET_MEAN, OPENAI_DATASET_STD

    if backbone_type == "sam":
        if stretch_to_square:
            preprocess = transforms.Compose([
                transforms.Resize(
                    size=(args.image_size, args.image_size),
                    interpolation=transforms.InterpolationMode.BICUBIC,
                    max_size=None,
                    antialias=None,
                ),
                transforms.CenterCrop(size=(args.image_size, args.image_size)),
                transforms.Lambda(lambda image: image.convert("RGB")),
                transforms.ToTensor(),
            ])
        else:
            preprocess = transforms.Compose([
                ResizeMaxSize(args.image_size, fill=0),
                transforms.Lambda(lambda image: image.convert("RGB")),
                transforms.ToTensor(),
            ])
    else:
        preprocess = image_transform(
            args.image_size,
            is_train=False,
            mean=mean,
            std=std,
            resize_longest_max=not stretch_to_square,
        )

    if stretch_to_square:
        if backbone_type != "sam":
            preprocess.transforms[0] = transforms.Resize(
                size=(args.image_size, args.image_size),
                interpolation=transforms.InterpolationMode.BICUBIC,
                max_size=None,
                antialias=None,
            )
            preprocess.transforms[1] = transforms.CenterCrop(size=(args.image_size, args.image_size))
        target_transform = transforms.Compose([
            transforms.Resize((args.image_size, args.image_size), interpolation=transforms.InterpolationMode.NEAREST),
            transforms.CenterCrop(args.image_size),
            transforms.ToTensor(),
        ])
    else:
        target_transform = transforms.Compose([
            ResizePadMask(args.image_size),
            transforms.ToTensor(),
        ])
    return preprocess, target_transform
