import torchvision.transforms as T
from torchvision.transforms import InterpolationMode

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def build_transforms(target_image_size):
    """
    Creates basic train/validation transforms.

    Only the image size is configurable from outside; adjust the rest directly
    in this module when needed.
    """
    padded_size = int(target_image_size * 1.1)
    resize_ground = T.Resize(padded_size)
    resize_satellite = T.Resize((target_image_size, target_image_size))
    normalize = T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)

    ground = T.Compose([
        resize_ground,
        T.RandomCrop(target_image_size),
        T.RandomHorizontalFlip(),
        T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.02),
        T.RandomGrayscale(p=0.1),
        T.ToTensor(),
        normalize,
    ])

    satellite = T.Compose([
        resize_satellite,
        # T.RandomRotation(degrees=10, interpolation=InterpolationMode.BILINEAR),
        # T.RandomHorizontalFlip(),
        # T.RandomVerticalFlip(),
        # T.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.05, hue=0.01),
        T.ToTensor(),
        normalize,
    ])

    return {
        "ground": ground,
        "satellite": satellite,
    }
