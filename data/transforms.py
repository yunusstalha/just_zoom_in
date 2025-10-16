import torch
import torchvision.transforms as T
from torchvision.transforms import InterpolationMode

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

class TwoCropTransform:
    """
    Custom transform to split a wide image (e.g., 1Kx2K) into two square
    patches and stack them. This preserves all image content without distortion.
    """
    def __init__(self, target_size):
        # Base transform to apply to each individual square crop
        self.base_transform = T.Compose([
            T.Resize((target_size, target_size), interpolation=InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])

    def __call__(self, img):
        """
        Args:
            img (PIL.Image): The input wide image.
        
        Returns:
            torch.Tensor: A tensor of shape (2, C, H, W).
        """
        w, h = img.size
        # Assuming the image is twice as wide as it is tall (e.g., 1K x 2K)
        # Crop the left and right halves
        left_half = img.crop((0, 0, w // 2, h))
        right_half = img.crop((w // 2, 0, w, h))
        
        # Apply the base transform to each half and stack them
        transformed_left = self.base_transform(left_half)
        transformed_right = self.base_transform(right_half)
        
        return torch.stack([transformed_left, transformed_right])

def build_transforms(target_image_size):
    """
    Creates basic train/validation transforms. The ground transform now uses
    the TwoCropTransform to handle wide images correctly.
    """
    ground_transform = TwoCropTransform(target_image_size)

    satellite_transform = T.Compose([
        T.Resize((target_image_size, target_image_size)),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])

    return {
        "ground": ground_transform,
        "satellite": satellite_transform,
    }