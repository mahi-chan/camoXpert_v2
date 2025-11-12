import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
import numpy as np


class AdvancedCODAugmentation:
    """Advanced augmentations specifically for camouflaged object detection"""

    def __init__(self, img_size=416, is_train=True):
        if is_train:
            self.transform = A.Compose([
                # Geometric augmentations
                A.Resize(img_size, img_size),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.3),
                A.RandomRotate90(p=0.3),
                A.ShiftScaleRotate(
                    shift_limit=0.1,
                    scale_limit=0.2,
                    rotate_limit=30,
                    p=0.5
                ),

                # Advanced color augmentations
                A.OneOf([
                    A.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.15, p=1.0),
                    A.RGBShift(r_shift_limit=25, g_shift_limit=25, b_shift_limit=25, p=1.0),
                    A.HueSaturationValue(hue_shift_limit=25, sat_shift_limit=35, val_shift_limit=25, p=1.0),
                    A.CLAHE(clip_limit=4.0, tile_grid_size=(8, 8), p=1.0),
                ], p=0.7),

                # Lighting variations
                A.OneOf([
                    A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=1.0),
                    A.RandomGamma(gamma_limit=(70, 130), p=1.0),
                    A.RandomToneCurve(scale=0.2, p=1.0),
                ], p=0.5),

                # Blur augmentations
                A.OneOf([
                    A.GaussianBlur(blur_limit=(3, 9), p=1.0),
                    A.MotionBlur(blur_limit=9, p=1.0),
                    A.MedianBlur(blur_limit=9, p=1.0),
                ], p=0.3),

                # Noise augmentations
                A.OneOf([
                    A.GaussNoise(var_limit=(10.0, 60.0), p=1.0),
                    A.ISONoise(color_shift=(0.01, 0.07), intensity=(0.1, 0.6), p=1.0),
                    A.MultiplicativeNoise(multiplier=(0.85, 1.15), p=1.0),
                ], p=0.3),

                # Weather/environmental effects
                A.OneOf([
                    A.RandomRain(slant_lower=-15, slant_upper=15, drop_length=25, p=1.0),
                    A.RandomFog(fog_coef_lower=0.1, fog_coef_upper=0.4, alpha_coef=0.15, p=1.0),
                    A.RandomShadow(shadow_roi=(0, 0.4, 1, 1), num_shadows_lower=1, num_shadows_upper=3, p=1.0),
                    A.RandomSunFlare(flare_roi=(0, 0, 1, 0.5), src_radius=150, p=1.0),
                ], p=0.25),

                # Spatial dropout (occlusion)
                A.CoarseDropout(
                    max_holes=10, max_height=40, max_width=40,
                    min_holes=3, min_height=20, min_width=20,
                    fill_value=0, p=0.3
                ),

                # Distortions (texture variation)
                A.OneOf([
                    A.GridDistortion(num_steps=7, distort_limit=0.4, p=1.0),
                    A.ElasticTransform(alpha=1.5, sigma=60, alpha_affine=60, p=1.0),
                    A.OpticalDistortion(distort_limit=0.6, shift_limit=0.6, p=1.0),
                ], p=0.3),

                # Advanced: Pixel-level augmentations
                A.OneOf([
                    A.Sharpen(alpha=(0.2, 0.5), lightness=(0.5, 1.0), p=1.0),
                    A.Emboss(alpha=(0.2, 0.5), strength=(0.2, 0.7), p=1.0),
                    A.Posterize(num_bits=4, p=1.0),
                    A.Equalize(p=1.0),
                ], p=0.2),

                # Normalization
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2()
            ])
        else:
            self.transform = A.Compose([
                A.Resize(img_size, img_size),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2()
            ])

    def __call__(self, image, mask):
        return self.transform(image=image, mask=mask)


class MixUpAugmentation:
    """MixUp augmentation for improved generalization"""

    def __init__(self, alpha=0.3):
        self.alpha = alpha

    def __call__(self, img1, mask1, img2, mask2):
        if np.random.rand() > 0.5:
            lam = np.random.beta(self.alpha, self.alpha)
            img = lam * img1 + (1 - lam) * img2
            mask = lam * mask1 + (1 - lam) * mask2
            return img, mask
        return img1, mask1