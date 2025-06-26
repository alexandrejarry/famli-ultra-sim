
import math
import torch
import torch.nn.functional as F
from torch import nn
from torchvision import transforms
from torchvision.transforms import functional as VF

# from pl_bolts.transforms.dataset_normalizations import (
#     imagenet_normalization
# )

from monai.transforms import (        
    BorderPad,
    CenterSpatialCrop,
    EnsureChannelFirst,    
    Compose,      
    NormalizeIntensity,      
    RandAffine,       
    RandAxisFlip, 
    RandRotate,
    RandFlip,
    RandZoom,
    RepeatChannel,
    Resize,
    ScaleIntensityRange,
    ScaleIntensity,
    ToTensor, 
    RandAdjustContrast,
    RandGaussianNoise,
    RandGaussianSmooth,
    RandSpatialCrop,
    LoadImaged,        
    EnsureChannelFirstd,
    RandAxisFlipd,
    RandAffined,
    RandRotated,
    RandZoomd,
    Resized,
    ScaleIntensityRanged,
    ToTensord
)

from monai import transforms as monai_transforms

from torch.nn.utils.rnn import pad_sequence

class EnsureNumChannels:
    def __init__(self, num_channels: int = 3):
        self.num_channels = num_channels

    def __call__(self, x):
        if x.shape[0] != self.num_channels:
            x = RepeatChannel(self.num_channels)(x)
        return x

### TRANSFORMS
class SaltAndPepper:    
    def __init__(self, prob=0.2):
        self.prob = prob
    def __call__(self, x):
        noise_tensor = torch.rand(x.shape)
        salt = torch.max(x)
        pepper = torch.min(x)
        x[noise_tensor < self.prob/2] = salt
        x[noise_tensor > 1-self.prob/2] = pepper
        return x

# class Moco2TrainTransforms:
#     def __init__(self, height: int = 128):

#         # image augmentation functions
#         self.train_transform = transforms.Compose(
#             [
#                 # transforms.RandomResizedCrop(height, scale=(0.2, 1.0)),
#                 # transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),  # not strengthened
#                 # transforms.RandomGrayscale(p=0.2),
#                 transforms.Pad(64),
#                 transforms.RandomCrop(height),
#                 transforms.RandomHorizontalFlip(),
#                 transforms.RandomApply([SaltAndPepper(0.05)]),
#                 transforms.RandomApply([transforms.GaussianBlur(5, sigma=(0.1, 2.0))], p=0.5)
#                 # transforms.RandomRotation(180),
#             ]
#         )

#     def __call__(self, inp):
#         q = self.train_transform(inp)
#         k = self.train_transform(inp)
#         return q, k
# class Moco2TrainTransforms:
#     def __init__(self, height: int = 128):

#         # image augmentation functions
#         self.train_transform = transforms.Compose(
#             [
#                 transforms.RandomHorizontalFlip(),
#                 # transforms.RandomRotation(180),
#                 transforms.Pad(32),
#                 transforms.RandomCrop(height),
#                 transforms.RandomApply([SaltAndPepper(0.05)]),
#                 transforms.RandomApply([transforms.ColorJitter(brightness=[.5, 1.8], contrast=[0.5, 1.8], saturation=[.5, 1.8], hue=[-.2, .2])], p=0.8),
#                 transforms.RandomApply([transforms.GaussianBlur(5, sigma=(0.1, 2.0))], p=0.5),
#             ]
#         )

#     def __call__(self, inp):
#         q = self.train_transform(inp)
#         k = self.train_transform(inp)
#         return q, k

class Moco2TrainTransforms:
    def __init__(self, height: int = 128):

        # image augmentation functions
        self.train_transform = transforms.Compose(
            [
                # ScaleIntensityRange(a_min=0.0, a_max=255.0, b_min=0.0, b_max=1.0),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(180),
                transforms.RandomResizedCrop(size=height, scale=(0.25, 1.0), ratio=(0.75, 1.3333333333333333)),
                transforms.RandomApply([transforms.GaussianBlur(5, sigma=(0.1, 2.0))], p=0.5),
                # transforms.RandomApply([SaltAndPepper(0.05)]),
                transforms.ColorJitter(brightness=[0.5, 1.5], contrast=[0.5, 1.5], saturation=[0.5, 1.5], hue=[-.2, .2])
            ]
        )

    def __call__(self, inp):
        q = self.train_transform(inp)
        k = self.train_transform(inp)
        return q, k

class Moco2EvalTransforms:
    """Moco 2 augmentation:
    https://arxiv.org/pdf/2003.04297.pdf
    """

    def __init__(self, height: int = 128):

        self.eval_transform = transforms.Compose(
            [
                # ScaleIntensityRange(a_min=0.0, a_max=255.0, b_min=0.0, b_max=1.0),
                transforms.RandomCrop(height)
            ]
        )

    def __call__(self, inp):
        q = self.eval_transform(inp)
        k = self.eval_transform(inp)
        return q, k

class Moco2TestTransforms:
    """Moco 2 augmentation:
    https://arxiv.org/pdf/2003.04297.pdf
    """

    def __init__(self, height: int = 128):

        self.test_transform = Moco2EvalTransforms(height).eval_transform

    def __call__(self, inp):
        return self.test_transform(inp)

class SimCLRTrainTransforms:
    def __init__(self, height: int = 128):

        # image augmentation functions
        self.train_transform = transforms.Compose(
            [
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(180),
                # transforms.RandomResizedCrop(size=height, scale=(0.25, 1.0), ratio=(0.75, 1.3333333333333333)),
                # transforms.RandomApply([transforms.GaussianBlur(5, sigma=(0.1, 2.0))], p=0.5),
                # transforms.RandomApply([SaltAndPepper(0.05)]),
                transforms.ColorJitter(brightness=[0.5, 1.5], contrast=[0.5, 1.5], saturation=[0.5, 1.5], hue=[-.2, .2]),
                transforms.Pad(64),
                transforms.RandomCrop(height),
                GaussianNoise()
            ]
        )

    def __call__(self, inp):
        q = self.train_transform(inp)
        k = self.train_transform(inp)
        return q, k

class SimCLREvalTransforms:
    def __init__(self, height: int = 128):

        self.eval_transform = transforms.Compose(
            [
                transforms.CenterCrop(height)
            ]
        )

    def __call__(self, inp):
        q = self.eval_transform(inp)
        k = self.eval_transform(inp)
        return q, k


class SimCLRTestTransforms:
    def __init__(self, height: int = 128):

        self.test_transform = SimCLREvalTransforms(height).eval_transform

    def __call__(self, inp):
        return self.test_transform(inp)

class SimTrainTransforms:
    def __init__(self, height: int = 128):

        # image augmentation functions
        self.train_transform = transforms.Compose(
            [
                ScaleIntensityRange(a_min=0.0, a_max=255.0, b_min=0.0, b_max=1.0),
                transforms.ColorJitter(brightness=[0.5, 1.5], contrast=[0.5, 1.5], saturation=[0.5, 1.5], hue=[-.2, .2]),
                transforms.RandomHorizontalFlip(),
                transforms.RandomChoice([
                    transforms.Compose([transforms.RandomRotation(180), transforms.Pad(64), transforms.RandomCrop(height)]),
                    transforms.RandomResizedCrop(size=height, scale=(0.4, 1.0), ratio=(0.75, 1.3333333333333333))
                    ])
            ]
        )

    def __call__(self, inp):
        q = self.train_transform(inp)
        k = self.train_transform(inp)
        return q, k

class SimTrainTransformsV2:
    def __init__(self, height: int = 128):

        # image augmentation functions
        self.train_transform = transforms.Compose(
            [
                ScaleIntensityRange(a_min=0.0, a_max=255.0, b_min=0.0, b_max=1.0),
                transforms.ColorJitter(brightness=[0.5, 1.5], contrast=[0.5, 1.5], saturation=[0.5, 1.5], hue=[-.2, .2]),
                transforms.RandomHorizontalFlip(),
                transforms.Compose([transforms.RandomRotation(180), transforms.Pad(32), transforms.RandomCrop(height)])
            ]
        )

    def __call__(self, inp):
        q = self.train_transform(inp)
        k = self.train_transform(inp)
        return q, k

#additional transforms
class SimTrainTransformsV3:
    def __init__(self, height: int = 128):

        # image augmentation functions
        self.train_transform = transforms.Compose(
            [
                ScaleIntensityRange(a_min=0.0, a_max=255.0, b_min=0.0, b_max=1.0),
                transforms.ColorJitter(brightness=[0.5, 1.5], contrast=[0.5, 1.5], saturation=[0.5, 1.5], hue=[-.2, .2]),
                transforms.RandomHorizontalFlip(),
                transforms.Compose([transforms.RandomRotation(180), transforms.Pad(32), transforms.RandomCrop(height)]),
                transforms.Sobel(3)
            ]
        )

    def __call__(self, inp):
        q = self.train_transform(inp)
        k = self.train_transform(inp)
        return q, k

class SimEvalTransforms:
    def __init__(self, height: int = 128):

        self.eval_transform = transforms.Compose(
            [
                ScaleIntensityRange(a_min=0.0, a_max=255.0, b_min=0.0, b_max=1.0),
                transforms.CenterCrop(height)
            ]
        )

    def __call__(self, inp):
        q = self.eval_transform(inp)
        k = self.eval_transform(inp)
        return q, k


class SimTestTransforms:
    def __init__(self, height: int = 128):

        self.test_transform = SimEvalTransforms(height).eval_transform

    def __call__(self, inp):
        return self.test_transform(inp)

# class USClassTrainTransforms:
#     def __init__(self, height: int = 128):

#         self.train_transform = transforms.Compose(
#             [
#                 ScaleIntensityRange(a_min=0.0, a_max=255.0, b_min=0.0, b_max=1.0),
#                 transforms.ColorJitter(brightness=[0.5, 1.5], contrast=[0.5, 1.5], saturation=[0.5, 1.5], hue=[-.2, .2]),
#                 transforms.RandomHorizontalFlip(),
#                 transforms.RandomChoice([
#                     transforms.Compose([transforms.RandomRotation(180), transforms.Pad(64), transforms.RandomCrop(height)]),
#                     transforms.RandomResizedCrop(size=height, scale=(0.4, 1.0), ratio=(0.75, 1.3333333333333333))
#                     ])
#             ]
#         )

#     def __call__(self, inp):
#         return self.train_transform(inp)
class USClassTrainTransforms:
    def __init__(self, height: int = 256):

        self.train_transform = transforms.Compose(
            [
                ScaleIntensityRange(a_min=0.0, a_max=255.0, b_min=0.0, b_max=1.0),
                transforms.ColorJitter(brightness=[0.5, 1.5], contrast=[0.5, 1.5], saturation=[0.5, 1.5], hue=[-.2, .2]),
                transforms.RandomHorizontalFlip(),
                transforms.Compose([transforms.RandomRotation(180), transforms.Pad(64), transforms.RandomCrop(height)])
            ]
        )

    def __call__(self, inp):
        return self.train_transform(inp)


class USCutTrainTransforms:
    def __init__(self, size: int = 256):

        self.train_transform = transforms.Compose(
            [
                ScaleIntensityRange(a_min=0.0, a_max=255.0, b_min=0.0, b_max=1.0),
                # transforms.RandomHorizontalFlip(),
                transforms.CenterCrop(size)
            ]
        )

    def __call__(self, inp):
        return self.train_transform(inp)

class USClassEvalTransforms:

    def __init__(self, size=256):

        self.test_transform = transforms.Compose(
            [   
                ScaleIntensityRange(a_min=0.0, a_max=255.0, b_min=0.0, b_max=1.0),
                transforms.CenterCrop(size),
            ]
        )

    def __call__(self, inp):
        return self.test_transform(inp)

class USTrainTransforms:
    def __init__(self, height: int = 128):

        self.train_transform = transforms.Compose(
            [
                ScaleIntensityRange(a_min=0.0, a_max=255.0, b_min=0.0, b_max=1.0),
                transforms.ColorJitter(brightness=[0.5, 1.5], contrast=[0.5, 1.5], saturation=[0.5, 1.5], hue=[-.2, .2]),
                transforms.RandomHorizontalFlip(),
                transforms.RandomChoice([
                    transforms.Compose([transforms.RandomRotation(180), transforms.Pad(64), transforms.RandomCrop(height)]),
                    transforms.RandomResizedCrop(size=height, scale=(0.4, 1.0), ratio=(0.75, 1.3333333333333333))
                    ])
            ]
        )

    def __call__(self, inp):
        return self.train_transform(inp)



class RandomFrames:
    def __init__(self, num_frames=50):
        self.num_frames=num_frames
    def __call__(self, x):
        if self.num_frames > 0:
            idx = torch.randint(x.size(0), (self.num_frames,))
            idx = torch.sort(idx).values
            x = x[idx]
        return x

class NormI:
    def __init__(self):
        self.subtrahend=torch.tensor((0.485, 0.456, 0.406))
        self.divisor=torch.tensor((0.229, 0.224, 0.225))
    def __call__(self, x):
        return (x - self.subtrahend)/self.divisor
    


class USTrainGATransforms:
    def __init__(self, height: int = 128, num_frames=-1):

        self.train_transform = transforms.Compose(
            [
                RandomFrames(num_frames),
                EnsureChannelFirst(strict_check=False, channel_dim='no_channel'),
                ScaleIntensityRange(a_min=0.0, a_max=255.0, b_min=0.0, b_max=1.0),
                RepeatChannel(3),                
                BorderPad(spatial_border=[-1, 32, 32]),
                RandSpatialCrop(roi_size=[-1, 256, 256], random_size=False),
                transforms.Lambda(lambda x: torch.permute(x, (1,0,2,3))),
                transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
            ]
        )

    def __call__(self, inp):
        return self.train_transform(inp)

class USEvalGATransforms:
    def __init__(self, height: int = 128, num_frames=-1):

        self.eval_transform = transforms.Compose(
            [
                RandomFrames(num_frames),
                EnsureChannelFirst(strict_check=False, channel_dim='no_channel'),                
                ScaleIntensityRange(a_min=0.0, a_max=255.0, b_min=0.0, b_max=1.0),
                RepeatChannel(3),                
                transforms.Lambda(lambda x: torch.permute(x, (1,0,2,3))),
                transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
            ]
        )

    def __call__(self, inp):
        return self.eval_transform(inp)

class USEvalTransforms:

    def __init__(self, size=256, unsqueeze=False):

        self.test_transform = transforms.Compose(
            [                
                transforms.CenterCrop(size),
                ScaleIntensityRange(a_min=0.0, a_max=255.0, b_min=0.0, b_max=1.0),
                # imagenet_normalization(),
                transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
            ]
        )
        self.unsqueeze = unsqueeze

    def __call__(self, inp):
        inp = self.test_transform(inp)
        if self.unsqueeze:
            return inp.unsqueeze(dim=0)
        return inp

class US3DTrainTransforms:

    def __init__(self, size=128):
        # image augmentation functions        
        self.train_transform = Compose(
            [
                AddChannel(),                
                RandFlip(prob=0.5),
                RandRotate(prob=0.5, range_x=math.pi, range_y=math.pi, range_z=math.pi, mode="nearest", padding_mode='zeros'),
                CenterSpatialCrop(size),
                ScaleIntensityRange(a_min=0.0, a_max=255.0, b_min=0.0, b_max=1.0),
                RandAdjustContrast(prob=0.5),
                RandGaussianNoise(prob=0.5),
                RandGaussianSmooth(prob=0.5)
            ]
        )
    def __call__(self, inp):
        return self.train_transform(inp)


class US3DEvalTransforms:

    def __init__(self, size=128):

        self.test_transform = transforms.Compose(
            [
                AddChannel(),
                CenterSpatialCrop(size),
                ScaleIntensityRange(a_min=0.0, a_max=255.0, b_min=0.0, b_max=1.0)
            ]
        )

    def __call__(self, inp):        
        return self.test_transform(inp)


class GaussianNoise(nn.Module):    
    def __init__(self, mean=0.0, std=0.1):
        super(GaussianNoise, self).__init__()
        self.mean = torch.tensor(0.0)
        self.std = torch.tensor(0.1)
    def forward(self, x):
        return x + torch.normal(mean=self.mean, std=self.std, size=x.size(), device=x.device)

class EffnetDecodeTrainTransforms:
    def __init__(self, height: int = 128):

        # image augmentation functions
        self.train_transform = transforms.Compose(
            [
                ScaleIntensityRange(a_min=0.0, a_max=255.0, b_min=0.0, b_max=1.0),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(180),                
                transforms.RandomResizedCrop(size=height, scale=(0.25, 1.0), ratio=(0.75, 1.3333333333333333)),
                transforms.ColorJitter(brightness=[0.5, 1.5], contrast=[0.5, 1.5], saturation=[0.5, 1.5], hue=[-.2, .2]),
                GaussianNoise(),
                transforms.GaussianBlur(5, sigma=(0.1, 2.0)),
            ]
        )

    def __call__(self, inp):
        q = self.train_transform(inp)
        k = self.train_transform(inp)
        return q, k

class EffnetDecodeEvalTransforms:
    def __init__(self, height: int = 128):

        self.eval_transform = transforms.Compose(
            [
                ScaleIntensityRange(a_min=0.0, a_max=255.0, b_min=0.0, b_max=1.0),
                transforms.CenterCrop(height)
            ]
        )

    def __call__(self, inp):
        q = self.eval_transform(inp)
        k = self.eval_transform(inp)
        return q, k

class EffnetDecodeTestTransforms:
    def __init__(self, height: int = 128):

        self.test_transform = EffnetDecodeEvalTransforms(height).eval_transform

    def __call__(self, inp):
        return self.test_transform(inp)


class AutoEncoderTrainTransforms:
    def __init__(self, height: int = 128):

        # image augmentation functions
        self.train_transform = transforms.Compose(
            [
                ScaleIntensityRange(a_min=0.0, a_max=255.0, b_min=0.0, b_max=1.0),
                transforms.ColorJitter(brightness=[0.5, 1.5], contrast=[0.5, 1.5], saturation=[0.5, 1.5], hue=[-.2, .2]),
                transforms.RandomHorizontalFlip(),
                transforms.RandomChoice([
                    transforms.Compose([transforms.RandomRotation(180), transforms.Pad(64), transforms.RandomCrop(height)]),
                    transforms.RandomResizedCrop(size=height, scale=(0.4, 1.0), ratio=(0.75, 1.3333333333333333))
                    # transforms.RandomResizedCrop(size=height, scale=(0.9, 1.0), ratio=(0.9, 1.1))
                    ])
                # transforms.RandomRotation(30),
                # transforms.RandomResizedCrop(size=height, scale=(0.6, 1.0), ratio=(0.8, 1.2)),
            ]
        )
        # self.train_transform = transforms.Compose(
        #     [
        #         ScaleIntensityRange(a_min=0.0, a_max=255.0, b_min=0.0, b_max=1.0),
        #         transforms.ColorJitter(brightness=[0.5, 1.5], contrast=[0.5, 1.5], saturation=[0.5, 1.5], hue=[-.2, .2]),
        #         transforms.RandomHorizontalFlip(),
        #         transforms.RandomApply([transforms.RandomRotation(180)]),
        #         transforms.RandomApply([transforms.RandomResizedCrop(size=height, scale=(0.4, 1.0), ratio=(0.75, 1.3333333333333333))]),
        #     ]
        # )

    def __call__(self, inp):
        q = self.train_transform(inp)
        k = self.train_transform(inp)
        return q, k

class AutoEncoderEvalTransforms:
    def __init__(self, height: int = 128):

        self.eval_transform = transforms.Compose(
            [
                ScaleIntensityRange(a_min=0.0, a_max=255.0, b_min=0.0, b_max=1.0),
                transforms.CenterCrop(height)
            ]
        )

    def __call__(self, inp):
        q = self.eval_transform(inp)
        k = self.eval_transform(inp)
        return q, k

class AutoEncoderTestTransforms:
    def __init__(self, height: int = 128):

        self.test_transform = AutoEncoderEvalTransforms(height).eval_transform

    def __call__(self, inp):
        return self.test_transform(inp)



class DiffusionTrainTransforms:
    def __init__(self, height: int = 256):

        # image augmentation functions
        self.train_transform = transforms.Compose(
            [
                EnsureChannelFirst(strict_check=False, channel_dim='no_channel'),
                ScaleIntensityRange(a_min=0.0, a_max=255.0, b_min=0.0, b_max=1.0),                
                transforms.RandomHorizontalFlip(),
                transforms.RandomChoice([
                    transforms.Compose([transforms.RandomRotation(180), transforms.Pad(64), transforms.RandomCrop(height)]),
                    transforms.RandomResizedCrop(size=height, scale=(0.4, 1.0), ratio=(0.75, 1.3333333333333333))                
                ])
            ]
        )

    def __call__(self, inp):
        return self.train_transform(inp)        

class DiffusionEvalTransforms:
    def __init__(self, height: int = 256):

        self.eval_transform = transforms.Compose(
            [
                EnsureChannelFirst(strict_check=False, channel_dim='no_channel'),
                # ScaleIntensityRange(a_min=0.0, a_max=255.0, b_min=0.0, b_max=1.0),
                transforms.CenterCrop(height)
            ]
        )

    def __call__(self, inp):
        return self.eval_transform(inp)


class DiffusionV2TrainTransforms:
    def __init__(self, height: int = 64):

        # image augmentation functions
        self.train_transform = transforms.Compose(
            [
                FirstChannelOnly(),
                transforms.Resize(height),
                ScaleIntensityRange(a_min=0.0, a_max=255.0, b_min=0.0, b_max=1.0),                
                transforms.RandomHorizontalFlip(),
                # transforms.RandomRotation(180)
            ]
        )

    def __call__(self, inp):
        return self.train_transform(inp)        

class DiffusionV2EvalTransforms:
    def __init__(self, height: int = 64):

        self.eval_transform = transforms.Compose(
            [                
                FirstChannelOnly(),
                transforms.Resize(height),
                ScaleIntensityRange(a_min=0.0, a_max=255.0, b_min=0.0, b_max=1.0),                
            ]
        )

    def __call__(self, inp):
        return self.eval_transform(inp)



class LabelTrainTransforms:
    def __init__(self, height: int = 256):

        # image augmentation functions
        self.train_transform = transforms.Compose(
            [   
                ToTensor(),             
                transforms.RandomHorizontalFlip(),
                transforms.RandomChoice([
                    transforms.Compose([transforms.RandomRotation(30), transforms.Pad(32), transforms.RandomCrop(height)]),
                    transforms.RandomResizedCrop(size=height, scale=(0.4, 1.0), ratio=(0.75, 1.3333333333333333))                
                ])
            ]
        )

    def __call__(self, inp):
        return self.train_transform(inp)        

class LabelEvalTransforms:
    def __init__(self, height: int = 256):

        self.eval_transform = transforms.Compose(
            [
                ToTensor(),
                transforms.CenterCrop(height)
            ]
        )

    def __call__(self, inp):
        return self.eval_transform(inp)

class MustTrainTransforms:
    def __init__(self, height: int = 256):

        # image augmentation functions
        self.train_transform = transforms.Compose(
            [
                EnsureChannelFirst(strict_check=False, channel_dim='no_channel'),
                ScaleIntensityRange(a_min=0.0, a_max=255.0, b_min=0.0, b_max=1.0),                
                transforms.RandomHorizontalFlip()
            ]
        )

    def __call__(self, inp):
        return self.train_transform(inp)        

class MustEvalTransforms:
    def __init__(self, height: int = 256):

        self.eval_transform = transforms.Compose(
            [
                EnsureChannelFirst(strict_check=False, channel_dim='no_channel'),
                ScaleIntensityRange(a_min=0.0, a_max=255.0, b_min=0.0, b_max=1.0),
                transforms.CenterCrop(height)
            ]
        )

    def __call__(self, inp):
        return self.eval_transform(inp)


class FirstChannelOnly:
    def __call__(self, inp):
        return inp[0:1]


class Transpose:
    """Transposes the given PIL Image (swaps width and height)."""
    def __call__(self, img):
        """
        Args:
            img (PIL Image): Image to be transformed.

        Returns:
            PIL Image: Transformed image.
        """
        # Rotate the image by 90 degrees
        rotated_img = VF.rotate(img, -90, expand=True)
        # Flip the rotated image horizontally
        transposed_img = VF.hflip(rotated_img)
        return transposed_img

    def __repr__(self):
        return self.__class__.__name__ + '()'

    def __repr__(self):
        return self.__class__.__name__ + '()'


class RealUSTrainTransforms:
    def __init__(self, height: int = 256):

        # image augmentation functions
        self.train_transform = transforms.Compose(
            [
                EnsureChannelFirst(strict_check=False, channel_dim='no_channel'),
                Resize(spatial_size=(height, height), mode='bilinear'),
                ScaleIntensityRange(a_min=0.0, a_max=255.0, b_min=0.0, b_max=1.0),
                Transpose(),
                transforms.RandomHorizontalFlip()
            ]
        )

    def __call__(self, inp):
        return self.train_transform(inp)        

class RealUSEvalTransforms:
    def __init__(self, height: int = 256):

        self.eval_transform = transforms.Compose(
            [
                EnsureChannelFirst(strict_check=False, channel_dim='no_channel'),
                Resize(spatial_size=(height, height), mode='bilinear'),
                Transpose(),
                ScaleIntensityRange(a_min=0.0, a_max=255.0, b_min=0.0, b_max=1.0),
            ]
        )

    def __call__(self, inp):
        return self.eval_transform(inp)

class RealUSTrainTransformsV2:
    def __init__(self, height: int = 256):

        # image augmentation functions
        self.train_transform = transforms.Compose(
            [
                EnsureChannelFirst(strict_check=False, channel_dim=0),
                FirstChannelOnly(),
                Resize(spatial_size=(height, height), mode='bilinear'),
                ScaleIntensityRange(a_min=0.0, a_max=255.0, b_min=0.0, b_max=1.0),
                Transpose(),
                transforms.RandomHorizontalFlip()
            ]
        )

    def __call__(self, inp):
        return self.train_transform(inp)        

class RealUSEvalTransformsV2:
    def __init__(self, height: int = 256):

        self.eval_transform = transforms.Compose(
            [
                EnsureChannelFirst(strict_check=False, channel_dim=0),
                FirstChannelOnly(),
                # Resize(spatial_size=(height, height), mode='bilinear'),
                ScaleIntensityRange(a_min=0.0, a_max=255.0, b_min=0.0, b_max=1.0),
                Transpose(),
            ]
        )

    def __call__(self, inp):
        return self.eval_transform(inp)
    

class BlindSweepWTagTrainTransforms:
    def __init__(self, height: int = 256):

        # image augmentation functions
        self.train_transform = transforms.Compose(
            [
                RandomFrames(num_frames=160),
                EnsureChannelFirst(strict_check=False, channel_dim='no_channel'),
                ScaleIntensityRange(a_min=0.0, a_max=255.0, b_min=0.0, b_max=1.0)                
            ]
        )

    def __call__(self, inp):
        return self.train_transform(inp)        

class BlindSweepWTagEvalTransforms:
    def __init__(self, height: int = 256):

        self.eval_transform = transforms.Compose(
            [
                RandomFrames(num_frames=160),
                EnsureChannelFirst(strict_check=False, channel_dim='no_channel'),                
                ScaleIntensityRange(a_min=0.0, a_max=255.0, b_min=0.0, b_max=1.0)
            ]
        )

    def __call__(self, inp):
        return self.eval_transform(inp)



class DiffusionTrainTransformsPaired:
    def __init__(self, height: int = 256):

        # image augmentation functions
        self.train_transform = transforms.Compose(
            [
                EnsureChannelFirst(strict_check=False, channel_dim='no_channel')
                # ScaleIntensity()
            ]
        )

    def __call__(self, inp):
        return self.train_transform(inp)        

class DiffusionEvalTransformsPaired:
    def __init__(self, height: int = 256):

        self.eval_transform = transforms.Compose(
            [
                EnsureChannelFirst(strict_check=False, channel_dim='no_channel')
                # ScaleIntensity()
            ]
        )

    def __call__(self, inp):
        return self.eval_transform(inp)





class ZSample:
    def __call__(self, x):   
        z_mu = x['z_mu']
        z_sigma = x['z_sigma']
        return self.sampling(z_mu, z_sigma)

    def sampling(self, z_mu: torch.Tensor, z_sigma: torch.Tensor) -> torch.Tensor:
            """
            From the mean and sigma representations resulting of encoding an image through the latent space,
            obtains a noise sample resulting from sampling gaussian noise, multiplying by the variance (sigma) and
            adding the mean.

            Args:
                z_mu: Bx[Z_CHANNELS]x[LATENT SPACE SIZE] mean vector obtained by the encoder when you encode an image
                z_sigma: Bx[Z_CHANNELS]x[LATENT SPACE SIZE] variance vector obtained by the encoder when you encode an image

            Returns:
                sample of shape Bx[Z_CHANNELS]x[LATENT SPACE SIZE]
            """
            eps = torch.randn_like(z_sigma)
            z_vae = z_mu + eps * z_sigma
            return torch.tanh(z_vae)

class ZGanTrainTransforms:
    def __init__(self, height: int = 64):

        # image augmentation functions
        self.train_transform = transforms.Compose(
            [                
                EnsureChannelFirstd(keys=["z_mu", "z_sigma"], channel_dim=0),  
                ZSample(),
                transforms.RandomHorizontalFlip(),
                transforms.Compose([transforms.RandomRotation(180), transforms.Pad(8), transforms.RandomCrop(height)])
            ]
        )

    def __call__(self, inp):
        return self.train_transform(inp)        

class ZGanEvalTransforms:
    def __init__(self, height: int = 64):

        self.eval_transform = transforms.Compose(
            [                
                EnsureChannelFirstd(keys=["z_mu", "z_sigma"], channel_dim=0),
                ZSample()
            ]
        )

    def __call__(self, inp):
        return self.eval_transform(inp)


class PadAndCrop:

    """
    Pad the image with specified pixels on top and bottom, 
    and then crop it back to the original size.
    """

    def __init__(self, pad_height=0, pad_bottom=0):
        """
        :param pad_height: Number of pixels to pad on the top.
        :param pad_bottom: Number of pixels to pad on the bottom.
        """
        self.pad_height = pad_height
        self.pad_bottom = pad_bottom

    def __call__(self, img):
        """
        :param img: PIL Image or Tensor to be padded and cropped.
        :return: Padded and cropped image. (padding_left,padding_right, padding_top,padding_bottom)padding_top,padding_bottom)
        """
        # Pad the image
        img_padded = VF.pad(img, (0, self.pad_bottom, 0, self.pad_height), padding_mode='constant', fill=0)
        _, _, h, w = img.shape
        
        # return img_padded
        
        return VF.crop(img_padded, top=self.pad_height, left=0, height=h, width= w) 
        
class LotusTrainTransforms:
    def __init__(self, height: int = 256):

        # image augmentation functions
        self.train_transform = Compose(
            [   
                EnsureChannelFirstd(keys=['img', 'seg'], channel_dim=-1),
                ToTensord(keys=['img', 'seg']),
                Resized(keys=['img', 'seg'], spatial_size=(height, height), mode=['bilinear', 'nearest']),
                ScaleIntensityRanged(keys=['img'], a_min=0.0, a_max=255.0, b_min=0.0, b_max=1.0),                
                RandRotated(keys=['img', 'seg'], range_x=math.pi, mode=['bilinear', 'nearest'], prob=1.0, padding_mode='zeros'),
                RandAffined(keys=['img', 'seg'], prob=0.8, shear_range=(0.1, 0.1), mode=['bilinear', 'nearest'], padding_mode='zeros'),
                RandAxisFlipd(keys=['img', 'seg'], prob=0.5),
                RandZoomd(keys=['img', 'seg'], min_zoom=0.8, max_zoom=1.1, mode=['area', 'nearest'], prob=0.5, padding_mode='constant')
            ]
        )

    def __call__(self, inp):
        return self.train_transform(inp)  
    
class LotusEvalTransforms:
    def __init__(self, height: int = 256):

        # image augmentation functions
        self.eval_transform = Compose(
            [   
                EnsureChannelFirstd(keys=['img', 'seg'], channel_dim=-1),
                Resized(keys=['img', 'seg'], spatial_size=(height, height), mode=['bilinear', 'nearest']),
                ToTensord(keys=['img', 'seg']),
                ScaleIntensityRanged(keys=['img'], a_min=0.0, a_max=255.0, b_min=0.0, b_max=1.0),
            ]
        )

    def __call__(self, inp):
        return self.eval_transform(inp)
    
class VolumeSlicingTrainTransforms:
    def __init__(self, spatial_size: tuple = (-1, -1, -1)):

        # image augmentation functions
        self.train_transform = Compose(
            [   
                Resize(spatial_size=spatial_size, mode='nearest'),
                RandRotate(range_x=math.pi, mode='nearest', prob=1.0, padding_mode='zeros'),
                RandAffine(prob=0.5, shear_range=(0.1, 0.1), mode='nearest', padding_mode='zeros'),
                RandAxisFlip(prob=0.5),
                RandZoom(min_zoom=0.5, max_zoom=1.25, mode='nearest', prob=0.5, padding_mode='constant')
            ]
        )

    def __call__(self, inp):
        return self.train_transform(inp)  
    
class LotusEvalTransforms:
    def __init__(self, height: int = 256):

        # image augmentation functions
        self.eval_transform = Compose(
            [   
                EnsureChannelFirstd(keys=['img', 'seg'], channel_dim=-1),
                Resized(keys=['img', 'seg'], spatial_size=(height, height), mode=['bilinear', 'nearest']),
                ToTensord(keys=['img', 'seg']),
                ScaleIntensityRanged(keys=['img'], a_min=0.0, a_max=255.0, b_min=0.0, b_max=1.0),
            ]
        )

    def __call__(self, inp):
        return self.eval_transform(inp)

class LotusEvalTransformsRandom:
    def __init__(self, height: int = 256):

        # image augmentation functions
        self.eval_transform = Compose(
            [   
                EnsureChannelFirstd(keys=['img', 'seg'], channel_dim=-1),
                Resized(keys=['img', 'seg'], spatial_size=(height, height), mode=['bilinear', 'nearest']),
                ScaleIntensityRanged(keys=['img'], a_min=0.0, a_max=255.0, b_min=0.0, b_max=1.0),
                RandZoomd(keys=['img', 'seg'], min_zoom=0.8, max_zoom=1.1, mode=['area', 'nearest'], prob=0.9, padding_mode='constant'),
                RandRotated(keys=['img', 'seg'], range_x=math.pi, mode=['bilinear', 'nearest'], prob=1.0),
                RandAxisFlipd(keys=['img', 'seg'], prob=0.5),
                ToTensord(keys=['img', 'seg'])
            ]
        )


class DinoUSEvalTransforms:
    def __init__(self, height: int = 224):

        self.eval_transform = transforms.Compose(
            [
                transforms.Resize(height),
                ScaleIntensityRange(a_min=0.0, a_max=255.0, b_min=0.0, b_max=1.0),
                EnsureNumChannels(3),
                transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ]
        )

    def __call__(self, inp):
        return self.eval_transform(inp)
    
class ToVF:
    def __init__(self, keys, threshold=0.0, threshold_key=None, use_v=False, key_v=None):
        self.keys = keys
        self.threshold = threshold
        self.threshold_key = threshold_key
        self.use_v = use_v
        self.key_v = key_v
    def __call__(self, x):

        threshold_loc = None
        if self.threshold_key is not None:
            threshold_loc = x[self.threshold_key].reshape(-1) > self.threshold

        for key in self.keys:
            x[key] = self.get_grid_VF(x[key], threshold_loc, use_v=(self.use_v and key == self.key_v))
        return x
    
    def compute_grid(self, X):
        c, d, h, w = X.shape

        mesh_grid_params = [torch.arange(end=s, device=X.device) for s in (d, h, w)]
        mesh_grid_idx = torch.stack(torch.meshgrid(mesh_grid_params), dim=-1).squeeze().to(torch.float32)
        
        return mesh_grid_idx

    def get_grid_VF(self, x, threshold_loc=None, use_v=False):

        V = self.compute_grid(x)
        V = V.reshape(-1, 3)

        x = x.reshape(-1, 1)
        
        if threshold_loc is None:
            threshold_loc = torch.ones_like(x.squeeze(1)).to(torch.bool)
            
        V_filtered = V[threshold_loc]
        F_filtered = x[threshold_loc]

        if use_v:
            F_filtered = torch.cat([F_filtered, V_filtered], dim=-1)

        return V_filtered, F_filtered
    
class SampleVF:
    def __init__(self, keys, num_samples=250000):
        self.keys = keys
        self.num_samples = num_samples
    def __call__(self, x):
        V_, F_ = x[self.keys[0]]
        
        min_samples = min(self.num_samples, V_.shape[0])
        idx = torch.randperm(V_.shape[0])[:min_samples]
        idx = torch.sort(idx).values

        for key in self.keys:
            V, F = x[key]
            x[key] = (V[idx], F[idx])

        return x
    
class FluidTrainTransforms:
    def __init__(self):

        # image augmentation functions
        self.train_transform = Compose(
            [   
                ToTensord(keys=['img', 'seg']),
                EnsureChannelFirstd(keys=['img', 'seg'], channel_dim='no_channel'),
                ScaleIntensityRanged(keys=['img'], a_min=0.0, a_max=255.0, b_min=0.0, b_max=1.0),                
                # RandRotated(keys=['img', 'seg'], range_x=math.pi, mode=['bilinear', 'nearest'], prob=1.0, padding_mode='zeros'),
                RandAffined(keys=['img', 'seg'], prob=0.8, shear_range=(0.1, 0.1), mode=['bilinear', 'nearest'], padding_mode='zeros'),
                RandAxisFlipd(keys=['img', 'seg'], prob=0.5),
                RandZoomd(keys=['img', 'seg'], min_zoom=0.8, max_zoom=1.1, mode=['area', 'nearest'], prob=0.5, padding_mode='constant'),
                ToVF(keys=['img', 'seg'], use_v=True, key_v='img'), 
                # SampleVF(keys=['img', 'seg'], num_samples=500000)
            ]
        )

    def __call__(self, inp):
        return self.train_transform(inp)  
    
class FluidEvalTransforms:
    def __init__(self):

        # image augmentation functions
        self.eval_transform = Compose(
            [   
                ToTensord(keys=['img', 'seg']),
                EnsureChannelFirstd(keys=['img', 'seg'], channel_dim='no_channel'),
                ScaleIntensityRanged(keys=['img'], a_min=0.0, a_max=255.0, b_min=0.0, b_max=1.0),
                ToVF(keys=['img', 'seg'], use_v=True, key_v='img'),
                # SampleVF(keys=['img', 'seg'], num_samples=500000)
            ]
        )

    def __call__(self, inp):
        return self.eval_transform(inp)
    
class RandomRotation3D:
    def __init__(self, max_angle=180, mode='nearest'):
        """
        Initialize the random rotation transform.

        Args:
            max_angle (int): Maximum rotation angle in degrees for each axis.
        """
        self.max_angle = max_angle
        self.mode = mode

    def get_random_rotation_matrix(self, device):
        """Generate a random 3D rotation matrix (angle in degrees)."""
        angles = torch.rand(3, device=device) * 2 * self.max_angle - self.max_angle
        angles = angles * torch.pi / 180.0  # to radians

        cx, cy, cz = torch.cos(angles)
        sx, sy, sz = torch.sin(angles)

        Rx = torch.tensor([
            [1, 0, 0],
            [0, cx, -sx],
            [0, sx, cx]
        ], device=device)

        Ry = torch.tensor([
            [cy, 0, sy],
            [0, 1, 0],
            [-sy, 0, cy]
        ], device=device)

        Rz = torch.tensor([
            [cz, -sz, 0],
            [sz, cz, 0],
            [0, 0, 1]
        ], device=device)

        return Rz @ Ry @ Rx  # Combined rotation

    def rotate_3d_tensor(self, x):
        """
        Rotate a 3D tensor [B, C, D, H, W] with independent random rotations.
        """
        device = x.device
        B, C, D, H, W = x.shape

        # Create a batch of affine matrices
        affines = []
        for _ in range(B):
            R = self.get_random_rotation_matrix(device)
            A = torch.eye(4, device=device)
            A[:3, :3] = R
            affines.append(A[:3])  # [3, 4]

        affines = torch.stack(affines, dim=0)  # [B, 3, 4]

        # Create grid for each sample in the batch
        grid = F.affine_grid(affines, size=x.shape, align_corners=True)  # [B, D, H, W, 3]

        rotated = F.grid_sample(x, grid, mode=self.mode, padding_mode='zeros', align_corners=True)
        return rotated

    def __call__(self, x):
        return self.rotate_3d_tensor(x)
    
class RandomScale3D:
    def __init__(self, scale_range=(0.9, 1.1)):
        """
        Initialize the random scale transform.

        Args:
            scale_range (tuple): Min and max scale factor for each axis.
        """
        self.scale_range = scale_range

    def get_random_scale_matrix(self, device):
        """
        Generate a random 3D scaling matrix.

        Returns:
            torch.Tensor: [3, 3] scaling matrix
        """
        scales = torch.empty(3, device=device).uniform_(*self.scale_range)
        S = torch.diag(scales)
        return S

    def scale_3d_tensor(self, x):
        """
        Scale a 3D tensor [B, C, D, H, W] with random scaling.

        Returns:
            torch.Tensor: Scaled tensor [B, C, D, H, W]
        """
        device = x.device
        B, C, D, H, W = x.shape

        # Create batch of affine scale matrices
        affines = []
        for _ in range(B):
            S = self.get_random_scale_matrix(device)
            A = torch.eye(4, device=device)
            A[:3, :3] = S
            affines.append(A[:3])  # [3, 4]

        affines = torch.stack(affines, dim=0)  # [B, 3, 4]

        # Create grid and sample
        grid = F.affine_grid(affines, size=x.shape, align_corners=False)  # [B, D, H, W, 3]
        scaled = F.grid_sample(x, grid, mode='bilinear', padding_mode='border', align_corners=False)
        return scaled

    def __call__(self, x):
        return self.scale_3d_tensor(x)
    

class Resize3D:
    def __init__(self, target_size, mode='nearest'):
        """
        Resize a 3D tensor [B, C, D, H, W] to the target size.

        Args:
            target_size (tuple): (D, H, W), use -1 to keep that dimension unchanged
        """
        assert len(target_size) == 3, "target_size must be a tuple of (D, H, W)"
        self.target_size = target_size
        self.mode = mode

    def __call__(self, x):
        """
        Resize the input tensor.

        Args:
            x (torch.Tensor): [B, C, D, H, W]

        Returns:
            torch.Tensor: resized tensor [B, C, D', H', W']
        """
        assert x.ndim == 5, "Input tensor must have shape [B, C, D, H, W]"
        B, C, D, H, W = x.shape

        out_D = self.target_size[0] if self.target_size[0] != -1 else D
        out_H = self.target_size[1] if self.target_size[1] != -1 else H
        out_W = self.target_size[2] if self.target_size[2] != -1 else W

        resized = F.interpolate(x, size=(out_D, out_H, out_W), mode=self.mode)
        return resized
    
class DiffusorTrainTransform:
    def __init__(self, target_size: tuple = (-1, -1, -1)):

        # image augmentation functions
        self.train_transform = transforms.Compose(
            [
                Resize3D(target_size=target_size, mode='nearest'),
                RandomRotation3D(max_angle=180, mode='nearest'),
                RandomScale3D(scale_range=(0.5, 1.25)),
            ]
        )

    def __call__(self, inp):
        return self.train_transform(inp)