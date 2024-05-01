from PIL import ImageFilter
import random
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
from torchvision import transforms
from aug import Global_Location_Scale_Augmentation


normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
augmentation_p = transforms.Compose([
    transforms.ToTensor(),
    transforms.RandomResizedCrop(224, scale=(0.2, 1.)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    normalize
])

transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225]),
                transforms.Resize((224,224))
            ])

def scale_to_uint8(image):
    min_val, max_val = image.min(), image.max()
    scaled_image = ((image - min_val) / (max_val - min_val) * 255).astype(np.uint8)
    return scaled_image

class BezierTransform:
    """Take two random crops of one image as the query and key."""

    def __init__(self, base_transform):
        self.base_transform = base_transform

    def __call__(self, x):
        import random
        if random.random() < 0.5:
            x1 = np.array(x)/255    
            x1 = Global_Location_Scale_Augmentation(x1)
            x1 = scale_to_uint8(x1)
            x1 = Image.fromarray(x1)
            q = self.base_transform(x1)
        else:
            q = self.base_transform(x)        
            
        k = self.base_transform(x)
        x = transform(x)
        return [q, k, x]


class GaussianBlur(object):
    """Gaussian blur augmentation in SimCLR https://arxiv.org/abs/2002.05709"""

    def __init__(self, sigma=[.1, 2.]):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x
