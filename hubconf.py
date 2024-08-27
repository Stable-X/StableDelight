import os
import glob
from typing import Optional, Tuple
import torch
import numpy as np
from torchvision import transforms
from PIL import Image, ImageOps
import torch.nn.functional as F

dependencies = ["torch", "numpy",  "diffusers", "PIL"]
from stabledelight.pipeline_yoso_delight import YosoDelightPipeline

def pad_to_square(image: Image.Image) -> Tuple[Image.Image, Tuple[int, int], Tuple[int, int, int, int]]:
    """Pad the input image to make it square."""
    width, height = image.size
    size = max(width, height)
    
    delta_w = size - width
    delta_h = size - height
    padding = (delta_w // 2, delta_h // 2, delta_w - (delta_w // 2), delta_h - (delta_h // 2))
    
    padded_image = ImageOps.expand(image, padding)
    return padded_image, image.size, padding

def resize_image(image: Image.Image, resolution: int) -> Tuple[Image.Image, Tuple[int, int], Tuple[float, float]]:
    """Resize the image while maintaining aspect ratio and then pad to nearest multiple of 64."""
    if not isinstance(image, Image.Image):
        raise ValueError("Expected a PIL Image object")
    
    np_image = np.array(image)
    height, width = np_image.shape[:2]

    scale = resolution / max(height, width)
    new_height = int(np.round(height * scale / 64.0)) * 64
    new_width = int(np.round(width * scale / 64.0)) * 64

    resized_image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
    return resized_image, (height, width), (new_height / height, new_width / width)

def center_crop(image: Image.Image) -> Tuple[Image.Image, Tuple[int, int], Tuple[float, float, float, float]]:
    """Crop the center of the image to make it square."""
    width, height = image.size
    crop_size = min(width, height)
    
    left = (width - crop_size) / 2
    top = (height - crop_size) / 2
    right = (width + crop_size) / 2
    bottom = (height + crop_size) / 2

    cropped_image = image.crop((left, top, right, bottom))
    return cropped_image, image.size, (left, top, right, bottom)

class Predictor:
    """Predictor class for Stable Diffusion models."""

    def __init__(self, model):
        self.model = model
        try:
            import xformers
            self.model.enable_xformers_memory_efficient_attention()
        except ImportError:
            pass

    def to(self, device, dtype=torch.float16):
        self.model.to(device, dtype)
        return self
    
    @torch.no_grad()
    def __call__(self, img: Image.Image, image_resolution=768, mode='stable', 
                 preprocess=None) -> Image.Image:
        if img.mode == 'RGBA':
            img = img.convert('RGB')

        original_size = img.size

        if preprocess == 'pad':
            img, original_size, padding_info = pad_to_square(img)
        elif preprocess == 'crop':
            img, original_size, crop_info = center_crop(img)
        
        img, original_dims, scaling_factors = resize_image(img, image_resolution)
        if mode == 'stable':
            init_latents = torch.zeros([1, 4, img.size[1] // 8, img.size[0] // 8], 
                                    device="cuda", dtype=torch.float16)
        else:
            init_latents = None

        pipe_out = self.model(img, latents=init_latents)
        pred_diffuse = (pipe_out.prediction.clip(-1, 1) + 1) / 2
        pred_diffuse = (pred_diffuse[0] * 255).astype(np.uint8)
        pred_diffuse = Image.fromarray(pred_diffuse)
        
        new_dims = (int(original_dims[1]), int(original_dims[0])) # reverse the shape (width, height)
        pred_diffuse = pred_diffuse.resize(new_dims, Image.Resampling.LANCZOS)

        if preprocess == 'pad':
            left, top, right, bottom = padding_info[0], padding_info[1], original_dims[0] - padding_info[2], original_dims[1] - padding_info[3]
            pred_diffuse = pred_diffuse.crop((left, top, right, bottom))
        elif preprocess == 'crop':
            left, top, right, bottom = crop_info
            pred_diffuse_with_bg = Image.new("RGB", original_size)
            pred_diffuse_with_bg.paste(pred_diffuse, (int(left), int(top)))
            pred_diffuse = pred_diffuse_with_bg
        else:
            # If no preprocessing, resize the output image to the original size
            pred_diffuse = pred_diffuse.resize(original_size, Image.Resampling.LANCZOS)

        return pred_diffuse
    
    def generate_reflection_score(self, rgb_image, diffuse_image, kernel_size=15):
        """
        Generate a reflection score by comparing grayscale RGB and diffuse images using PyTorch.
        
        :param rgb_image: RGB image as a PIL Image
        :param diffuse_image: Diffuse image as a PIL Image  
        :param kernel_size: Size of the box kernel for local smoothing
        :return: reflection score as a PIL Image
        """
        
        # Set device  
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Convert RGB and diffuse images to grayscale
        rgb_gray = rgb_image.convert('L')
        diffuse_gray = diffuse_image.convert('L')
        
        # Load and convert images to PyTorch tensors
        to_tensor = transforms.ToTensor()
        rgb_tensor = to_tensor(rgb_gray).to(device)
        diffuse_tensor = to_tensor(diffuse_gray).to(device)
        
        # Ensure both images have the same shape
        assert rgb_tensor.shape == diffuse_tensor.shape, "Grayscale RGB and diffuse images must have the same dimensions"
        
        residuals = torch.abs(rgb_tensor - diffuse_tensor)
        
        # Create box kernel
        box_kernel = torch.ones(1, 1, kernel_size, kernel_size, device=device) / (kernel_size ** 2)
        
        # Apply local smoothing
        smoothed_residuals = F.conv2d(residuals.unsqueeze(0), box_kernel, padding=kernel_size//2).squeeze(0)
        
        # Compute patch values
        patch_size = 16
        patch_values = F.avg_pool2d(smoothed_residuals.unsqueeze(0), kernel_size=patch_size, stride=1, padding=patch_size//2).squeeze(0)
        
        # Use patch values as the reflection score
        score = smoothed_residuals
        
        # Normalize the score to [0, 255] range and convert to uint8
        score = (score - score.min()) / (score.max() - score.min())
        score = score * 255
        score = score[0].cpu().numpy().astype(np.uint8)
        
        # Convert the score to a PIL Image
        score_image = Image.fromarray(score)
        
        return score_image

    def generate_specular_image(self, rgb_image, diffuse_image):
        """
        Generate specular image by subtracting the diffuse image from the RGB image.
        
        :param rgb_image: RGB image as a PIL Image 
        :param diffuse_image: Diffuse image as a PIL Image
        :return: Specular image as a PIL Image
        """
        
        # Convert images to numpy arrays  
        rgb_np = np.array(rgb_image)
        diffuse_np = np.array(diffuse_image)
        
        # Subtract diffuse from RGB (clipping to avoid underflow)
        specular_np = np.clip(rgb_np.astype(int) - diffuse_np.astype(int), 0, 255).astype(np.uint8)
        
        # Convert back to PIL Image
        specular_image = Image.fromarray(specular_np)
        
        return specular_image
    
    def __repr__(self):
        return f"Predictor(model={self.model})"

def StableDelight_turbo(local_cache_dir: Optional[str] = None, device="cuda:0", yoso_version='yoso-delight-v0-4-base') -> Predictor:
    """Load the StableDelight_turbo pipeline for a faster inference."""
    
    yoso_weight_path = os.path.join(local_cache_dir if local_cache_dir else "Stable-X", yoso_version)
    pipe = YosoDelightPipeline.from_pretrained(yoso_weight_path, 
                                               trust_remote_code=True, safety_checker=None, variant="fp16", 
                                               torch_dtype=torch.float16, t_start=0).to(device)

    return Predictor(pipe)

def save_mask_as_image(mask_tensor, output_path):
    """
    Save the PyTorch tensor mask as a grayscale image.
    
    :param mask_tensor: PyTorch tensor containing the mask
    :param output_path: Path to save the output image  
    """
    # Convert to numpy array
    mask_np = mask_tensor.cpu().numpy().squeeze()
    
    # Convert to 8-bit unsigned integer
    mask_np = (mask_np * 255).astype(np.uint8)
    
    # Create and save image
    Image.fromarray(mask_np).save(output_path)
    
def process_all_images(base_dir):
    """
    Process all images in the given directory structure.
    
    :param base_dir: Base directory containing 'color' and 'diffuse' subdirectories  
    """
    color_dir = os.path.join(base_dir, 'color') 
    diffuse_dir = os.path.join(base_dir, 'diffuse')
    reflection_dir = os.path.join(base_dir, 'reflection_mask')
    specular_dir = os.path.join(base_dir, 'specular')
    
    # Create output directories if they don't exist
    os.makedirs(reflection_dir, exist_ok=True)
    os.makedirs(specular_dir, exist_ok=True)
    
    # Initialize predictor
    predictor = StableDelight_turbo(local_cache_dir='./weights', device="cuda:0")
    
    # Process each image
    for rgb_path in glob.glob(os.path.join(color_dir, '*.png')):
        filename = os.path.basename(rgb_path)
        diffuse_path = os.path.join(diffuse_dir, filename)
        
        if os.path.exists(diffuse_path):
            print(f"Processing {filename}")
            
            mask_output_path = os.path.join(reflection_dir, f"{os.path.splitext(filename)[0]}.png") 
            specular_output_path = os.path.join(specular_dir, f"{os.path.splitext(filename)[0]}.png")
            
            predictor.process_image(rgb_path, diffuse_path, mask_output_path, specular_output_path)
        else:
            print(f"Diffuse image not found for {filename}")

def _test_run():
    import argparse

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--input", "-i", type=str, required=True, help="Input image file")
    parser.add_argument("--output", "-o", type=str, required=True, help="Output image file")

    args = parser.parse_args()
    predictor = StableDelight_turbo(local_cache_dir='./weights', device="cuda:0")
    
    image = Image.open(args.input)
    with torch.inference_mode():
        diffuse_image = predictor(image) 
    diffuse_image.save(args.output)
    reflection_score = predictor.generate_reflection_score(image, diffuse_image)
    reflection_score.save(args.output[:-4]+ '_mask.png')

if __name__ == "__main__":
    _test_run()
