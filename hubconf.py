import os
import glob
from typing import Optional, Tuple, List
import torch
import numpy as np
from torchvision import transforms
from PIL import Image, ImageOps
import torch.nn.functional as F
from tqdm import tqdm

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
    def __init__(self, model):
        self.model = model
        self.tile_size = 768  # Fixed tile size
        self.latent_tile_size = 96  # 768 // 8
        self.device = model.device
        try:
            import xformers
            self.model.enable_xformers_memory_efficient_attention()
        except ImportError:
            pass

    def to(self, device, dtype=torch.float16):
        self.model.to(device, dtype)
        self.device = device
        return self

    def resize_for_tiling(self, img: Image.Image, splits_vertical: int, splits_horizontal: int, tile_overlap: int) -> Image.Image:
        target_height = self.tile_size + (splits_vertical - 1) * (self.tile_size - tile_overlap)
        target_width = self.tile_size + (splits_horizontal - 1) * (self.tile_size - tile_overlap)
        return img.resize((target_width, target_height), Image.Resampling.LANCZOS)

    def split_image(self, img: torch.Tensor, splits_vertical: int, splits_horizontal: int, tile_overlap: int):
        _, c, h, w = img.shape
        tiles = []
        positions = []
        
        for row in range(splits_vertical):
            for col in range(splits_horizontal):
                row_start, row_end, col_start, col_end = self._tile2pixel_indices(row, col, self.tile_size, self.tile_size, tile_overlap, tile_overlap)
                tile = img[:, :, row_start:row_end, col_start:col_end]
                tiles.append(tile)
                positions.append([row_start, col_start, row_end - row_start, col_end - col_start])
        
        return tiles, torch.tensor(positions, device=self.device)

    def split_latents(self, latents: torch.Tensor, splits_vertical: int, splits_horizontal: int, tile_overlap: int):
        _, c, h, w = latents.shape
        latent_tiles = []
        
        for row in range(splits_vertical):
            for col in range(splits_horizontal):
                row_start, row_end, col_start, col_end = self._tile2latent_indices(row, col, self.tile_size, self.tile_size, tile_overlap, tile_overlap)
                latent_tile = latents[:, :, row_start:row_end, col_start:col_end]
                latent_tiles.append(latent_tile)

    def assemble_results(self, tiles: List[torch.Tensor], positions: torch.Tensor, output_shape: Tuple[int, int, int, int], 
                         splits_vertical: int, splits_horizontal: int, tile_overlap: int):
        _, c, h, w = output_shape
        output = torch.zeros(output_shape, device=self.device)
        weights_sum = torch.zeros((1, 1, h, w), device=self.device)        
        tile_weights = self._create_tile_weights(self.tile_size, self.tile_size, splits_vertical, splits_horizontal, tile_overlap)
        
        for tile, pos, weight in zip(tiles, positions, tile_weights):
            row_start, col_start, tile_h, tile_w = pos
            output[:, :, row_start:row_start+tile_h, col_start:col_start+tile_w] += tile * weight
            weights_sum[:, :, row_start:row_start+tile_h, col_start:col_start+tile_w] += weight
        
        # Normalize by the sum of weights
        output = output / weights_sum.clamp(min=1e-8)
        return output.clamp(0, 1)

    def _create_tile_weights(self, tile_height: int, tile_width: int, splits_vertical: int, splits_horizontal: int, tile_overlap: int):
        weights = []
        for row in range(splits_vertical):
            for col in range(splits_horizontal):
                weight = self._gaussian_weights(tile_height, tile_width)
                
                # Adjust weights for edge tiles
                if row == 0:
                    weight[:tile_overlap//2, :] = 1
                if row == splits_vertical - 1:
                    weight[-tile_overlap//2:, :] = 1
                if col == 0:
                    weight[:, :tile_overlap//2] = 1
                if col == splits_horizontal - 1:
                    weight[:, -tile_overlap//2:] = 1
                
                weights.append(weight)
        return weights

    def _gaussian_weights(self, tile_height: int, tile_width: int):
        center_y, center_x = tile_height // 2, tile_width // 2
        y = torch.arange(tile_height, device=self.device)
        x = torch.arange(tile_width, device=self.device)
        y, x = torch.meshgrid(y, x, indexing='ij')
        
        gaussian = torch.exp(-((x - center_x)**2 + (y - center_y)**2) / (2 * (min(tile_width, tile_height) / 4)**2))
        return gaussian.unsqueeze(0).unsqueeze(0)  # Add batch and channel dimensions

    def _tile2pixel_indices(self, tile_row: int, tile_col: int, tile_width: int, tile_height: int, tile_row_overlap: int, tile_col_overlap: int):
        row_start = 0 if tile_row == 0 else tile_row * (tile_height - tile_row_overlap)
        row_end = row_start + tile_height
        col_start = 0 if tile_col == 0 else tile_col * (tile_width - tile_col_overlap)
        col_end = col_start + tile_width
        return row_start, row_end, col_start, col_end

    def _tile2latent_indices(self, tile_row: int, tile_col: int, tile_width: int, tile_height: int, tile_row_overlap: int, tile_col_overlap: int):
        px_row_start, px_row_end, px_col_start, px_col_end = self._tile2pixel_indices(tile_row, tile_col, tile_width, tile_height, tile_row_overlap, tile_col_overlap)
        return px_row_start // 8, px_row_end // 8, px_col_start // 8, px_col_end // 8

    @torch.no_grad()
    def __call__(self, img: Image.Image, splits_vertical: int = 1, splits_horizontal: int = 1, 
                 tile_overlap: int = 384, mode: str = 'stable') -> Image.Image:
        img, original_size, padding_info = pad_to_square(img)
        padded_size = img.size
        
        # Resize image for tiling
        img_resized = self.resize_for_tiling(img, splits_vertical, splits_horizontal, tile_overlap)
        img_tensor = transforms.ToTensor()(img_resized).unsqueeze(0).to(self.device)
        
        # Initialize latents for the entire image
        if mode == 'stable':
            init_latents = torch.zeros([1, 4, img_tensor.shape[2] // 8, img_tensor.shape[3] // 8],
                                       device=self.device, dtype=torch.float16)
        else:
            init_latents = torch.randn([1, 4, img_tensor.shape[2] // 8, img_tensor.shape[3] // 8],
                                       device=self.device, dtype=torch.float16)

        # Split image and latents into tiles
        tiles, positions = self.split_image(img_tensor, splits_vertical, splits_horizontal, tile_overlap)
        latent_tiles = self.split_latents(init_latents, splits_vertical, splits_horizontal, tile_overlap)

        results = []
        for tile, latent_tile in tqdm(zip(tiles, latent_tiles), total=len(tiles), desc="Processing tiles"):
            pipe_out = self.model(tile, latents=latent_tile)
            pred_diffuse = (pipe_out.prediction.clip(-1, 1) + 1) / 2
            pred_diffuse = torch.tensor(pred_diffuse, device=self.device).permute(0, 3, 1, 2)
            results.append(pred_diffuse)

        # Assemble results
        assembled_result = self.assemble_results(results, positions, img_tensor.shape, 
                                                 splits_vertical, splits_horizontal, tile_overlap)
        
        # Convert to PIL Image and resize to padded size
        pred_diffuse = transforms.ToPILImage()(assembled_result.squeeze().cpu())
        pred_diffuse = pred_diffuse.resize(padded_size, Image.Resampling.LANCZOS)

        # Crop to original size
        left, top, right, bottom = padding_info[0], padding_info[1], padded_size[0] - padding_info[2], padded_size[1] - padding_info[3]
        pred_diffuse = pred_diffuse.crop((left, top, right, bottom))
        
        return pred_diffuse
    
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

    args = parser.parse_args()
    predictor = StableDelight_turbo(local_cache_dir='./weights', device="cuda:0")
    
    image = Image.open(args.input)
    with torch.inference_mode():
        diffuse_image = predictor(image) 
    diffuse_image.save(args.input[:-4]+ '_out.png')

if __name__ == "__main__":
    _test_run()
