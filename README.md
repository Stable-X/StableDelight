# StableDelight
[![Hugging Face Space](https://img.shields.io/badge/🤗%20Hugging%20Face%20-Space-yellow)](https://huggingface.co/spaces/Stable-X/StableDelight)
[![Hugging Face Model](https://img.shields.io/badge/🤗%20Hugging%20Face%20-Model-green)](https://huggingface.co/Stable-X/yoso-delight-v0-4-base)
[![License](https://img.shields.io/badge/License-Apache--2.0-929292)](https://www.apache.org/licenses/LICENSE-2.0)

![17541724684116_ pic_hd](https://github.com/user-attachments/assets/26713cf5-4f2c-40a3-b6e7-61309b2e175a)

StableDelight is a cutting-edge solution for real-time reflection removal from textured surfaces. Building upon the success of [StableNormal](https://github.com/Stable-X/StableNormal), which focused on enhancing stability in monocular normal estimation, StableDelight takes this concept further by applying it to the challenging task of reflection removal.

## Background
StableDelight is inspired by our previous work, [StableNormal](https://github.com/Stable-X/StableNormal), which introduced a novel approach to tailoring diffusion priors for monocular normal estimation. The key innovation of StableNormal was its focus on enhancing estimation stability by reducing the inherent stochasticity of diffusion models (such as Stable Diffusion). This resulted in "Stable-and-Sharp" normal estimation that outperformed multiple baselines.

## Installation:

Please run following commands to build package:
```
git clone https://github.com/Stable-X/StableDelight.git
cd StableDelight
pip install -r requirements.txt
pip install -e .
```
or directly build package:
```
pip install git+https://github.com/Stable-X/StableDelight.git
```

## Usage
To use the StableDelight pipeline, you can instantiate the model and apply it to an image as follows:

```python
import torch
from PIL import Image

# Load an image
input_image = Image.open("path/to/your/image.jpg")

# Create predictor instance
predictor = torch.hub.load("Stable-X/StableDelight", "StableDelight_turbo", trust_repo=True)

# Apply the model to the image
delight_image = predictor(input_image)

# Save or display the result
delight_image.save("output/delight.png")
```
