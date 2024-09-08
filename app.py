# Copyright 2024 Anton Obukhov, ETH Zurich. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# --------------------------------------------------------------------------
# If you find this code useful, we kindly ask you to cite our paper in your work.
# Please find bibtex at: https://github.com/prs-eth/Marigold#-citation
# More information about the method can be found at https://marigoldmonodepth.github.io
# --------------------------------------------------------------------------
from __future__ import annotations

import functools
import os
import tempfile

import gradio as gr
import imageio as imageio
import numpy as np
import spaces
import torch as torch
torch.backends.cuda.matmul.allow_tf32 = True
from PIL import Image
from gradio_imageslider import ImageSlider
from tqdm import tqdm

from pathlib import Path
import gradio
from gradio.utils import get_cache_folder
from stabledelight import YosoDelightPipeline

class Examples(gradio.helpers.Examples):
    def __init__(self, *args, directory_name=None, **kwargs):
        super().__init__(*args, **kwargs, _initiated_directly=False)
        if directory_name is not None:
            self.cached_folder = get_cache_folder() / directory_name
            self.cached_file = Path(self.cached_folder) / "log.csv"
        self.create()


default_seed = 2024
default_batch_size = 1

default_image_processing_resolution = 2048
default_video_out_max_frames = 60

def process_image_check(path_input):
    if path_input is None:
        raise gr.Error(
            "Missing image in the first pane: upload a file or use one from the gallery below."
        )

def resize_image(input_image, resolution):
    # Ensure input_image is a PIL Image object
    if not isinstance(input_image, Image.Image):
        raise ValueError("input_image should be a PIL Image object")

    # Convert image to numpy array
    input_image_np = np.asarray(input_image)

    # Get image dimensions
    H, W, C = input_image_np.shape
    H = float(H)
    W = float(W)
    
    # Calculate the scaling factor
    k = float(resolution) / min(H, W)
    
    # Determine new dimensions
    H *= k
    W *= k
    H = int(np.round(H / 64.0)) * 64
    W = int(np.round(W / 64.0)) * 64
    
    # Resize the image using PIL's resize method
    img = input_image.resize((W, H), Image.Resampling.LANCZOS)
    
    return img

def process_image(
    pipe,
    path_input,
):
    name_base, name_ext = os.path.splitext(os.path.basename(path_input))
    print(f"Processing image {name_base}{name_ext}")

    path_output_dir = tempfile.mkdtemp()
    path_out_png = os.path.join(path_output_dir, f"{name_base}_delight.png")
    input_image = Image.open(path_input)
    pipe_out = pipe(
        input_image,
        match_input_resolution=False,
        processing_resolution=default_image_processing_resolution
    )

    processed_frame = (pipe_out.prediction.clip(-1, 1) + 1) / 2
    processed_frame = (processed_frame[0] * 255).astype(np.uint8)
    processed_frame = Image.fromarray(processed_frame)
    processed_frame.save(path_out_png)
    yield [input_image, path_out_png]

def process_video(
    pipe,
    path_input,
    out_max_frames=default_video_out_max_frames,
    target_fps=10,
    progress=gr.Progress(),
):
    if path_input is None:
        raise gr.Error(
            "Missing video in the first pane: upload a file or use one from the gallery below."
        )

    name_base, name_ext = os.path.splitext(os.path.basename(path_input))
    print(f"Processing video {name_base}{name_ext}")

    path_output_dir = tempfile.mkdtemp()
    path_out_vis = os.path.join(path_output_dir, f"{name_base}_delight.mp4")

    init_latents = None
    reader, writer = None, None
    try:
        reader = imageio.get_reader(path_input)

        meta_data = reader.get_meta_data()
        fps = meta_data["fps"]
        size = meta_data["size"]
        duration_sec = meta_data["duration"]

        writer = imageio.get_writer(path_out_vis, fps=target_fps)

        out_frame_id = 0
        pbar = tqdm(desc="Processing Video", total=duration_sec)

        for frame_id, frame in enumerate(reader):
            if frame_id % (fps // target_fps) != 0:
                continue
            else:
                out_frame_id += 1
                pbar.update(1)
            if out_frame_id > out_max_frames:
                break

            frame_pil = Image.fromarray(frame)
            pipe_out = pipe(
                frame_pil,
                match_input_resolution=False,
                latents=init_latents,
                processing_resolution=default_image_processing_resolution
            )

            if init_latents is None:
                init_latents = pipe_out.gaus_noise
            processed_frame = (pipe_out.prediction.clip(-1, 1) + 1) / 2
            processed_frame = processed_frame[0]
            _processed_frame = imageio.core.util.Array(processed_frame)
            writer.append_data(_processed_frame)
            
            yield (
                [frame_pil, processed_frame],
                None,
            )
    finally:

        if writer is not None:
            writer.close()

        if reader is not None:
            reader.close()

    yield (
        [frame_pil, processed_frame],
        [path_out_vis,]
    )


def run_demo_server(pipe):
    process_pipe_image = spaces.GPU(functools.partial(process_image, pipe))
    process_pipe_video = spaces.GPU(
        functools.partial(process_video, pipe), duration=120
    )

    gradio_theme = gr.themes.Default()

    with gr.Blocks(
        theme=gradio_theme,
        title="Stable Delight Estimation",
        css="""
            #download {
                height: 118px;
            }
            .slider .inner {
                width: 5px;
                background: #FFF;
            }
            .viewport {
                aspect-ratio: 4/3;
            }
            .tabs button.selected {
                font-size: 20px !important;
                color: crimson !important;
            }
            h1 {
                text-align: center;
                display: block;
            }
            h2 {
                text-align: center;
                display: block;
            }
            h3 {
                text-align: center;
                display: block;
            }
            .md_feedback li {
                margin-bottom: 0px !important;
            }
        """,
        head="""
            <script async src="https://www.googletagmanager.com/gtag/js?id=G-1FWSVCGZTG"></script>
            <script>
                window.dataLayer = window.dataLayer || [];
                function gtag() {dataLayer.push(arguments);}
                gtag('js', new Date());
                gtag('config', 'G-1FWSVCGZTG');
            </script>
        """,
    ) as demo:
        gr.Markdown(
            """
            # StableDelight: Removing Reflections from Textured Surfaces in a Single Image
            <p align="center">
        """
        )

        with gr.Tabs(elem_classes=["tabs"]):
            with gr.Tab("Image"):
                with gr.Row():
                    with gr.Column():
                        image_input = gr.Image(
                            label="Input Image",
                            type="filepath",
                        )
                        with gr.Row():
                            image_submit_btn = gr.Button(
                                value="Delightning", variant="primary"
                            )
                            image_reset_btn = gr.Button(value="Reset")
                    with gr.Column():
                        image_output_slider = ImageSlider(
                            label="Delight outputs",
                            type="filepath",
                            show_download_button=True,
                            show_share_button=True,
                            interactive=False,
                            elem_classes="slider",
                            position=0.25,
                        )

                Examples(
                    fn=process_pipe_image,
                    examples=sorted([
                        os.path.join("files", "image", name)
                        for name in os.listdir(os.path.join("files", "image"))
                    ]),
                    inputs=[image_input],
                    outputs=[image_output_slider],
                    cache_examples=False,
                    directory_name="examples_image",
                )

            with gr.Tab("Video"):
                with gr.Row():
                    with gr.Column():
                        video_input = gr.Video(
                            label="Input Video",
                            sources=["upload", "webcam"],
                        )
                        with gr.Row():
                            video_submit_btn = gr.Button(
                                value="Delighting", variant="primary"
                            )
                            video_reset_btn = gr.Button(value="Reset")
                    with gr.Column():
                        processed_frames = ImageSlider(
                            label="Realtime Visualization",
                            type="filepath",
                            show_download_button=True,
                            show_share_button=True,
                            interactive=False,
                            elem_classes="slider",
                            position=0.25,
                        )
                        video_output_files = gr.Files(
                            label="Delight outputs",
                            elem_id="download",
                            interactive=False,
                        )
                Examples(
                    fn=process_pipe_video,
                    examples=sorted([
                        os.path.join("files", "video", name)
                        for name in os.listdir(os.path.join("files", "video"))
                    ]),
                    inputs=[video_input],
                    outputs=[processed_frames, video_output_files],
                    directory_name="examples_video",
                    cache_examples=False,
                )

        ### Image tab
        image_submit_btn.click(
            fn=process_image_check,
            inputs=image_input,
            outputs=None,
            preprocess=False,
            queue=False,
        ).success(
            fn=process_pipe_image,
            inputs=[
                image_input,
            ],
            outputs=[image_output_slider],
            concurrency_limit=1,
        )

        image_reset_btn.click(
            fn=lambda: (
                None,
                None,
                None,
            ),
            inputs=[],
            outputs=[
                image_input,
                image_output_slider,
            ],
            queue=False,
        )

        ### Video tab

        video_submit_btn.click(
            fn=process_pipe_video,
            inputs=[video_input],
            outputs=[processed_frames, video_output_files],
            concurrency_limit=1,
        )

        video_reset_btn.click(
            fn=lambda: (None, None, None),
            inputs=[],
            outputs=[video_input, processed_frames, video_output_files],
            concurrency_limit=1,
        )

        ### Server launch

        demo.queue(
            api_open=False,
        ).launch(
            server_name="0.0.0.0",
            server_port=7860,
        )


def main():
    os.system("pip freeze")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    pipe = YosoDelightPipeline.from_pretrained(
        'weights/yoso-delight-v0-4-base', trust_remote_code=True, variant="fp16", 
        torch_dtype=torch.float16, t_start=0).to(device)
    # pipe.push_to_hub('Stable-X/yoso-delight-v0-4-base', variant="fp16")
    try:
        import xformers
        pipe.enable_xformers_memory_efficient_attention()
    except:
        pass  # run without xformers

    run_demo_server(pipe)


if __name__ == "__main__":
    main()
