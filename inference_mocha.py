import sys
import torch
import torch.nn as nn
from diffsynth import ModelManager, WanVideoMoChaPipeline, save_video, VideoData
from diffsynth.set_condition_branch import set_stand_in
import torch, os, imageio, argparse
from torchvision.transforms import v2
from einops import rearrange
import pandas as pd
import torchvision
from PIL import Image
import random
import numpy as np
import json

class VideoRefDataset(torch.utils.data.Dataset):
    def __init__(self, data_path, args, max_num_frames=161, frame_interval=1, num_frames=161, height=480, width=832):
        metadata = pd.read_csv(data_path)
        self.video_path = metadata["source_video"]
        self.mask_path = metadata["source_mask"]
        self.ref_path_1 = metadata["reference_1"]
        self.ref_path_2 = []
        for ref_name in metadata["reference_2"]:
            if pd.isna(ref_name) or ref_name == 'None':
                self.ref_path_2.append("None")
            else:
                self.ref_path_2.append(ref_name)
        
        self.max_num_frames = max_num_frames
        self.frame_interval = frame_interval
        self.num_frames = num_frames
        self.height = height
        self.width = width
        self.args = args
            
        self.frame_process = v2.Compose([
            v2.CenterCrop(size=(height, width)),
            v2.Resize(size=(height, width), antialias=True),
            v2.ToTensor(),
            v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])

        self.mask_process = v2.Compose([
            v2.CenterCrop(size=(height // 8, width // 8)),
            v2.Resize(size=(height // 8, width // 8), antialias=True),
            v2.ToTensor(),
            v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])
        
        
    def crop_and_resize(self, image, isMask = False):
        width, height = image.size
        scale = max(self.width / width, self.height / height)
        if isMask:
            scale /= 8
        image = torchvision.transforms.functional.resize(
            image,
            (round(height*scale), round(width*scale)),
            interpolation=torchvision.transforms.InterpolationMode.BILINEAR
        )
        return image


    def load_frames_using_imageio(self, file_path, max_num_frames, start_frame_id, interval, num_frames, frame_process):
        reader = imageio.get_reader(file_path)
        if reader.count_frames() < max_num_frames or reader.count_frames() - 1 < start_frame_id + (num_frames - 1) * interval:
            num_frames = 1 + (reader.count_frames() - 1) //4 * 4
        
        frames = []
        first_frame = None
        for frame_id in range(num_frames):
            frame = reader.get_data(start_frame_id + frame_id * interval)
            frame = Image.fromarray(frame)
            frame = self.crop_and_resize(frame)
            if first_frame is None:
                first_frame = np.array(frame)
            frame = frame_process(frame)
            frames.append(frame)
        reader.close()

        frames = torch.stack(frames, dim=0)
        frames = rearrange(frames, "T C H W -> C T H W")

        return frames


    def is_image(self, file_path):
        file_ext_name = file_path.split(".")[-1]
        if file_ext_name.lower() in ["jpg", "jpeg", "png", "webp"]:
            return True
        return False

    def load_image_frame(self, file_path, isMask=False):
        image = Image.open(file_path).convert('RGB')
        image = self.crop_and_resize(image, isMask)
        if isMask:
            image = self.mask_process(image)
        else:
            image = self.frame_process(image)
        image = image.unsqueeze(1)
        return image
    
    def load_video(self, file_path, isMask = False):
        if self.is_image(file_path):
            if isMask:
                return self.load_image_frame(file_path, isMask=True)
            else:
                return self.load_image_frame(file_path)

        frames = self.load_frames_using_imageio(file_path, self.max_num_frames, 0, self.frame_interval, self.num_frames, self.frame_process)
        return frames

    def __getitem__(self, data_id):
        video_path = self.video_path[data_id]
        mask_path = self.mask_path[data_id]

        video = self.load_video(video_path)
        if video is None:
            raise ValueError(f"{video_path} is not a valid video.")

        source_mask = self.load_video(mask_path, isMask=True)
        mask_cond = torch.sign(source_mask[0:1, 0:1, :, :]).repeat(16, 1, 1, 1)

        ref_path_1 = self.ref_path_1[data_id]
        ref_path_2 = self.ref_path_2[data_id]

        first_ref = self.load_video(ref_path_1)

        if pd.isna(ref_path_2) or ref_path_2 == 'None':
            second_ref = "None"
        else:
            second_ref = self.load_video(ref_path_2)

        data = {"video": video, "video_path": video_path, "mask": mask_cond, "first_ref": first_ref, "second_ref": second_ref}

        return data
    

    def __len__(self):
        return len(self.video_path)

def parse_args():
    parser = argparse.ArgumentParser(description="MoCha Inference Code")
    parser.add_argument(
        "--data_path",
        type=str,
        default="./data/test_data.csv",
        help="The path of the Video.",
    )
    parser.add_argument(
        "--ckpt_path",
        type=str,
        default="./checkpoints/step18500.ckpt",
        help="Path to save the model.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./results",
        help="Path to save the results.",
    )
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=1,
        help="Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process.",
    )
    parser.add_argument(
        "--cfg_scale",
        type=float,
        default=5.0,
    )
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()

    # Load Wan2.1 pre-trained models
    model_manager = ModelManager(torch_dtype=torch.bfloat16, device="cpu")
    model_manager.load_models([
        ["/path/to/diffusion_pytorch_model-00001-of-00006.safetensors", "/path/to/diffusion_pytorch_model-00002-of-00006.safetensors", "/path/to/diffusion_pytorch_model-00003-of-00006.safetensors", "/path/to/diffusion_pytorch_model-00004-of-00006.safetensors", "/path/to/diffusion_pytorch_model-00005-of-00006.safetensors", "/path/to/diffusion_pytorch_model-00006-of-00006.safetensors"],
        "/path/to/Wan2.1-T2V-14B/models_t5_umt5-xxl-enc-bf16.pth",
        "/path/to/Wan2.1-T2V-14B/Wan2.1_VAE.pth",
    ])
    pipe = WanVideoMoChaPipeline.from_model_manager(model_manager, device="cuda")

    # Load checkpoint
    state_dict = torch.load(args.ckpt_path, map_location="cpu")
    pipe.dit.load_state_dict(state_dict, strict=True)
    pipe.to("cuda")
    pipe.to(dtype=torch.bfloat16)

    output_dir = args.output_dir
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Prepare Dataset (Source video, Mask, Reference Image)
    dataset = VideoRefDataset(
        args.data_path,
        args,
    )
    dataloader = torch.utils.data.DataLoader(
        dataset,
        shuffle=False,
        batch_size=1,
        num_workers=args.dataloader_num_workers
    )

    # Inference
    for batch_idx, batch in enumerate(dataloader):
        source_video = batch["video"]
        source_mask = batch["mask"]
        first_ref = batch["first_ref"]
        second_ref = batch["second_ref"]
        cond_path_name = os.path.basename(batch["video_path"][0])
        if second_ref[0] == "None":
            second_ref = None

        video = pipe(
            prompt=" ",
            negative_prompt="色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走",
            source_video=source_video,
            source_mask=source_mask,
            first_ref=first_ref,
            second_ref=second_ref,
            cfg_scale=args.cfg_scale,
            num_inference_steps=6,
            num_frames=81,
            seed=0, tiled=True
        )

        save_video(video, os.path.join(output_dir, f"{os.path.splitext(cond_path_name)[0]}_replaced.mp4"), fps=30, quality=5)