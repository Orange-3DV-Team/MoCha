<div align="center" style="margin-top: 0px; margin-bottom: 0px;">
<img src="https://github.com/user-attachments/assets/a119b789-9e40-48e5-ab86-5a2a0ee8f221" width="50%"/>
</div>

# MoCha: End-to-End Video Character Replacement without Structural Guidance

<div align="center">
  <a href="https://github.com/Orange-3DV-Team/MoCha-Code"><img src="https://img.shields.io/static/v1?label=Code&message=Github&color=blue"></a> &ensp;
  <a href="https://orange-3dv-team.github.io/MoCha"><img src="https://img.shields.io/static/v1?label=Project%20Page&message=Web&color=green"></a> &ensp;
  <a href="https://huggingface.co/Orange-3DV-Team/MoCha"><img src="https://img.shields.io/static/v1?label=MoCha&message=HuggingFace&color=yellow"></a> &ensp;
  <a href="https://github.com/kijai/ComfyUI-WanVideoWrapper/pull/1501"><img src="https://img.shields.io/static/v1?label=workflow&message=ComfyUI&color=purple"></a> &ensp;
  <!-- <a href="https://github.com/user-attachments/assets/3544c61d-87ab-48ba-95bf-f9ec6fce3327"><img src=https://img.shields.io/badge/WeChat-07C160?label=Group&message=WeChat></a> &ensp; -->
</div>

https://github.com/user-attachments/assets/5fe4d80f-3aa4-4d45-8a29-a4d51f6d339d

https://github.com/user-attachments/assets/b9a5525f-cebf-4295-9738-7dd755daacf4


## 🔥 Updates
- __[2025.10.21]__: Try our work with [ComfyUI workflow](https://github.com/kijai/ComfyUI-WanVideoWrapper/pull/1501)!
- __[2025.10.21]__: Release the [inference code](https://github.com/Orange-3DV-Team/MoCha-Code).
- __[2025.10.20]__: Release the [project page](https://orange-3dv-team.github.io/MoCha).



## 📝 Abstract
Controllable video character replacement with a user-provided one remains a challenging problem due to the lack of qualified paired-video data. 
Prior works have predominantly adopted a reconstruction-based paradigm reliant on per-frame masks and explicit structural guidance (e.g., pose, depth). This reliance, however, renders them fragile in complex scenarios involving occlusions, rare poses, character-object interactions, or complex illumination, often resulting in visual artifacts and temporal discontinuities.
In this paper, we propose MoCha, a novel framework that bypasses these limitations, which requires only a single first-frame mask and re-renders the character by unifying different conditions into a single token stream.
Further, MoCha adopts a condition-aware RoPE to support multi-reference images and variable-length video generation.
To overcome the data bottleneck, we construct a comprehensive data synthesis pipeline to collect qualified paired-training videos. Extensive experiments show that our method substantially outperforms existing state-of-the-art approaches.

## ☕ Getting Started with MoCha

### Inference

Step 1: Clone this repository

```shell
git clone https://github.com/Orange-3DV-Team/MoCha.git
cd MoCha
```

Step 2: Set up the environment

```shell
# 1. Create conda environment
conda create -n MoCha python==3.10

# 2. Activate the environment
conda activate MoCha

# 3. Install pip dependencies
pip install -r requirements.txt
```

Step 3: Download the pretrained checkpoints
1. Download the pre-trained Wan2.1 models from [huggingface](https://huggingface.co/Wan-AI/Wan2.1-T2V-14B)
2. Download the pre-trained MoCha checkpoint

Please download from [huggingface](https://huggingface.co/Orange-3DV-Team/MoCha) and place it in ```./checkpoints```.

Step 4: Test the example videos

```shell
python inference_mocha.py
```

### Test your own video

To start your own character replacement with MoCha, the following three inputs are required:

- Source Video: The original video with the character to be replaced.
- Designation Mask for the First Frame: A mask marking the source character to be replaced in the first frame of Source Video.
- Reference Images: Reference Images of the new character for replacement with clean background. We recommend uploading at least one high-quality, front-facing facial close-up.


https://github.com/user-attachments/assets/3ae70cc1-9bf2-47ca-b6c2-bf7c84f0054f


Then organize your test data following the structure of the ```./data/test_data.csv```.

- ```source_video```: Path to Source Video.
- ```source_mask```: Path to Designation Mask.
- ```reference_1```: Path to first Reference Image.
- ```reference_2```: Path to second Reference Image. This image needs to be a high-quality, front-facing facial close-up. (You can even zoom up your first reference image!) **If you really cannot provide this reference image, leave it as** ```None```.

Finally, test your videos by:

```shell
python inference_mocha.py --data_path path/to/your/data.csv
```

## 🌟 Citation

Please leave us a star 🌟 and cite our repo if you find our work helpful.
```
@inproceedings{orange2025mocha,
  title={MoCha: End-to-End Video Character Replacement without Structural Guidance}, 
  author={Orange Team},
  booktitle = {GitHub Repository},
  year={2025},
  url={https://github.com/Orange-3DV-Team/MoCha}, 
}
```






