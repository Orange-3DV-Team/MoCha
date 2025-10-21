<div align="center" style="margin-top: 0px; margin-bottom: 0px;">
<img src="https://github.com/user-attachments/assets/a119b789-9e40-48e5-ab86-5a2a0ee8f221" width="50%"/>
</div>

# MoCha: End-to-End Video Character Replacement without Structural Guidance

<div align="center">
  <a href="https://github.com/Orange-3DV-Team/MoCha-Code"><img src="https://img.shields.io/static/v1?label=Code&message=Github&color=blue"></a> &ensp;
  <a href="https://orange-3dv-team.github.io/MoCha"><img src="https://img.shields.io/static/v1?label=Project%20Page&message=Web&color=green"></a> &ensp;
  <a href="https://huggingface.co/Orange-3DV-Team/MoCha"><img src="https://img.shields.io/static/v1?label=MoCha&message=HuggingFace&color=yellow"></a> &ensp;
</div>

https://github.com/user-attachments/assets/00d23513-2d7e-4ebd-b922-420d90fdf340





## 🔥 Updates
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
1. Download the pre-trained Wan2.1 models
2. Download the pre-trained MoCha checkpoint

Please download from [huggingface](https://huggingface.co/Orange-3DV-Team/MoCha) and place it in ```./checkpoints```.

Step 4: Test the example videos

```shell
python inference_mocha.py
```

### Test your own video

If you want to test your own videos, you need to prepare your test data following the structure of the ```example_test_data``` folder.


If you want to test your own videos, you need to prepare your test data following the structure of the ```example_test_data``` folder. This includes N mp4 videos, each with at least 81 frames, and a ```metadata.csv``` file that stores their paths and corresponding captions. You can refer to the [Prompt Extension section](https://github.com/Wan-Video/Wan2.1?tab=readme-ov-file#2-using-prompt-extension) in Wan2.1 for guidance on preparing video captions. 

```shell
python inference_recammaster.py --cam_type 1 --dataset_path path/to/your/data
```

## 🌟 Citation

Please leave us a star 🌟 and cite our repo if you find our work helpful.
```
@article{bai2025recammaster,
  title={ReCamMaster: Camera-Controlled Generative Rendering from A Single Video},
  author={Bai, Jianhong and Xia, Menghan and Fu, Xiao and Wang, Xintao and Mu, Lianrui and Cao, Jinwen and Liu, Zuozhu and Hu, Haoji and Bai, Xiang and Wan, Pengfei and others},
  journal={arXiv preprint arXiv:2503.11647},
  year={2025}
}
```


