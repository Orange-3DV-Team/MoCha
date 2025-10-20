<div align="center" style="margin-top: 0px; margin-bottom: 0px;">
<img src="https://github.com/user-attachments/assets/a119b789-9e40-48e5-ab86-5a2a0ee8f221" width="50%"/>
</div>

# MoCha: End-to-End Video Character Replacement without Structural Guidance

<div align="center">
  <a href="https://github.com/Orange-3DV-Team/MoCha-Code"><img src="https://img.shields.io/static/v1?label=Code&message=Github&color=blue"></a> &ensp;
  <a href="https://orange-3dv-team.github.io/MoCha"><img src="https://img.shields.io/static/v1?label=Project%20Page&message=Web&color=green"></a> &ensp;
  <a href=""><img src="https://img.shields.io/static/v1?label=MoCha&message=HuggingFace&color=yellow"></a> &ensp;
</div>

## 🔥 Updates
- __[2025.10.20]__: Release the [project page](https://orange-3dv-team.github.io/MoCha).



## Abstract
Controllable video character replacement with a user-provided one remains a challenging problem due to the lack of qualified paired-video data. 
Prior works have predominantly adopted a reconstruction-based paradigm reliant on per-frame masks and explicit structural guidance (e.g., pose, depth). This reliance, however, renders them fragile in complex scenarios involving occlusions, rare poses, character-object interactions, or complex illumination, often resulting in visual artifacts and temporal discontinuities.
In this paper, we propose MoCha, a novel framework that bypasses these limitations, which requires only a single first-frame mask and re-renders the character by unifying different conditions into a single token stream.
Further, MoCha adopts a condition-aware RoPE to support multi-reference images and variable-length video generation.
To overcome the data bottleneck, we construct a comprehensive data synthesis pipeline to collect qualified paired-training videos. Extensive experiments show that our method substantially outperforms existing state-of-the-art approaches.



