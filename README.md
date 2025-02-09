# ComfyUI-RealESRGAN_Upscaler [![ComfyUI - Homepage](https://img.shields.io/badge/ComfyUI-Homepage-aa00ee)](https://github.com/comfyanonymous/ComfyUI) [![ComfyUI - Manager](https://img.shields.io/badge/ComfyUI-Manager-2aeeef)](https://github.com/ltdrdata/ComfyUI-Manager)


> [!IMPORTANT]  
> <p align="justify">🚧 This documentation is still under 
> construction. The development of the upscaler is a ongoing 
> activity. There might be small differenes in comparison of 
> node and documentation.</p>

> [!CAUTION]
> <p align="justify">Please note, that the node will run, only
> if an error in one Python package file is fixed. See the 
> related section for informations on this topic.</p> 

# Preface

<p align="justify">This node uses the RealESRGAN model from
xinntao [1]. This is my personal favourite upscaling model and 
upscaling approach.</p>

## Upscaling

<p align="justify">One can set the scaling factor in steps of 0.1
in the node. For the moment there is no known limit for the scaling
factor. I used for example an unrealistic scaling factor of 30.0 
for upscaling of a test image. The limiting factor is in this case 
the size of the image file.</p>

## Node Preview

<img src="./images/node_preview.png" alt="node preview" width="512">
<p><i>Figure 1: Main node preview</i></p>

![Bildschirmfoto vom 2025-02-08 16-32-33](https://github.com/user-attachments/assets/eefc86af-87fb-4fe0-bf41-df37dc93eb5a)

## Workflow Preview

![image](https://github.com/user-attachments/assets/8ac47db6-6293-44d3-98e0-aae302bab020)

## Final Words

Have fun. Be inspired!

# Reference

[1] https://github.com/xinntao/Real-ESRGAN
