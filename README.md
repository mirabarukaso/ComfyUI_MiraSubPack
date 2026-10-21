# ComfyUI_MiraSubPack
More test nodes for my [ComfyUI_Mira](https://github.com/mirabarukaso/ComfyUI_Mira)


## Model (Mira/SubPack/Model)
- **Sage FP16 Switch (Mira SubPack)** `SageFP16Switch_MiraSubPack`: place after the model loader. Sets this model's compute dtype to fp16 and (optional) casts attention q/k/v to fp16 for SageAttention. Per-model, no global patch; bypass to disable. Useful on GPUs without bf16 (e.g. Turing).
- **Load Diffusion Model FP16 Storage (Mira SubPack)** `UNETLoaderFP16Storage_MiraSubPack`: UNETLoader variant storing weights as fp16 (half VRAM of fp32).

