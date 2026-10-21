# ComfyUI_MiraSubPack
More test nodes for my [ComfyUI_Mira](https://github.com/mirabarukaso/ComfyUI_Mira)


## Model (Mira/SubPack/Model)
- **Sage FP16 Switch (Mira SubPack)** `SageFP16Switch_MiraSubPack`: place after the model loader. Sets this model's compute dtype to fp16 and (optional) casts attention q/k/v to fp16 for SageAttention. Per-model, no global patch; bypass to disable. Useful on GPUs without bf16 (e.g. Turing).
- **Load Diffusion Model FP16 Storage (Mira SubPack)** `UNETLoaderFP16Storage_MiraSubPack`: UNETLoader variant storing weights as fp16 (half VRAM of fp32).

### SageFP16Switch benchmark: MOSS-Image SFT, bf16 vs int8 (Titan RTX, sm75)
- GPU: Titan RTX (Turing sm75, no hardware bf16). Orig env: Python 3.13, torch 2.9.0+cu130, sageattention 2.2.0. Test venv: torch 2.14.1+cu132, triton-windows 3.2, sageattention 1.0.6.
- Workflow: MOSS-Image SFT + sft 4-step LoRA v2.0 + style LoRA, SkipLayerGuidance (single layer 22), Qwen3.5-9B int8 text encoder, flux2 VAE, 1536x2304, 6 steps, euler/simple, cfg 2.0.
- Seeds: 777 warm-up (discarded), timed on 2056035064 and 2056035065. s/image = mean server execution time of the 2 timed runs (model load excluded).
- LPIPS (AlexNet), mean of both seeds, vs **bf16 / default attention / orig env**. Sage rows use `SageFP16Switch_MiraSubPack` (compute_fp16 + sage_cast).

| Weights | Env / torch | Attention | Compute dtype | s/image | Speedup | LPIPS |
|---|---|---|---|---|---|---|
| bf16 | orig / 2.9.0 | default | fp32 (bf16 upcast) | 94.0 | 1.00x | baseline |
| bf16 | orig / 2.9.0 | `--use-pytorch-cross-attention` | fp32 | 94.5 | 0.99x | 0.0000 |
| bf16 | venv / 2.14.1 | sage 1.0.6 + switch | fp16 | 20.1 | 4.67x | 0.1856 |
| bf16 | orig / 2.9.0 | sage 2.2.0 + switch | fp16 | 21.5 | 4.38x | 0.1902 |
| int8 | orig / 2.9.0 | default | fp32 (manual cast) | 18.2 | 5.16x | 0.2304 |
| int8 | orig / 2.9.0 | `--use-pytorch-cross-attention` | fp32 (manual cast) | 18.3 | 5.13x | 0.2304 |
| int8 | venv / 2.14.1 | sage 1.0.6 + switch | fp16 | 13.7 | 6.89x | 0.2440 |
| int8 | orig / 2.9.0 | sage 2.2.0 + switch | fp16 | 14.3 | 6.59x | 0.2378 |

Notes:
- Default attention already resolves to PyTorch SDPA on this build, so "default" and `--use-pytorch-cross-attention` are the same path (identical images).
- On sm75, sage 1.0.6 and sage 2.2.0 run the same Triton kernel; the small speed gap also includes different torch versions.
- int8 is much faster than bf16 here likely because the int8 weights avoid the full fp32 matmul path, but this was not profiled.
- No sage failures or fallbacks; the switch confirmed fp16 q/k/v in all sage runs.

<img src="examples/SageFP16Switch_MOSS_sft_benchmark.jpg" width="900">

