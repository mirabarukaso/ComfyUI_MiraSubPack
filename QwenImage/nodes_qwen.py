# https://huggingface.co/Phr00t/Qwen-Image-Edit-Rapid-AIO/blob/main/fixed-textencode-node/nodes_qwen.py

import node_helpers
import comfy.utils
import math
from typing_extensions import override
from comfy_api.latest import ComfyExtension, io

CAT="Mira/SubPack/QwenImage"

class TextEncodeQwenImageEditMod(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="TextEncodeQwenImageEdit_MiraSubPack",
            display_name="Text Encode QwenImage Edit Mira",
            category=CAT,
            inputs=[
                io.Clip.Input("clip"),
                io.String.Input("prompt", multiline=True, dynamic_prompts=True),
                io.Vae.Input("vae", optional=True),
                io.Image.Input("image", optional=True),
            ],
            outputs=[
                io.Conditioning.Output(),
            ],
        )

    @classmethod
    def execute(cls, clip, prompt, vae=None, image=None) -> io.NodeOutput:
        ref_latent = None
        if image is None:
            images = []
        else:
            samples = image.movedim(-1, 1)
            total = int(1024 * 1024)

            scale_by = math.sqrt(total / (samples.shape[3] * samples.shape[2]))
            width = round(samples.shape[3] * scale_by)
            height = round(samples.shape[2] * scale_by)

            s = comfy.utils.common_upscale(samples, width, height, "area", "disabled")
            image = s.movedim(1, -1)
            images = [image[:, :, :, :3]]
            if vae is not None:
                ref_latent = vae.encode(image[:, :, :, :3])

        tokens = clip.tokenize(prompt, images=images)
        conditioning = clip.encode_from_tokens_scheduled(tokens)
        if ref_latent is not None:
            conditioning = node_helpers.conditioning_set_values(conditioning, {"reference_latents": [ref_latent]}, append=True)
        return io.NodeOutput(conditioning)


class TextEncodeQwenImageEditPlusMod(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="TextEncodeQwenImageEditPlus_MiraSubPack",
            display_name="Text Encode QwenImage Edit Plus Mira",
            category=CAT,
            inputs=[
                io.Clip.Input("clip"),
                io.String.Input("system_prompt", default="Describe key details of the input image (including any objects, characters, poses, facial features, clothing, setting, textures and style), then explain how the user's text instruction should alter, modify or recreate the image. Generate a new image that meets the user's requirements, which can vary from a small change to a completely new image using inputs as a guide.", multiline=True, dynamic_prompts=True),
                io.String.Input("prompt", multiline=True, dynamic_prompts=True),
                io.Vae.Input("vae", optional=True),
                io.Image.Input("image1", optional=True),
                io.Image.Input("image2", optional=True),
                io.Image.Input("image3", optional=True),
                io.Image.Input("image4", optional=True),
                io.Latent.Input("target_latent", optional=True),
                io.Combo.Input("crop_method", default="disabled", options=["disabled", "center"]),   
                io.Int.Input("vl_image_size", default=512, optional=False, min=256, max=2048, step=8),
                io.Combo.Input("reference_latents_method", default="none", options=["none", "offset", "index", "uxo/uno", "index_timestep_zero"]),   
                io.Int.Input("reference_latents_size", default=1024, optional=True, min=256, max=4096, step=32),                
            ],
            outputs=[
                io.Conditioning.Output(),
            ],
        )

    @classmethod
    def execute(cls, clip, system_prompt, prompt, vae=None, image1=None, image2=None, image3=None, image4=None, target_latent=None, crop_method="center", vl_image_size=512, reference_latents_size=1024, reference_latents_method="none") -> io.NodeOutput:
        ref_latents = []
        images = [image1, image2, image3, image4]
        images_vl = []
        llama_template = "<|im_start|>system\n__REPLACE_SYSTEM__<|im_end|>\n<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n".replace("__REPLACE_SYSTEM__", system_prompt)
        image_prompt = ""

        for i, image in enumerate(images):
            if image is not None:
                samples = image.movedim(-1, 1)
                total = int(vl_image_size * vl_image_size)

                scale_by = math.sqrt(total / (samples.shape[3] * samples.shape[2]))
                width = round(samples.shape[3] * scale_by)
                height = round(samples.shape[2] * scale_by)

                s = comfy.utils.common_upscale(samples, width, height, "lanczos", crop_method)
                images_vl.append(s.movedim(1, -1))
                if vae is not None:
                    twidth = 0
                    theight = 0
                    if target_latent is None:                        
                        total = int(reference_latents_size * reference_latents_size)
                        scale_by = math.sqrt(total / (samples.shape[3] * samples.shape[2]))
                        twidth = round(samples.shape[3] * scale_by / 32.0) * 32
                        theight = round(samples.shape[2] * scale_by / 32.0) * 32
                    else:
                        twidth = target_latent["samples"].shape[-1] * 8
                        theight = target_latent["samples"].shape[-2] * 8
                    print("twidth, theight", twidth, theight)
                    print("samples.shape[3], samples.shape[2]", samples.shape[3], samples.shape[2])
                    if samples.shape[3] == twidth and samples.shape[2] == theight:
                        s = samples
                    else:
                        s = comfy.utils.common_upscale(samples, twidth, theight, "lanczos", crop_method)
                    ref_latents.append(vae.encode(s.movedim(1, -1)[:, :, :, :3]))

                image_prompt += "Picture {}: <|vision_start|><|image_pad|><|vision_end|>".format(i + 1)

        tokens = clip.tokenize(image_prompt + prompt, images=images_vl, llama_template=llama_template)
        conditioning = clip.encode_from_tokens_scheduled(tokens)
        if len(ref_latents) > 0:
            conditioning = node_helpers.conditioning_set_values(conditioning, {"reference_latents": ref_latents}, append=True)

        if reference_latents_method != "none":
            '''
            offset: Samples reference latents at (main timestep ± offset), making reference guidance move dynamically with the denoising progress.
            index: Uses a fixed absolute timestep index to read reference latents, independent of the main generation step, giving stable, static influence.
            uxo/uno: Injects reference in U-Net feature space instead of by timestep; uxo spreads influence deeper/stronger, uno keeps it lighter and closer to original U-Net layers.
            index_timestep_zero: Always uses reference latents at timestep 0 (initial noise) as a prior/seed, shaping the global structure and noise distribution rather than guiding mid-denoise steps.
            '''
            
            # refer https://github.com/comfyanonymous/ComfyUI/blob/d9a76cf66e3fc6b0047692a07bc1d24f20e16e20/comfy_extras/nodes_flux.py#L152
            if "uxo" in reference_latents_method or "uso" in reference_latents_method:
                reference_latents_method = "uxo"
            
            conditioning = node_helpers.conditioning_set_values(conditioning, {"reference_latents_method": reference_latents_method})
        return io.NodeOutput(conditioning)

class QwenExtensionMod(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [
            TextEncodeQwenImageEditMod,
            TextEncodeQwenImageEditPlusMod,
        ]


async def comfy_entrypoint() -> QwenExtensionMod:
    return QwenExtensionMod()