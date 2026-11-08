import torch
from contextlib import contextmanager
from comfy_api.latest import io

CAT = "Mira/SubPack/Utils"


@contextmanager
def safe_vae_context(vae):
    """Temporarily disable fused kitchen fp16_conv3d path for VAE encode/decode.

    On sm_75 (e.g. TITAN RTX), Flux2 fused VAE + allow_fp16_accumulation can hit
    unsupported CUTLASS Mma<16,8,16> and CUDA 719. Restores flags after the call
    so global --fast fp16_accumulation still applies to sampling.
    """
    mm = torch.backends.cuda.matmul
    had_acc = hasattr(mm, "allow_fp16_accumulation")
    old_acc = mm.allow_fp16_accumulation if had_acc else None
    if had_acc:
        mm.allow_fp16_accumulation = False

    restored = []
    fsm = getattr(vae, "first_stage_model", None)
    if fsm is not None:
        for m in fsm.modules():
            if hasattr(m, "fused_norm") and m.fused_norm:
                restored.append((m, True))
                m.fused_norm = False
    try:
        yield
    finally:
        for m, v in restored:
            m.fused_norm = v
        if had_acc:
            mm.allow_fp16_accumulation = old_acc


def _encode_one(vae, pixels, mode, tile_size, overlap, temporal_size, temporal_overlap):
    if mode == "tiled":
        return vae.encode_tiled(
            pixels,
            tile_x=tile_size,
            tile_y=tile_size,
            overlap=overlap,
            tile_t=temporal_size,
            overlap_t=temporal_overlap,
        )
    return vae.encode(pixels)


def _decode_tiled_args(vae, tile_size, overlap, temporal_size, temporal_overlap):
    if tile_size < overlap * 4:
        overlap = tile_size // 4
    if temporal_size < temporal_overlap * 2:
        temporal_overlap = temporal_overlap // 2
    temporal_compression = vae.temporal_compression_decode()
    if temporal_compression is not None:
        temporal_size = max(2, temporal_size // temporal_compression)
        temporal_overlap = max(1, min(temporal_size // 2, temporal_overlap // temporal_compression))
    else:
        temporal_size = None
        temporal_overlap = None
    compression = vae.spacial_compression_decode()
    return (
        tile_size // compression,
        tile_size // compression,
        overlap // compression,
        temporal_size,
        temporal_overlap,
    )


def _decode_one(vae, latent, mode, tile_size, overlap, temporal_size, temporal_overlap):
    if mode == "tiled":
        tile_x, tile_y, ov, tile_t, overlap_t = _decode_tiled_args(
            vae, tile_size, overlap, temporal_size, temporal_overlap
        )
        images = vae.decode_tiled(
            latent,
            tile_x=tile_x,
            tile_y=tile_y,
            overlap=ov,
            tile_t=tile_t,
            overlap_t=overlap_t,
        )
    else:
        images = vae.decode(latent)
    if len(images.shape) == 5:
        images = images.reshape(-1, images.shape[-3], images.shape[-2], images.shape[-1])
    return images


class VAEEncode_Mira(io.ComfyNode):
    """
    Encode images to latent with batch VRAM optimization, safe fused-VAE bypass,
    and optional tiled encode.
    """
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="VAEEncode_MiraSubPack",
            display_name="VAE Encode (Mira SubPack)",
            category=CAT,
            description=(
                "Encode images to latent. Batch>1 processes one image at a time. "
                "Temporarily disables fused kitchen fp16 VAE path (sm_75 safe). "
                "mode=tiled uses VAE encode_tiled."
            ),
            inputs=[
                io.Image.Input("pixels"),
                io.Vae.Input("vae"),
                io.Combo.Input(
                    "mode",
                    options=["direct", "tiled"],
                    default="direct",
                    tooltip="direct: vae.encode; tiled: vae.encode_tiled (lower VRAM peak).",
                ),
                io.Int.Input(
                    "tile_size",
                    default=512,
                    min=64,
                    max=4096,
                    step=64,
                    advanced=True,
                    tooltip="Pixel tile size for tiled encode.",
                ),
                io.Int.Input(
                    "overlap",
                    default=64,
                    min=0,
                    max=4096,
                    step=32,
                    advanced=True,
                    tooltip="Pixel overlap for tiled encode.",
                ),
                io.Int.Input(
                    "temporal_size",
                    default=64,
                    min=8,
                    max=4096,
                    step=4,
                    advanced=True,
                    tooltip="Video VAEs only: frames to encode at a time.",
                ),
                io.Int.Input(
                    "temporal_overlap",
                    default=8,
                    min=4,
                    max=4096,
                    step=4,
                    advanced=True,
                    tooltip="Video VAEs only: frame overlap.",
                ),
            ],
            outputs=[
                io.Latent.Output(display_name="samples"),
            ],
            is_output_node=True,
        )

    @classmethod
    def execute(
        cls,
        pixels,
        vae,
        mode="direct",
        tile_size=512,
        overlap=64,
        temporal_size=64,
        temporal_overlap=8,
    ) -> io.NodeOutput:
        batch_size = pixels.shape[0]

        with safe_vae_context(vae):
            if batch_size == 1:
                latent = _encode_one(
                    vae, pixels, mode, tile_size, overlap, temporal_size, temporal_overlap
                )
                samples = latent["samples"] if isinstance(latent, dict) else latent
                return io.NodeOutput({"samples": samples})

            latent_list = []
            for i in range(batch_size):
                single_latent = _encode_one(
                    vae,
                    pixels[i : i + 1],
                    mode,
                    tile_size,
                    overlap,
                    temporal_size,
                    temporal_overlap,
                )
                samples = (
                    single_latent["samples"]
                    if isinstance(single_latent, dict)
                    else single_latent
                )
                latent_list.append(samples.cpu() if hasattr(samples, "cpu") else samples)

            combined = torch.cat(latent_list, dim=0)
            return io.NodeOutput({"samples": combined})


class VAEDecode_Mira(io.ComfyNode):
    """
    Decode latents to images with batch VRAM optimization, safe fused-VAE bypass,
    and optional tiled decode.
    """
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="VAEDecode_MiraSubPack",
            display_name="VAE Decode (Mira SubPack)",
            category=CAT,
            description=(
                "Decode latents to images. Batch>1 processes one latent at a time. "
                "Temporarily disables fused kitchen fp16 VAE path (sm_75 safe). "
                "mode=tiled uses VAE decode_tiled."
            ),
            inputs=[
                io.Latent.Input("samples"),
                io.Vae.Input("vae"),
                io.Combo.Input(
                    "mode",
                    options=["direct", "tiled"],
                    default="direct",
                    tooltip="direct: vae.decode; tiled: vae.decode_tiled (lower VRAM peak).",
                ),
                io.Int.Input(
                    "tile_size",
                    default=512,
                    min=64,
                    max=4096,
                    step=32,
                    advanced=True,
                    tooltip="Pixel tile size for tiled decode (divided by VAE compression).",
                ),
                io.Int.Input(
                    "overlap",
                    default=64,
                    min=0,
                    max=4096,
                    step=32,
                    advanced=True,
                    tooltip="Pixel overlap for tiled decode (divided by VAE compression).",
                ),
                io.Int.Input(
                    "temporal_size",
                    default=64,
                    min=8,
                    max=4096,
                    step=4,
                    advanced=True,
                    tooltip="Video VAEs only: frames to decode at a time.",
                ),
                io.Int.Input(
                    "temporal_overlap",
                    default=8,
                    min=4,
                    max=4096,
                    step=4,
                    advanced=True,
                    tooltip="Video VAEs only: frame overlap.",
                ),
            ],
            outputs=[
                io.Image.Output(display_name="image"),
            ],
            is_output_node=True,
        )

    @classmethod
    def execute(
        cls,
        samples,
        vae,
        mode="direct",
        tile_size=512,
        overlap=64,
        temporal_size=64,
        temporal_overlap=8,
    ) -> io.NodeOutput:
        latent = samples["samples"]
        if hasattr(latent, "is_nested") and latent.is_nested:
            latent = latent.unbind()[0]

        batch_size = latent.shape[0]

        with safe_vae_context(vae):
            if batch_size == 1:
                images = _decode_one(
                    vae, latent, mode, tile_size, overlap, temporal_size, temporal_overlap
                )
                return io.NodeOutput(images)

            image_list = []
            for i in range(batch_size):
                single_image = _decode_one(
                    vae,
                    latent[i : i + 1],
                    mode,
                    tile_size,
                    overlap,
                    temporal_size,
                    temporal_overlap,
                )
                image_list.append(single_image.cpu())

            return io.NodeOutput(torch.cat(image_list, dim=0))
