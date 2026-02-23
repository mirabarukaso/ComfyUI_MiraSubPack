import torch
import math
import comfy.sample
import comfy.samplers
import comfy.utils
import latent_preview
from comfy_api.latest import io
import node_helpers

CAT = "Mira/SubPack/Image Tiled Upscaler"

TILE_ALIGMENT_TIP = "Align tile dimensions to multiples of this value (e.g., 8 SDXL, 16 FLUX.2, 32 Qwen Image)."

# ==========================================
# Custom Node Type Definition
# ==========================================
@io.comfytype(io_type="mira_image_tiled_upscaler_pipeline")
class MiraITUPipeline:
    Type = list  # Python type hint

    class Input(io.Input):
        def __init__(self, id: str, **kwargs):
            super().__init__(id, **kwargs)

    class Output(io.Output):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
MiraITUPipeline = io.Custom("mira_image_tiled_upscaler_pipeline")

class MiraITUPipelineExtract(io.ComfyNode):
    """
    Extract Upscaled Pipeline Info
    """    
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MiraITUPipelineExtract_MiraSubPack",
            display_name="Mira ITU Pipeline Extract",
            category=CAT,
            description="Extract upscaled pipeline info.",
            inputs=[
                MiraITUPipeline.Input("mira_itu_pipeline", optional=False, tooltip="Upscaled pipeline info."),
            ],
            outputs=[
                io.Int.Output(display_name="full_width"),
                io.Int.Output(display_name="full_height"),
                io.Int.Output(display_name="tile_width"),
                io.Int.Output(display_name="tile_height"),
                io.Int.Output(display_name="overlap"),
                io.Float.Output(display_name="overlap_feather_rate"),
                io.Int.Output(display_name="pixel_alignment"),
            ],
        )
    
    @classmethod
    def execute(cls, mira_itu_pipeline) -> io.NodeOutput:
        (full_width, full_height, tile_width, tile_height, overlap, overlap_feather_rate, pixel_alignment) = mira_itu_pipeline
        return io.NodeOutput(full_width, full_height, tile_width, tile_height, overlap, overlap_feather_rate, pixel_alignment)
    
class MiraITUPipelineCombine(io.ComfyNode):
    """
    Combine Upscaled Pipeline Info
    """    
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MiraITUPipelineCombine_MiraSubPack",
            display_name="Mira ITU Pipeline Combine",
            category=CAT,
            description="Combine upscaled pipeline info.",
            inputs=[
                io.Int.Input("full_width", max=65536, tooltip="Full image width."),
                io.Int.Input("full_height", max=65536, tooltip="Full image height."),
                io.Int.Input("tile_width", max=8192, tooltip="Tile width."),
                io.Int.Input("tile_height", max=8192, tooltip="Tile height."),
                io.Int.Input("overlap", max=2048, tooltip="Overlap pixels."),
                io.Float.Input("overlap_feather_rate", max=8.0, tooltip="Overlap feather rate."),
                io.Int.Input("pixel_alignment", max=256, tooltip="Pixel alignment for tile calculations."),
            ],
            outputs=[
                MiraITUPipeline.Output(display_name="mira_itu_pipeline"),
            ],
        )
    
    @classmethod
    def execute(cls, full_width, full_height, tile_width, tile_height, overlap, overlap_feather_rate, pixel_alignment) -> io.NodeOutput:
        return io.NodeOutput((full_width, full_height, tile_width, tile_height, overlap, overlap_feather_rate, pixel_alignment))

# ==========================================
# Common Helper
# ==========================================            
class FeatherBlendHelper:
    """
    Shared Feathering Blend Helper Class
    """
    @staticmethod
    def get_geometric_mask(tile_x, tile_y, tile_w, tile_h, full_w, full_h, feather, device, channels=None):
        """
        Create a feather mask based on the tile's geometric position in the full image.
        
        Args:
            tile_x, tile_y: Position of the tile
            tile_w, tile_h: Dimensions of the tile
            full_w, full_h: Dimensions of the full image
            feather: Feathering width in pixels
            device: torch device
            channels: If provided, expands mask to [H, W, C]
            
        Returns:
            mask: Tensor [H, W] or [H, W, C] with values 0.0 to 1.0
        """
        # Start with a solid mask of 1.0
        mask = torch.ones((tile_h, tile_w), device=device, dtype=torch.float32)
        
        if feather < 1:
            if channels:
                return mask[..., None].expand(-1, -1, channels)
            return mask

        # Create cosine ramp gradients (smoother than linear, eliminates derivative
        # discontinuities at tile boundaries that cause visible seams)
        t = torch.linspace(0.0, 1.0, feather, device=device, dtype=torch.float32)
        ramp = 0.5 - 0.5 * torch.cos(t * math.pi)
        
        # Feather Top (If not at the very top of the image)
        if tile_y > 0:
            f_len = min(feather, tile_h)
            mask[:f_len, :] *= ramp[:f_len, None]
            
        # Feather Bottom (If not at the very bottom of the image)
        if tile_y + tile_h < full_h:
            f_len = min(feather, tile_h)
            mask[-f_len:, :] *= ramp.flip(0)[:f_len, None]
            
        # Feather Left (If not at the very left of the image)
        if tile_x > 0:
            f_len = min(feather, tile_w)
            mask[:, :f_len] *= ramp[None, :f_len]

        # Feather Right (If not at the very right of the image)
        if tile_x + tile_w < full_w:
            f_len = min(feather, tile_w)
            mask[:, -f_len:] *= ramp.flip(0)[None, :f_len]

        if channels:
            return mask[..., None].expand(-1, -1, channels)
        
        return mask

class TileHelper:
    @staticmethod
    def _find_optimal_tile_size(W, H, base_tile_size, overlap, max_deviation, max_aspect_ratio=1.33, pixel_alignment=8):
        """
        Find the optimal tile dimensions separately for width and height.

        Args:
            W: Image width
            H: Image height
            base_tile_size: Target tile size
            overlap: Overlap pixels between tiles
            max_deviation: Maximum allowed deviation from base_tile_size
            max_aspect_ratio: Maximum aspect ratio (e.g., 1.33 for 4:3, 1.25 for 5:4, 1.5 for 3:2)

        Returns:
            (tile_width, tile_height): Optimal tile width and height
        """
        if base_tile_size <= overlap:
            aligned = (base_tile_size // pixel_alignment) * pixel_alignment
            return aligned, aligned

        def find_best_for_dimension(length):
            """Find the optimal size for a single dimension"""
            best_effective = base_tile_size
            best_score = float('inf')

            for adj in range(-max_deviation, max_deviation + 1):
                effective = base_tile_size + adj
                if effective <= overlap:
                    continue

                step = effective - overlap
                if step <= 0:
                    continue

                # Calculate the number of tiles needed
                n_tiles = math.ceil(length / step)

                # Actual coverage range
                coverage = (n_tiles - 1) * step + effective

                # Extra pixels beyond the image dimension
                extra = coverage - length

                if extra < 0:
                    continue

                # Scoring: fewer extra pixels is better; smaller deviation from base_tile_size is better
                score = extra + abs(adj) * 0.1
                if score < best_score:
                    best_score = score
                    best_effective = effective

            return best_effective

        # Find optimal values separately for width and height
        best_width = find_best_for_dimension(W)
        best_height = find_best_for_dimension(H)

        # Apply aspect ratio constraint
        current_ratio = max(best_width, best_height) / max(1, min(best_width, best_height))

        if current_ratio > max_aspect_ratio:
            # Adjust to satisfy the aspect ratio constraint
            # Prioritize reducing the larger dimension to meet the ratio
            if best_width > best_height:
                target_width = int(best_height * max_aspect_ratio)
                # Find the closest value within the allowed deviation range
                best_width = max(base_tile_size - max_deviation,
                                 min(base_tile_size + max_deviation, target_width))
            else:
                target_height = int(best_width * max_aspect_ratio)
                best_height = max(base_tile_size - max_deviation,
                                  min(base_tile_size + max_deviation, target_height))

        # Align to the requested pixel grid
        best_width = (best_width // pixel_alignment) * pixel_alignment
        best_height = (best_height // pixel_alignment) * pixel_alignment

        # Verify coverage and increase tile size if necessary
        def ensure_coverage(length, tile_size):
            """Ensure tiles fully cover the image"""
            step = tile_size - overlap
            if step <= 0:
                return tile_size

            n_tiles = math.ceil(length / step)
            coverage = (n_tiles - 1) * step + tile_size

            while coverage < length:
                tile_size += pixel_alignment
                step = tile_size - overlap
                n_tiles = math.ceil(length / step)
                coverage = (n_tiles - 1) * step + tile_size

            return tile_size

        best_width = ensure_coverage(W, best_width)
        best_height = ensure_coverage(H, best_height)

        return best_width, best_height

    @staticmethod
    def _calculate_tiles(width, height, tile_width, tile_height, overlap, pixel_alignment):
        """
        Calculate tile divisions using the specified tile width and height.

        Args:
            width: Image width
            height: Image height
            tile_width: Tile width
            tile_height: Tile height
            overlap: Overlap pixels

        Returns:
            List of (x, y, w, h): Coordinates and dimensions of each tile
        """
        tiles = []
        step_x = tile_width - overlap
        step_y = tile_height - overlap

        # Calculate the number of tiles needed
        tiles_x = math.ceil((width - overlap) / step_x) if width > tile_width else 1
        tiles_y = math.ceil((height - overlap) / step_y) if height > tile_height else 1

        for i in range(tiles_y):
            for j in range(tiles_x):
                x = j * step_x
                y = i * step_y

                # Align the last column/row tiles to the edge
                if x + tile_width > width:
                    x = width - tile_width
                if y + tile_height > height:
                    y = height - tile_height

                # Ensure coordinates are non-negative (for small images)
                x = max(0, x)
                y = max(0, y)

                # Align to pixel_alignment grid
                x = (int(x) // pixel_alignment) * pixel_alignment
                y = (int(y) // pixel_alignment) * pixel_alignment
                w = tile_width
                h = tile_height

                # Crop tile size if the image is smaller than the tile
                w = min(w, width - x)
                h = min(h, height - y)
                w = (w // pixel_alignment) * pixel_alignment
                h = (h // pixel_alignment) * pixel_alignment

                if w > 0 and h > 0:
                    tiles.append((x, y, w, h))

        # Remove duplicates and sort (first by y, then by x)
        return sorted(list(set(tiles)), key=lambda t: (t[1], t[0]))
    
    @staticmethod
    def _center_crop_to_alignment(image, pixel_alignment):
        """
        Center crop image dimensions to be aligned with pixel_alignment.
        If W or H is not divisible by pixel_alignment, crops the excess pixels
        symmetrically (center crop).
        
        Args:
            image: Input tensor [H, W, C] or [H, W]
            pixel_alignment: Target alignment value
            
        Returns:
            Cropped image tensor and info dict with original/cropped dimensions
        """
        H, W = image.shape[0], image.shape[1]
        
        # Calculate aligned dimensions
        aligned_W = (W // pixel_alignment) * pixel_alignment
        aligned_H = (H // pixel_alignment) * pixel_alignment
        
        # Check if cropping is needed
        needs_crop_W = W != aligned_W
        needs_crop_H = H != aligned_H
        
        if needs_crop_W or needs_crop_H:
            # Calculate crop amounts (center crop)
            crop_W = W - aligned_W
            crop_H = H - aligned_H
            
            # Center crop: distribute crop pixels symmetrically
            left_crop = crop_W // 2
            top_crop = crop_H // 2
            
            right_end = W - (crop_W - left_crop)
            bottom_end = H - (crop_H - top_crop)
            
            # Perform center crop
            cropped_image = image[top_crop:bottom_end, left_crop:right_end, ...]
            
            info = {
                "original_width": W,
                "original_height": H,
                "aligned_width": aligned_W,
                "aligned_height": aligned_H,
                "cropped": True,
                "crop_top": top_crop,
                "crop_left": left_crop,
                "crop_amount_w": crop_W,
                "crop_amount_h": crop_H,
            }
            
            return cropped_image, info
        else:
            # No cropping needed
            info = {
                "original_width": W,
                "original_height": H,
                "aligned_width": aligned_W,
                "aligned_height": aligned_H,
                "cropped": False,
                "crop_top": 0,
                "crop_left": 0,
                "crop_amount_w": 0,
                "crop_amount_h": 0,
            }
            return image, info
        
# ==========================================
# Ksampler with Tagger Support
# ==========================================    
class ImageTiledKSamplerWithTagger(io.ComfyNode):
    """
    Ksampler with Tagger Support for Tiled Image Sampling
    """    
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="ImageTiledKSamplerWithTagger_MiraSubPack",
            display_name="Tiled Image KSampler with Tagger",
            category=CAT,
            description="Perform tiled image sampling with dynamic tagger-based prompts for each tile.",
            inputs=[
                io.Model.Input("model"),
                io.Clip.Input("clip"),                
                io.Latent.Input("tiled_samples", tooltip="Tiled latents input from VAE."),
                io.String.Input("common_positive", default="", multiline=True, tooltip="Common positive prompt for all tiles."),
                io.String.Input("common_negative", default="bad quality, worst quality, worst detail, sketch", multiline=True, tooltip="Common negative prompt for all tiles."),
                io.String.Input("tagger_text", default="", multiline=True, tooltip="Tagger output text mapping for tiles, one line per tile."),
                io.Int.Input("seed", default=0, min=0, max=0xffffffffffffffff),
                io.Int.Input("steps", default=16, min=1, max=100), 
                io.Float.Input("cfg", default=7.0, min=0.0, max=32.0, step=0.1),
                io.Combo.Input("sampler_name", default="euler_ancestral", options=comfy.samplers.KSampler.SAMPLERS),
                io.Combo.Input("scheduler", default="beta", options=comfy.samplers.KSampler.SCHEDULERS),
                io.Float.Input("denoise", default=0.35, min=0.0, max=1.0, step=0.01),
                io.Combo.Input("mode", default="Normal", options=["Normal", "Reference"],
                    tooltip="Normal: ignore ref_latents and noise boost. (SDXL/Z Image)\n"
                            "Reference: use ref_latents with optional noise boost. (Flux.2)"),
                io.Float.Input("noise_boost", default=0.0, min=0.0, max=1.0, step=0.05,
                    tooltip="Extra noise injected into ref_latents before conditioning to encourage detail generation.\n"
                            "Perturbs the reference that guides generation, effective even at denoise=1.0.\n"
                            "0.0 = no boost (original behavior), 0.1~0.3 = subtle detail enhancement,\n"
                            "0.3~0.6 = moderate (recommended for upscale with ref_latents),\n"
                            "0.6~1.0 = aggressive (more creative, may deviate from original).\n"
                            "Only effective when ref_latents is connected."),
                io.Combo.Input("noise_injection_method", default="adaptive",
                    options=["uniform", "high_frequency", "adaptive"],
                    tooltip="Noise injection method:\n"
                            "uniform: Standard Gaussian noise, uniform across all regions.\n"
                            "high_frequency: Emphasizes high-frequency detail noise, better for textures.\n"
                            "adaptive: Adds more noise to flat/blurry regions, less to detailed areas."),
                io.Clip.Input("clip_negative", optional=True),
                io.Latent.Input("ref_latents", optional=True),
            ],
            outputs=[
                io.Latent.Output(display_name="tiled_latents"),
            ],
        )
    
    @classmethod
    def execute(cls, model, clip, tiled_samples, common_positive, common_negative, tagger_text,
             seed, steps, cfg, sampler_name, scheduler, denoise, mode="Reference", noise_boost=0.0, noise_injection_method="adaptive",
               clip_negative = None, ref_latents = None
               ) -> io.NodeOutput:
        negative_conditioning = None
        if clip_negative:
            negative_tokens = clip_negative.tokenize(common_negative)
            negative_conditioning = clip_negative.encode_from_tokens_scheduled(negative_tokens)    
        else:
            negative_tokens = clip.tokenize(common_negative)
            negative_conditioning = clip.encode_from_tokens_scheduled(negative_tokens)
            
        del negative_tokens

        batch_latents = tiled_samples["samples"]
        print(f"[MiraSubPack:AutoTiledTagger] Using {len(batch_latents)} tiles.")
        print(f"[MiraSubPack:AutoTiledTagger] tagger_text (for copy to SAA)\n{tagger_text}")
        
        if mode == "Normal" and ref_latents is not None:
            print("    >Mode=Normal: ref_latents will be ignored.")
        elif ref_latents is not None:
            print("    >Using provided reference latents for this tile.")

        if mode == "Normal" and noise_boost > 0:
            print(f"[MiraSubPack:AutoTiledTagger] Mode={mode}: noise_boost is ignored.")
        elif noise_boost > 0:
            print(f"[MiraSubPack:AutoTiledTagger] Noise boost: {noise_boost:.2f}, method: {noise_injection_method}")
                            
        # Parse tagger text mapping
        mapping = tagger_text.splitlines()
        tile_latents = None
        for idx in range(len(batch_latents)):
            # Build dynamic prompt
            if tagger_text != "":
                dynamic_prompt = common_positive
                tags_str = ""
                if idx < len(mapping):
                    tags_str = mapping[idx].replace('(', r'\(').replace(')', r'\)')
                    dynamic_prompt = f"{common_positive}, {tags_str}" if common_positive else tags_str
                print(f"    Tags: {tags_str}")
            else:
                dynamic_prompt = common_positive

            # Tokenize
            positive_tokens = clip.tokenize(dynamic_prompt)
            positive_conditioning = clip.encode_from_tokens_scheduled(positive_tokens)
            
            if ref_latents is not None and mode != "Normal":
                all_ref_latent = ref_latents["samples"]

                if len(all_ref_latent) == len(batch_latents):
                    ref_latent = all_ref_latent[idx].unsqueeze(0)
                else:
                    print(f"    >Warning: ref_latents count {len(all_ref_latent)} does not match batch_latents count {len(batch_latents)}. Using the first ref_latent for all tiles.")
                    ref_latent = all_ref_latent[0].unsqueeze(0)
                
                if mode == "Reference":
                    # Inject noise into ref_latent (not single_latent) to perturb the reference
                    # that guides generation. At high denoise (e.g. 1.0), single_latent is replaced
                    # by pure noise, so injecting there is meaningless. The ref_latent is what
                    # actually anchors the output via conditioning - perturbing it gives the model
                    # freedom to generate new details instead of faithfully reproducing the blurry original.
                    if noise_boost > 0:
                        ref_latent = cls._inject_noise(
                            ref_latent, noise_boost, noise_injection_method, seed + idx
                        )
                        print(f"    Noise boost applied to ref_latent (boost={noise_boost:.2f}, method={noise_injection_method})")

                    if clip_negative:
                        negative_tokens = clip_negative.tokenize(common_negative)
                        negative_conditioning = clip_negative.encode_from_tokens_scheduled(negative_tokens)
                    else:
                        negative_tokens = clip.tokenize(common_negative)
                        negative_conditioning = clip.encode_from_tokens_scheduled(negative_tokens)

                    positive_conditioning = node_helpers.conditioning_set_values(positive_conditioning, {"reference_latents": [ref_latent]}, append=True)
                    negative_conditioning = node_helpers.conditioning_set_values(negative_conditioning, {"reference_latents": [ref_latent]}, append=True)

                    del positive_tokens
                    del negative_tokens                    
            
            single_latent = batch_latents[idx].unsqueeze(0)  # [C, H, W] -> [1, C, H, W]
                
            print(f"  > Sampling Tile {idx+1}/{len(batch_latents)}: {single_latent.shape[3]}x{single_latent.shape[2]}")
            print(f"    Tile latent shape: {single_latent.shape}")
            sampled_tile = cls._sample_single(
                model, positive_conditioning, negative_conditioning, {"samples": single_latent},
                seed, steps, cfg, sampler_name, scheduler, denoise
            )            
            tile_latents = torch.cat([tile_latents, sampled_tile["samples"]], dim=0) if tile_latents is not None else sampled_tile["samples"]
            
            if ref_latents is not None and mode == "Reference":
                del positive_conditioning
                del negative_conditioning
                torch.cuda.empty_cache()
                
        return io.NodeOutput({"samples": tile_latents})
    
    @staticmethod
    def _crop_latent(samples, x, y, width, height, pixel_alignment):
        latent = samples["samples"]
        lx, ly, lw, lh = x//pixel_alignment, y//pixel_alignment, width//pixel_alignment, height//pixel_alignment
        cropped = latent[:, :, ly:ly+lh, lx:lx+lw].clone()
        # Padding if necessary (usually handled by clamp above, but safe to keep)
        if cropped.shape[2] != lh or cropped.shape[3] != lw:
             # Basic padding if size mismatch
             pad_h = lh - cropped.shape[2]
             pad_w = lw - cropped.shape[3]
             cropped = torch.nn.functional.pad(cropped, (0, max(0, pad_w), 0, max(0, pad_h)), mode='replicate')
        return {"samples": cropped}        
            
    @staticmethod
    def _inject_noise(latent, noise_boost, method, seed):
        """
        Inject additional noise into latent to encourage detail generation.
        This perturbs the starting point away from the blurry original,
        giving the sampler more room to reconstruct with enhanced detail.
        
        Args:
            latent: Input latent tensor [1, C, H, W]
            noise_boost: Strength of noise injection (0.0 to 1.0)
            method: 'uniform', 'high_frequency', or 'adaptive'
            seed: Random seed for reproducibility
        """
        generator = torch.manual_seed(seed)
        l = latent.clone()
        
        # Scale noise relative to the latent's own magnitude
        latent_std = l.std().clamp(min=1e-6)
        
        if method == "uniform":
            # Standard Gaussian noise scaled by latent statistics
            boost_noise = torch.randn(l.shape, dtype=l.dtype, device=l.device, generator=generator)
            l = l + boost_noise * latent_std * noise_boost
            
        elif method == "high_frequency":
            # Generate noise, then subtract a blurred version to keep only high-frequency components.
            # This emphasizes texture/detail-scale perturbations rather than broad color shifts.
            raw_noise = torch.randn(l.shape, dtype=l.dtype, device=l.device, generator=generator)
            
            # Simple box-blur approximation for low-frequency extraction (kernel_size=5)
            kernel_size = 5
            padding = kernel_size // 2
            # Average pool each channel to get low-frequency component
            low_freq = torch.nn.functional.avg_pool2d(
                raw_noise, kernel_size=kernel_size, stride=1, padding=padding
            )
            # High-frequency = original - low_frequency
            hf_noise = raw_noise - low_freq
            # Normalize to unit variance for consistent scaling
            hf_std = hf_noise.std().clamp(min=1e-6)
            hf_noise = hf_noise / hf_std
            
            l = l + hf_noise * latent_std * noise_boost * 1.5  # 1.5x because HF noise is weaker perceptually
            
        elif method == "adaptive":
            # Add more noise to flat/blurry regions (low local variance),
            # less noise to regions that already have detail (high local variance).
            # This targets exactly the areas that need more detail generation.
            raw_noise = torch.randn(l.shape, dtype=l.dtype, device=l.device, generator=generator)
            
            # Compute local variance using a sliding window
            kernel_size = 7
            padding = kernel_size // 2
            
            # Local mean and variance
            local_mean = torch.nn.functional.avg_pool2d(
                l, kernel_size=kernel_size, stride=1, padding=padding
            )
            local_var = torch.nn.functional.avg_pool2d(
                (l - local_mean) ** 2, kernel_size=kernel_size, stride=1, padding=padding
            )
            
            # Inverse variance map: high where flat, low where detailed
            # Normalize to [0, 1] range
            local_std = (local_var + 1e-8).sqrt()
            max_std = local_std.amax(dim=(-2, -1), keepdim=True).clamp(min=1e-6)
            detail_map = local_std / max_std  # 0=flat, 1=detailed
            
            # Invert: more noise where flat (detail_map is low)
            # Use smooth ramp: flat areas get up to 2x boost, detailed areas get 0.3x
            noise_scale = 0.3 + 1.7 * (1.0 - detail_map.clamp(0, 1))
            
            l = l + raw_noise * latent_std * noise_boost * noise_scale
        
        return l
    
    @staticmethod
    def _sample_single(model, positive, negative, latent, seed, steps, cfg, sampler_name, scheduler, denoise):
        l = latent["samples"]
        noise = torch.randn(l.shape, dtype=l.dtype, device=l.device, generator=torch.manual_seed(seed))
        callback = latent_preview.prepare_callback(model, steps)
        sampled = comfy.sample.sample(model, noise, steps, cfg, sampler_name, scheduler, 
                                      positive, negative, l, denoise=denoise, seed=seed, 
                                      force_full_denoise=True, callback=callback)
        return {"samples": sampled}

# ==========================================
# Latent Merging Utilities
# ==========================================
class OverlappedLatentMerge(io.ComfyNode):
    """
    Merge overlapped latent tiles.
    Uses geometric feathering and weighting boost for large overlaps.
    """    
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="OverlappedLatentMerge_MiraSubPack",
            display_name="Overlapped Latent Merge",
            category=CAT,
            description="Merge overlapped latent tiles using geometric feathering and overlap priority.",
            inputs=[
                io.Latent.Input("tiled_latents", optional=False, tooltip="Tiled latents input."),
                MiraITUPipeline.Input("mira_itu_pipeline",optional=False, tooltip="Mira Image Tiled Upscale pipeline info from tiling node."),
                io.Float.Input("feather_rate_override", default=0, min=0, max=4.0, step=0.1, tooltip="Override fathering rate multiplier if value is not 0."),
                io.Float.Input("pixel_alignment", default=8.0, min=1.0, max=64.0, step=1.0, tooltip="Pixel alignment for tile calculations (e.g., 8 for 8-pixel grid)."),
            ],
            outputs=[
                io.Latent.Output()
            ],
        )

    @classmethod
    def execute(cls, tiled_latents, mira_itu_pipeline, feather_rate_override) -> io.NodeOutput:
        (full_width, full_height, tile_width, tile_height, overlap, overlap_feather_rate, pixel_alignment) = mira_itu_pipeline
        if round(feather_rate_override,2) != 0:
            print(f"[MiraSubPack:OverlappedImageMerge] Override feather_rate to {feather_rate_override} ")
            overlap_feather_rate = feather_rate_override
            
        device = tiled_latents["samples"].device
        dtype = tiled_latents["samples"].dtype
        batch_latents = tiled_latents["samples"]
        
        # 1. Recalculate tile positions
        tiles = TileHelper._calculate_tiles(full_width, full_height, tile_width, tile_height, overlap, pixel_alignment)
        
        # 2. Setup Canvas
        lw = full_width // pixel_alignment
        lh = full_height // pixel_alignment
        channels = batch_latents.shape[1]
        
        output = torch.zeros((1, channels, lh, lw), device=device, dtype=dtype)
        weight_map = torch.zeros((1, 1, lh, lw), device=device, dtype=torch.float32)
        
        # 3. Feathering params
        # Use overlap_feather_rate directly (feather = overlap × rate).
        # rate=2.0 (default) → feather=128px for overlap=64, extending 64px
        # beyond the overlap zone on each side for smooth blending.
        feather_px = int(overlap * overlap_feather_rate)
        feather_px = int(min(max(tile_width, tile_height) * 0.25, feather_px))
        feather_px = max(feather_px, 1)
        l_feather = int(feather_px // pixel_alignment)
        
        # Track previous tile end positions to detect overlap ratio
        # row_y -> max_x_end
        row_last_x_end = {} 
        # col_x -> max_y_end
        col_last_y_end = {}

        print(f"[MiraSubPack:OverlappedLatentMerge] Merging {len(tiles)} tiles for canvas {full_width}x{full_height}...")
        for idx, (x, y, w, h) in enumerate(tiles):
            # Extract current tile
            tile_latent = batch_latents[idx] # [C, H, W]
            
            lx, ly = x // pixel_alignment, y // pixel_alignment
            lw_tile, lh_tile = w // pixel_alignment, h // pixel_alignment
            
            # Ensure dimensions match (handling potential rounding in calculation vs tensor)
            tile_latent = tile_latent[:, :lh_tile, :lw_tile]
            
            # Get Geometric Mask (Feathers based on image boundaries)
            # This solves Issue 1: First tile gets feathered if it's not at x=0
            mask = FeatherBlendHelper.get_geometric_mask(
                lx, ly, lw_tile, lh_tile, lw, lh, l_feather, device
            )
            mask = mask[None, None, :, :] # [1, 1, H, W]

            # --- Issue 3: The 50% Rule ---
            # Determine if this tile overlaps significantly with previous content
            # and should "dominate" (overwrite) the previous content.
            
            boost_weight = 1.0
            
            # Check Horizontal Overlap with previous tile in this row
            if y in row_last_x_end:
                prev_end = row_last_x_end[y]
                overlap_amount = prev_end - lx
                # If overlap is > 50% of the tile width
                if overlap_amount > (lw_tile * 0.5):
                    boost_weight = 10.0
                    print(f"  > Tile {idx}: Horizontal overlap > 50% ({overlap_amount}/{lw_tile}), boosting weight.")
            
            # Check Vertical Overlap (less common in row-by-row but good for robustness)
            if x in col_last_y_end:
                prev_end = col_last_y_end[x]
                overlap_amount = prev_end - ly
                if overlap_amount > (lh_tile * 0.5):
                    boost_weight = max(boost_weight, 10.0)
                    print(f"  > Tile {idx}: Vertical overlap > 50% ({overlap_amount}/{lh_tile}), boosting weight.")

            if boost_weight > 1.0:
                mask = mask * boost_weight
            
            # --- Accumulation ---
            target_region = output[:, :, ly:ly+lh_tile, lx:lx+lw_tile]
            target_region += tile_latent[None, ...] * mask
            output[:, :, ly:ly+lh_tile, lx:lx+lw_tile] = target_region
            
            weight_region = weight_map[:, :, ly:ly+lh_tile, lx:lx+lw_tile]
            weight_region += mask
            weight_map[:, :, ly:ly+lh_tile, lx:lx+lw_tile] = weight_region
            
            # Update trackers
            row_last_x_end[y] = lx + lw_tile
            col_last_y_end[x] = ly + lh_tile

        # 4. Normalize
        weight_map = weight_map.clamp(min=1e-5)
        output = output / weight_map
        
        return io.NodeOutput({"samples": output})

# ==========================================
# Image Merging Utilities
# ==========================================    
class OverlappedImageMerge(io.ComfyNode):
    """
    Merge tiled images with corrected feathering and overlap dominance.
    Not good for latents due to potential color shifts, but works well for final image merging.
    """
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="OverlappedImageMerge_MiraSubPack",
            display_name="Overlapped Image Merge",
            category=CAT,
            description="Merge tiled images using geometric feathering and overlap priority.",
            inputs=[
                io.Image.Input("tiled_images", optional=False, tooltip="Tiled images input."),
                MiraITUPipeline.Input("mira_itu_pipeline",optional=False, tooltip="Mira Image Tiled Upscale pipeline info from tiling node."),
                io.Float.Input("feather_rate_override", default=0, min=-0.1, max=4.0, step=0.1, tooltip="Override feathering rate multiplier if value is not 0. Use -0.1 to disable feathering and directly overwrite tiles."),
            ],
            outputs=[
                io.Image.Output()
            ],
        )
        
    @classmethod
    def execute(cls, tiled_images, mira_itu_pipeline, feather_rate_override) -> io.NodeOutput:
        (full_width, full_height, tile_width, tile_height, overlap, overlap_feather_rate, pixel_alignment) = mira_itu_pipeline
        direct_overwrite = round(feather_rate_override, 2) == -0.1
        if direct_overwrite:
            print("[MiraSubPack:OverlappedImageMerge] Feathering disabled (-0.1), using direct overwrite.")
        elif round(feather_rate_override,2) != 0:
            print(f"[MiraSubPack:OverlappedImageMerge] Override feather_rate to {feather_rate_override} ")
            overlap_feather_rate = feather_rate_override
        
        device = tiled_images.device
        N, H, W, C = tiled_images.shape
        
        # 1. Calculate tile positions
        tiles = TileHelper._calculate_tiles(full_width, full_height, tile_width, tile_height, overlap, pixel_alignment)
        
        # 2. Setup Canvas
        # canvas needs to hold color, so it uses C channels (usually 3)
        canvas = torch.zeros((full_height, full_width, C), device=device, dtype=torch.float32)
        # weight_map only needs to track accumulated weight, so it uses 1 channel
        weight_map = torch.zeros((full_height, full_width, 1), device=device, dtype=torch.float32)

        if not direct_overwrite:
            feather = max(overlap * 4, int(overlap * overlap_feather_rate))
            feather = int(min(max(tile_width, tile_height) * 0.25, feather))
        
        row_last_x_end = {}
        col_last_y_end = {}

        print(f"[MiraSubPack:OverlappedImageMerge] Merging {len(tiles)} tiles...")

        for idx, (x, y, w, h) in enumerate(tiles):
            if idx >= N: break
            
            tile = tiled_images[idx] # [H, W, C]
            
            # Crop tile to expected size (safety check)
            tile = tile[:h, :w, :]
            
            if direct_overwrite:
                canvas[y:y+h, x:x+w, :] = tile
            else:
                # Get Geometric Mask (Single Channel [H, W])
                # We use channels=None to get a 2D mask [h, w] first
                mask_2d = FeatherBlendHelper.get_geometric_mask(
                    x, y, w, h, full_width, full_height, feather, device, channels=None
                )
                
                # --- Overlap Dominance Logic ---
                boost_weight = 1.0
                
                # Horizontal Check
                if y in row_last_x_end:
                    prev_end = row_last_x_end[y]
                    overlap_amt = prev_end - x
                    if overlap_amt > (w * 0.5):
                        boost_weight = 10.0
                
                # Vertical Check
                if x in col_last_y_end:
                    prev_end = col_last_y_end[x]
                    overlap_amt = prev_end - y
                    if overlap_amt > (h * 0.5):
                        boost_weight = max(boost_weight, 10.0)
                
                if boost_weight > 1.0:
                    mask_2d = mask_2d * boost_weight

                # Prepare masks for broadcasting
                # mask_3ch for Image: [H, W, 1] -> broadcasts to [H, W, 3] during multiplication
                mask_expanded = mask_2d[:, :, None] 
                    
                # Accumulate
                # canvas (3ch) += tile (3ch) * mask (1ch, broadcasts to 3ch) -> Works
                canvas[y:y+h, x:x+w, :] += tile * mask_expanded
                
                # weight_map (1ch) += mask (1ch) -> Works (Fixes the RuntimeError)
                weight_map[y:y+h, x:x+w, :] += mask_expanded
            
            # Update trackers
            row_last_x_end[y] = x + w
            col_last_y_end[x] = y + h

        # 3. Normalize
        if direct_overwrite:
            return io.NodeOutput(canvas.unsqueeze(0))

        weight_map = weight_map.clamp(min=1e-5)
        canvas = canvas / weight_map
        
        return io.NodeOutput(canvas.unsqueeze(0))
    
# ==========================================
# Tiled Image Color Correction
# ==========================================
class TiledImageColorCorrection(io.ComfyNode):
    """
    Color correction for tiled images using pre-ksampler reference tiles.
    Corrects color/luminance shift in each tile by directly comparing with the
    corresponding reference tile (same index), so no pipeline recalculation is needed.
    Supports applying color correction and luminance correction independently or together.
    """    
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="TiledImageColorCorrection_MiraSubPack",
            display_name="Tiled Image Color Correction",
            category=CAT,
            description="Correct color/luminance shift in tiled images using pre-ksampler reference tiles. Supports combining color and luminance corrections.",
            inputs=[
                io.Image.Input("tiled_images", optional=False, tooltip="Sampled tiled images to be corrected (post-ksampler)."),
                io.Image.Input("reference_tiles", optional=False, tooltip="Original tiles before ksampler (e.g. tiled_images output from ImageCropTiles). Must have the same tile count as tiled_images."),
                io.Combo.Input("color_correction_method",
                    default="histogram_match",
                    options=["none", "histogram_match", "color_transfer"],
                    tooltip="Color correction method applied to each tile:\n"
                            "none: Skip color correction.\n"
                            "Histogram Match: Full per-channel distribution matching.\n"
                            "Color Transfer: Mean & std matching in CIE LAB space (Reinhard 2001)."),
                io.Float.Input("color_correction_strength", default=1.0, min=0.0, max=1.0, step=0.05,
                    tooltip="Blend strength for color correction: 0=no correction, 1=full correction."),
                io.Float.Input("luminance_correction_strength", default=0.0, min=0.0, max=1.0, step=0.05,
                    tooltip="Blend strength for luminance (brightness+contrast) correction applied AFTER color correction.\n"
                            "0=skip, 1=full luminance match.\n"
                            "Can be combined with any color_correction_method."),
                io.Float.Input("edge_preserving_smooth", default=0.0, min=0.0, max=1.0, step=0.1,
                    tooltip="Optional edge-preserving smoothing to reduce color artifacts/stains.\n"
                            "0=no smoothing, 0.3-0.5=recommended for artifact reduction, 1.0=maximum smoothing.\n"
                            "Applied as final post-processing step."),
            ],
            outputs=[
                io.Image.Output(display_name="corrected_tiles"),
            ],
        )
    
    @classmethod
    def execute(cls, tiled_images, reference_tiles, color_correction_method, color_correction_strength, luminance_correction_strength, edge_preserving_smooth) -> io.NodeOutput:
        device = tiled_images.device
        N = tiled_images.shape[0]
        N_ref = reference_tiles.shape[0]

        # Early return if no correction needed
        if color_correction_strength == 0 and luminance_correction_strength == 0:
            print("[MiraSubPack:TiledImageColorCorrection] All correction strengths are 0, skipping.")
            return io.NodeOutput(tiled_images)

        if N_ref != N:
            print(f"[MiraSubPack:TiledImageColorCorrection] ⚠ Warning: reference_tiles count ({N_ref}) != tiled_images count ({N}). Will process min({N_ref}, {N}) tiles.")

        active_methods = []
        if color_correction_method != "none" and color_correction_strength > 0:
            active_methods.append(f"{color_correction_method}(s={color_correction_strength:.2f})")
        if luminance_correction_strength > 0:
            active_methods.append(f"luminance(s={luminance_correction_strength:.2f})")
        method_str = " + ".join(active_methods) if active_methods else "none"
        print(f"[MiraSubPack:TiledImageColorCorrection] Correcting {N} tiles | method: {method_str}")

        correction_stats = {
            "mean_delta": 0.0,
            "max_delta": 0.0,
        }

        corrected_tiles = []
        process_count = min(N, N_ref)

        for idx in range(N):
            tile = tiled_images[idx]  # [H, W, C]

            if idx >= process_count:
                # No reference available for this tile, pass through
                corrected_tiles.append(tile)
                continue

            ref_tile = reference_tiles[idx]  # [H, W, C]

            # Resize reference tile to match sampled tile size if needed
            # (e.g. when upscaling was involved between crop and decode)
            if ref_tile.shape[0] != tile.shape[0] or ref_tile.shape[1] != tile.shape[1]:
                ref_tile = torch.nn.functional.interpolate(
                    ref_tile.permute(2, 0, 1).unsqueeze(0),
                    size=(tile.shape[0], tile.shape[1]),
                    mode='bilinear',
                    align_corners=False
                ).squeeze(0).permute(1, 2, 0)

            corrected_tile = tile

            # --- Step 1: Color correction ---
            if color_correction_method != "none" and color_correction_strength > 0:
                if color_correction_method == "histogram_match":
                    color_corrected = cls._histogram_match(tile, ref_tile, device)
                elif color_correction_method == "color_transfer":
                    color_corrected = cls._color_transfer(tile, ref_tile, device)
                else:
                    color_corrected = tile
                corrected_tile = torch.lerp(tile, color_corrected, color_correction_strength)

            # --- Step 2: Luminance correction (on top of step 1 result) ---
            if luminance_correction_strength > 0:
                lum_corrected = cls._luminance_correction(corrected_tile, ref_tile, device)
                corrected_tile = torch.lerp(corrected_tile, lum_corrected, luminance_correction_strength)

            # --- Step 3: Optional edge-preserving smoothing to reduce artifacts ---
            if edge_preserving_smooth > 0:
                smoothed = cls._edge_preserving_smooth(corrected_tile, edge_preserving_smooth, device)
                corrected_tile = smoothed

            corrected_tile = corrected_tile.clamp(0, 1)

            # Track correction magnitude relative to original tile
            delta = (corrected_tile - tile).abs().mean().item()
            correction_stats["mean_delta"] += delta
            correction_stats["max_delta"] = max(correction_stats["max_delta"], delta)

            corrected_tiles.append(corrected_tile)

        # Stack corrected tiles back into batch
        output = torch.stack(corrected_tiles, dim=0)

        # Print summary statistics
        if process_count > 0:
            correction_stats["mean_delta"] /= process_count
        print(f"  Mean delta: {correction_stats['mean_delta']:.6f}, Max delta: {correction_stats['max_delta']:.6f}")

        return io.NodeOutput(output)
    
    @staticmethod
    def _match_channel_histogram(src_flat: torch.Tensor, ref_flat: torch.Tensor, device) -> torch.Tensor:
        """
        Improved histogram matching using exact sort-matching (Quantile Matching).
        
        This avoids the "foggy/blocky" artifacts caused by binning (discretization) 
        in traditional CDF matching. It matches the statistical distribution 
        while preserving the local gradients of the source.
        """
        src_flat = src_flat.view(-1)
        ref_flat = ref_flat.view(-1)
        
        n_src = src_flat.numel()
        n_ref = ref_flat.numel()
        
        if n_src == 0 or n_ref == 0:
            return src_flat

        # 1. Get Source Indices (argsort)
        # We need to preserve the relative ordering (rank) of the source pixels.
        # This tells us: "Pixel i was the k-th smallest value in the source image"
        src_indices = torch.argsort(src_flat)
        
        # 2. Sort Reference to get the Target Quantile Function
        # We want the reference values sorted min to max
        ref_sorted, _ = torch.sort(ref_flat)

        # 3. Interpolate Reference values to match Source count
        # We map the k-th rank of Source to the k-th rank of Reference.
        if n_src == n_ref:
            mapped_sorted = ref_sorted
        else:
            # We need to sample the reference distribution at n_src points
            # Treat ref_sorted as signal y, defined on x = [0, 1]
            # We want values at linspace(0, 1, n_src)
            try:
                # Optimized 1D interpolation
                # Input: [Batch, Channel, Length] -> [1, 1, n_ref]
                input_tensor = ref_sorted.view(1, 1, -1)
                # Output size: n_src
                mapped_sorted = torch.nn.functional.interpolate(
                    input_tensor, size=n_src, mode='linear', align_corners=True
                ).view(-1)
            except Exception:
                # Fallback for very weird shapes or errors
                mapped_sorted = ref_sorted

        # 4. Map back to original positions
        # The pixel that was k-th smallest in Source (src_indices[k]) 
        # gets the value mapped_sorted[k]
        
        # Create an empty tensor to scatter values back
        matched = torch.zeros_like(src_flat)
        matched.scatter_(0, src_indices, mapped_sorted)
        
        return matched

    @staticmethod
    def _histogram_match(tile, reference, device):
        """
        Match histogram of tile to reference image in CIE LAB space.

        Performing histogram matching per-channel in RGB destroys the inter-channel
        correlation (the R, G, B values of each pixel are correlated; remapping them
        independently with different rank orderings dismantles valid colour combinations),
        which produces the characteristic foggy colour blotches.

        LAB is perceptually decorrelated: L = luminance, a/b = chrominance axes.
        Matching each axis independently in LAB is safe and avoids hue artefacts.
        """
        rgb = tile[..., :3]
        rgb_ref = reference[..., :3]

        lab = TiledImageColorCorrection._rgb_to_lab(rgb)           # [H, W, 3]
        lab_ref = TiledImageColorCorrection._rgb_to_lab(rgb_ref)   # [H, W, 3]

        corrected_lab = lab.clone()
        for c in range(3):
            src_flat = lab[..., c].flatten()
            ref_flat = lab_ref[..., c].flatten()
            corrected_lab[..., c] = TiledImageColorCorrection._match_channel_histogram(
                src_flat, ref_flat, device
            ).reshape(lab[..., c].shape)

        corrected_rgb = TiledImageColorCorrection._lab_to_rgb(corrected_lab)

        # Preserve alpha if present
        if tile.shape[2] > 3:
            corrected_rgb = torch.cat([corrected_rgb, tile[..., 3:]], dim=-1)

        return corrected_rgb
    
    @staticmethod
    def _rgb_to_lab(img: torch.Tensor) -> torch.Tensor:
        """
        Convert linear-sRGB [H, W, 3] in [0,1] to CIE LAB [H, W, 3].
        L in [0, 100], a/b roughly in [-128, 127].
        Uses the standard D65 illuminant.
        """
        # --- sRGB gamma linearization ---
        img = img.clamp(0.0, 1.0)
        linear = torch.where(img <= 0.04045,
                             img / 12.92,
                             ((img + 0.055) / 1.055) ** 2.4)

        # --- Linear RGB → CIE XYZ (D65) ---
        M = torch.tensor([
            [0.4124564, 0.3575761, 0.1804375],
            [0.2126729, 0.7151522, 0.0721750],
            [0.0193339, 0.1191920, 0.9503041],
        ], dtype=img.dtype, device=img.device)
        xyz = linear.reshape(-1, 3) @ M.T           # [N, 3]
        xyz = xyz.reshape(img.shape)

        # --- Normalize by D65 white point ---
        white = torch.tensor([0.95047, 1.00000, 1.08883],
                             dtype=img.dtype, device=img.device)
        xyz = xyz / white

        # --- CIE f function ---
        delta = 6.0 / 29.0
        f = torch.where(xyz > delta ** 3,
                        xyz ** (1.0 / 3.0),
                        xyz / (3.0 * delta ** 2) + 4.0 / 29.0)

        L = 116.0 * f[..., 1] - 16.0
        a = 500.0 * (f[..., 0] - f[..., 1])
        b = 200.0 * (f[..., 1] - f[..., 2])
        return torch.stack([L, a, b], dim=-1)

    @staticmethod
    def _lab_to_rgb(lab: torch.Tensor) -> torch.Tensor:
        """
        Convert CIE LAB [H, W, 3] back to sRGB [H, W, 3] in [0, 1].
        """
        L, a, b = lab[..., 0], lab[..., 1], lab[..., 2]
        fy = (L + 16.0) / 116.0
        fx = a / 500.0 + fy
        fz = fy - b / 200.0

        delta = 6.0 / 29.0
        def f_inv(t):
            return torch.where(t > delta,
                               t ** 3,
                               3.0 * delta ** 2 * (t - 4.0 / 29.0))

        white = torch.tensor([0.95047, 1.00000, 1.08883],
                             dtype=lab.dtype, device=lab.device)
        xyz = torch.stack([f_inv(fx), f_inv(fy), f_inv(fz)], dim=-1) * white

        # --- CIE XYZ → Linear RGB ---
        M_inv = torch.tensor([
            [ 3.2404542, -1.5371385, -0.4985314],
            [-0.9692660,  1.8760108,  0.0415560],
            [ 0.0556434, -0.2040259,  1.0572252],
        ], dtype=lab.dtype, device=lab.device)
        linear = xyz.reshape(-1, 3) @ M_inv.T
        linear = linear.reshape(lab.shape)

        # --- Linear → sRGB gamma ---
        srgb = torch.where(linear <= 0.0031308,
                           linear * 12.92,
                           1.055 * linear.clamp(min=0.0) ** (1.0 / 2.4) - 0.055)
        return srgb.clamp(0.0, 1.0)

    @staticmethod
    def _color_transfer(tile, reference, device):
        """
        Transfer color statistics from reference to tile using CIE LAB space.
        
        Implements the Reinhard et al. (2001) method: convert to LAB (perceptually
        uniform, decorrelated channels), match mean & std per channel, convert back.
        
        Advantages over plain RGB stat matching:
        - LAB channels are nearly decorrelated → per-channel correction doesn't
          introduce spurious hue shifts.
        - L, a, b carry luminance and chrominance separately, so the transfer is
          perceptually meaningful.
        - std ratio is clamped conservatively only when tile has negligible variance
          to prevent noise amplification, not to cap the correction range.
        """
        rgb = tile[..., :3]
        rgb_ref = reference[..., :3]

        lab      = TiledImageColorCorrection._rgb_to_lab(rgb)
        lab_ref  = TiledImageColorCorrection._rgb_to_lab(rgb_ref)

        corrected_lab = lab.clone()
        for c in range(3):
            t_mean = lab[..., c].mean()
            t_std  = lab[..., c].std().clamp(min=1e-6)
            r_mean = lab_ref[..., c].mean()
            r_std  = lab_ref[..., c].std().clamp(min=1e-6)

            # Ratio cap: only prevent extreme noise amplification in near-flat regions
            std_ratio = (r_std / t_std).clamp(max=4.0)

            corrected_lab[..., c] = (lab[..., c] - t_mean) * std_ratio + r_mean

        corrected_rgb = TiledImageColorCorrection._lab_to_rgb(corrected_lab)

        # Preserve alpha if present
        if tile.shape[2] > 3:
            corrected_rgb = torch.cat([corrected_rgb, tile[..., 3:]], dim=-1)

        return corrected_rgb
    
    @staticmethod
    def _luminance_correction(tile, reference, device):
        """
        Correct luminance (brightness + contrast) while preserving chrominance (hue/saturation).

        Improvements over the original ratio-only approach:
        1. Uses perceptual luminance Y = 0.2126R + 0.7152G + 0.0722B (BT.709),
           not a simple 3-channel average.
        2. Matches both mean AND std of Y, so contrast differences are also corrected.
        3. Applies correction per-pixel via the Y ratio map, keeping colour ratios
           (Cb, Cr components) intact — true chrominance preservation.
        4. Wider clamp range (0.1–10.0) to handle dark/bright mismatches without
           artificially capping the correction.
        """
        corrected = tile.clone()

        if tile.shape[2] < 3:
            return corrected

        rgb     = tile[..., :3]          # [H, W, 3]
        rgb_ref = reference[..., :3]

        # BT.709 luminance coefficients
        lum_w = torch.tensor([0.2126, 0.7152, 0.0722],
                             dtype=rgb.dtype, device=device)

        # Per-pixel luminance maps
        tile_Y = (rgb     * lum_w).sum(dim=-1, keepdim=True).clamp(min=1e-7)  # [H, W, 1]
        ref_Y  = (rgb_ref * lum_w).sum(dim=-1, keepdim=True).clamp(min=1e-7)

        # Match mean and std of Y
        t_mean = tile_Y.mean()
        t_std  = tile_Y.std().clamp(min=1e-6)
        r_mean = ref_Y.mean()
        r_std  = ref_Y.std().clamp(min=1e-6)

        # Target per-pixel Y: same relative structure, shifted to reference distribution
        target_Y = (tile_Y - t_mean) * (r_std / t_std) + r_mean
        target_Y = target_Y.clamp(min=1e-7)

        # Per-pixel scaling ratio — preserves chrominance by scaling all channels equally
        ratio = (target_Y / tile_Y).clamp(0.1, 10.0)   # wide range, no artificial cap

        corrected[..., :3] = (rgb * ratio).clamp(0.0, 1.0)

        # Keep alpha unchanged if present
        if tile.shape[2] > 3:
            corrected[..., 3:] = tile[..., 3:]

        return corrected
    
    @staticmethod
    def _edge_preserving_smooth(tile, strength, device):
        """
        Apply edge-preserving smoothing to reduce color artifacts while maintaining detail.
        
        Uses a guided filter approach with spatial and range components to smooth
        color artifacts while preserving edges and fine details.
        
        Args:
            tile: Input image [H, W, C]
            strength: Smoothing strength (0-1)
            device: torch device
            
        Returns:
            Smoothed image [H, W, C]
        """
        if strength <= 0:
            return tile
        
        # Only process RGB channels
        if tile.shape[2] < 3:
            return tile
        
        rgb = tile[..., :3].clone()  # [H, W, 3]
        H, W = rgb.shape[0], rgb.shape[1]
        
        # Adaptive kernel size and sigma based on strength
        kernel_size = int(3 + strength * 6)  # 3-9 pixels
        if kernel_size % 2 == 0:
            kernel_size += 1
        
        sigma_range = 0.05 + strength * 0.15   # Color similarity sigma
        
        # Convert to [C, H, W] for processing
        rgb_chw = rgb.permute(2, 0, 1)  # [3, H, W]
        
        # Simple domain transform based smoothing (faster approximation of bilateral)
        # Iterate with increasing radius for better edge preservation
        smoothed = rgb_chw.clone()
        
        num_iterations = max(1, int(strength * 3))  # 1-3 iterations
        
        for _ in range(num_iterations):
            # Compute spatial averages with varying kernel
            padded = torch.nn.functional.pad(smoothed, (kernel_size//2, kernel_size//2, kernel_size//2, kernel_size//2), mode='reflect')
            
            # Create simple box filter kernel
            kernel_1d = torch.ones(kernel_size, device=device, dtype=torch.float32) / kernel_size
            kernel_2d = kernel_1d[None, :] * kernel_1d[:, None]  # [K, K]
            kernel_2d = kernel_2d.unsqueeze(0).unsqueeze(0)  # [1, 1, K, K]
            
            # Apply separable filtering for efficiency
            # Horizontal pass
            h_smoothed = torch.nn.functional.conv2d(
                padded.unsqueeze(0),  # [1, 3, H_pad, W_pad]
                kernel_2d.repeat(3, 1, 1, 1),  # [3, 1, K, K]
                groups=3,
                padding=0
            ).squeeze(0)  # [3, H, W]
            
            # Compute edge weights based on color difference
            color_diff = (h_smoothed - smoothed).abs().mean(dim=0, keepdim=True)  # [1, H, W]
            edge_weight = torch.exp(-color_diff / sigma_range)  # High weight for similar colors
            
            # Blend smoothed with original based on edge detection
            # Strong edges (large color diff) -> low weight -> keep original
            # Flat regions (small color diff) -> high weight -> use smoothed
            smoothed = torch.lerp(smoothed, h_smoothed, edge_weight * strength)
        
        # Convert back to [H, W, C]
        smoothed_rgb = smoothed.permute(1, 2, 0)  # [H, W, 3]
        
        # Preserve alpha if present
        if tile.shape[2] > 3:
            result = torch.cat([smoothed_rgb, tile[..., 3:]], dim=-1)
        else:
            result = smoothed_rgb
        
        return result.clamp(0, 1)

# ==========================================
# Image Crop Utilities
# ==========================================    
class ImageCropTiles(io.ComfyNode):
    """
    Crop image into overlapping tiles. 
    Kept original adaptability logic but ensured compatibility with new grid system.
    """
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="ImageCropTiles_MiraSubPack",
            display_name="Image Crop to Tiles",
            category=CAT,
            description="Crop image into overlapping tiles.",
            inputs=[
                io.Image.Input("image", optional=False),
                io.Int.Input("tile_size", default=1024, min=512, max=4096, step=64),
                io.Int.Input("overlap", default=128, min=64, max=256, step=64),
                io.Float.Input("overlap_feather_rate", default=2.0, min=0.1, max=4.0, step=0.1,
                               tooltip="Feather width = overlap × rate. rate=1.0: feather only within overlap zone; rate=2.0 (recommended): extends 1× overlap beyond the boundary for smooth blending."),
                io.Boolean.Input("adaptable_tile_size", default=True),
                io.Float.Input("adaptable_max_deviation_ratio", default=0.25, min=0.1, max=0.5, step=0.05),
                io.Float.Input("adaptable_max_aspect_ratio", default=1.33, min=1.0, max=2.0, step=0.01, 
                               tooltip="Max aspect ratio (W/H or H/W) for adaptable tile sizing.\n5:4=1.25, 4:3=1.33, 16:9=1.78, 21:9=2.33."),
                io.Int.Input("pixel_alignment", default=8, min=8, max=256, step=8, tooltip=TILE_ALIGMENT_TIP   ),
            ],
            outputs=[
                io.Image.Output(display_name="tiled_images"),                             
                MiraITUPipeline.Output(display_name="mira_itu_pipeline"),
                io.String.Output(display_name="mira_itu_pipeline_info"),
                io.Image.Output(display_name="cropped_full_image"),
            ],
            is_output_node=True
        )
        
    @classmethod
    def execute(cls, image, tile_size, overlap, overlap_feather_rate, adaptable_tile_size, adaptable_max_deviation_ratio=0.25, adaptable_max_aspect_ratio=1.33, pixel_alignment=8) -> io.NodeOutput:
        if not isinstance(image, torch.Tensor): raise ValueError("Input 'image' must be a torch.Tensor")        
        if image.ndim == 3: image = image.unsqueeze(0)
        source = image[0]
        H, W, _ = source.shape

        # Input validation
        if tile_size <= overlap:
            print(f"[MiraSubPack:ImageCropTiles] ⚠ Warning: tile_size ({tile_size}) must be larger than overlap ({overlap})")
            tile_size = overlap + pixel_alignment
            print(f"  Auto-adjusted tile_size to {tile_size}")
        
        if W < pixel_alignment or H < pixel_alignment:
            raise ValueError(f"Image dimensions ({W}x{H}) are too small for pixel_alignment ({pixel_alignment})")

        # ===== Center Crop Preprocessing =====
        # Align image dimensions to pixel_alignment using center crop
        source, crop_info = TileHelper._center_crop_to_alignment(source, pixel_alignment)
        H, W = source.shape[0], source.shape[1]
        
        if crop_info["cropped"]:
            print("[MiraSubPack:ImageCropTiles] ⚠ Image center cropped for pixel alignment:")
            print(f"  Original: {crop_info['original_width']}x{crop_info['original_height']}")
            print(f"  Cropped: {W}x{H}")
            print(f"  Crop amount: W={crop_info['crop_amount_w']}px, H={crop_info['crop_amount_h']}px")
            print(f"  Crop position: top={crop_info['crop_top']}, left={crop_info['crop_left']}")

        print(f"[MiraSubPack:ImageCropTiles] Processing image: {W}x{H}")
        print(f"  Tile size: {tile_size}, Overlap: {overlap}, Pixel alignment: {pixel_alignment}")

        effective_tile_width, effective_tile_height = tile_size, tile_size
        if adaptable_tile_size:
            value = int(round(tile_size * adaptable_max_deviation_ratio))
            adaptable_max_deviation = (value // pixel_alignment) * pixel_alignment
            print(f"[MiraSubPack:ImageCropTiles] adaptable_max_deviation set to {adaptable_max_deviation} pixels.")
            effective_tile_width, effective_tile_height = TileHelper._find_optimal_tile_size(W, H, tile_size, overlap, adaptable_max_deviation, adaptable_max_aspect_ratio, pixel_alignment)

        tiles = TileHelper._calculate_tiles(W, H, effective_tile_width, effective_tile_height, overlap, pixel_alignment)
        
        tile_list = []
        for x, y, w, h in tiles:
            tile_img = source[y:y+h, x:x+w, :]
            tile_list.append(tile_img)

        cropped_tiles = torch.stack(tile_list, dim=0)
        pipeline = (W, H, effective_tile_width, effective_tile_height, overlap, overlap_feather_rate, pixel_alignment)
        upscaled_pipeline_info = f"Full: {W}x{H}\nTile: {len(tiles)} -> {effective_tile_width}x{effective_tile_height}\nOverlap: {overlap}\nFeatherRate: {overlap_feather_rate}\nOriginalTileSize: {tile_size}\nAdaptable: {adaptable_tile_size}\nMaxDeviationRatio: {adaptable_max_deviation_ratio}\nMaxAspectRatio: {adaptable_max_aspect_ratio}\nPixelAlignment: {pixel_alignment}"
        # Output the cropped full image for color correction reference
        cropped_full_image = source.unsqueeze(0)  # [H, W, C] -> [1, H, W, C]
        return io.NodeOutput(cropped_tiles, pipeline, upscaled_pipeline_info, cropped_full_image)    

class ImageCropTilesByPixels(io.ComfyNode):
    """
    Crop image into overlapping tiles based on maximum pixels per tile.
    Automatically determines optimal tile size based on image dimensions and max pixels per tile.
    """
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="ImageCropTilesByPixels_MiraSubPack",
            display_name="Image Crop to Tiles by Pixels",
            category=CAT,
            description="Crop image into overlapping tiles based on maximum pixels per tile.",
            inputs=[
                io.Image.Input("image", optional=False),
                io.Float.Input("max_pixels_per_tile", default=1.5, min=1.0, max=8.0, step=0.1,
                            tooltip="Maximum pixels per tile in millions (e.g., 1.0M = 1048576 pixels = 1024x1024)."),                
                io.Int.Input("overlap", default=128, min=64, max=256, step=64),
                io.Float.Input("overlap_feather_rate", default=2.0, min=0.1, max=4.0, step=0.1,
                               tooltip="Feather width = overlap × rate. rate=1.0: feather only within overlap zone; rate=2.0 (recommended): extends 1× overlap beyond the boundary for smooth blending."),
                io.Boolean.Input("adaptable_tile_size", default=True),
                io.Float.Input("adaptable_max_deviation_ratio", default=0.25, min=0.1, max=0.5, step=0.05),
                io.Float.Input("adaptable_max_aspect_ratio", default=1.33, min=1.0, max=2.0, step=0.01, 
                               tooltip="Max aspect ratio (W/H or H/W) for adaptable tile sizing.\n5:4=1.25, 4:3=1.33, 16:9=1.78, 21:9=2.33."),
                io.Int.Input("pixel_alignment", default=8, min=8, max=256, step=8, tooltip=TILE_ALIGMENT_TIP   ),
            ],
            outputs=[
                io.Image.Output(display_name="tiled_images"),                             
                MiraITUPipeline.Output(display_name="mira_itu_pipeline"),
                io.String.Output(display_name="mira_itu_pipeline_info"),
                io.Image.Output(display_name="cropped_full_image"),
            ],
            is_output_node=True
        )
        
    @classmethod
    def execute(cls, image, max_pixels_per_tile, overlap, overlap_feather_rate, adaptable_tile_size, adaptable_max_deviation_ratio=0.25, adaptable_max_aspect_ratio=1.33, pixel_alignment=8) -> io.NodeOutput:
        if not isinstance(image, torch.Tensor): 
            raise ValueError("Input 'image' must be a torch.Tensor")        
        if image.ndim == 3: 
            image = image.unsqueeze(0)
        
        source = image[0]
        H, W, _ = source.shape

        # Input validation
        if W < pixel_alignment or H < pixel_alignment:
            raise ValueError(f"Image dimensions ({W}x{H}) are too small for pixel_alignment ({pixel_alignment})")

        # ===== Center Crop Preprocessing =====
        # Align image dimensions to pixel_alignment using center crop
        source, crop_info = TileHelper._center_crop_to_alignment(source, pixel_alignment)
        H, W = source.shape[0], source.shape[1]
        
        if crop_info["cropped"]:
            print("[MiraSubPack:ImageCropTilesByPixels] ⚠ Image center cropped for pixel alignment:")
            print(f"  Original: {crop_info['original_width']}x{crop_info['original_height']}")
            print(f"  Cropped: {W}x{H}")
            print(f"  Crop amount: W={crop_info['crop_amount_w']}px, H={crop_info['crop_amount_h']}px")
            print(f"  Crop position: top={crop_info['crop_top']}, left={crop_info['crop_left']}")

        # Convert megapixels to pixels (1.0M = 1048576 pixels)
        max_pixels_value = int(max_pixels_per_tile * 1048576)
        
        # Calculate base tile size from max pixels
        # Start with assumption of square tiles
        base_tile_size = int(math.sqrt(max_pixels_value))
        # Align to pixel_alignment
        base_tile_size = (base_tile_size // pixel_alignment) * pixel_alignment
        base_tile_size = max(pixel_alignment * 8, base_tile_size)  # Minimum 8x pixel_alignment
        
        # Ensure tile_size is larger than overlap
        if base_tile_size <= overlap:
            print(f"[MiraSubPack:ImageCropTilesByPixels] ⚠ Warning: calculated tile_size ({base_tile_size}) must be larger than overlap ({overlap})")
            base_tile_size = overlap + pixel_alignment
            print(f"  Auto-adjusted tile_size to {base_tile_size}")
        
        actual_pixels = base_tile_size * base_tile_size
        print(f"[MiraSubPack:ImageCropTilesByPixels] Calculated base tile size: {base_tile_size}x{base_tile_size}")
        print(f"  Max pixels per tile: {max_pixels_per_tile}M ({max_pixels_value} pixels)")
        print(f"  Actual pixels per tile: {actual_pixels}")
        print(f"  Image size: {W}x{H}")

        effective_tile_width, effective_tile_height = base_tile_size, base_tile_size
        if adaptable_tile_size:
            value = int(round(base_tile_size * adaptable_max_deviation_ratio))
            adaptable_max_deviation = (value // pixel_alignment) * pixel_alignment
            print(f"[MiraSubPack:ImageCropTilesByPixels] adaptable_max_deviation set to {adaptable_max_deviation} pixels.")
            effective_tile_width, effective_tile_height = TileHelper._find_optimal_tile_size(W, H, base_tile_size, overlap, adaptable_max_deviation, adaptable_max_aspect_ratio, pixel_alignment)

        tiles = TileHelper._calculate_tiles(W, H, effective_tile_width, effective_tile_height, overlap, pixel_alignment)
        
        tile_list = []
        for x, y, w, h in tiles:
            tile_img = source[y:y+h, x:x+w, :]
            tile_list.append(tile_img)

        cropped_tiles = torch.stack(tile_list, dim=0)
        pipeline = (W, H, effective_tile_width, effective_tile_height, overlap, overlap_feather_rate, pixel_alignment)
        upscaled_pipeline_info = f"Full: {W}x{H}\nTile: {len(tiles)} -> {effective_tile_width}x{effective_tile_height}\nOverlap: {overlap}\nFeatherRate: {overlap_feather_rate}\nMaxPixelsPerTile: {max_pixels_per_tile}M\nAdaptable: {adaptable_tile_size}\nMaxDeviationRatio: {adaptable_max_deviation_ratio}\nMaxAspectRatio: {adaptable_max_aspect_ratio}\nPixelAlignment: {pixel_alignment}"
        print(upscaled_pipeline_info)
        # Output the cropped full image for color correction reference
        cropped_full_image = source.unsqueeze(0)  # [H, W, C] -> [1, H, W, C]
        return io.NodeOutput(cropped_tiles, pipeline, upscaled_pipeline_info, cropped_full_image)    
    
# ==========================================
# Latent Crop Utilities - Optimized Version
# ==========================================    
class LatentUpscaleAndCropTiles(io.ComfyNode):
    """
    Advanced latent upscaler using Bislerp (Hybrid Interpolation) and Variance Matching.
    Outputs tiled latents compatible with OverlappedLatentMerge.
    """
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="LatentUpscaleAndCropTiles_MiraSubPack",
            display_name="Latent Upscale then Crop to Tiles (Advanced)",
            category=CAT,
            description="Upscales latents with statistical correction and splits them into overlapping tiles.",
            inputs=[
                io.Latent.Input("latent", optional=False, tooltip="Input latent to upscale and tile."),
                io.Float.Input("scale_factor", default=1.25, min=0.5, max=8.0, step=0.05),
                io.Combo.Input("upscale_method", default="bislerp", 
                               options=["bislerp", "nearest-exact", "bilinear", "bicubic", "area"],
                               tooltip="Method for upscaling. 'bislerp' combines nearest-exact and bicubic for best results."),
                io.Float.Input("bislerp_strength", default=0.35, min=0.0, max=1.0, step=0.05, 
                               tooltip="Weight of nearest-exact vs bicubic. Higher is sharper."),
                io.Boolean.Input("variance_matching", default=True, 
                                 tooltip="Maintains latent distribution to prevent color/contrast drift."),
                io.Boolean.Input("multi_stage", default=True, tooltip="Iterative upscaling for factors > 2.0."),
                io.Float.Input("noise_strength", default=0.0, min=0.0, max=1.0, step=0.01),
                io.Int.Input("seed", default=0, min=0, max=0xffffffffffffffff),
                io.Int.Input("tile_size", default=1024, min=512, max=4096, step=64),
                io.Int.Input("overlap", default=128, min=64, max=256, step=64),
                io.Float.Input("overlap_feather_rate", default=2.0, min=0.1, max=4.0, step=0.1,
                               tooltip="Feather width = overlap × rate. rate=2.0 (recommended) for overlap=64 gives feather=128px."),
                io.Boolean.Input("adaptable_tile_size", default=True),
                io.Float.Input("adaptable_max_deviation_ratio", default=0.25, step=0.05),
                io.Float.Input("adaptable_max_aspect_ratio", default=1.33, step=0.01),
                io.Int.Input("pixel_alignment", default=8, min=8, max=256, step=8, tooltip=TILE_ALIGMENT_TIP   ),
            ],
            outputs=[
                io.Latent.Output(display_name="tiled_latents"),
                io.Latent.Output(display_name="full_latent"),
                io.Latent.Output(display_name="original_tiled_latents"),
                MiraITUPipeline.Output(),
                io.String.Output(display_name="mira_itu_pipeline_info"),
                io.Int.Output(display_name="original_tile_size"),
                io.Int.Output(display_name="original_width"),
                io.Int.Output(display_name="original_height"),                
            ],
            is_output_node=True
        )

    def _apply_upscale(self, samples, target_h, target_w, method, bislerp_strength, apply_vm, orig_std, orig_mean):
        """Internal helper for high-quality latent upscaling."""
        if method == "bislerp":
            # Hybrid interpolation: Mix Nearest-Exact for structure and Bicubic for smoothness
            s1 = torch.nn.functional.interpolate(samples, size=(target_h, target_w), mode="nearest-exact")
            s2 = torch.nn.functional.interpolate(samples, size=(target_h, target_w), mode="bicubic", align_corners=False)
            upscaled = s1 * bislerp_strength + s2 * (1.0 - bislerp_strength)
        else:
            align = False if method in ["bilinear", "bicubic"] else None
            upscaled = torch.nn.functional.interpolate(samples, size=(target_h, target_w), mode=method, align_corners=align)

        if apply_vm:
            # Variance Matching: Corrects the flattening effect of interpolation
            # to maintain original contrast and texture energy
            current_std = upscaled.std()
            current_mean = upscaled.mean()
            if current_std > 1e-6:
                upscaled = (upscaled - current_mean) * (orig_std / current_std) + orig_mean
        
        return upscaled

    @classmethod
    def execute(cls, latent, scale_factor, upscale_method, bislerp_strength, variance_matching, multi_stage, 
                noise_strength, seed, tile_size, overlap, overlap_feather_rate, 
                adaptable_tile_size, adaptable_max_deviation_ratio, adaptable_max_aspect_ratio, pixel_alignment) -> io.NodeOutput:
        
        samples = latent["samples"]
        B, C, latent_h, latent_w = samples.shape
        device = samples.device
        
        # Calculate target dimensions in pixel space with strict alignment
        orig_width_px = latent_w * pixel_alignment
        orig_height_px = latent_h * pixel_alignment
        
        # Calculate ideal target dimensions
        ideal_width_px = orig_width_px * scale_factor
        ideal_height_px = orig_height_px * scale_factor
        
        # Align to pixel_alignment grid
        target_width_px = max(pixel_alignment, (int(ideal_width_px) // pixel_alignment) * pixel_alignment)
        target_height_px = max(pixel_alignment, (int(ideal_height_px) // pixel_alignment) * pixel_alignment)
        
        # Calculate actual scale factors after alignment
        actual_scale_w = target_width_px / orig_width_px
        actual_scale_h = target_height_px / orig_height_px
        actual_scale_factor = (actual_scale_w + actual_scale_h) / 2
        
        # Warn if scale factor was adjusted significantly
        scale_diff = abs(actual_scale_factor - scale_factor)
        if scale_diff > 0.01:  # More than 1% difference
            print("[MiraSubPack:LatentUpscaleAndCropTiles] ⚠ Scale factor adjusted for pixel alignment:")
            print(f"  Requested: {scale_factor:.4f}x")
            print(f"  Actual: {actual_scale_factor:.4f}x (W: {actual_scale_w:.4f}x, H: {actual_scale_h:.4f}x)")
            print(f"  Original: {orig_width_px}x{orig_height_px} → Target: {target_width_px}x{target_height_px}")
            print(f"  Pixel alignment: {pixel_alignment}")
        
        target_latent_w = target_width_px // pixel_alignment
        target_latent_h = target_height_px // pixel_alignment

        # Store original stats for Variance Matching
        orig_std, orig_mean = samples.std(), samples.mean()
        
        current_samples = samples.clone()        
        
        # --- Upscaling Logic ---
        if multi_stage and scale_factor > 2.0:
            remaining_scale = scale_factor
            while remaining_scale > 1.0:
                # Step by 2.0x or the final remaining fraction
                step_scale = 2.0 if remaining_scale >= 2.0 else remaining_scale
                remaining_scale /= step_scale
                
                is_last = remaining_scale <= 1.0
                step_h = target_latent_h if is_last else int(current_samples.shape[2] * step_scale)
                step_w = target_latent_w if is_last else int(current_samples.shape[3] * step_scale)
                
                current_samples = cls()._apply_upscale(current_samples, step_h, step_w, upscale_method, 
                                                      bislerp_strength, variance_matching, orig_std, orig_mean)
                if is_last: break
        else:
            current_samples = cls()._apply_upscale(current_samples, target_latent_h, target_latent_w, upscale_method, 
                                                  bislerp_strength, variance_matching, orig_std, orig_mean)

        # --- Noise Injection (Using Lerp for energy conservation) ---
        if noise_strength > 0:
            generator = torch.manual_seed(seed)
            noise = torch.randn(current_samples.shape, dtype=current_samples.dtype, device=device, generator=generator)
            # Mix noise instead of raw addition to prevent variance explosion
            current_samples = torch.lerp(current_samples, noise, noise_strength * 0.2)

        # --- Tiling Logic (Batch-Safe) ---
        eff_tile_w, eff_tile_h = tile_size, tile_size
        if adaptable_tile_size:
            max_dev = (int(tile_size * adaptable_max_deviation_ratio) // pixel_alignment) * pixel_alignment
            eff_tile_w, eff_tile_h = TileHelper._find_optimal_tile_size(
                target_width_px, target_height_px, tile_size, overlap, max_dev, adaptable_max_aspect_ratio, pixel_alignment
            )
        
        tile_coords = TileHelper._calculate_tiles(target_width_px, target_height_px, eff_tile_w, eff_tile_h, overlap, pixel_alignment)
        
        all_tiles = []
        for b in range(B):
            single_latent = current_samples[b]
            for x, y, w, h in tile_coords:
                lx, ly = int(x // pixel_alignment), int(y // pixel_alignment)
                lw, lh = int(w // pixel_alignment), int(h // pixel_alignment)
                # Crop with boundary safety
                tile = single_latent[:, ly:min(ly+lh, target_latent_h), lx:min(lx+lw, target_latent_w)].clone()
                all_tiles.append(tile)

        tiled_output = torch.stack(all_tiles, dim=0)
        
        # --- Crop original-size latent with same tiling scheme ---
        # Directly calculate original tile coordinates by dividing upscaled coordinates by actual scale factor
        # This ensures the same number of tiles and matching positions
        print("[MiraSubPack:LatentUpscaleAndCropTiles] Cropping original latent with same tile scheme for reference...")
        print(f"  Original latent size: {latent_w}x{latent_h} (pixels: {orig_width_px}x{orig_height_px})")
        
        original_tile_coords = []
        for x, y, w, h in tile_coords:
            # Scale down coordinates and dimensions using actual scale factors
            orig_x = int(x / actual_scale_w)
            orig_y = int(y / actual_scale_h)
            orig_w = int(w / actual_scale_w)
            orig_h = int(h / actual_scale_h)
            # Align to pixel_alignment grid
            orig_x = (orig_x // pixel_alignment) * pixel_alignment
            orig_y = (orig_y // pixel_alignment) * pixel_alignment
            orig_w = (orig_w // pixel_alignment) * pixel_alignment
            orig_h = (orig_h // pixel_alignment) * pixel_alignment
            # Ensure coordinates don't exceed original image boundaries
            orig_x = min(orig_x, orig_width_px - pixel_alignment)
            orig_y = min(orig_y, orig_height_px - pixel_alignment)
            # Ensure dimensions are valid and within bounds
            orig_w = max(pixel_alignment, min(orig_w, orig_width_px - orig_x))
            orig_h = max(pixel_alignment, min(orig_h, orig_height_px - orig_y))
            original_tile_coords.append((orig_x, orig_y, orig_w, orig_h))
        
        print(f"  Original tile count: {len(original_tile_coords)} (same as upscaled: {len(tile_coords)})")
        print(f"  Original tile size range: {min(w for _,_,w,_ in original_tile_coords)}~{max(w for _,_,w,_ in original_tile_coords)} x {min(h for _,_,_,h in original_tile_coords)}~{max(h for _,_,_,h in original_tile_coords)}")
        print(f"  Original tile coordinates (x, y, w, h): {original_tile_coords[:5]}...")  # Print first 5 for brevity
        
        original_all_tiles = []
        orignial_samples = samples.clone()
        for b in range(B):
            single_latent = orignial_samples[b]
            for x, y, w, h in original_tile_coords:
                lx, ly = int(x // pixel_alignment), int(y // pixel_alignment)
                lw, lh = int(w // pixel_alignment), int(h // pixel_alignment)
                # Crop with boundary safety
                tile = single_latent[:, ly:min(ly+lh, latent_h), lx:min(lx+lw, latent_w)].clone()
                original_all_tiles.append(tile)
        
        original_tiled_output = torch.stack(original_all_tiles, dim=0) if original_all_tiles else samples
        
        # Metadata and Pipeline Info
        pipeline = (target_width_px, target_height_px, eff_tile_w, eff_tile_h, overlap, overlap_feather_rate, pixel_alignment)
        upscaled_pipeline_info = f"Full: {target_width_px}x{target_height_px}\nTile: {len(all_tiles)} -> {eff_tile_w}x{eff_tile_h}\nOverlap: {overlap}\nFeatherRate: {overlap_feather_rate}\nOriginalTileSize: {tile_size}\nPixelAlignment: {pixel_alignment}"
        info = f"{upscaled_pipeline_info}\n\nScale: {actual_scale_factor:.4f}x (requested: {scale_factor}x)\nMethod: {upscale_method}\nTiles: {len(tile_coords)} per batch\nSize: {eff_tile_w}x{eff_tile_h}"

        return io.NodeOutput(
            {"samples": tiled_output}, 
            {"samples": current_samples}, 
            {"samples": original_tiled_output},
            pipeline, info, tile_size, orig_width_px, orig_height_px
        )

# ==========================================
# Upscale Factor Calculator
# ==========================================
class MiraImageUpscaleCalculator(io.ComfyNode):
    """
    Recalculate upscale factor and resize image if it exceeds pixel limit.
    """
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MiraImageUpscaleCalculator_MiraSubPack",
            display_name="Mira Image Upscale Calculator",
            category=CAT,
            description="Recalculate upscale factor and resize image if it exceeds pixel limit.",
            inputs=[
                io.Image.Input("image", tooltip="Input image."),
                io.Float.Input("target_upscale_factor", default=2.0, min=1.0, max=16.0, step=0.1, tooltip="Target upscale factor relative to the ORIGINAL image size."),
                io.Float.Input("limit_megapixels", default=4.0, min=1, max=64.0, step=0.5, tooltip="Maximum allowed megapixels for the input image. If exceeded, image is downscaled to fit."),
                io.Int.Input("pixel_alignment", default=8, min=8, max=128, step=8, tooltip="Pixel alignment for resizing."),
                io.Combo.Input("downscale_method", default="bicubic", options=["nearest-exact", "bilinear", "area", "bicubic", "lanczos"], tooltip="Method for downscaling if limit is exceeded."),
            ],
            outputs=[
                io.Image.Output(display_name="processed_image"),
                io.Float.Output(display_name="new_upscale_factor"),
            ],
        )

    @classmethod
    def execute(cls, image, target_upscale_factor, limit_megapixels, pixel_alignment, downscale_method) -> io.NodeOutput:
        N, H, W, C = image.shape
        limit_pixels = int(limit_megapixels * 1024 * 1024)

        # 0. Center Crop to pixel_alignment BEFORE any calculation
        # This matches VAE behavior which might crop if not aligned.
        aligned_width = (W // pixel_alignment) * pixel_alignment
        aligned_height = (H // pixel_alignment) * pixel_alignment
        
        if aligned_width != W or aligned_height != H:
            print(f"[MiraSubPack:UpscaleCalculator] aligning input image to {pixel_alignment}px grid...")
            crop_h = H - aligned_height
            crop_w = W - aligned_width
            
            # Center crop logic
            top = crop_h // 2
            left = crop_w // 2
            
            # Perform crop on the batch [N, H, W, C]
            image = image[:, top:top+aligned_height, left:left+aligned_width, :]
            
            # Update Dimensions
            print(f"  Cropped Input: {aligned_width}x{aligned_height} (Original: {W}x{H})")
            H, W = aligned_height, aligned_width

        current_pixels = W * H
        
        # Calculate original target dimensions
        target_width = int(round(W * target_upscale_factor))
        target_height = int(round(H * target_upscale_factor))
        
        # Align target dimensions to pixel_alignment
        target_width = (target_width // pixel_alignment) * pixel_alignment
        target_height = (target_height // pixel_alignment) * pixel_alignment
        
        if current_pixels <= limit_pixels:
            print(f"[MiraSubPack:UpscaleCalculator] Image size {W}x{H} ({current_pixels/1e6:.2f}MP) is within limit {limit_megapixels}MP.")
            return io.NodeOutput(image, target_upscale_factor)
            
        print(f"[MiraSubPack:UpscaleCalculator] Image size {W}x{H} ({current_pixels/1e6:.2f}MP) exceeds limit {limit_megapixels}MP. Downscaling...")
        
        scale = math.sqrt(limit_pixels / current_pixels)
        new_width = int(W * scale)
        new_height = int(H * scale)
        
        # Align new dimensions
        new_width = (new_width // pixel_alignment) * pixel_alignment
        new_height = (new_height // pixel_alignment) * pixel_alignment
        
        # Ensure at least 1 alignment block
        new_width = max(pixel_alignment, new_width)
        new_height = max(pixel_alignment, new_height)
        
        # Resize image
        # Permute to [N, C, H, W] for interpolation
        image_permuted = image.permute(0, 3, 1, 2)
        processed_image = comfy.utils.common_upscale(image_permuted, new_width, new_height, downscale_method, "disabled")
        processed_image = processed_image.permute(0, 2, 3, 1)
        
        # Recalculate upscale factor to reach ORIGINAL target size from NEW downscaled size
        # We use the max ratio to ensure coverage
        factor_w = target_width / new_width
        factor_h = target_height / new_height
        new_factor = max(factor_w, factor_h)
        
        print(f"[MiraSubPack:UpscaleCalculator] Resized to {new_width}x{new_height}. New upscale factor: {new_factor:.4f} (Original target: {target_width}x{target_height})")
        
        return io.NodeOutput(processed_image, new_factor)
    
