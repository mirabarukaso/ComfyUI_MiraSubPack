import torch
import math
import comfy.sample
import comfy.samplers
import latent_preview
from comfy_api.latest import io
import node_helpers

CAT = "Mira/SubPack/Image Tiled Upscaler"

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
                io.Int.Input("full_width", optional=False, tooltip="Full image width."),
                io.Int.Input("full_height", optional=False, tooltip="Full image height."),
                io.Int.Input("tile_width", optional=False, tooltip="Tile width."),
                io.Int.Input("tile_height", optional=False, tooltip="Tile height."),
                io.Int.Input("overlap", optional=False, tooltip="Overlap pixels."),
                io.Float.Input("overlap_feather_rate", optional=False, tooltip="Overlap feather rate."),
                io.Int.Input("pixel_alignment", optional=False, tooltip="Pixel alignment for tile calculations."),
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

        # Create ramp gradients
        ramp = torch.linspace(0, 1, feather, device=device, dtype=torch.float32)
        
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
                io.Clip.Input("clip_negative", optional=True),
                io.Latent.Input("ref_latents", optional=True),
            ],
            outputs=[
                io.Latent.Output(display_name="tiled_latents"),
            ],
        )
    
    @classmethod
    def execute(cls, model, clip, tiled_samples, common_positive, common_negative, tagger_text,
               seed, steps, cfg, sampler_name, scheduler, denoise, clip_negative = None, ref_latents = None
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
        
        if ref_latents is not None:
            print("    >Using provided reference latents for this tile.")
                            
        # Parse tagger text mapping
        mapping = tagger_text.splitlines()
        tile_latents = None
        for idx in range(len(batch_latents)):                    
            if tagger_text != "":
                # Dynamic prompt construction
                dynamic_prompt = common_positive
                tags_str = ""
                if idx < len(mapping):
                    tags_str = mapping[idx].replace('(', r'\(').replace(')', r'\)')
                    dynamic_prompt = f"{common_positive}, {tags_str}" if common_positive else tags_str
                                
                print(f"    Tags: {tags_str}")
                positive_tokens = clip.tokenize(dynamic_prompt)
                positive_conditioning = clip.encode_from_tokens_scheduled(positive_tokens)
            else:
                positive_tokens = clip.tokenize(common_positive)
                positive_conditioning = clip.encode_from_tokens_scheduled(positive_tokens)
            
            if ref_latents is not None:
                all_ref_latent = ref_latents["samples"]                
                ref_latent = all_ref_latent[idx].unsqueeze(0)                                                
                
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
                torch.cuda.empty_cache()                        
            
            single_latent = batch_latents[idx].unsqueeze(0)  # [C, H, W] -> [1, C, H, W]                
            print(f"  > Sampling Tile {idx+1}/{len(batch_latents)}: {single_latent.shape[3]}x{single_latent.shape[2]}")
            print(f"    Tile latent shape: {single_latent.shape}")
            sampled_tile = cls._sample_single(
                model, positive_conditioning, negative_conditioning, {"samples": single_latent},
                seed, steps, cfg, sampler_name, scheduler, denoise
            )            
            tile_latents = torch.cat([tile_latents, sampled_tile["samples"]], dim=0) if tile_latents is not None else sampled_tile["samples"]
            
            if ref_latents is not None:
                del positive_conditioning
                del negative_conditioning
                
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
        feather_px = max(overlap * 4, int(overlap * overlap_feather_rate))
        feather_px = min(max(tile_width, tile_height) * 0.25, feather_px)
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
                io.Float.Input("feather_rate_override", default=0, min=0, max=4.0, step=0.1, tooltip="Override fathering rate multiplier if value is not 0."),
            ],
            outputs=[
                io.Image.Output()
            ],
        )
        
    @classmethod
    def execute(cls, tiled_images, mira_itu_pipeline, feather_rate_override) -> io.NodeOutput:
        (full_width, full_height, tile_width, tile_height, overlap, overlap_feather_rate, pixel_alignment) = mira_itu_pipeline
        if round(feather_rate_override,2) != 0:
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
        weight_map = weight_map.clamp(min=1e-5)
        canvas = canvas / weight_map
        
        return io.NodeOutput(canvas.unsqueeze(0))

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
                io.Int.Input("overlap", default=64, min=64, max=256, step=64),
                io.Float.Input("overlap_feather_rate", default=1.0, min=0.1, max=4.0, step=0.1, tooltip="Feathering rate multiplier."),
                io.Boolean.Input("adaptable_tile_size", default=True),
                io.Float.Input("adaptable_max_deviation_ratio", default=0.25, min=0.1, max=0.5, step=0.05),
                io.Float.Input("adaptable_max_aspect_ratio", default=1.33, min=1.0, max=2.0, step=0.01, 
                               tooltip="Max aspect ratio (W/H or H/W) for adaptable tile sizing.\n5:4=1.25, 4:3=1.33, 16:9=1.78, 21:9=2.33."),
                io.Int.Input("pixel_alignment", default=8, min=8, max=256, step=8, tooltip="Align tile dimensions to multiples of this value (e.g., 8 for SDXL, 16 for FLUX.2)."),
            ],
            outputs=[
                io.Image.Output(display_name="tiled_images"),                             
                MiraITUPipeline.Output(display_name="mira_itu_pipeline"),
                io.String.Output(display_name="mira_itu_pipeline_info"),
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
        return io.NodeOutput(cropped_tiles, pipeline, upscaled_pipeline_info)    

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
                io.Int.Input("overlap", default=64, min=64, max=256, step=64),
                io.Float.Input("overlap_feather_rate", default=1.0, min=0.1, max=4.0, step=0.1, 
                               tooltip="Feathering rate multiplier."),
                io.Boolean.Input("adaptable_tile_size", default=True),
                io.Float.Input("adaptable_max_deviation_ratio", default=0.25, min=0.1, max=0.5, step=0.05),
                io.Float.Input("adaptable_max_aspect_ratio", default=1.33, min=1.0, max=2.0, step=0.01, 
                               tooltip="Max aspect ratio (W/H or H/W) for adaptable tile sizing.\n5:4=1.25, 4:3=1.33, 16:9=1.78, 21:9=2.33."),
                io.Int.Input("pixel_alignment", default=8, min=8, max=256, step=8, tooltip="Align tile dimensions to multiples of this value (e.g., 8 for SDXL, 16 for FLUX.2)."   ),
            ],
            outputs=[
                io.Image.Output(display_name="tiled_images"),                             
                MiraITUPipeline.Output(display_name="mira_itu_pipeline"),
                io.String.Output(display_name="mira_itu_pipeline_info"),
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
        return io.NodeOutput(cropped_tiles, pipeline, upscaled_pipeline_info)    
    
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
                io.Int.Input("overlap", default=64, min=64, max=256, step=64),
                io.Float.Input("overlap_feather_rate", default=1.0, min=0.1, max=4.0, step=0.1),
                io.Boolean.Input("adaptable_tile_size", default=True),
                io.Float.Input("adaptable_max_deviation_ratio", default=0.25, step=0.05),
                io.Float.Input("adaptable_max_aspect_ratio", default=1.33, step=0.01),
                io.Int.Input("pixel_alignment", default=8, tooltip="8 for SD1.5/SDXL, 16 for FLUX."),
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
    
class LatentUpscaleSimple(io.ComfyNode):
    """
    Upscales input latent.
    """
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="LatentUpscaleSimple_MiraSubPack",
            display_name="Latent Upscale with Add noise",
            category=CAT,
            description="Upscale latent.",
            inputs=[
                io.Latent.Input("latent", optional=False, tooltip="Input latent to upscale and tile."),
                io.Float.Input("scale_factor", default=1.25, min=0.5, max=8.0, step=0.05,
                              tooltip="Upscaling factor (e.g., 2.0 = double size)."),
                io.Combo.Input("upscale_method", default="nearest",
                              options=["nearest", "bilinear", "bicubic", "area", "nearest-exact"],
                              tooltip="Interpolation method for upscaling."),
                io.Boolean.Input("multi_stage", default=True,
                                tooltip="Use multi-stage upscaling for factors > 2.0 (smoother results)."),
                io.Float.Input("noise_strength", default=0.0, min=0.0, max=1.0, step=0.01,
                              tooltip="Add noise to upscaled latent (helps with detail generation)."),
                io.Int.Input("seed", default=0, min=0, max=0xffffffffffffffff,
                            tooltip="Seed for noise generation."),
                io.Int.Input("pixel_alignment", default=8, min=8, max=256, step=8, tooltip="Align dimensions to multiples of this value (e.g., 8 for SDXL, 16 for FLUX.2)."   ),
            ],
            outputs=[
                io.Latent.Output(display_name="sample"),             
            ],
            is_output_node=True
        )
    
    @classmethod
    def execute(cls, latent, scale_factor, upscale_method, multi_stage, noise_strength, seed, pixel_alignment) -> io.NodeOutput:
        """
        Upscale latent and add noise.
        """
        samples = latent["samples"]
        B, C, latent_h, latent_w = samples.shape
        
        # Original dimensions in pixel space
        orig_width = latent_w * pixel_alignment
        orig_height = latent_h * pixel_alignment
        
        # Calculate ideal target dimensions
        ideal_width = orig_width * scale_factor
        ideal_height = orig_height * scale_factor
        
        # Align to pixel_alignment
        new_width = (int(ideal_width) // pixel_alignment) * pixel_alignment
        new_height = (int(ideal_height) // pixel_alignment) * pixel_alignment
        new_width = max(pixel_alignment, new_width)
        new_height = max(pixel_alignment, new_height)
        
        # Calculate actual scale factors after alignment
        actual_scale_w = new_width / orig_width
        actual_scale_h = new_height / orig_height
        actual_scale_factor = (actual_scale_w + actual_scale_h) / 2
        
        # Warn if scale factor was adjusted significantly
        scale_diff = abs(actual_scale_factor - scale_factor)
        if scale_diff > 0.01:  # More than 1% difference
            print("[MiraSubPack:LatentUpscaleSimple] ⚠ Scale factor adjusted for pixel alignment:")
            print(f"  Requested: {scale_factor:.4f}x")
            print(f"  Actual: {actual_scale_factor:.4f}x (W: {actual_scale_w:.4f}x, H: {actual_scale_h:.4f}x)")
            print(f"  Original: {orig_width}x{orig_height} → Target: {new_width}x{new_height}")
            print(f"  Pixel alignment: {pixel_alignment}")
        
        new_latent_w = new_width // pixel_alignment
        new_latent_h = new_height // pixel_alignment
        
        print("[MiraSubPack:LatentUpscalerAdvanced] Upscaling latent:")
        print(f"  Original: {orig_width}x{orig_height} ({latent_w}x{latent_h} latent)")
        print(f"  Target: {new_width}x{new_height} ({new_latent_w}x{new_latent_h} latent)")
        print(f"  Scale: {actual_scale_factor:.3f}x")
        print(f"  Method: {upscale_method}")
        
        # Perform upscaling
        current_samples = samples
        
        if multi_stage and scale_factor >= 2.0:
            # Multi-stage upscaling
            stages = []
            remaining_scale = scale_factor
            
            while remaining_scale >= 2.0:
                stages.append(2.0)
                remaining_scale /= 2.0
            
            if remaining_scale > 1.0:
                stages.append(remaining_scale)
            
            print(f"  Multi-stage: {len(stages)} stages {stages}")
            
            for i, stage_scale in enumerate(stages):
                current_h = current_samples.shape[2]
                current_w = current_samples.shape[3]
                stage_h = int(current_h * stage_scale)
                stage_w = int(current_w * stage_scale)
                
                current_samples = torch.nn.functional.interpolate(
                    current_samples,
                    size=(stage_h, stage_w),
                    mode=upscale_method,
                    align_corners=False if upscale_method in ["bilinear", "bicubic"] else None
                )
                
                print(f"    Stage {i+1}: {current_h}x{current_w} -> {stage_h}x{stage_w}")
        else:
            # Single-stage upscaling
            current_samples = torch.nn.functional.interpolate(
                current_samples,
                size=(new_latent_h, new_latent_w),
                mode=upscale_method,
                align_corners=False if upscale_method in ["bilinear", "bicubic"] else None
            )
        
        # Add noise if requested
        if noise_strength > 0:
            noise = torch.randn(current_samples.shape, dtype=current_samples.dtype, device=current_samples.device, generator=torch.manual_seed(seed+1))
            current_samples = current_samples + noise * noise_strength
            print(f"  Added noise: strength={noise_strength:.3f}, seed={seed}")
        
        # Now split the upscaled latent into tiles
        upscaled_latent = current_samples[0]  # [C, H, W]
                
        return io.NodeOutput({"samples": upscaled_latent.unsqueeze(0)})
    