import torch
import math
import comfy.sample
import comfy.samplers
import latent_preview
from comfy_api.latest import io

CAT = "Mira/SubPack"

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
    def _calculate_tiles(width, height, tile_size, overlap):
        """Standard grid calculation."""
        tiles = []
        step_x = tile_size - overlap
        step_y = tile_size - overlap
        
        # Ensure we cover the whole area
        tiles_x = math.ceil((width - overlap) / step_x) if width > tile_size else 1
        tiles_y = math.ceil((height - overlap) / step_y) if height > tile_size else 1
        
        for i in range(tiles_y):
            for j in range(tiles_x):
                x = j * step_x
                y = i * step_y
                
                # Align last tiles to edges
                if x + tile_size > width: x = width - tile_size
                if y + tile_size > height: y = height - tile_size
                
                # Sanity check for small images
                x = max(0, x)
                y = max(0, y)
                
                # Snap to grid (8px)
                x = (int(x) // 8) * 8
                y = (int(y) // 8) * 8
                w = tile_size
                h = tile_size
                
                # Clamp dimensions if image is smaller than tile_size
                w = min(w, width - x)
                h = min(h, height - y)
                w = (w // 8) * 8
                h = (h // 8) * 8
                
                if w > 0 and h > 0:
                    tiles.append((x, y, w, h))
                    
        return sorted(list(set(tiles)), key=lambda t: (t[1], t[0]))

    @staticmethod
    def _find_optimal_tile_size(W, H, base_tile_size, overlap, max_deviation):
        if base_tile_size <= overlap: 
            aligned = (base_tile_size // 8) * 8
            return aligned
        
        longer = max(W, H)
        best_effective = base_tile_size
        best_score = float('inf')
        
        for adj in range(-max_deviation, max_deviation + 1):
            effective = base_tile_size + adj
            if effective <= overlap: 
                continue
            
            step = effective - overlap
            if step <= 0: 
                continue
            
            # Calculate number of tiles needed
            n_long = math.ceil(longer / step)
            
            # Actual coverage: (number of tiles - 1) * step + effective
            coverage = (n_long - 1) * step + effective
            
            # Extra pixels
            extra = coverage - longer
            
            # Must fully cover, extra >= 0 is guaranteed
            if extra < 0: 
                continue
            
            score = extra + abs(adj) * 0.1
            if score < best_score:
                best_score = score
                best_effective = effective
        
        # Ensure alignment to 8px
        best_effective = (best_effective // 8) * 8
        
        # Check coverage again
        step = best_effective - overlap
        n_long = math.ceil(longer / step)
        coverage = (n_long - 1) * step + best_effective
        
        # In case of insufficient coverage, align up to next multiple of 8
        while coverage < longer:
            best_effective += 8
            step = best_effective - overlap
            n_long = math.ceil(longer / step)
            coverage = (n_long - 1) * step + best_effective
        
        return best_effective
        
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
                
                #io.Int.Input("full_width", default=0, min=0, max=65536, step=8, tooltip="Full image width."),
                #io.Int.Input("full_height", default=0, min=0, max=65536, step=8, tooltip="Full image height."),
                #io.Int.Input("tile_size", default=1280, min=512, max=2048, step=64),
                #io.Int.Input("overlap", default=64, min=64, max=256, step=64),
            ],
            outputs=[
                io.Latent.Output(display_name="tiled_latents"),
            ],
        )
    
    @classmethod
    def execute(cls, model, clip, tiled_samples, common_positive, common_negative, tagger_text,
               seed, steps, cfg, sampler_name, scheduler, denoise 
               #full_width, full_height, tile_size, overlap
               ) -> io.NodeOutput:                                
        negative_tokens = clip.tokenize(common_negative)
        negative_conditioning = clip.encode_from_tokens_scheduled(negative_tokens)

        batch_latents = tiled_samples["samples"]
        print(f"[MiraSubPack:AutoTiledTagger] Using {len(batch_latents)} tiles.")
        
        # Parse tagger text mapping
        mapping = tagger_text.splitlines()
        tile_latents = None
        for idx in range(len(batch_latents)):
            # Dynamic prompt construction
            dynamic_prompt = common_positive
            tags_str = ""
            if idx < len(mapping):
                tags_str = mapping[idx].replace('(', r'\(').replace(')', r'\)')
                dynamic_prompt = f"{common_positive}, {tags_str}" if common_positive else tags_str
            
            single_latent = batch_latents[idx].unsqueeze(0)  # [C, H, W] -> [1, C, H, W]
            
            print(f"  > Sampling Tile {idx+1}/{len(batch_latents)}: {single_latent.shape[3]*8}x{single_latent.shape[2]*8}")
            print(f"    Tags: {tags_str}")
            positive_tokens = clip.tokenize(dynamic_prompt)
            positive_conditioning = clip.encode_from_tokens_scheduled(positive_tokens)
            
            print(f"    Tile latent shape: {single_latent.shape}")
            sampled_tile = cls._sample_single(
                model, positive_conditioning, negative_conditioning, {"samples": single_latent},
                seed, steps, cfg, sampler_name, scheduler, denoise
            )            
            tile_latents = torch.cat([tile_latents, sampled_tile["samples"]], dim=0) if tile_latents is not None else sampled_tile["samples"]
                    
        return io.NodeOutput({"samples": tile_latents})
    
    @staticmethod
    def _crop_latent(samples, x, y, width, height):
        latent = samples["samples"]
        lx, ly, lw, lh = x//8, y//8, width//8, height//8
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
                io.Int.Input("full_width", default=0, min=0, max=65536, step=8, tooltip="Full image width."),
                io.Int.Input("full_height", default=0, min=0, max=65536, step=8, tooltip="Full image height."),
                io.Int.Input("tile_size", default=1024, min=512, max=4096, step=64),
                io.Int.Input("overlap", default=64, min=64, max=256, step=64),
                io.Float.Input("overlap_feather_rate", default=1.0, min=0.1, max=4.0, step=0.1, tooltip="Feathering rate multiplier."),
            ],
            outputs=[
                io.Latent.Output()
            ],
        )

    @classmethod
    def execute(cls, tiled_latents, full_width, full_height, tile_size, overlap, overlap_feather_rate) -> io.NodeOutput:
        device = tiled_latents["samples"].device
        dtype = tiled_latents["samples"].dtype
        batch_latents = tiled_latents["samples"]
        
        # 1. Recalculate tile positions
        tiles = TileHelper._calculate_tiles(full_width, full_height, tile_size, overlap)
        
        # 2. Setup Canvas
        lw = full_width // 8
        lh = full_height // 8
        channels = batch_latents.shape[1]
        
        output = torch.zeros((1, channels, lh, lw), device=device, dtype=dtype)
        weight_map = torch.zeros((1, 1, lh, lw), device=device, dtype=torch.float32)
        
        # 3. Feathering params
        feather_px = max(overlap * 4, int(overlap * overlap_feather_rate))
        feather_px = min(tile_size * 0.25, feather_px)
        l_feather = int(feather_px // 8)
        
        # Track previous tile end positions to detect overlap ratio
        # row_y -> max_x_end
        row_last_x_end = {} 
        # col_x -> max_y_end
        col_last_y_end = {}

        print(f"[MiraSubPack:OverlappedLatentMerge] Merging {len(tiles)} tiles for canvas {full_width}x{full_height}...")
        for idx, (x, y, w, h) in enumerate(tiles):
            # Extract current tile
            tile_latent = batch_latents[idx] # [C, H, W]
            
            lx, ly = x // 8, y // 8
            lw_tile, lh_tile = w // 8, h // 8
            
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
                io.Int.Input("full_width", default=0, min=0, max=65536, step=8, tooltip="Full image width."),
                io.Int.Input("full_height", default=0, min=0, max=65536, step=8, tooltip="Full image height."),
                io.Int.Input("tile_size", default=1280, min=512, max=4096, step=64),
                io.Int.Input("overlap", default=64, min=64, max=256, step=64),
                io.Float.Input("overlap_feather_rate", default=1.0, min=0.1, max=4.0, step=0.1, tooltip="Feathering rate multiplier."),
            ],
            outputs=[
                io.Image.Output()
            ],
        )
        
    @classmethod
    def execute(cls, tiled_images, full_width, full_height, tile_size, overlap, overlap_feather_rate) -> io.NodeOutput:
        device = tiled_images.device
        N, H, W, C = tiled_images.shape
        
        # 1. Calculate tile positions
        tiles = TileHelper._calculate_tiles(full_width, full_height, tile_size, overlap)
        
        # 2. Setup Canvas
        # canvas needs to hold color, so it uses C channels (usually 3)
        canvas = torch.zeros((full_height, full_width, C), device=device, dtype=torch.float32)
        # weight_map only needs to track accumulated weight, so it uses 1 channel
        weight_map = torch.zeros((full_height, full_width, 1), device=device, dtype=torch.float32)
        
        feather = max(overlap * 4, int(overlap * overlap_feather_rate))
        feather = int(min(tile_size * 0.25, feather))
        
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
                io.Boolean.Input("adaptable_tile_size", default=True),
                io.Int.Input("adaptable_max_deviation", default=256, min=64, max=1024, step=64),
            ],
            outputs=[
                io.Image.Output(display_name="tiled_images"),
                io.Int.Output(display_name="full_width"),
                io.Int.Output(display_name="full_height"),
                io.Int.Output(display_name="effective_tile_size"),
                io.Int.Output(display_name="tile_overlap"),
                io.Int.Output(display_name="original_tile_size"),
            ],
            is_output_node=True
        )
        
    @classmethod
    def execute(cls, image, tile_size, overlap, adaptable_tile_size, adaptable_max_deviation=256) -> io.NodeOutput:
        if not isinstance(image, torch.Tensor): raise ValueError("Input 'image' must be a torch.Tensor")        
        if image.ndim == 3: image = image.unsqueeze(0)
        source = image[0]
        H, W, _ = source.shape

        effective_tile_size = tile_size
        if adaptable_tile_size:
            effective_tile_size = TileHelper._find_optimal_tile_size(W, H, tile_size, overlap, adaptable_max_deviation)

        tiles = TileHelper._calculate_tiles(W, H, effective_tile_size, overlap)
        
        tile_list = []
        for x, y, w, h in tiles:
            tile_img = source[y:y+h, x:x+w, :]
            tile_list.append(tile_img)

        cropped_tiles = torch.stack(tile_list, dim=0)
        return io.NodeOutput(cropped_tiles, W, H, effective_tile_size, overlap, tile_size)    
    
# ==========================================
# Latent Crop Utilities
# ==========================================    
class LatentUpscaleAndCropTiles(io.ComfyNode):
    """
    Advanced latent upscaler that outputs tiled latents for OverlappedLatentMerge.
    Upscales input latent and splits it into overlapping tiles.
    """
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="LatentUpscaleAndCropTiles_MiraSubPack",
            display_name="Latent Upscale then Crop to Tiles",
            category=CAT,
            description="Upscale latent and split into overlapping tiles for further processing.",
            inputs=[
                io.Latent.Input("latent", optional=False, tooltip="Input latent to upscale and tile."),
                io.Float.Input("scale_factor", default=2.0, min=0.5, max=8.0, step=0.25,
                              tooltip="Upscaling factor (e.g., 2.0 = double size)."),
                io.Combo.Input("upscale_method", default="bicubic",
                              options=["nearest", "bilinear", "bicubic", "area"],
                              tooltip="Interpolation method for upscaling."),
                io.Boolean.Input("multi_stage", default=True,
                                tooltip="Use multi-stage upscaling for factors > 2.0 (smoother results)."),
                io.Float.Input("noise_strength", default=0.0, min=0.0, max=1.0, step=0.01,
                              tooltip="Add noise to upscaled latent (helps with detail generation)."),
                io.Int.Input("seed", default=0, min=0, max=0xffffffffffffffff,
                            tooltip="Seed for noise generation."),
                io.Int.Input("tile_size", default=1024, min=512, max=4096, step=64,
                            tooltip="Size of each tile in pixels (will be converted to latent space)."),
                io.Int.Input("overlap", default=64, min=64, max=256, step=64,
                            tooltip="Overlap between tiles in pixels."),
                io.Boolean.Input("adaptable_tile_size", default=True),
                io.Int.Input("adaptable_max_deviation", default=256, min=64, max=1024, step=64),
            ],
            outputs=[
                io.Latent.Output(display_name="tiled_latents"),
                io.Int.Output(display_name="full_width"),
                io.Int.Output(display_name="full_height"),
                io.Int.Output(display_name="effective_tile_size"),
                io.Int.Output(display_name="overlap"),
                io.Int.Output(display_name="original_tile_size"),
                io.Int.Output(display_name="original_width"),
                io.Int.Output(display_name="original_height"),
            ],
            is_output_node=True
        )
    
    @classmethod
    def execute(cls, latent, scale_factor, upscale_method, multi_stage, 
                noise_strength, seed, tile_size, overlap, adaptable_tile_size, adaptable_max_deviation) -> io.NodeOutput:
        """
        Upscale latent and split into tiles for OverlappedLatentMerge.
        
        Returns tiled latents in batch format [N, C, H, W] where N is number of tiles.
        """
        samples = latent["samples"]
        B, C, latent_h, latent_w = samples.shape
        
        # Original dimensions in pixel space
        orig_width = latent_w * 8
        orig_height = latent_h * 8
        
        # Calculate target dimensions
        new_width = int(orig_width * scale_factor)
        new_height = int(orig_height * scale_factor)
        
        # Align to 8px
        new_width = (new_width // 8) * 8
        new_height = (new_height // 8) * 8
        new_width = max(8, new_width)
        new_height = max(8, new_height)
        
        new_latent_w = new_width // 8
        new_latent_h = new_height // 8
        
        print("[MiraSubPack:LatentUpscalerAdvanced] Upscaling latent:")
        print(f"  Original: {orig_width}x{orig_height} ({latent_w}x{latent_h} latent)")
        print(f"  Target: {new_width}x{new_height} ({new_latent_w}x{new_latent_h} latent)")
        print(f"  Scale: {scale_factor:.3f}x")
        print(f"  Method: {upscale_method}")
        
        # Perform upscaling
        current_samples = samples
        
        if multi_stage and scale_factor > 2.0:
            # Multi-stage upscaling
            stages = []
            remaining_scale = scale_factor
            
            while remaining_scale > 2.0:
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
            noise = torch.randn(current_samples.shape, dtype=current_samples.dtype, device=current_samples.device, generator=torch.manual_seed(seed))
            current_samples = current_samples + noise * noise_strength
            print(f"  Added noise: strength={noise_strength:.3f}, seed={seed}")
        
        # Now split the upscaled latent into tiles
        upscaled_latent = current_samples[0]  # [C, H, W]
        
        effective_tile_size = tile_size
        if adaptable_tile_size:
            effective_tile_size = TileHelper._find_optimal_tile_size(new_width, new_height, tile_size, overlap, adaptable_max_deviation)
        
        # Calculate tile positions in pixel space
        tiles = TileHelper._calculate_tiles(new_width, new_height, effective_tile_size, overlap)
        
        print(f"[MiraSubPack:LatentUpscalerAdvanced] Splitting into {len(tiles)} tiles:")
        print(f"  Tile size: {effective_tile_size}px, Overlap: {overlap}px")
        
        # Crop tiles from upscaled latent
        tile_list = []
        for idx, (x, y, w, h) in enumerate(tiles):
            # Convert to latent space coordinates
            lx, ly = x // 8, y // 8
            lw, lh = w // 8, h // 8
            
            # Crop tile from upscaled latent
            tile_latent = upscaled_latent[:, ly:ly+lh, lx:lx+lw].clone()
            tile_list.append(tile_latent)
            
            print(f"  Tile {idx+1}: position=({x},{y}) size={w}x{h} latent_size={lw}x{lh}")
        
        # Stack tiles into batch format [N, C, H, W]
        tiled_latents = torch.stack(tile_list, dim=0)
        
        print("[MiraSubPack:LatentUpscalerAdvanced] Output:")
        print(f"  Tiled latents shape: {tiled_latents.shape}")
        print(f"  Ready for OverlappedLatentMerge with full_size={new_width}x{new_height}")
        
        return io.NodeOutput(
            {"samples": tiled_latents},  # Format compatible with OverlappedLatentMerge
            new_width,
            new_height,
            effective_tile_size,
            overlap,
            tile_size,
            orig_width,
            orig_height
        )