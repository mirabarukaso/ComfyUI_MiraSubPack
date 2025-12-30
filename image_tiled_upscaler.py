"""
Thought process: Tile large image -> Use CL Tagger on tiles -> Generate local prompts -> Sample locally -> Merge
"""

import torch
import math
import os
import json
import gc
import cv2
import numpy as np
import onnxruntime as ort
from PIL import Image
import comfy.sample
import comfy.samplers
import latent_preview

# ==========================================
# 1. Setup
# ==========================================

CAT = "Mira/SubPack"
ONNX_PATH = ""

def get_onnx_models_from_path(base_path):
    """Recursively find all .onnx files in the given directory."""
    if not os.path.exists(base_path) or not os.path.isdir(base_path):
        return []
    onnx_models = []
    for root, dirs, files in os.walk(base_path):
        for file in files:
            if file.lower().endswith('.onnx'):
                full_path = os.path.join(root, file)
                relative_path = os.path.relpath(full_path, base_path).replace(os.sep, '/')
                onnx_models.append(relative_path)
    onnx_models.sort()
    return onnx_models

def init_onnx_path():
    """Locate ComfyUI root and set ONNX models path."""
    global ONNX_PATH
    current_file = os.path.abspath(__file__)
    base = os.path.dirname(current_file)
    while True:
        if os.path.basename(base) == "custom_nodes":
            comfy_root = os.path.dirname(base)
            ONNX_PATH = os.path.join(comfy_root, "models", "onnx")
            break
        new_base = os.path.dirname(base)
        if new_base == base:  # Reached filesystem root
            print("[MiraSubPack:TaggerSampler] Warning: Could not find ComfyUI root, using relative path.")
            ONNX_PATH = os.path.join(base, "models", "onnx")
            break
        base = new_base
    
    if not os.path.exists(ONNX_PATH):
        os.makedirs(ONNX_PATH, exist_ok=True)

init_onnx_path()
onnx_list = get_onnx_models_from_path(ONNX_PATH)
if not onnx_list:
    onnx_list = ["None"]

# ==========================================
# 2. CL Tagger
# ==========================================
class cl_tagger:
    """
    CL Tagger implementation (based on cella110n's model).
    Generates tags from image tiles using an ONNX model.
    """
    
    def __init__(self):
        self._mean = np.array([0.5, 0.5, 0.5], dtype=np.float32).reshape(3, 1, 1)
        self._std = np.array([0.5, 0.5, 0.5], dtype=np.float32).reshape(3, 1, 1)
        self._tag_mapping_cache = {}
        self._cpu_session = None
        self._gpu_session = None
    
    def get_tag_mapping(self, full_tag_map_path):
        """Load and cache tag mapping JSON."""
        if full_tag_map_path not in self._tag_mapping_cache:
            print("[MiraSubPack:ClTagger] Loading tag mapping: " + full_tag_map_path)
            self._tag_mapping_cache[full_tag_map_path] = self.load_tag_mapping(full_tag_map_path)        
        return self._tag_mapping_cache[full_tag_map_path]

    def get_session(self, model_path, session_method):
        """Create or reuse ONNX inference session (CPU or GPU)."""
        if session_method.startswith('CPU'):
            if not self._cpu_session:                
                self._cpu_session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
            return self._cpu_session
        elif session_method.startswith('GPU'):
            if not self._gpu_session:
                self._gpu_session = ort.InferenceSession(model_path, providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
            return self._gpu_session
        
    def pad_square_np(self, img_array: np.ndarray) -> np.ndarray:
        """Pad image to square with white borders."""
        h, w, _ = img_array.shape
        if h == w:
            return img_array
        new_size = max(h, w)
        pad_top = (new_size - h) // 2
        pad_bottom = new_size - h - pad_top
        pad_left = (new_size - w) // 2
        pad_right = new_size - w - pad_left
        return np.pad(img_array, ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0)),
                      mode='constant', constant_values=255)

    def preprocess_image(self, image, target_size=(448, 448)):
        """Preprocess PIL image for ONNX tagger input."""
        img = np.array(image)
        img = self.pad_square_np(img)
        img = cv2.resize(img, target_size, interpolation=cv2.INTER_CUBIC)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        img = img.astype(np.float32) / 255.0
        img = (img - 0.5) / 0.5
        img = np.expand_dims(np.transpose(img, (2, 0, 1)), axis=0)
        return img

    def load_tag_mapping(self, mapping_path):
        """Load tag mapping JSON and organize by category."""
        with open(mapping_path, 'r', encoding='utf-8') as f:
            tag_mapping_data = json.load(f)
        # Support multiple JSON formats
        if isinstance(tag_mapping_data, dict) and "idx_to_tag" in tag_mapping_data:
            idx_to_tag = {int(k): v for k, v in tag_mapping_data["idx_to_tag"].items()}
            tag_to_category = tag_mapping_data["tag_to_category"]
        elif isinstance(tag_mapping_data, dict):
            try:
                tag_mapping_data_int_keys = {int(k): v for k, v in tag_mapping_data.items()}
                idx_to_tag = {idx: data['tag'] for idx, data in tag_mapping_data_int_keys.items()}
                tag_to_category = {data['tag']: data['category'] for data in tag_mapping_data_int_keys.values()}
            except (KeyError, ValueError) as e:
                raise ValueError(f"Unsupported tag mapping format (dict): {e}.")
        else:
            raise ValueError("Unsupported tag mapping format: Expected a dictionary.")

        # Organize tags by category
        names = [None] * (max(idx_to_tag.keys()) + 1)
        rating, general, artist, character, copy_right, meta, quality, model_name = [], [], [], [], [], [], [], []
        for idx, tag in idx_to_tag.items():
            if idx >= len(names):
                names.extend([None] * (idx - len(names) + 1))
            names[idx] = tag
            category = tag_to_category.get(tag, 'Unknown')
            idx_int = int(idx)
            if category == 'Rating': rating.append(idx_int)
            elif category == 'General': general.append(idx_int)
            elif category == 'Artist': artist.append(idx_int)
            elif category == 'Character': character.append(idx_int)
            elif category == 'Copyright': copy_right.append(idx_int)
            elif category == 'Meta': meta.append(idx_int)
            elif category == 'Quality': quality.append(idx_int)
            elif category == 'Model': model_name.append(idx_int)

        return {
            "names": names,
            "rating": rating,
            "general": general,
            "character": character,
            "copyright": copy_right,
            "artist": artist,
            "meta": meta,
            "quality": quality,
            "model": model_name
        }
    
    def get_tags(self, probs, labels, gen_threshold, char_threshold):
        """Extract tags based on probabilities and thresholds."""
        result = {
            "rating": [], "general": [], "character": [], "copyright": [],
            "artist": [], "meta": [], "quality": [], "model": []
        }

        # Select highest probability for rating and quality
        for cat, indices in [("rating", labels["rating"]), ("quality", labels["quality"])]:
            valid_indices = np.array([i for i in indices if i < len(probs)])
            if len(valid_indices) > 0:
                cat_probs = probs[valid_indices]
                if len(cat_probs) > 0:
                    max_idx_local = np.argmax(cat_probs)
                    max_idx_global = valid_indices[max_idx_local]
                    if max_idx_global < len(labels["names"]) and labels["names"][max_idx_global] is not None:
                        result[cat].append((labels["names"][max_idx_global], float(cat_probs[max_idx_local])))

        # Threshold-based selection for other categories
        category_map = {
            "general": (labels["general"], gen_threshold),
            "character": (labels["character"], char_threshold),
            "copyright": (labels["copyright"], char_threshold),
            "artist": (labels["artist"], char_threshold),
            "meta": (labels["meta"], gen_threshold),
            "model": (labels["model"], gen_threshold)
        }

        for category, (indices, threshold) in category_map.items():
            valid_indices = np.array([i for i in indices if i < len(probs)])
            if len(valid_indices) > 0:
                cat_probs = probs[valid_indices]
                selected = cat_probs >= threshold
                sel_local = np.nonzero(selected)[0]
                if len(sel_local) > 0:
                    sel_global = valid_indices[sel_local]
                    sel_probs = cat_probs[sel_local]
                    for g_idx, prob in zip(sel_global, sel_probs):
                        if g_idx < len(labels["names"]) and labels["names"][g_idx] is not None:
                            result[category].append((labels["names"][g_idx], float(prob)))

        # Sort by probability descending
        for k in result:
            result[k] = sorted(result[k], key=lambda x: x[1], reverse=True)

        return result            
    
    def run_cl_tagger(self, image, full_model_path, full_tag_map_path, general, character, replace_space, categories, exclude, session_method):
        """Run tagger on image and return comma-separated tags."""
        input_tensor = self.preprocess_image(image)
        g_labels_data = self.get_tag_mapping(full_tag_map_path)
        session = self.get_session(full_model_path, session_method)
        
        input_name = session.get_inputs()[0].name
        output_name = session.get_outputs()[0].name
        outputs = session.run([output_name], {input_name: input_tensor})[0]
        
        if np.isnan(outputs).any() or np.isinf(outputs).any():
            print("[MiraSubPack:ClTagger] Warning: NaN/Inf in output. Clamping values.")
            outputs = np.nan_to_num(outputs, nan=0.0, posinf=1.0, neginf=0.0)

        probs = 1 / (1 + np.exp(-np.clip(outputs[0], -30, 30)))
                
        predictions = self.get_tags(probs, g_labels_data, general, character)

        categories_select = [c.strip() for c in categories.split(',') if c.strip()]
        
        output_tags = []

        for category in categories_select:
            if category not in predictions:
                continue
            tags_in_category = predictions.get(category, [])

            if category in ["rating", "quality"]:
                if tags_in_category:
                    tag_name = tags_in_category[0][0]
                    if replace_space:
                        tag_name = tag_name.replace("_", " ")
                    output_tags.append(tag_name)
                continue
                
            for tag, prob in tags_in_category:                
                if category == "meta" and any(p in tag.lower() for p in ['id', 'commentary', 'request', 'mismatch']):
                    continue
                if replace_space:
                    output_tags.append(tag.replace("_", " "))
                else:
                    output_tags.append(tag)
        
        exclude_list = [e.strip().lower() for e in exclude.split(',') if e.strip()]
        if exclude_list:
            filtered_tags = [tag for tag in output_tags if not any(ex in tag.lower() for ex in exclude_list)]
            output_tags = filtered_tags
        
        output_text = ", ".join(output_tags)
        print("[MiraSubPack:ClTagger] " + output_text)
        
        if session_method.endswith('Release'):
            self._cpu_session = None
            self._gpu_session = None
            gc.collect()
            
        return output_text    

# ==========================================
# 3. Core: ImageTiledKSamplerWithTagger
# ==========================================

class ImageTiledKSamplerWithTagger:
    """
    ComfyUI node: Image tiled KSampler with CL Tagger for local prompt generation on large images.
    """
    def __init__(self):
        self.tagger = cl_tagger()

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "clip": ("CLIP",),  
                "image": ("IMAGE",),
                "samples": ("LATENT",),
                "positive_text": ("STRING", {"multiline": True, "default": ""}),
                "negative_text": ("STRING", {"multiline": True, "default": "bad quality, worst quality, worst detail, sketch"}),
                
                # Tagger parameters
                "tagger_model": (onnx_list,),
                "tagger_general_threshold": ("FLOAT", {"default": 0.55, "min": 0.0, "max": 1.0, "step": 0.05}),
                "tagger_exclude_tags": ("STRING", {"default": ""}),
                "tagger_session_method": (['GPU', 'CPU'], ),
                
                # Sampler parameters
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "steps": ("INT", {"default": 16, "min": 1, "max": 100}),
                "cfg": ("FLOAT", {"default": 7.0, "min": 0.0, "max": 32.0, "step": 0.1}),
                "sampler_name": (comfy.samplers.KSampler.SAMPLERS,),
                "scheduler": (comfy.samplers.KSampler.SCHEDULERS,),
                "denoise": ("FLOAT", {"default": 0.35, "min": 0.0, "max": 1.0, "step": 0.01}),
                
                # Tiling parameters
                "tile_size": ("INT", {"default": 1280, "min": 512, "max": 2048, "step": 64}),
                "overlap": ("INT", {"default": 64, "min": 64, "max": 256, "step": 64}),
                "overlap_feather_rate": ("FLOAT", {"default": 1, "min": 0.1, "max": 4, "step": 0.1}),
            }
        }

    RETURN_TYPES = ("LATENT", "LATENT", "STRING", "INT", "INT", "INT", "INT")
    RETURN_NAMES = ("latent", "tiled_latents", "tile_logs", "full_width", "full_height", "tile_size", "overlap")
    FUNCTION = "sample"
    CATEGORY = CAT
    DESCRIPTION = "Image tiled KSampler with CL Tagger assistance for large images hi-res fix."

    def sample(self, model, clip, image, samples, positive_text, negative_text, 
               tagger_model, tagger_general_threshold, tagger_exclude_tags, tagger_session_method,
               seed, steps, cfg, sampler_name, scheduler, denoise, 
               tile_size, overlap, overlap_feather_rate):
        
        latent_image = samples["samples"]
        _, _, latent_h, latent_w = latent_image.shape
        pixel_h, pixel_w = latent_h * 8, latent_w * 8
        
        mapping = None
        full_model_path = None
        full_tag_map_path = None
        if tagger_model != "None":
            try:                
                full_model_path = os.path.join(ONNX_PATH, tagger_model)
                full_tag_map_path = full_model_path.replace('.onnx', '_tag_mapping.json')
                mapping = self.tagger
            except Exception as e:
                print(f"[MiraSubPack:AutoTiledTagger] Error loading tagger model: {e}")
                return ({"samples": latent_image}, {"samples": latent_image}, f"Error: {e}", pixel_w, pixel_h, tile_size, overlap)

        source_img_tensor = image[0] 
        src_h, src_w, _ = source_img_tensor.shape
        if src_h != pixel_h or src_w != pixel_w:
            print(f"[MiraSubPack:AutoTiledTagger] Warning: Input image size ({src_w}x{src_h}) mismatched with latent ({pixel_w}x{pixel_h}). Resizing for tagging.")
            img_permuted = image.permute(0, 3, 1, 2)  # B, C, H, W
            img_resized = torch.nn.functional.interpolate(img_permuted, size=(pixel_h, pixel_w), mode='nearest-exact')
            source_img_tensor = img_resized[0].permute(1, 2, 0)  # H, W, C

        tiles = self._calculate_tiles(pixel_w, pixel_h, tile_size, overlap)
        print(f"[MiraSubPack:AutoTiledTagger] Using {len(tiles)} tiles for {pixel_w}x{pixel_h} image.")
        
        negative_tokens = clip.tokenize(negative_text)
        negative_conditioning = clip.encode_from_tokens_scheduled(negative_tokens)

        output_latent = latent_image.clone()
        original_latent = latent_image.clone() 
        log_info = []

        # Phase 1: Sample all tiles independently
        sampled_tiles = []  # Store (x, y, w, h, sampled_latent)
        
        for idx, (x, y, w, h) in enumerate(tiles):
            print(f"  > Sampling Tile {idx+1}/{len(tiles)}: ({x},{y}) {w}x{h}")
            
            tile_img_tensor = source_img_tensor[y:y+h, x:x+w, :]
            tile_img_np = (tile_img_tensor.cpu().numpy() * 255).astype(np.uint8)
            tile_pil = Image.fromarray(tile_img_np)
            
            dynamic_prompt = positive_text
            tags_str = ""
            if mapping:
                tags_str = self.tagger.run_cl_tagger(tile_pil, full_model_path, full_tag_map_path, 
                                                     tagger_general_threshold, 1, True, "general", 
                                                     tagger_exclude_tags, tagger_session_method)
                tags_str = tags_str.replace('(', '\(').replace(')', '\)')
                dynamic_prompt = f"{positive_text}, {tags_str}" if positive_text else tags_str
            
            log_info.append(f"Tile {idx}: {tags_str}")
            
            positive_tokens = clip.tokenize(dynamic_prompt)
            positive_conditioning = clip.encode_from_tokens_scheduled(positive_tokens)
            
            tile_latent = self._crop_latent(samples, x, y, w, h)
            
            sampled_tile = self._sample_single(
                model, positive_conditioning, negative_conditioning, tile_latent,
                seed + idx, steps, cfg, sampler_name, scheduler, denoise  # Offset seed per tile for variation
            )            
            sampled_tiles.append((x, y, w, h, sampled_tile["samples"]))
            
        # Calculate actual feather width
        feather_width = max(overlap * 4, int(overlap * overlap_feather_rate))
        feather_width = min(tile_size * 0.25, feather_width) 
        feather_width = (feather_width // 8) * 8
        
        # Phase 2: Merge all sampled tiles with enhanced blending
        tile_latents = None
        placed_tiles = []  # Track placed tiles: [(x, y, x_end, y_end), ...]
        
        for x, y, w, h, tile_latent in sampled_tiles:
            print(f"  > Merging Tile at ({x},{y}) {w}x{h}")
            output_latent = self._place_tile_enhanced(
                output_latent, tile_latent, 
                original_latent, x, y, w, h, overlap, feather_width, placed_tiles
            )
            tile_latents = torch.cat([tile_latents, tile_latent], dim=0) if tile_latents is not None else tile_latent
        
        return ({"samples": output_latent}, {"samples": tile_latents}, "\n".join(log_info), pixel_w, pixel_h, tile_size, overlap)
    
    def _calculate_tiles(self, width, height, tile_size, overlap):
        """Calculate overlapping tile positions, aligned to latent grid (multiples of 8)."""
        tiles = []
        step_x = tile_size - overlap
        step_y = tile_size - overlap
        tiles_x = math.ceil((width - overlap) / step_x)
        tiles_y = math.ceil((height - overlap) / step_y)
        for i in range(tiles_y):
            for j in range(tiles_x):
                x = j * step_x
                y = i * step_y
                w = min(tile_size, width - x)
                h = min(tile_size, height - y)
                # Adjust edge tiles to full size if possible
                if w < tile_size and x > 0:
                    x = max(0, width - tile_size)
                    w = min(tile_size, width)
                if h < tile_size and y > 0:
                    y = max(0, height - tile_size)
                    h = min(tile_size, height)
                # Align to 8-pixel grid
                x, y = (x // 8) * 8, (y // 8) * 8
                w, h = ((w + 7) // 8) * 8, ((h + 7) // 8) * 8
                # Clamp oversized tiles
                if x + w > width:
                    x = (x // 8) * 8
                    w = width - x
                    w = (w // 8) * 8
                if y + h > height:
                    y = (y // 8) * 8
                    h = height - y
                    h = (h // 8) * 8
                if w > 0 and h > 0:
                    tiles.append((x, y, w, h))
        # Remove duplicates and sort top-to-bottom, left-to-right
        return sorted(list(set(tiles)), key=lambda t: (t[1], t[0]))

    def _crop_latent(self, samples, x, y, width, height):
        """Crop latent region, aligned to grid, with padding if too small."""
        latent = samples["samples"]
        lx, ly, lw, lh = x//8, y//8, width//8, height//8
        cropped = latent[:, :, ly:ly+lh, lx:lx+lw].clone()
        if cropped.shape[2] < 3 or cropped.shape[3] < 3:
            cropped = torch.nn.functional.pad(cropped, (0, max(0, 3 - cropped.shape[3]), 0, max(0, 3 - cropped.shape[2])), mode='replicate')
        return {"samples": cropped}
        
    def _place_tile_enhanced(self, output, tile, original_latent, 
                    x, y, width, height, overlap, feather_width, placed_tiles):
        """
        Enhanced tile merging with three-level consistency:
        1. Global light color alignment (30% strength)
        2. Strong overlap region calibration (80% strength)
        3. Feathered blending for smooth edges
        
        Handles excessive overlap by cropping tiles appropriately.
        overlap: Full overlap width used for sampling (provides context)
        feather_width: Actual feather width for blending (smaller = sharper)
        placed_tiles: List tracking previously placed tile bounds [(x, y, x_end, y_end), ...]
        """
        lx, ly, lw, lh = x//8, y//8, width//8, height//8
        l_overlap = overlap // 8  # Used for excessive overlap detection
        l_feather = feather_width // 8  # Smaller feather width in latent space
        batch, channels, tile_h, tile_w = tile.shape
        
        # Original latent space coordinates
        original_lx, original_ly = lx, ly
        
        # Detect and handle excessive overlap by checking all previously placed tiles
        crop_top = 0
        crop_left = 0
        
        for prev_x, prev_y, prev_x_end, prev_y_end in placed_tiles:
            prev_lx, prev_ly = prev_x // 8, prev_y // 8
            prev_lx_end, prev_ly_end = prev_x_end // 8, prev_y_end // 8
            
            # Check vertical overlap (same column)
            if abs(prev_lx - lx) < lw // 2 and prev_ly < ly < prev_ly_end:
                actual_overlap_y = prev_ly_end - ly
                if actual_overlap_y > l_overlap * 2:
                    excess = actual_overlap_y - l_overlap
                    crop_top = max(crop_top, excess)
                    ly = original_ly + crop_top
            
            # Check horizontal overlap (same row)
            if abs(prev_ly - ly) < lh // 2 and prev_lx < lx < prev_lx_end:
                actual_overlap_x = prev_lx_end - lx
                if actual_overlap_x > l_overlap * 2:
                    excess = actual_overlap_x - l_overlap
                    crop_left = max(crop_left, excess)
                    lx = original_lx + crop_left
        
        # Crop tile if needed
        if crop_top > 0 or crop_left > 0:
            tile = tile[:, :, crop_top:, crop_left:].clone()
            tile_h = tile.shape[2]
            tile_w = tile.shape[3]
            print(f"    [Crop] Excessive overlap detected, cropped tile by (top={crop_top}, left={crop_left})")
        
        # Final boundary check
        if ly + tile_h > output.shape[2]:
            tile_h = output.shape[2] - ly
            tile = tile[:, :, :tile_h, :]
        if lx + tile_w > output.shape[3]:
            tile_w = output.shape[3] - lx
            tile = tile[:, :, :, :tile_w]
        
        if tile_h <= 0 or tile_w <= 0:
            return output
        
        original_tile = original_latent[:, :, ly:ly+tile_h, lx:lx+tile_w]
        
        if original_tile.shape[2:] != tile.shape[2:]:
            print("  [Warning] Tile size mismatch during merge; falling back to basic blending.")
            return self._place_tile_basic(output, tile, lx*8, ly*8, width, height, feather_width)
        
        tile_corrected = tile.clone()
        
        # Level 1: Global color alignment (light, 30%)
        for c in range(channels):
            orig_c = original_tile[:, c:c+1, :, :]
            tile_c = tile[:, c:c+1, :, :]
            
            orig_mean, orig_std = orig_c.mean(), orig_c.std() + 1e-8
            tile_mean, tile_std = tile_c.mean(), tile_c.std() + 1e-8
            
            corrected = (tile_c - tile_mean) * (orig_std / tile_std) + orig_mean
            tile_corrected[:, c:c+1, :, :] = tile_c * 0.7 + corrected * 0.3
        
        # Level 2: Strong overlap calibration (80%)
        # Use smaller feather region for calibration to avoid over-blending
        if l_feather > 0:
            overlap_mask = self._create_overlap_mask(tile_h, tile_w, lx, ly, l_feather, output.shape, tile.device)
            
            overlap_pixels = overlap_mask.sum()
            if overlap_pixels > 10:
                for c in range(channels):
                    orig_c = original_tile[:, c:c+1, :, :]
                    tile_c = tile_corrected[:, c:c+1, :, :]
                    mask_c = overlap_mask[:, c:c+1, :, :]
                    
                    orig_mean = (orig_c * mask_c).sum() / overlap_pixels
                    tile_mean = (tile_c * mask_c).sum() / overlap_pixels
                    
                    orig_std = torch.sqrt(((orig_c - orig_mean) ** 2 * mask_c).sum() / overlap_pixels + 1e-8)
                    tile_std = torch.sqrt(((tile_c - tile_mean) ** 2 * mask_c).sum() / overlap_pixels + 1e-8)
                    
                    corrected = (tile_c - tile_mean) * (orig_std / tile_std) + orig_mean
                    tile_corrected[:, c:c+1, :, :] = tile_c * (1 - mask_c * 0.8) + corrected * mask_c * 0.8
        
        # Level 3: Feathered blending with smaller feather width
        blend_mask = self._create_feather_mask(tile_h, tile_w, lx, ly, l_feather, output.shape, tile.device)
        
        output[:, :, ly:ly+tile_h, lx:lx+tile_w] = (
            tile_corrected * blend_mask + 
            output[:, :, ly:ly+tile_h, lx:lx+tile_w] * (1 - blend_mask)
        )
        
        # Record this tile's final position for future overlap detection
        placed_tiles.append((lx * 8, ly * 8, (lx + tile_w) * 8, (ly + tile_h) * 8))
        
        return output
    
    def _create_overlap_mask(self, tile_h, tile_w, lx, ly, l_overlap, output_shape, device):
        """Binary mask for overlap regions."""
        mask = torch.zeros((tile_h, tile_w), device=device)
        
        if ly > 0:  # Top overlap
            mask[:min(l_overlap, tile_h), :] = 1
        if ly + tile_h < output_shape[2]:  # Bottom overlap
            mask[-min(l_overlap, tile_h):, :] = 1
        if lx > 0:  # Left overlap
            mask[:, :min(l_overlap, tile_w)] = 1
        if lx + tile_w < output_shape[3]:  # Right overlap
            mask[:, -min(l_overlap, tile_w):] = 1
        
        return mask.unsqueeze(0).unsqueeze(0).expand(1, output_shape[1], -1, -1)
    
    def _create_feather_mask(self, tile_h, tile_w, lx, ly, l_overlap, output_shape, device):
        """Smooth feathered mask based on distance to tile edges."""
        y_coords = torch.arange(tile_h, device=device, dtype=torch.float32).unsqueeze(1)
        x_coords = torch.arange(tile_w, device=device, dtype=torch.float32).unsqueeze(0)
        
        dist_top = y_coords.clamp(max=l_overlap) if ly > 0 else torch.full_like(y_coords, l_overlap)
        dist_bottom = (tile_h - 1 - y_coords).clamp(max=l_overlap) if ly + tile_h < output_shape[2] else torch.full_like(y_coords, l_overlap)
        dist_left = x_coords.clamp(max=l_overlap) if lx > 0 else torch.full_like(x_coords, l_overlap)
        dist_right = (tile_w - 1 - x_coords).clamp(max=l_overlap) if lx + tile_w < output_shape[3] else torch.full_like(x_coords, l_overlap)
        
        dist_v = torch.minimum(dist_top, dist_bottom)
        dist_h = torch.minimum(dist_left, dist_right)
        dist = torch.minimum(dist_v, dist_h)
        
        mask = (dist / l_overlap).clamp(0, 1)
        mask = mask * mask * (3 - 2 * mask)  # Smoothstep
        
        return mask.unsqueeze(0).unsqueeze(0)
    
    def _place_tile_basic(self, output, tile, x, y, width, height, overlap):
        """Fallback basic feathered blending."""
        lx, ly, lw, lh = x//8, y//8, width//8, height//8
        l_overlap = overlap // 8
        batch, _, tile_h, tile_w = tile.shape
        
        mask = self._create_feather_mask(tile_h, tile_w, lx, ly, l_overlap, output.shape, tile.device)
        
        target_h, target_w = output[:, :, ly:ly+tile_h, lx:lx+tile_w].shape[2:]
        if target_h != tile_h or target_w != tile_w:
            tile = tile[:, :, :target_h, :target_w]
            mask = mask[:, :, :target_h, :target_w]
        
        output[:, :, ly:ly+target_h, lx:lx+target_w] = tile * mask + output[:, :, ly:ly+target_h, lx:lx+target_w] * (1 - mask)
            
        return output
            
    def _sample_single(self, model, positive, negative, latent, seed, steps, cfg, sampler_name, scheduler, denoise):
        """Sample a single tile latent."""
        l = latent["samples"]
        noise = torch.randn(l.shape, dtype=l.dtype, device=l.device, generator=torch.manual_seed(seed))
        callback = latent_preview.prepare_callback(model, steps)
        sampled = comfy.sample.sample(model, noise, steps, cfg, sampler_name, scheduler, 
                                      positive, negative, l, denoise=denoise, seed=seed, 
                                      force_full_denoise=True, callback=callback)
        return {"samples": sampled}

# ==========================================
# 4. ImageTilesFeatherMerger
# ==========================================    
class ImageTilesFeatherMerger:
    """
    No reference image required. Uses full_width/full_height for canvas size.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),  # Batch of tiles: [N, H, W, 3] or [N, H, W, 4]
                "full_width": ("INT", {"default": 0, "min": 0, "max": 16384, "step": 8, "tooltip": "Full image width."}),
                "full_height": ("INT", {"default": 0, "min": 0, "max": 16384, "step": 8, "tooltip": "Full image height."}),
                "tile_size": ("INT", {"default": 1280, "min": 512, "max": 4096, "step": 64}),
                "overlap": ("INT", {"default": 64, "min": 64, "max": 256, "step": 64}),
                "overlap_feather_rate": ("FLOAT", {"default": 1, "min": 0.1, "max": 4, "step": 0.1}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "merge"
    CATEGORY = CAT
    DESCRIPTION = "Merge tiled image batch using OpenCV seamlessClone for natural feathered blending."

    def merge(self, images, full_width, full_height, tile_size, overlap, overlap_feather_rate):
        device = images.device
        N, H, W, C = images.shape

        step = tile_size - overlap
        cols = math.ceil((full_width - overlap) / step)
        rows = math.ceil((full_height - overlap) / step)

        canvas = torch.zeros((full_height, full_width, 3), device=device, dtype=torch.float32)
        weight_map = torch.zeros((full_height, full_width, 3), device=device, dtype=torch.float32)

        # Calculate actual feather width based on overlap_feather_rate
        feather = max(overlap * 4, int(overlap * overlap_feather_rate))
        feather = min(tile_size * 0.25, feather)
        
        ramp = torch.linspace(0, 1, feather, device=device, dtype=torch.float32)
        ramp_rev = ramp.flip(0)

        tile_idx = 0
        for r in range(rows):
            for c in range(cols):
                if tile_idx >= N:
                    break

                tile = images[tile_idx, :, :, :3].clone()
                actual_h, actual_w = tile.shape[:2]

                x = c * step
                y = r * step

                # Edge alignment to ensure tile fits within canvas
                x = min(x, full_width  - actual_w)
                y = min(y, full_height - actual_h)

                x = max(0, x)
                y = max(0, y)

                x_end = x + actual_w
                y_end = y + actual_h

                # Check for excessive overlap and crop tile if necessary
                crop_top = 0
                crop_left = 0
                
                if r > 0:
                    prev_y = (r - 1) * step
                    prev_y = min(prev_y, full_height - tile_size)
                    prev_y = max(0, prev_y)
                    prev_y_end = prev_y + tile_size
                    
                    actual_overlap_y = prev_y_end - y
                    if actual_overlap_y > overlap * 2:
                        excess = actual_overlap_y - overlap
                        crop_top = excess
                        y += excess
                        y_end = y + (actual_h - crop_top)
                
                if c > 0:
                    prev_x = (c - 1) * step
                    prev_x = min(prev_x, full_width - tile_size)
                    prev_x = max(0, prev_x)
                    prev_x_end = prev_x + tile_size
                    
                    actual_overlap_x = prev_x_end - x
                    if actual_overlap_x > overlap * 2:
                        excess = actual_overlap_x - overlap
                        crop_left = excess
                        x += excess
                        x_end = x + (actual_w - crop_left)

                # Crop tile if needed
                tile_cropped = tile[crop_top:, crop_left:].clone()
                crop_h, crop_w = tile_cropped.shape[:2]
                
                # Ensure tile fits within canvas boundaries
                if x_end > full_width:
                    crop_w = full_width - x
                    tile_cropped = tile_cropped[:, :crop_w]
                    x_end = full_width
                
                if y_end > full_height:
                    crop_h = full_height - y
                    tile_cropped = tile_cropped[:crop_h, :]
                    y_end = full_height

                actual_h, actual_w = tile_cropped.shape[:2]
                if actual_h <= 0 or actual_w <= 0:
                    tile_idx += 1
                    continue

                # Create feathering mask with dynamic feather width
                mask = torch.ones((actual_h, actual_w), device=device, dtype=torch.float32)
                if y > 0 and feather > 0:
                    mask[:feather, :] *= ramp[:min(feather, actual_h), None]
                if y_end < full_height and feather > 0:
                    mask[-feather:, :] *= ramp_rev[:min(feather, actual_h), None]
                if x > 0 and feather > 0:
                    mask[:, :feather] *= ramp[None, :min(feather, actual_w)]
                if x_end < full_width and feather > 0:
                    mask[:, -feather:] *= ramp_rev[None, :min(feather, actual_w)]

                mask = mask[..., None]
                mask3 = mask.expand(-1, -1, 3)

                # Tile blending with feathered mask
                canvas[y:y_end, x:x_end] += tile_cropped[:actual_h, :actual_w] * mask3
                weight_map[y:y_end, x:x_end] += mask3

                tile_idx += 1

        # Normalize by weight map to avoid darkening
        weight_map = weight_map.clamp(min=1e-5)
        canvas /= weight_map

        return (canvas.unsqueeze(0),)