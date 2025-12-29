"""
Thought process: Tile large image -> Use CL Tagger on tiles -> Generate local Prompts -> Sample locally -> Merge
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
    global ONNX_PATH
    current_file = os.path.abspath(__file__)
    base = os.path.dirname(current_file)
    while True:
        if os.path.basename(base) == "custom_nodes":
            comfy_root = os.path.dirname(base)
            ONNX_PATH = os.path.join(comfy_root, "models", "onnx")
            break
        new_base = os.path.dirname(base)
        if new_base == base:  # reached root
            print("[MiraSubPack:TaggerSampler] Warning: Could not find ComfyUI root, defaulting relative.")
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
    '''
    CL Tagger by cella110n https://huggingface.co/cella110n
    Few codes reference from https://huggingface.co/spaces/DraconicDragon/cl_tagger
    
    Inputs:
    image           - Image for tagger
    model_name      - Onnx model
    general         - General threshold
    character       - Character threshold
    replace_space   - Replace '_' with ' ' (space)
    categories      - Selected categories in generate tags, and order by input order
    exclude_tags    - Exclude tags
    session_method  - Tagger Model in CPU or GPU session. Release will release session after generate
        
    Outputs:
    tags            - Generated tags
    '''
    
    def __init__(self):
        self._mean = np.array([0.5, 0.5, 0.5], dtype=np.float32).reshape(3, 1, 1)
        self._std = np.array([0.5, 0.5, 0.5], dtype=np.float32).reshape(3, 1, 1)
        self._tag_mapping_cache = {}
        self._cpu_session = None
        self._gpu_session = None
    
    def get_tag_mapping(self, full_tag_map_path):
        if full_tag_map_path not in self._tag_mapping_cache:
            print("[MiraSubPack:ClTagger] Load tag mapping: " + full_tag_map_path)
            self._tag_mapping_cache[full_tag_map_path] = self.load_tag_mapping(full_tag_map_path)        
        return self._tag_mapping_cache[full_tag_map_path]

    def get_session(self, model_path, session_method):
        if session_method.startswith('CPU'):
            if not self._cpu_session:                
                self._cpu_session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
            return self._cpu_session
        elif session_method.startswith('GPU'):
            if not self._gpu_session:
                self._gpu_session = ort.InferenceSession(model_path, providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
            return self._gpu_session
        
            
    def pad_square_np(self, img_array: np.ndarray) -> np.ndarray:
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
        img = np.array(image)
        img = self.pad_square_np(img)
        img = cv2.resize(img, target_size, interpolation=cv2.INTER_CUBIC)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        img = img.astype(np.float32) / 255.0
        img = (img - 0.5) / 0.5
        img = np.expand_dims(np.transpose(img, (2, 0, 1)), axis=0)
        return img

    def load_tag_mapping(self, mapping_path):
        with open(mapping_path, 'r', encoding='utf-8') as f: tag_mapping_data = json.load(f)
        if isinstance(tag_mapping_data, dict) and "idx_to_tag" in tag_mapping_data:
            idx_to_tag = {int(k): v for k, v in tag_mapping_data["idx_to_tag"].items()}
            tag_to_category = tag_mapping_data["tag_to_category"]
        elif isinstance(tag_mapping_data, dict):
            try:
                tag_mapping_data_int_keys = {int(k): v for k, v in tag_mapping_data.items()}
                idx_to_tag = {idx: data['tag'] for idx, data in tag_mapping_data_int_keys.items()}
                tag_to_category = {data['tag']: data['category'] for data in tag_mapping_data_int_keys.values()}
            except (KeyError, ValueError) as e:
                raise ValueError(f"Unsupported tag mapping format (dict): {e}. Expected int keys with 'tag' and 'category'.")
        else:
            raise ValueError("Unsupported tag mapping format: Expected a dictionary.")

        names = [None] * (max(idx_to_tag.keys()) + 1)
        rating, general, artist, character, copy_right, meta, quality, model_name = [], [], [], [], [], [], [], []
        for idx, tag in idx_to_tag.items():
            if idx >= len(names): names.extend([None] * (idx - len(names) + 1))
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

        label_data = {
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
        return label_data
    
    def get_tags(self, probs, labels, gen_threshold, char_threshold):
        result = {
            "rating": [],
            "general": [],
            "character": [],
            "copyright": [],
            "artist": [],
            "meta": [],
            "quality": [],
            "model": []
        }

        # Rating (select max)
        if len(labels["rating"]) > 0:
            valid_indices = np.array([i for i in labels["rating"] if i < len(probs)])
            if len(valid_indices) > 0:
                rating_probs = probs[valid_indices]
                if len(rating_probs) > 0:
                    rating_idx_local = np.argmax(rating_probs)
                    rating_idx_global = valid_indices[rating_idx_local]
                    if rating_idx_global < len(labels["names"]) and labels["names"][rating_idx_global] is not None:
                        rating_name = labels["names"][rating_idx_global]
                        rating_conf = float(rating_probs[rating_idx_local])
                        result["rating"].append((rating_name, rating_conf))

        # Quality (select max)
        if len(labels["quality"]) > 0:
            valid_indices = np.array([i for i in labels["quality"] if i < len(probs)])
            if len(valid_indices) > 0:
                quality_probs = probs[valid_indices]
                if len(quality_probs) > 0:
                    quality_idx_local = np.argmax(quality_probs)
                    quality_idx_global = valid_indices[quality_idx_local]
                    if quality_idx_global < len(labels["names"]) and labels["names"][quality_idx_global] is not None:
                        quality_name = labels["names"][quality_idx_global]
                        quality_conf = float(quality_probs[quality_idx_local])
                        result["quality"].append((quality_name, quality_conf))

        # Threshold-based categories
        category_map = {
            "general": (labels["general"], gen_threshold),
            "character": (labels["character"], char_threshold),
            "copyright": (labels["copyright"], char_threshold),
            "artist": (labels["artist"], char_threshold),
            "meta": (labels["meta"], gen_threshold),
            "model": (labels["model"], gen_threshold)
        }

        for category, (indices, threshold) in category_map.items():
            if len(indices) > 0:
                valid_indices = np.array([i for i in indices if i < len(probs)])
                if len(valid_indices) > 0:
                    category_probs = probs[valid_indices]
                    mask = category_probs >= threshold
                    selected_indices_local = np.nonzero(mask)[0]
                    if len(selected_indices_local) > 0:
                        selected_indices_global = valid_indices[selected_indices_local]
                        selected_probs = category_probs[selected_indices_local]
                        for idx_global, prob_val in zip(selected_indices_global, selected_probs):
                            if idx_global < len(labels["names"]) and labels["names"][idx_global] is not None:
                                result[category].append((labels["names"][idx_global], float(prob_val)))

        # Sort results by probability
        for k in result:
            result[k] = sorted(result[k], key=lambda x: x[1], reverse=True)

        return result            
    
    def run_cl_tagger(self, image, full_model_path, full_tag_map_path, general, character, replace_space, categories, exclude, session_method):
        input_tensor = self.preprocess_image(image)
        g_labels_data = self.get_tag_mapping(full_tag_map_path)
        session = self.get_session(full_model_path, session_method)
        
        input_name = session.get_inputs()[0].name
        output_name = session.get_outputs()[0].name
        outputs = session.run([output_name], {input_name: input_tensor})[0]
        
        if np.isnan(outputs).any() or np.isinf(outputs).any():
            print("[MiraSubPack:ClTagger]Warning: NaN or Inf detected in model output. Clamping...")
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
            filtered_tags = []
            for tag in output_tags:
                tag_l = tag.lower()
                hit = False
                for ex in exclude_list:
                    if ex in tag_l:
                        hit = True
                        break
                if not hit:
                    filtered_tags.append(tag)
            output_tags = filtered_tags
        
        output_text = ", ".join(output_tags)
        print("[MiraSubPack:ClTagger] " + output_text)
        
        if session_method.endswith('Release'):
            session = None
            self._cpu_session = None
            self._gpu_session = None
            gc.collect()
            
        return output_text    
    
# ==========================================
# 3. Core: AutoTiledKSamplerWithTagger
# ==========================================

class AutoTiledKSamplerWithTagger:
    """
    A VLM/Tagger assisted auto-tiled sampler node
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
                "negative_text": ("STRING", {"multiline": True, "default": "lowres, bad anatomy"}),
                
                # Tagger
                "tagger_model": (onnx_list,),
                "tagger_general_threshold": ("FLOAT", {"default": 0.55, "min": 0.0, "max": 1.0, "step": 0.05}),
                "tagger_character_threshold": ("FLOAT", {"default": 0.6, "min": 0.0, "max": 1.0, "step": 0.05}),
                "exclude_tags": ("STRING", {"default": ""}),
                
                # Sampler
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "steps": ("INT", {"default": 16, "min": 1, "max": 10000}),
                "cfg": ("FLOAT", {"default": 7.0, "min": 0.0, "max": 100.0, "step": 0.1}),
                "sampler_name": (comfy.samplers.KSampler.SAMPLERS,),
                "scheduler": (comfy.samplers.KSampler.SCHEDULERS,),
                "denoise": ("FLOAT", {"default": 0.35, "min": 0.0, "max": 1.0, "step": 0.01}),
                
                # Tiling
                "tile_size": ("INT", {"default": 1024, "min": 512, "max": 2048, "step": 64}),
                "overlap": ("INT", {"default": 128, "min": 0, "max": 512, "step": 8}),
            }
        }
    
    RETURN_TYPES = ("LATENT", "STRING")
    RETURN_NAMES = ("latent", "log")
    FUNCTION = "sample"
    CATEGORY = CAT
    DESCRIPTION = "Auto Tiled KSampler with CL Tagger assistance for large images."

    def sample(self, model, clip, image, samples, positive_text, negative_text, 
               tagger_model, tagger_general_threshold, tagger_character_threshold, exclude_tags,
               seed, steps, cfg, sampler_name, scheduler, denoise, 
               tile_size, overlap):
        
        latent_image = samples["samples"]
        batch_size, _, latent_h, latent_w = latent_image.shape
        pixel_h, pixel_w = latent_h * 8, latent_w * 8
        
        mapping = None
        if tagger_model != "None":
            try:                
                full_model_path = os.path.join(ONNX_PATH, tagger_model)
                full_tag_map_path = full_model_path.replace('.onnx', '_tag_mapping.json')
                mapping = self.tagger
            except Exception as e:
                print(f"[MiraSubPack:AutoTiledTagger] Error loading model: {e}")
                return ({"samples": latent_image}, f"Error: {e}")

        source_img_tensor = image[0] 
        src_h, src_w, _ = source_img_tensor.shape
        if src_h != pixel_h or src_w != pixel_w:
            print(f"[MiraSubPack:AutoTiledTagger] Warning: Image input size ({src_w}x{src_h}) does not match latent size ({pixel_w}x{pixel_h}). Resizing image for tagging.")
            img_permuted = image.permute(0, 3, 1, 2)  # B, C, H, W
            img_resized = torch.nn.functional.interpolate(img_permuted, size=(pixel_h, pixel_w), mode='bilinear')
            source_img_tensor = img_resized[0].permute(1, 2, 0)  # H, W, C

        tiles = self._calculate_tiles(pixel_w, pixel_h, tile_size, overlap)
        print(f"[MiraSubPack:AutoTiledTagger] Strategy: {len(tiles)} tiles for {pixel_w}x{pixel_h}")
        
        negative_tokens = clip.tokenize(negative_text)
        negative_conditioning = clip.encode_from_tokens_scheduled(negative_tokens)

        output_latent = latent_image.clone()
        log_info = []

        for idx, (x, y, w, h) in enumerate(tiles):
            print(f"  > Processing Tile {idx+1}/{len(tiles)}: {x},{y} ({w}x{h})...")
            
            tile_img_tensor = source_img_tensor[y:y+h, x:x+w, :]
            tile_img_np = (tile_img_tensor.cpu().numpy() * 255).astype(np.uint8)
            tile_pil = Image.fromarray(tile_img_np)
            
            dynamic_prompt = ""
            if mapping:                
                tags_str = self.tagger.run_cl_tagger(tile_pil, full_model_path, full_tag_map_path, tagger_general_threshold, tagger_character_threshold, True, "general", exclude_tags, "GPU")
                dynamic_prompt = tags_str
                if positive_text != "":                
                    dynamic_prompt = f"{positive_text}, {tags_str}"
            else:
                tags_str = ""
                dynamic_prompt = positive_text                        
            
            log_info.append(f"Tile {idx}: {tags_str}")
            
            positive_tokens = clip.tokenize(dynamic_prompt)
            positive_conditioning = clip.encode_from_tokens_scheduled(positive_tokens)
            
            tile_latent = self._crop_latent(samples, x, y, w, h)
            
            sampled_tile = self._sample_single(
                model, positive_conditioning, negative_conditioning, tile_latent,
                seed, steps, cfg, sampler_name, scheduler, denoise
            )
            
            self._place_tile(output_latent, sampled_tile["samples"], x, y, w, h, overlap)
            
        return ({"samples": output_latent}, "\n".join(log_info))
    
    def _calculate_tiles(self, width, height, tile_size, overlap):
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
                if w < tile_size and x > 0: x = max(0, width - tile_size); w = min(tile_size, width)
                if h < tile_size and y > 0: y = max(0, height - tile_size); h = min(tile_size, height)
                x, y = (x // 8) * 8, (y // 8) * 8
                w, h = ((w + 7) // 8) * 8, ((h + 7) // 8) * 8
                if x + w > width: x = (x//8)*8; w = width - x; w = (w//8)*8
                if y + h > height: y = (y//8)*8; h = height - y; h = (h//8)*8
                if w > 0 and h > 0: tiles.append((x, y, w, h))
        return sorted(list(set(tiles)), key=lambda t: (t[1], t[0]))

    def _crop_latent(self, samples, x, y, width, height):
        latent = samples["samples"]
        lx, ly, lw, lh = x//8, y//8, width//8, height//8
        cropped = latent[:, :, ly:ly+lh, lx:lx+lw].clone()
        if cropped.shape[2] < 3 or cropped.shape[3] < 3:
            cropped = torch.nn.functional.pad(cropped, (0, max(0,3-cropped.shape[3]), 0, max(0,3-cropped.shape[2])), mode='replicate')
        return {"samples": cropped}

    def _place_tile(self, output, tile, x, y, width, height, overlap):
        lx, ly, lw, lh = x//8, y//8, width//8, height//8
        l_overlap = overlap // 8
        batch, _, tile_h, tile_w = tile.shape
        mask = torch.ones((tile_h, tile_w), device=tile.device)
        if ly > 0: 
            for i in range(min(l_overlap, tile_h)): mask[i, :] *= (i/l_overlap)
        if ly + tile_h < output.shape[2]: 
            for i in range(min(l_overlap, tile_h)): mask[tile_h-1-i, :] *= (i/l_overlap)
        if lx > 0: 
            for j in range(min(l_overlap, tile_w)): mask[:, j] *= (j/l_overlap)
        if lx + tile_w < output.shape[3]: 
            for j in range(min(l_overlap, tile_w)): mask[:, tile_w-1-j, ] *= (j/l_overlap)
        
        mask = mask.unsqueeze(0).unsqueeze(0)
        target_h, target_w = output[:, :, ly:ly+tile_h, lx:lx+tile_w].shape[2:]
        if target_h != tile_h or target_w != tile_w:
            tile = tile[:, :, :target_h, :target_w]
            mask = mask[:, :, :target_h, :target_w]
            
        output[:, :, ly:ly+target_h, lx:lx+target_w] = \
            tile * mask + output[:, :, ly:ly+target_h, lx:lx+target_w] * (1 - mask)

    def _sample_single(self, model, positive, negative, latent, seed, steps, cfg, sampler_name, scheduler, denoise):
        l = latent["samples"]
        noise = torch.randn(l.shape, dtype=l.dtype, device=l.device, generator=torch.manual_seed(seed))
        callback = latent_preview.prepare_callback(model, steps)
        return {"samples": comfy.sample.sample(model, noise, steps, cfg, sampler_name, scheduler, 
                                            positive, negative, l, denoise, seed, 
                                            force_full_denoise=True, callback=callback)}