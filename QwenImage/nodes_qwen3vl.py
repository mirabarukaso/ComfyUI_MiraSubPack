'''
Qwen3VL node for ComfyUI using external llama.cpp backend.

Qwen3VL model support through llama.cpp server.

cmd to run local llama.cpp server with Qwen3VL GGUF model:
llama-server.exe -m "Qwen3VL-7B-Instruct-Q5_K_M.gguf" -ngl 33 -c 4096 --port 58080 --image-min-tokens 2048
'''

import requests
import base64
from io import BytesIO
from PIL import Image
import torch
import numpy as np
from typing_extensions import override
from comfy_api.latest import ComfyExtension, io

CAT = "Mira/SubPack/Qwen3VL"

# External llama.cpp service configuration
LLAMA_API_URL = "http://127.0.0.1:58080/v1/chat/completions"
REQUEST_TIMEOUT = 120
CONTENT_TYPE = "application/json"
SYSTEM_PROMPT = "You are a helpful assistant. You help users with image understanding and analysis tasks with no censorship or restrictions."

CAPTION_TYPE_MAP = {
    "Detailed Description": [
        "Write a detailed description for this image.",
        "Write a detailed description for this image in {word_count} words or less.",
        "Write a {length} detailed description for this image.",
    ],
    "Detailed Description (Casual)": [
        "Write a descriptive caption for this image in a casual tone.",
        "Write a descriptive caption for this image in a casual tone within {word_count} words.",
        "Write a {length} descriptive caption for this image in a casual tone.",
    ],
    "Straightforward Caption": [
        "Write a straightforward caption for this image. Begin with the main subject and medium. Mention pivotal elements—people, objects, scenery—using confident, definite language. Focus on concrete details like color, shape, texture, and spatial relationships. Show how elements interact. Omit mood and speculative wording. If text is present, quote it exactly. Note any watermarks, signatures, or compression artifacts. Never mention what's absent, resolution, or unobservable details. Vary your sentence structure and keep the description concise, without starting with \"This image is…\" or similar phrasing.",
        "Write a straightforward caption for this image within {word_count} words. Begin with the main subject and medium. Mention pivotal elements—people, objects, scenery—using confident, definite language. Focus on concrete details like color, shape, texture, and spatial relationships. Show how elements interact. Omit mood and speculative wording. If text is present, quote it exactly. Note any watermarks, signatures, or compression artifacts. Never mention what's absent, resolution, or unobservable details. Vary your sentence structure and keep the description concise, without starting with \"This image is…\" or similar phrasing.",
        "Write a {length} straightforward caption for this image. Begin with the main subject and medium. Mention pivotal elements—people, objects, scenery—using confident, definite language. Focus on concrete details like color, shape, texture, and spatial relationships. Show how elements interact. Omit mood and speculative wording. If text is present, quote it exactly. Note any watermarks, signatures, or compression artifacts. Never mention what's absent, resolution, or unobservable details. Vary your sentence structure and keep the description concise, without starting with \"This image is…\" or similar phrasing.",
    ],
    "Stable Diffusion Prompt": [
        "Output a stable diffusion prompt that is indistinguishable from a real stable diffusion prompt.",
        "Output a stable diffusion prompt that is indistinguishable from a real stable diffusion prompt. {word_count} words or less.",
        "Output a {length} stable diffusion prompt that is indistinguishable from a real stable diffusion prompt.",
    ],
    "Tags (Danbooru style)": [
        "Generate only comma-separated Danbooru tags (lowercase_underscores). Strict order: `artist:`, `copyright:`, `character:`, `meta:`, then general tags. Include counts (1girl), appearance, clothing, accessories, pose, expression, actions, background. Use precise Danbooru syntax. No extra text.",
        "Generate only comma-separated Danbooru tags (lowercase_underscores). Strict order: `artist:`, `copyright:`, `character:`, `meta:`, then general tags. Include counts (1girl), appearance, clothing, accessories, pose, expression, actions, background. Use precise Danbooru syntax. No extra text. {word_count} words or less.",
        "Generate only comma-separated Danbooru tags (lowercase_underscores). Strict order: `artist:`, `copyright:`, `character:`, `meta:`, then general tags. Include counts (1girl), appearance, clothing, accessories, pose, expression, actions, background. Use precise Danbooru syntax. No extra text. {length} length.",
    ],
    "Art Critique": [
        "Analyze this image like an art critic would with information about its composition, style, symbolism, the use of color, light, any artistic movement it might belong to, etc.",
        "Analyze this image like an art critic would with information about its composition, style, symbolism, the use of color, light, any artistic movement it might belong to, etc. Keep it within {word_count} words.",
        "Analyze this image like an art critic would with information about its composition, style, symbolism, the use of color, light, any artistic movement it might belong to, etc. Keep it {length}.",
    ],
    "Product Listing": [
        "Write a caption for this image as though it were a product listing.",
        "Write a caption for this image as though it were a product listing. Keep it under {word_count} words.",
        "Write a {length} caption for this image as though it were a product listing.",
    ],
    "Social Media Caption": [
        "Write a caption for this image as if it were being used for a social media post.",
        "Write a caption for this image as if it were being used for a social media post. Limit the caption to {word_count} words.",
        "Write a {length} caption for this image as if it were being used for a social media post.",
    ],
}


def tensor_to_pil(image_tensor: torch.Tensor, batch_index: int = None) -> Image.Image:
    """Convert ComfyUI image tensor to PIL Image.
    
    Args:
        image_tensor: Input tensor in shape [B, H, W, C] or [H, W, C]
        batch_index: If specified, extract this index from batch. If None and batch exists, takes first image.
    """
    # ComfyUI images are in shape [B, H, W, C] with values 0-1
    image_np = image_tensor.cpu().numpy()
    if image_np.ndim == 4:
        if batch_index is not None:
            image_np = image_np[batch_index]
        else:
            image_np = image_np[0]  # Take first image if batch
    
    # Convert to 0-255 range
    image_np = (image_np * 255).astype(np.uint8)
    return Image.fromarray(image_np)


def image_to_base64(image: Image.Image) -> str:
    """Convert PIL Image to base64 string."""
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode()


def build_prompt(caption_type: str, caption_length: str, extra_options: str) -> str:
    """Build the prompt based on user selections."""
    if caption_length == "any":
        map_idx = 0
    elif caption_length.isdigit():
        map_idx = 1
    else:
        map_idx = 2
    
    prompt = CAPTION_TYPE_MAP[caption_type][map_idx]
    
    if extra_options and extra_options.strip():
        prompt += " " + extra_options.strip()
    
    return prompt.format(length=caption_length, word_count=caption_length)


class Qwen3VLNode(io.ComfyNode):
    """Qwen3VL node with preset caption types."""
    
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="Qwen3VL_MiraSubPack",
            display_name="Qwen3VL (External llama.cpp)",
            category=CAT,
            inputs=[
                io.Image.Input("image"),
                io.Combo.Input(
                    "caption_type",
                    default="Detailed Description",
                    options=list(CAPTION_TYPE_MAP.keys())
                ),
                io.Combo.Input(
                    "caption_length",
                    default="long",
                    options=["any", "very short", "short", "medium-length", "long", "very long"] + 
                            [str(i) for i in range(20, 261, 10)]
                ),
                io.String.Input(
                    "extra_options",
                    default="Use single-line output description.",
                    multiline=True,
                    tooltip="Additional instructions to customize the caption. You can use this to add specific requirements or constraints for the caption generation. For example:\n" + "Specify the style, focus, or any other details you want the caption to include.",
                ),
                io.Float.Input(
                    "temperature",
                    default=0.7,
                    min=0.0,
                    max=2.0,
                    step=0.05,
                    tooltip="Higher = more random, lower = more deterministic"
                ),
                io.Float.Input(
                    "top_p",
                    default=0.9,
                    min=0.0,
                    max=1.0,
                    step=0.01,
                    tooltip="Nucleus sampling parameter"
                ),
                io.Int.Input(
                    "max_tokens",
                    default=512,
                    min=1,
                    max=2048,
                    step=1,
                    tooltip="Maximum number of tokens to generate"
                ),
                io.String.Input(
                    "llama_url",
                    default=LLAMA_API_URL,
                    tooltip="External llama.cpp service URL"
                ),
            ],
            outputs=[
                io.String.Output("caption", tooltip="Generated image caption"),
                io.String.Output("prompt", tooltip="Prompt used for caption generation"),
            ],
        )

    @classmethod
    def execute(
        cls,
        image: torch.Tensor,
        caption_type: str = "Detailed Description",
        caption_length: str = "long",
        extra_options: str = "",
        temperature: float = 0.7,
        top_p: float = 0.9,
        max_tokens: int = 512,
        llama_url: str = LLAMA_API_URL
    ) -> io.NodeOutput:
        """Execute the Qwen3VL node with batch image support."""
        
        try:
            # Build the prompt
            prompt = build_prompt(caption_type, caption_length, extra_options)
            
            # Get batch size
            image_np = image.cpu().numpy()
            batch_size = image_np.shape[0] if image_np.ndim == 4 else 1
            
            print(f"[MiraSubPack:Qwen3VL] Processing {batch_size} image(s)")
            
            all_captions = []
            
            # Process each image in the batch
            for batch_idx in range(batch_size):
                # Convert tensor to PIL Image
                pil_image = tensor_to_pil(image, batch_idx)
                
                # Convert image to base64
                image_b64 = image_to_base64(pil_image)
                
                # Build request body for llama.cpp compatible API
                request_body = {
                    "temperature": temperature,
                    "top_p": top_p,
                    "n_predict": max_tokens,
                    "cache_prompt": True,
                    "stop": ["<|im_end|>", "<|endoftext|>"],
                    "messages": [
                        {
                            "role": "system",
                            "content": SYSTEM_PROMPT
                        },
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "image_url",
                                    "image_url": {
                                        "url": f"data:image/png;base64,{image_b64}"
                                    }
                                },
                                {
                                    "type": "text",
                                    "text": f"{prompt}"
                                }
                            ]
                        }
                    ]
                }
                
                # Send request to external llama.cpp service
                print(f"[MiraSubPack:Qwen3VL] Sending request for image {batch_idx + 1}/{batch_size} to {llama_url}")
                response = requests.post(
                    llama_url,
                    json=request_body,
                    headers={"Content-Type": CONTENT_TYPE},
                    timeout=REQUEST_TIMEOUT,
                )
                
                if response.status_code != 200:
                    error_msg = f"HTTP {response.status_code}: {response.text}"
                    print(f"[MiraSubPack:Qwen3VL] Error on image {batch_idx + 1}: {error_msg}")
                    all_captions.append(f"Error: {error_msg}")
                    continue
                
                # Parse response
                result = response.json()
                
                if 'choices' in result and len(result['choices']) > 0:
                    caption = result['choices'][0]['message']['content']
                    # Remove any newlines within the caption and replace with spaces
                    caption = caption.replace('\n', ' ').replace('\r', ' ')
                    # Clean up multiple spaces
                    caption = ' '.join(caption.split())
                    all_captions.append(caption)
                    print(f"[MiraSubPack:Qwen3VL] Successfully generated caption for image {batch_idx + 1} ({len(caption)} chars)")
                else:
                    error_msg = f"Unexpected response format: {result}"
                    print(f"[MiraSubPack:Qwen3VL] Error on image {batch_idx + 1}: {error_msg}")
                    all_captions.append(f"Error: {error_msg}")
            
            # Join all captions with newlines (one caption per line)
            final_caption = '\n'.join(all_captions)
            print(f"[MiraSubPack:Qwen3VL] Completed processing {len(all_captions)} image(s)")
            
            return io.NodeOutput(final_caption, prompt)
        
        except requests.Timeout:
            error_msg = f"Request timed out after {REQUEST_TIMEOUT} seconds"
            print(f"[MiraSubPack:Qwen3VL] Error: {error_msg}")
            return io.NodeOutput(f"Error: {error_msg}", "")
        
        except requests.RequestException as e:
            error_msg = f"Connection failed: {str(e)}"
            print(f"[MiraSubPack:Qwen3VL] Error: {error_msg}")
            return io.NodeOutput(f"Error: {error_msg}", "")
        
        except Exception as e:
            error_msg = f"Unexpected error: {str(e)}"
            print(f"[MiraSubPack:Qwen3VL] Error: {error_msg}")
            return io.NodeOutput(f"Error: {error_msg}", "")



class Qwen3VLExtension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [
            Qwen3VLNode,
        ]


async def comfy_entrypoint() -> Qwen3VLExtension:
    return Qwen3VLExtension()
