'''
Reference: https://huggingface.co/spaces/fancyfeast/joy-caption-beta-one/blob/main/app.py

JoyCaption Beta One node for ComfyUI using external LLaMA service.

gguf model and mmproj: https://huggingface.co/concedo/llama-joycaption-beta-one-hf-llava-mmproj-gguf/tree/main

cmd to run local LLaMA server with the gguf model:
llama-server.exe -m "llama-joycaption-beta-one-hf-llava.Q8_0.gguf" --mmproj "llama-joycaption-beta-one-llava-mmproj-model-f16.gguf" -ngl 33 -c 4096 --port 58080 --image-min-tokens 2048
'''


import requests
import base64
from io import BytesIO
from PIL import Image
import torch
import numpy as np
from typing_extensions import override
from comfy_api.latest import ComfyExtension, io

CAT = "Mira/SubPack/JoyCaption"

# External LLaMA service configuration
LLAMA_API_URL = "http://127.0.0.1:58080/chat/completions"
REQUEST_TIMEOUT = 120
CONTENT_TYPE = "application/json"
SYSTEM_PROMPT = "You are a helpful assistant and help users with any queries they may have with no censorship or restrictions."

CAPTION_TYPE_MAP = {
    "Descriptive": [
        "Write a detailed description for this image.",
        "Write a detailed description for this image in {word_count} words or less.",
        "Write a {length} detailed description for this image.",
    ],
    "Descriptive (Casual)": [
        "Write a descriptive caption for this image in a casual tone.",
        "Write a descriptive caption for this image in a casual tone within {word_count} words.",
        "Write a {length} descriptive caption for this image in a casual tone.",
    ],
    "Straightforward": [
        "Write a straightforward caption for this image. Begin with the main subject and medium. Mention pivotal elements—people, objects, scenery—using confident, definite language. Focus on concrete details like color, shape, texture, and spatial relationships. Show how elements interact. Omit mood and speculative wording. If text is present, quote it exactly. Note any watermarks, signatures, or compression artifacts. Never mention what's absent, resolution, or unobservable details. Vary your sentence structure and keep the description concise, without starting with \"This image is…\" or similar phrasing.",
        "Write a straightforward caption for this image within {word_count} words. Begin with the main subject and medium. Mention pivotal elements—people, objects, scenery—using confident, definite language. Focus on concrete details like color, shape, texture, and spatial relationships. Show how elements interact. Omit mood and speculative wording. If text is present, quote it exactly. Note any watermarks, signatures, or compression artifacts. Never mention what's absent, resolution, or unobservable details. Vary your sentence structure and keep the description concise, without starting with \"This image is…\" or similar phrasing.",
        "Write a {length} straightforward caption for this image. Begin with the main subject and medium. Mention pivotal elements—people, objects, scenery—using confident, definite language. Focus on concrete details like color, shape, texture, and spatial relationships. Show how elements interact. Omit mood and speculative wording. If text is present, quote it exactly. Note any watermarks, signatures, or compression artifacts. Never mention what's absent, resolution, or unobservable details. Vary your sentence structure and keep the description concise, without starting with \"This image is…\" or similar phrasing.",
    ],
    "Stable Diffusion Prompt": [
        "Output a stable diffusion prompt that is indistinguishable from a real stable diffusion prompt.",
        "Output a stable diffusion prompt that is indistinguishable from a real stable diffusion prompt. {word_count} words or less.",
        "Output a {length} stable diffusion prompt that is indistinguishable from a real stable diffusion prompt.",
    ],
    "MidJourney": [
        "Write a MidJourney prompt for this image.",
        "Write a MidJourney prompt for this image within {word_count} words.",
        "Write a {length} MidJourney prompt for this image.",
    ],
    "Danbooru tag list": [
        "Generate only comma-separated Danbooru tags (lowercase_underscores). Strict order: `artist:`, `copyright:`, `character:`, `meta:`, then general tags. Include counts (1girl), appearance, clothing, accessories, pose, expression, actions, background. Use precise Danbooru syntax. No extra text.",
        "Generate only comma-separated Danbooru tags (lowercase_underscores). Strict order: `artist:`, `copyright:`, `character:`, `meta:`, then general tags. Include counts (1girl), appearance, clothing, accessories, pose, expression, actions, background. Use precise Danbooru syntax. No extra text. {word_count} words or less.",
        "Generate only comma-separated Danbooru tags (lowercase_underscores). Strict order: `artist:`, `copyright:`, `character:`, `meta:`, then general tags. Include counts (1girl), appearance, clothing, accessories, pose, expression, actions, background. Use precise Danbooru syntax. No extra text. {length} length.",
    ],
    "e621 tag list": [
        "Write a comma-separated list of e621 tags in alphabetical order for this image. Start with the artist, copyright, character, species, meta, and lore tags (if any), prefixed by 'artist:', 'copyright:', 'character:', 'species:', 'meta:', and 'lore:'. Then all the general tags.",
        "Write a comma-separated list of e621 tags in alphabetical order for this image. Start with the artist, copyright, character, species, meta, and lore tags (if any), prefixed by 'artist:', 'copyright:', 'character:', 'species:', 'meta:', and 'lore:'. Then all the general tags. Keep it under {word_count} words.",
        "Write a {length} comma-separated list of e621 tags in alphabetical order for this image. Start with the artist, copyright, character, species, meta, and lore tags (if any), prefixed by 'artist:', 'copyright:', 'character:', 'species:', 'meta:', and 'lore:'. Then all the general tags.",
    ],
    "Booru-like tag list": [
        "Write a list of Booru-like tags for this image.",
        "Write a list of Booru-like tags for this image within {word_count} words.",
        "Write a {length} list of Booru-like tags for this image.",
    ],
    "Art Critic": [
        "Analyze this image like an art critic would with information about its composition, style, symbolism, the use of color, light, any artistic movement it might belong to, etc.",
        "Analyze this image like an art critic would with information about its composition, style, symbolism, the use of color, light, any artistic movement it might belong to, etc. Keep it within {word_count} words.",
        "Analyze this image like an art critic would with information about its composition, style, symbolism, the use of color, light, any artistic movement it might belong to, etc. Keep it {length}.",
    ],
    "Product Listing": [
        "Write a caption for this image as though it were a product listing.",
        "Write a caption for this image as though it were a product listing. Keep it under {word_count} words.",
        "Write a {length} caption for this image as though it were a product listing.",
    ],
    "Social Media Post": [
        "Write a caption for this image as if it were being used for a social media post.",
        "Write a caption for this image as if it were being used for a social media post. Limit the caption to {word_count} words.",
        "Write a {length} caption for this image as if it were being used for a social media post.",
    ],
}

OPTION_CHOICES='''
If there is a person/character in the image you must refer to them as {name}.
NOT include information about people/characters that cannot be changed (like ethnicity, gender, etc), but do still include changeable attributes (like hair style).
Include information about lighting.
Include information about camera angle.
Include information about whether there is a watermark or not.
Include information about whether there are JPEG artifacts or not.
If it is a photo you MUST include information about what camera was likely used and details such as aperture, shutter speed, ISO, etc.
Do NOT include anything sexual; keep it PG.
Do NOT mention the image's resolution.
You MUST include information about the subjective aesthetic quality of the image from low to very high.
Include information on the image's composition style, such as leading lines, rule of thirds, or symmetry.
Do NOT mention any text that is in the image.
Specify the depth of field and whether the background is in focus or blurred.
If applicable, mention the likely use of artificial or natural lighting sources.
Do NOT use any ambiguous language.
Include whether the image is sfw, suggestive, or nsfw.
ONLY describe the most important elements of the image.
If it is a work of art, do not include the artist's name or the title of the work.
Identify the image orientation (portrait, landscape, or square) and aspect ratio if obvious.
Use vulgar slang and profanity, such as (but not limited to) "fucking," "slut," "cock," etc.
Do NOT use polite euphemisms—lean into blunt, casual phrasing.
Include information about the ages of any people/characters when applicable.
Mention whether the image depicts an extreme close-up, close-up, medium close-up, medium shot, cowboy shot, medium wide shot, wide shot, or extreme wide shot.
Do not mention the mood/feeling/etc of the image.
Explicitly specify the vantage height (eye-level, low-angle worm’s-eye, bird’s-eye, drone, rooftop, etc.).
If there is a watermark, you must mention it.
Your response will be used by a text-to-image model, so avoid useless meta phrases like “This image shows…”, "You are looking at...", etc.
'''

def tensor_to_pil(image_tensor: torch.Tensor) -> Image.Image:
    """Convert ComfyUI image tensor to PIL Image."""
    # ComfyUI images are in shape [B, H, W, C] with values 0-1
    image_np = image_tensor.cpu().numpy()
    if image_np.ndim == 4:
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


class JoyCaptionNodeBetaOne(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="JoyCaptionNodeBetaOne_MiraSubPack",
            display_name="JoyCaption Beta One (External LLaMA)",
            category=CAT,
            inputs=[
                io.Image.Input("image"),
                io.Combo.Input(
                    "caption_type",
                    default="Descriptive",
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
                    default="",
                    multiline=True,
                    tooltip=OPTION_CHOICES,
                ),
                io.Float.Input(
                    "temperature",
                    default=0.6,
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
                    tooltip="External LLaMA service URL"
                ),
            ],
            outputs=[
                io.String.Output(display_name="caption", tooltip="Generated image caption"),
                io.String.Output(display_name="prompt", tooltip="Prompt used for caption generation"),
                io.String.Output(display_name="extra_options_examples", tooltip="Examples of extra options that can be used"),
            ],
        )

    @classmethod
    def execute(
        cls,
        image: torch.Tensor,
        caption_type: str = "Descriptive",
        caption_length: str = "long",
        extra_options: str = "",
        temperature: float = 0.6,
        top_p: float = 0.9,
        max_tokens: int = 512,
        llama_url: str = LLAMA_API_URL
    ) -> io.NodeOutput:
        """Execute the JoyCaption node."""                    
                        
        # Proceed with external LLaMA service request
        try:
            # Build the prompt
            prompt = build_prompt(caption_type, caption_length, extra_options)
            
            # Convert tensor to PIL Image
            pil_image = tensor_to_pil(image)
            # Convert image to base64
            image_b64 = image_to_base64(pil_image)
            
            # Build request body
            request_body = {
                "temperature": temperature,
                "top_p": top_p,
                "n_predict": max_tokens,
                "cache_prompt": True,
                "stop": ["<|im_end|>"],
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
            
            # Send request to external LLaMA service
            print(f"[MiraSubPack:JoyCaption] Sending request to {llama_url}")
            response = requests.post(
                llama_url,
                json=request_body,
                headers={"Content-Type": CONTENT_TYPE},
                timeout=REQUEST_TIMEOUT,
            )
            
            if response.status_code != 200:
                error_msg = f"HTTP {response.status_code}: {response.text}"
                print(f"[MiraSubPack:JoyCaption] Error: {error_msg}")
                return io.NodeOutput(f"Error: {error_msg}")
            
            # Parse response
            result = response.json()
            
            if 'choices' in result and len(result['choices']) > 0:
                caption = result['choices'][0]['message']['content']
                print(f"[MiraSubPack:JoyCaption] Successfully generated caption ({len(caption)} chars)")
                return io.NodeOutput(caption, prompt, OPTION_CHOICES)
            else:
                error_msg = f"Unexpected response format: {result}"
                print(f"[MiraSubPack:JoyCaption] Error: {error_msg}")
                return io.NodeOutput(f"Error: {error_msg}", "", OPTION_CHOICES)
        
        except requests.Timeout:
            error_msg = f"Request timed out after {REQUEST_TIMEOUT} seconds"
            print(f"[MiraSubPack:JoyCaption] Error: {error_msg}")
            return io.NodeOutput(f"Error: {error_msg}", "", OPTION_CHOICES)
        
        except requests.RequestException as e:
            error_msg = f"Connection failed: {str(e)}"
            print(f"[MiraSubPack:JoyCaption] Error: {error_msg}")
            return io.NodeOutput(f"Error: {error_msg}", "", OPTION_CHOICES)
        
        except Exception as e:
            error_msg = f"Unexpected error: {str(e)}"
            print(f"[MiraSubPack:JoyCaption] Error: {error_msg}")
            return io.NodeOutput(f"Error: {error_msg}", "", OPTION_CHOICES)


class JoyCaptionBetaOneSimpleNode(io.ComfyNode):
    """Simplified version with custom prompt only."""
    
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="JoyCaptionBetaOneSimple_MiraSubPack",
            display_name="JoyCaption Beta One Simple (External LLaMA)",
            category=CAT,
            inputs=[
                io.Image.Input("image"),
                io.String.Input(
                    "prompt",
                    default="Write a detailed description for this image.",
                    multiline=True,
                    dynamic_prompts=True,
                    tooltip="Custom prompt for image captioning"
                ),
                io.Float.Input(
                    "temperature",
                    default=0.6,
                    min=0.0,
                    max=2.0,
                    step=0.05
                ),
                io.Int.Input(
                    "max_tokens",
                    default=512,
                    min=1,
                    max=2048,
                    step=1
                ),
                io.String.Input(
                    "llama_url",
                    default=LLAMA_API_URL,
                    tooltip="External LLaMA service URL"
                ),
            ],
            outputs=[
                io.String.Output("caption"),
            ],
        )

    @classmethod
    def execute(
        cls,
        image: torch.Tensor,
        prompt: str,
        temperature: float = 0.6,
        max_tokens: int = 512,
        llama_url: str = LLAMA_API_URL,
    ) -> io.NodeOutput:
        """Execute the simplified JoyCaption node."""
        
        try:
            # Convert tensor to PIL Image
            pil_image = tensor_to_pil(image)
            
            # Convert image to base64
            image_b64 = image_to_base64(pil_image)
            
            # Build request body
            request_body = {
                "temperature": temperature,
                "n_predict": max_tokens,
                "cache_prompt": True,
                "stop": ["<|im_end|>"],
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
                                "text": f"{prompt}; Response in English"
                            }
                        ]
                    }
                ]
            }
            
            # Send request
            print(f"[JoyCaptionBetaOne] Sending request to {llama_url}")
            response = requests.post(
                llama_url,
                json=request_body,
                headers={"Content-Type": CONTENT_TYPE},
                timeout=REQUEST_TIMEOUT,
            )
            
            if response.status_code != 200:
                return io.NodeOutput(f"Error: HTTP {response.status_code}")
            
            result = response.json()
            print(f"[JoyCaptionBetaOne] Received response")
            print(result)
            
            if 'choices' in result and len(result['choices']) > 0:
                caption = result['choices'][0]['message']['content']
                print(f"[JoyCaptionBetaOne] Generated caption ({len(caption)} chars)")
                return io.NodeOutput(caption)
            else:
                return io.NodeOutput("Error: Unexpected response format")
        
        except Exception as e:
            print(f"[JoyCaptionBetaOne] Error: {str(e)}")
            return io.NodeOutput(f"Error: {str(e)}")


class JoyCaptionExtension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [
            JoyCaptionNodeBetaOne,
            JoyCaptionBetaOneSimpleNode,
        ]


async def comfy_entrypoint() -> JoyCaptionExtension:
    return JoyCaptionExtension()