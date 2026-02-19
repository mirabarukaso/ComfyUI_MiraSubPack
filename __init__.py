from .JoyCaption.beta_one import JoyCaptionNodeBetaOne, JoyCaptionBetaOneSimpleNode
from .QwenImage.nodes_qwen import TextEncodeQwenImageEditMod, TextEncodeQwenImageEditPlusMod
from .QwenImage.nodes_qwen3vl import Qwen3VLNode
from .image_util import ImageMergeByPixelAlign
from .image_tiled_upscaler import MiraITUPipelineExtract, MiraITUPipelineCombine, ImageTiledKSamplerWithTagger, OverlappedImageMerge, OverlappedLatentMerge, \
    TiledImageColorCorrection, ImageCropTiles, ImageCropTilesByPixels, LatentUpscaleAndCropTiles, LatentUpscaleSimple
from .utils import VAEEncode_Mira, VAEDecode_Mira


def __init__(self):
    pass
    
NODE_CLASS_MAPPINGS = {
    "JoyCaptionNodeBetaOne_MiraSubPack": JoyCaptionNodeBetaOne,
    "JoyCaptionBetaOneSimple_MiraSubPack": JoyCaptionBetaOneSimpleNode,
    
    "TextEncodeQwenImageEdit_MiraSubPack": TextEncodeQwenImageEditMod,
    "TextEncodeQwenImageEditPlus_MiraSubPack": TextEncodeQwenImageEditPlusMod,

    "Qwen3VL_MiraSubPack": Qwen3VLNode,
    
    "ImageMergeByPixelAlign_MiraSubPack": ImageMergeByPixelAlign,
    
    "MiraITUPipelineExtract_MiraSubPack": MiraITUPipelineExtract,
    "MiraITUPipelineCombine_MiraSubPack": MiraITUPipelineCombine,
    "ImageTiledKSamplerWithTagger_MiraSubPack": ImageTiledKSamplerWithTagger,
    "OverlappedImageMerge_MiraSubPack": OverlappedImageMerge,
    "OverlappedLatentMerge_MiraSubPack": OverlappedLatentMerge,
    "TiledImageColorCorrection_MiraSubPack": TiledImageColorCorrection,
    "ImageCropTiles_MiraSubPack": ImageCropTiles,
    "ImageCropTilesByPixels_MiraSubPack": ImageCropTilesByPixels,
    "LatentUpscaleAndCropTiles_MiraSubPack": LatentUpscaleAndCropTiles,
    "LatentUpscaleSimple_MiraSubPack": LatentUpscaleSimple,
    
    "VAEEncode_MiraSubPack": VAEEncode_Mira,
    "VAEDecode_MiraSubPack": VAEDecode_Mira,
}    

NODE_DISPLAY_NAME_MAPPINGS = {
    "JoyCaptionNodeBetaOne_MiraSubPack": "JoyCaption Beta One (External llama.cpp)",
    "JoyCaptionBetaOneSimple_MiraSubPack": "JoyCaption Beta One Simple (External llama.cpp)",
    
    "Qwen3VL_MiraSubPack": "Qwen3VL (External llama.cpp)",
    
    "TextEncodeQwenImageEdit_MiraSubPack": "Text Encode QwenImage Edit Mira",
    "TextEncodeQwenImageEditPlus_MiraSubPack": "Text Encode QwenImage Edit Plus Mira",
    
    "ImageMergeByPixelAlign_MiraSubPack": "Image Merge By Pixel Align",
    
    "MiraITUPipelineExtract_MiraSubPack": "Mira ITU Pipeline Extract",
    "MiraITUPipelineCombine_MiraSubPack": "Mira ITU Pipeline Combine",
    "ImageTiledKSamplerWithTagger_MiraSubPack": "Tiled Image KSampler with Tagger",
    "OverlappedImageMerge_MiraSubPack": "Overlapped Image Merge",
    "OverlappedLatentMerge_MiraSubPack": "Overlapped Latent Merge",
    "TiledImageColorCorrection_MiraSubPack": "Tiled Image Color Correction",
    "ImageCropTiles_MiraSubPack": "Image Crop to Tiles",
    "ImageCropTilesByPixels_MiraSubPack": "Image Crop to Tiles by Pixels",
    "LatentUpscaleAndCropTiles_MiraSubPack": "Latent Upscale then Crop to Tiles",
    "LatentUpscaleSimple_MiraSubPack": "Latent Upscale with Add noise",
    
    "VAEEncode_MiraSubPack": "VAE Encode (Mira SubPack)",
    "VAEDecode_MiraSubPack": "VAE Decode (Mira SubPack)",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]