from .JoyCaption.beta_one import JoyCaptionNodeBetaOne, JoyCaptionBetaOneSimpleNode
from .QwenImage.nodes_qwen import TextEncodeQwenImageEditMod, TextEncodeQwenImageEditPlusMod 
from .image_util import ImageMergeByPixelAlign
from .image_tiled_upscaler import ImageTiledKSamplerWithTagger, OverlappedImageMerge, OverlappedLatentMerge, ImageCropTiles

def __init__(self):
    pass
    
NODE_CLASS_MAPPINGS = {
    "JoyCaptionNodeBetaOne_MiraSubPack": JoyCaptionNodeBetaOne,
    "JoyCaptionBetaOneSimple_MiraSubPack": JoyCaptionBetaOneSimpleNode,
    
    "TextEncodeQwenImageEdit_MiraSubPack": TextEncodeQwenImageEditMod,
    "TextEncodeQwenImageEditPlus_MiraSubPack": TextEncodeQwenImageEditPlusMod,
    
    "ImageMergeByPixelAlign_MiraSubPack": ImageMergeByPixelAlign,
    
    "ImageTiledKSamplerWithTagger": ImageTiledKSamplerWithTagger,
    "OverlappedImageMerge": OverlappedImageMerge,
    "OverlappedLatentMerge": OverlappedLatentMerge,
    "ImageCropTiles": ImageCropTiles
}    

NODE_DISPLAY_NAME_MAPPINGS = {
    "JoyCaptionNodeBetaOne_MiraSubPack": "JoyCaption Beta One (External LLaMA)",
    "JoyCaptionBetaOneSimple_MiraSubPack": "JoyCaption Beta One Simple (External LLaMA)",
    
    "TextEncodeQwenImageEdit_MiraSubPack": "Text Encode QwenImage Edit Mira",
    "TextEncodeQwenImageEditPlus_MiraSubPack": "Text Encode QwenImage Edit Plus Mira",
    
    "ImageMergeByPixelAlign_MiraSubPack": "Image Merge By Pixel Align",
    
    "ImageTiledKSamplerWithTagger": "Image Tiled KSampler",
    "OverlappedImageMerge": "Overlapped Image Merge",
    "OverlappedLatentMerge": "Overlapped Latent Merge",
    "ImageCropTiles": "Image Crop to Tiles",
    
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]