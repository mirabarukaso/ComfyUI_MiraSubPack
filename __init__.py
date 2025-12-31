from .JoyCaption.beta_one import JoyCaptionNodeBetaOne, JoyCaptionBetaOneSimpleNode
from .QwenImage.nodes_qwen import TextEncodeQwenImageEditMod, TextEncodeQwenImageEditPlusMod 
from .image_util import ImageMergeByPixelAlign
from .image_tiled_upscaler import MiraITUPipelineExtract, MiraITUPipelineCombine, ImageTiledKSamplerWithTagger, OverlappedImageMerge, OverlappedLatentMerge, ImageCropTiles, LatentUpscaleAndCropTiles

def __init__(self):
    pass
    
NODE_CLASS_MAPPINGS = {
    "JoyCaptionNodeBetaOne_MiraSubPack": JoyCaptionNodeBetaOne,
    "JoyCaptionBetaOneSimple_MiraSubPack": JoyCaptionBetaOneSimpleNode,
    
    "TextEncodeQwenImageEdit_MiraSubPack": TextEncodeQwenImageEditMod,
    "TextEncodeQwenImageEditPlus_MiraSubPack": TextEncodeQwenImageEditPlusMod,
    
    "ImageMergeByPixelAlign_MiraSubPack": ImageMergeByPixelAlign,
    
    "MiraITUPipelineExtract_MiraSubPack": MiraITUPipelineExtract,
    "MiraITUPipelineCombine_MiraSubPack": MiraITUPipelineCombine,
    "ImageTiledKSamplerWithTagger_MiraSubPack": ImageTiledKSamplerWithTagger,
    "OverlappedImageMerge_MiraSubPack": OverlappedImageMerge,
    "OverlappedLatentMerge_MiraSubPack": OverlappedLatentMerge,
    "ImageCropTiles_MiraSubPack": ImageCropTiles,
    "LatentUpscaleAndCropTiles_MiraSubPack": LatentUpscaleAndCropTiles,
}    

NODE_DISPLAY_NAME_MAPPINGS = {
    "JoyCaptionNodeBetaOne_MiraSubPack": "JoyCaption Beta One (External LLaMA)",
    "JoyCaptionBetaOneSimple_MiraSubPack": "JoyCaption Beta One Simple (External LLaMA)",
    
    "TextEncodeQwenImageEdit_MiraSubPack": "Text Encode QwenImage Edit Mira",
    "TextEncodeQwenImageEditPlus_MiraSubPack": "Text Encode QwenImage Edit Plus Mira",
    
    "ImageMergeByPixelAlign_MiraSubPack": "Image Merge By Pixel Align",
    
    "MiraITUPipelineExtract_MiraSubPack": "Mira ITU Pipeline Extract",
    "MiraITUPipelineCombine_MiraSubPack": "Mira ITU Pipeline Combine",
    "ImageTiledKSamplerWithTagger_MiraSubPack": "Tiled Image KSampler with Tagger",
    "OverlappedImageMerge_MiraSubPack": "Overlapped Image Merge",
    "OverlappedLatentMerge_MiraSubPack": "Overlapped Latent Merge",
    "ImageCropTiles_MiraSubPack": "Image Crop to Tiles",
    "LatentUpscaleAndCropTiles_MiraSubPack": "Latent Upscale then Crop to Tiles",    
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]