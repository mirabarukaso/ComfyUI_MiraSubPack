from .image_util import ImageMergeByPixelAlign
from .image_tiled_upscaler import MiraITUPipelineExtract, MiraITUPipelineCombine, ImageTiledKSamplerWithTagger, OverlappedImageMerge, OverlappedLatentMerge, \
    TiledImageColorCorrection, ImageCropTiles, ImageCropTilesByPixels, LatentUpscaleAndCropTiles, MiraImageUpscaleCalculator
from .image_tiles_llm import ImageTilesToLLM
from .utils import VAEEncode_Mira, VAEDecode_Mira, VAESafePassthrough_Mira
from .Model.nodes_sage_fp16 import SageFP16Switch_Mira, UNETLoaderFP16Storage_Mira


NODE_CLASS_MAPPINGS = {
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
    "MiraImageUpscaleCalculator_MiraSubPack": MiraImageUpscaleCalculator,
    "ImageTilesToLLM_MiraSubPack": ImageTilesToLLM,

    "VAEEncode_MiraSubPack": VAEEncode_Mira,
    "VAEDecode_MiraSubPack": VAEDecode_Mira,
    "VAESafePassthrough_MiraSubPack": VAESafePassthrough_Mira,

    "SageFP16Switch_MiraSubPack": SageFP16Switch_Mira,
    "UNETLoaderFP16Storage_MiraSubPack": UNETLoaderFP16Storage_Mira,
}

NODE_DISPLAY_NAME_MAPPINGS = {
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
    "MiraImageUpscaleCalculator_MiraSubPack": "Mira Image Upscale Calculator",
    "ImageTilesToLLM_MiraSubPack": "Image Tiles to LLM",

    "VAEEncode_MiraSubPack": "VAE Encode (Mira SubPack)",
    "VAEDecode_MiraSubPack": "VAE Decode (Mira SubPack)",
    "VAESafePassthrough_MiraSubPack": "VAE Safe Passthrough (Mira SubPack)",

    "SageFP16Switch_MiraSubPack": "Sage FP16 Switch (Mira SubPack)",
    "UNETLoaderFP16Storage_MiraSubPack": "Load Diffusion Model FP16 Storage (Mira SubPack)",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
