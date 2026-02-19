import torch
from comfy_api.latest import io

CAT = "Mira/SubPack/Utils"

class VAEEncode_Mira(io.ComfyNode):
    """
    Upscales input latent with VRAM optimization for batch processing.
    """
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="VAEEncode_MiraSubPack",
            display_name="VAE Encode (Mira SubPack)",
            category=CAT,
            description="Encode images to latent space with VRAM optimization for batch images.",
            inputs=[
                io.Image.Input("pixels", ),
                io.Vae.Input("vae", ),
            ],
            outputs=[
                io.Latent.Output(display_name="samples"),
            ],
            is_output_node=True
        )
    
    @classmethod
    def execute(cls, pixels, vae) -> io.NodeOutput:
        """
        Encode pixels to latent space.
        
        If only 1 image: use standard VAE encoding
        If multiple images: process each image separately and store in RAM to avoid VRAM OOM
        """
        batch_size = pixels.shape[0]
        
        # If only single image, use standard encoding
        if batch_size == 1:
            latent = vae.encode(pixels)
            latents = {
                'samples': latent
            }  
            return io.NodeOutput(latents)
        
        # For multiple images, process each separately to save VRAM
        latent_list = []        
        for i in range(batch_size):
            # Extract single image from batch
            single_image = pixels[i:i+1]
            
            # Encode single image
            single_latent = vae.encode(single_image)
            
            # Add to dict if not already, to handle cases where vae.encode returns dict or tensor
            single_latent_dict = {
                'samples': single_latent['samples'] if isinstance(single_latent, dict) else single_latent
            }
            
            latent_list.append(single_latent_dict)
        
        # Concatenate all latents back together
        if isinstance(latent_list[0], dict):
            combined_samples = torch.cat([l['samples'] for l in latent_list], dim=0)
        else:
            combined_samples = torch.cat(latent_list, dim=0)
                
        latent = {
            'samples': combined_samples
        }    
        
        return io.NodeOutput(latent)


class VAEDecode_Mira(io.ComfyNode):
    """
    Decodes latent images back into pixel space with VRAM optimization for batch processing.
    """
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="VAEDecode_MiraSubPack",
            display_name="VAE Decode (Mira SubPack)",
            category=CAT,
            description="Decode latent images to pixel space with VRAM optimization for batch latents.",
            inputs=[
                io.Latent.Input("samples", ),
                io.Vae.Input("vae", ),
            ],
            outputs=[
                io.Image.Output(display_name="image"),
            ],
            is_output_node=True
        )
    
    @classmethod
    def execute(cls, samples, vae) -> io.NodeOutput:
        """
        Decode latent to pixel space.
        
        If only 1 latent: use standard VAE decoding
        If multiple latents: process each latent separately and store in RAM to avoid VRAM OOM
        """
        # Handle nested latents
        latent = samples["samples"]
        if hasattr(latent, 'is_nested') and latent.is_nested:
            latent = latent.unbind()[0]
        
        batch_size = latent.shape[0]
        
        # If only single latent, use standard decoding
        if batch_size == 1:
            images = vae.decode(latent)
            # Handle 5D tensors (combine batches)
            if len(images.shape) == 5:
                images = images.reshape(-1, images.shape[-3], images.shape[-2], images.shape[-1])
            return io.NodeOutput(images)
        
        # For multiple latents, process each separately to save VRAM
        image_list = []
        
        for i in range(batch_size):
            # Extract single latent from batch
            single_latent = latent[i:i+1]
            
            # Decode single latent
            single_image = vae.decode(single_latent)
            
            # Handle 5D tensors
            if len(single_image.shape) == 5:
                single_image = single_image.reshape(-1, single_image.shape[-3], single_image.shape[-2], single_image.shape[-1])
            
            # Move to CPU to free VRAM
            single_image_cpu = single_image.cpu()
            
            image_list.append(single_image_cpu)
        
        # Concatenate all images back together
        combined_images = torch.cat(image_list, dim=0)
        
        return io.NodeOutput(combined_images)
        