# Per-model fp16 switch for GPUs without bf16 (e.g. Turing sm75), so SageAttention gets fp16 q/k/v.
# No global monkeypatch: everything is applied to a cloned ModelPatcher.

import logging
import torch
import folder_paths
import comfy.sd
from comfy_api.latest import io

logger = logging.getLogger(__name__)

CAT = "Mira/SubPack/Model"


def _dtype_name(dt):
    return str(dt).replace("torch.", "")


def _make_attention_override(prev):
    """Per-model optimized_attention_override: cast non-fp16 q/k/v to fp16, run the active attention, cast back."""
    state = {"logged": False}

    def override(func, q, k, v, *args, **kwargs):
        dt = q.dtype
        if not state["logged"]:
            state["logged"] = True
            logger.info("[SageFP16Switch] attention first call: q/k/v dtype=%s head_dim=%d -> %s", _dtype_name(dt), q.shape[-1],
                        "pass-through (already fp16)" if dt == torch.float16 else "casting to float16")
        call = (lambda *a, **kw: prev(func, *a, **kw)) if prev is not None else func
        if dt == torch.float16:
            return call(q, k, v, *args, **kwargs)
        return call(q.half(), k.half(), v.half(), *args, **kwargs).to(dt)

    override._mira_sage_fp16 = True
    return override


class SageFP16Switch_Mira(io.ComfyNode):
    """
    Run this model in fp16 (compute dtype) and optionally cast attention inputs to fp16 for SageAttention.
    Bypass the node to keep the original behavior.
    """
    NODE_ID = "SageFP16Switch_MiraSubPack"
    DISPLAY = "Sage FP16 Switch (Mira SubPack)"
    DEPRECATED = False

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id=cls.NODE_ID,
            display_name=cls.DISPLAY,
            category=CAT,
            description="Per-model fp16 switch: sets compute dtype to fp16 and installs a per-model attention fp16 cast for SageAttention. Bypass to disable.",
            search_aliases=["sage fp16", "fp16 switch", "turing fp16"],
            is_deprecated=cls.DEPRECATED,
            inputs=[
                io.Model.Input("model", tooltip="Diffusion model (after loader)."),
                io.Boolean.Input("compute_fp16", default=True, tooltip="Run this model's compute in fp16 (skip if already fp16). Same mechanism as ModelComputeDtype."),
                io.Boolean.Input("sage_cast", default=True, tooltip="Cast q/k/v to fp16 before attention (SageAttention) and back. Skips if already fp16."),
            ],
            outputs=[
                io.Model.Output(display_name="model"),
            ],
        )

    @classmethod
    def execute(cls, model, compute_fp16, sage_cast) -> io.NodeOutput:
        m = model.clone()
        base = m.model
        weight_dtype = base.get_dtype()
        # Do NOT read base.manual_cast_dtype directly: the BaseModel object is shared by every ModelPatcher clone,
        # and set_model_compute_dtype() is an object patch that is applied to it while *another* patcher is loaded
        # (e.g. a ModelComputeDtype branch that ran earlier). Resolve this patcher's own value instead.
        if "manual_cast_dtype" in m.object_patches:
            manual_cast = m.object_patches["manual_cast_dtype"]
        else:
            manual_cast = m.object_patches_backup.get("manual_cast_dtype", getattr(base, "manual_cast_dtype", None)) \
                if hasattr(m, "object_patches_backup") else getattr(base, "manual_cast_dtype", None)
        effective = manual_cast if manual_cast is not None else weight_dtype
        done = []

        if compute_fp16:
            # set_model_compute_dtype is idempotent and cheap; always apply it so the result never depends on
            # which patcher happened to be loaded when this node executed.
            m.set_model_compute_dtype(torch.float16)
            done.append(f"compute {_dtype_name(effective)} -> float16")

        if sage_cast:
            # copy transformer_options so the source patcher is not mutated
            transformer_options = dict(m.model_options.get("transformer_options", {}))
            prev = transformer_options.get("optimized_attention_override")
            if not getattr(prev, "_mira_sage_fp16", False):
                transformer_options["optimized_attention_override"] = _make_attention_override(prev)
                m.model_options["transformer_options"] = transformer_options
                done.append("per-model attention fp16 cast installed")

        logger.info("[SageFP16Switch] %s: weight dtype %s, manual cast %s; %s", base.__class__.__name__,
                    _dtype_name(weight_dtype), _dtype_name(manual_cast), "; ".join(done) or "nothing")
        return io.NodeOutput(m)


class UNETLoaderFP16Storage_Mira(io.ComfyNode):
    """
    UNETLoader variant that stores diffusion model weights as fp16 at load time (halves VRAM vs fp32 storage).
    """
    NODE_ID = "UNETLoaderFP16Storage_MiraSubPack"
    DISPLAY = "Load Diffusion Model FP16 Storage (Mira SubPack)"
    DEPRECATED = False

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id=cls.NODE_ID,
            display_name=cls.DISPLAY,
            category=CAT,
            description="Load a diffusion model with fp16 weight storage (model_options dtype=fp16). Separate cache entry, does not affect models loaded by other loaders.",
            search_aliases=["unet fp16", "load diffusion model fp16"],
            is_deprecated=cls.DEPRECATED,
            inputs=[
                io.Combo.Input("unet_name", options=folder_paths.get_filename_list("diffusion_models"), tooltip="Diffusion model file."),
            ],
            outputs=[
                io.Model.Output(display_name="model"),
            ],
        )

    @classmethod
    def execute(cls, unet_name) -> io.NodeOutput:
        path = folder_paths.get_full_path_or_raise("diffusion_models", unet_name)
        model = comfy.sd.load_diffusion_model(path, model_options={"dtype": torch.float16})
        logger.info("[UNETLoaderFP16Storage] loaded %s with fp16 weight storage (weight dtype %s, manual cast %s)", unet_name,
                    _dtype_name(model.model.get_dtype()), _dtype_name(getattr(model.model, "manual_cast_dtype", None)))
        return io.NodeOutput(model)

