import sys
import types
from pathlib import Path
from torch import Tensor
import torch
from torchao.quantization import float8_weight_only, int8_weight_only, quantize_
import math

sys.path.extend([str(Path(__file__).parent), str(Path(__file__).parent.parent)])

from comfy.model_patcher import ModelPatcher
from comfy.sd import VAE

# from _flux_forward_orig import forward_orig


torch.backends.cudnn.allow_tf32 = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.set_float32_matmul_precision("medium")


def has_affordable_memory(device: torch.device) -> bool:
    free_memory, _ = torch.cuda.mem_get_info(device)
    free_memory_gb = free_memory / (1024**3)
    return free_memory_gb > 24


def is_newer_than_ada_lovelace(device: torch.device) -> int:
    cc_major, cc_minor = torch.cuda.get_device_capability(device)
    return cc_major * 10 + cc_minor >= 89

def timestep_embedding(t: Tensor, dim, max_period=10000, time_factor: float = 1000.0):
    """
    Create sinusoidal timestep embeddings.
    :param t: a 1-D Tensor of N indices, one per batch element.
                      These may be fractional.
    :param dim: the dimension of the output.
    :param max_period: controls the minimum frequency of the embeddings.
    :return: an (N, D) Tensor of positional embeddings.
    """
    t = time_factor * t
    half = dim // 2
    freqs = torch.exp(-math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half).to(
        t.device
    )

    args = t[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    if torch.is_floating_point(t):
        embedding = embedding.to(t)
    return embedding

def forward_orig(
        self,
        img: Tensor,
        img_ids: Tensor,
        txt: Tensor,
        txt_ids: Tensor,
        timesteps: Tensor,
        y: Tensor,
        guidance: Tensor = None,
    ) -> Tensor:
        if img.ndim != 3 or txt.ndim != 3:
            raise ValueError("Input img and txt tensors must have 3 dimensions.")

        # running on sequences img
        img = self.img_in(img)
        vec = self.time_in(timestep_embedding(timesteps, 256).to(img.dtype))
        if self.params.guidance_embed:
            if guidance is None:
                raise ValueError("Didn't get guidance strength for guidance distilled model.")
            vec = vec + self.guidance_in(timestep_embedding(guidance, 256).to(img.dtype))

        vec = vec + self.vector_in(y)
        txt = self.txt_in(txt)

        ids = torch.cat((txt_ids, img_ids), dim=1)
        pe = self.pe_embedder(ids)

        for block in self.double_blocks:
            img, txt = block(img=img, txt=txt, vec=vec, pe=pe)

        img = torch.cat((txt, img), 1)
        for block in self.single_blocks:
            img = block(img, vec=vec, pe=pe)
        img = img[:, txt.shape[1] :, ...]

        img = self.final_layer(img, vec)  # (N, T, patch_size ** 2 * out_channels)
        return img

class FluxAccelerator:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL",),
                "vae": ("VAE",),
                "do_compile": ("BOOLEAN", {"default": True}),
                "mmdit_skip_blocks": ("STRING", {"default": "3,12"}),
                "dit_skip_blocks": ("STRING", {"default": ""}),
            }
        }

    RETURN_TYPES = ("MODEL", "VAE")
    FUNCTION = "acclerate"
    CATEGORY = "advanced/model"

    def __init__(self):
        self._compiled = False
        self._quantized = False

    def acclerate(
        self,
        model: ModelPatcher,
        vae: VAE,
        do_compile: bool,
        mmdit_skip_blocks: str,
        dit_skip_blocks: str,
    ) -> tuple[ModelPatcher, VAE]:
        diffusion_model = model.model.diffusion_model
        ae = vae.first_stage_model

        if not self._quantized:
            if ae.parameters().__next__().dtype in (
                torch.float8_e4m3fn,
                torch.float8_e5m2,
                torch.float8_e4m3fnuz,
                torch.float8_e5m2fnuz,
                torch.int8,
            ):
                pass
            elif is_newer_than_ada_lovelace(torch.device(0)):
                quantize_(ae, float8_weight_only())
            else:
                quantize_(ae, int8_weight_only())

            self._quantized = True

        if do_compile and not self._compiled:
            compile_mode = (
                "reduce-overhead"
                if has_affordable_memory(torch.device(0))
                else "default"
            )

            diffusion_model = diffusion_model.to(memory_format=torch.channels_last)
            diffusion_model = torch.compile(
                diffusion_model,
                mode=compile_mode,
                fullgraph=True,
            )

            ae = ae.to(memory_format=torch.channels_last)
            ae = torch.compile(
                ae,
                mode=compile_mode,
                fullgraph=True,
            )

            self.compiled = True

        model.model.diffusion_model = diffusion_model
        vae.first_stage_model = ae

        model.model.diffusion_model.mmdit_skip_blocks_ = [
            int(x) for x in mmdit_skip_blocks.split(",") if x
        ]
        model.model.diffusion_model.dit_skip_blocks_ = [
            int(x) for x in dit_skip_blocks.split(",") if x
        ]

        diffusion_model.forward_orig = types.MethodType(forward_orig, diffusion_model)

        return (model, vae)


