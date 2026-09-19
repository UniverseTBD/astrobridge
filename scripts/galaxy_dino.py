import types
import torch
import torch.nn.functional as F
import numpy as np

import dinov2.models.vision_transformer as vits

PATCH_SIZE = 14
IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

IMG_SIZE = 518
DINO_INPUT_SIZE = 154 

def load_model(ckpt_path, device, num_register_tokens):
    model = vits.vit_small(
        patch_size=PATCH_SIZE,
        num_register_tokens=num_register_tokens,
        init_values=1.0,
        block_chunks=0,
        img_size=IMG_SIZE, 
    )
    ckpt = torch.load(ckpt_path, map_location="mps")
    model.load_state_dict(ckpt)
    return model.to(device).eval()

def _attention_forward_with_maps(self, x, is_causal=False):
    B, N, C = x.shape
    qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads)
    q, k, v = torch.unbind(qkv, 2)
    q, k, v = [t.transpose(1, 2) for t in [q, k, v]]
    scale = (C // self.num_heads) ** -0.5
    attn_weights = (q @ k.transpose(-2, -1) * scale).softmax(dim=-1)
    self._attn_weights = attn_weights
    out = (attn_weights @ v).transpose(1, 2).contiguous().view(B, N, C)
    return self.proj_drop(self.proj(out))

def patch_last_attention(model):
    last_attn = model.blocks[-1].attn
    last_attn.forward = types.MethodType(_attention_forward_with_maps, last_attn)
    return last_attn

def get_attention_map(rgb_array, out_size=None):
    """
    rgb_array: (H, W, 3) float array in [0, 1] — your native cutout resolution (e.g. 160x160).
    out_size: (H, W) to resize the attention map to, default = rgb_array's own size.
    Returns: (H, W) numpy array, mean attention across heads, normalized to [0, 1].
    """
    h, w = rgb_array.shape[:2]
    if out_size is None:
        out_size = (h, w)

    x = torch.from_numpy(rgb_array).permute(2, 0, 1).float().unsqueeze(0)
    x = F.interpolate(x, size=(DINO_INPUT_SIZE, DINO_INPUT_SIZE), mode="bilinear", align_corners=False)
    x = (x - IMAGENET_MEAN) / IMAGENET_STD
    x = x.to(device)

    with torch.no_grad():
        dino_model.forward_features(x)

    attn = dino_last_attn._attn_weights
    num_prefix_tokens = NUM_REGISTERS + 1
    cls_attn = attn[0, :, 0, num_prefix_tokens:]

    grid_size = DINO_INPUT_SIZE // PATCH_SIZE  # 154 // 14 = 11
    cls_attn_grid = cls_attn.reshape(-1, 1, grid_size, grid_size)

    attn_map = F.interpolate(cls_attn_grid, size=out_size, mode="bilinear", align_corners=False)
    attn_map = attn_map.mean(dim=0).squeeze(0).cpu().numpy()

    return (attn_map - attn_map.min()) / (attn_map.max() - attn_map.min() + 1e-8)

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
NUM_REGISTERS = 4

dino_model = load_model("models/dinov2_vits14_reg4_pretrain.pth", device, NUM_REGISTERS)
dino_last_attn = patch_last_attention(dino_model)