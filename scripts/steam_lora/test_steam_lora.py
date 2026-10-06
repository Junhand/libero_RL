import sys, torch
sys.path.insert(0, "/workspace/libero_RL/tmp/rlinf_steam_dev")
from rlinf.models.embodiment.value_model.steam.configuration import SteamConfig
from rlinf.models.embodiment.value_model.steam.modeling_steam import SteamBackbone
M = "/workspace/libero_RL/models"
def build(**kw):
    base = dict(freeze_language_model=False, freeze_vision_encoder=False); base.update(kw)
    cfg = SteamConfig(vision_repo_id=f"{M}/siglip-so400m-patch14-384", language_repo_id=f"{M}/gemma-3-270m", num_bins=32, dtype="bfloat16", **base)
    return SteamBackbone(cfg)
for name, kw in (("freeze vision", dict(freeze_vision_encoder=True)), ("freeze both", dict(freeze_vision_encoder=True, freeze_language_model=True))):
    m = build(**kw); tot = sum(p.numel() for p in m.parameters()); tr = sum(p.numel() for p in m.parameters() if p.requires_grad)
    nl = sum(1 for n, _ in m.named_parameters() if "lora_A" in n)
    print(f"{name:16s} total {tot/1e6:7.1f}M trainable {tr/1e6:7.2f}M ({100*tr/tot:.2f}%)  lora_A modules: {nl}")
    if name.startswith("lora"):
        print("   example names:", [n for n, _ in m.named_parameters() if "lora_" in n][:2], [n for n, _ in m.named_parameters() if "base_layer" in n][:1])
        print("   dtypes of lora params:", {p.dtype for n, p in m.named_parameters() if "lora_" in n})
    del m
