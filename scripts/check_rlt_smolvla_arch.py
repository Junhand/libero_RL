import sys, time, importlib.util, torch
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy, make_att_2d_masks

P = "/workspace/libero_RL/models/smolvla_libero"
dev = "cuda"
pol = SmolVLAPolicy.from_pretrained(P).to(dev).eval()
cfg = pol.config; m = pol.model
print("chunk_size", cfg.chunk_size, "n_action_steps", cfg.n_action_steps, "num_steps", cfg.num_steps,
      "num_vlm_layers", cfg.num_vlm_layers, "attention_mode", cfg.attention_mode, "prefix_length", cfg.prefix_length)
print("vlm hidden", m.vlm_with_expert.config.text_config.hidden_size, "expert hidden", m.vlm_with_expert.expert_hidden_size)
print("image features:", {k: tuple(v.shape) for k, v in cfg.image_features.items()}, "| state", tuple(cfg.robot_state_feature.shape), "| action", tuple(cfg.action_feature.shape))
print("params (M): total %.0f  vlm %.0f  expert %.0f" % (
    sum(p.numel() for p in pol.parameters())/1e6,
    sum(p.numel() for p in m.vlm_with_expert.vlm.parameters())/1e6,
    sum(p.numel() for p in m.vlm_with_expert.lm_expert.parameters())/1e6))

B = 2
H, W = cfg.resize_imgs_with_padding
imgs = [torch.rand(B, 3, H, W, device=dev)*2-1 for _ in cfg.image_features]
img_masks = [torch.ones(B, dtype=torch.bool, device=dev) for _ in imgs]
tok = m.vlm_with_expert.processor.tokenizer
t = tok(["put both the alphabet soup and the tomato sauce in the basket\n"]*B, padding="max_length", max_length=cfg.tokenizer_max_length, return_tensors="pt")
lang_tokens, lang_masks = t["input_ids"].to(dev), t["attention_mask"].bool().to(dev)
state = torch.randn(B, cfg.max_state_dim, device=dev)

with torch.no_grad():
    embs, pad, att = m.embed_prefix(imgs, img_masks, lang_tokens, lang_masks, state=state)
    print("prefix embs", tuple(embs.shape))
    att2d = make_att_2d_masks(pad, att); pos = torch.cumsum(pad, dim=1) - 1
    out, kv = m.vlm_with_expert.forward(attention_mask=att2d, position_ids=pos, past_key_values=None,
                                        inputs_embeds=[embs, None], use_cache=True)
    print("vlm forward returns:", type(out).__name__, [None if o is None else tuple(o.shape) for o in out])
    z = out[0]
    print("prefix last-hidden: shape", tuple(z.shape), "dtype", z.dtype, "| finite:", bool(torch.isfinite(z).all()),
          "| std %.3f" % z.float().std().item())
    n_img_tokens = embs.shape[1] - cfg.tokenizer_max_length - 1
    print("tokens: image-part ~", n_img_tokens, " lang", cfg.tokenizer_max_length, " state 1")

    # reference chunk (stochastic flow-matching sample)
    a1 = m.sample_actions(imgs, img_masks, lang_tokens, lang_masks, state)
    a2 = m.sample_actions(imgs, img_masks, lang_tokens, lang_masks, state)
    print("reference chunk", tuple(a1.shape), "| differs between samples:", bool((a1-a2).abs().max() > 1e-6))

# RLinf's generic RLT token transformer on top of the (image-only) prefix hidden states
spec = importlib.util.spec_from_file_location("rlt_tt", "/workspace/libero_RL/RLinf/rlinf/models/embodiment/modules/rlt_token_transformer.py")
mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
hid = z.shape[-1]; M = n_img_tokens
rlt = mod.RLTTokenTransformer(input_dim=hid, embed_dim=hid, prefix_seq_len=M, num_layers=2, num_heads=8).to(dev)
opt = torch.optim.AdamW(rlt.parameters(), lr=1e-4)
torch.cuda.reset_peak_memory_stats()
t0 = time.time()
for i in range(20):
    loss, info = rlt(z[:, :M].float())
    opt.zero_grad(); loss.backward(); opt.step()
    if i in (0, 19): print("RLT recon step", i, "mse %.4f" % loss.item(), "| z_rl", tuple(info["z_rl"].shape))
print("RLT params (M): %.1f | 20 steps %.1fs | peak GPU mem %.2f GB" % (sum(p.numel() for p in rlt.parameters())/1e6, time.time()-t0, torch.cuda.max_memory_allocated()/1e9))
