"""
Loads the official pretrained GPT-2 weights (downloaded from the Hugging Face
Hub) into our from-scratch GPTModel implementation.

HF's GPT2 uses Conv1D layers (weight shape = [in_features, out_features]),
while our model uses nn.Linear (weight shape = [out_features, in_features]),
so weights that come from Conv1D layers need to be transposed on the way in.
"""
import torch

from .model import GPTModel


def _assign(target: torch.nn.Parameter, source: torch.Tensor) -> torch.nn.Parameter:
    source = source.clone().detach()
    if target.shape != source.shape:
        raise ValueError(f"Shape mismatch: target {target.shape}, source {source.shape}")
    return torch.nn.Parameter(source)


def load_pretrained_gpt2(model_size="gpt2", cfg=None):
    """
    model_size: one of "gpt2" (124M), "gpt2-medium" (355M),
                "gpt2-large" (774M), "gpt2-xl" (1558M)
    """
    from transformers import GPT2LMHeadModel

    size_to_dims = {
        "gpt2": {"emb_dim": 768, "n_layers": 12, "n_heads": 12},
        "gpt2-medium": {"emb_dim": 1024, "n_layers": 24, "n_heads": 16},
        "gpt2-large": {"emb_dim": 1280, "n_layers": 36, "n_heads": 20},
        "gpt2-xl": {"emb_dim": 1600, "n_layers": 48, "n_heads": 25},
    }
    if cfg is None:
        from .model import GPT_CONFIG_124M
        cfg = GPT_CONFIG_124M.copy()
    cfg = cfg.copy()
    cfg.update(size_to_dims[model_size])

    hf_model = GPT2LMHeadModel.from_pretrained(model_size)
    sd = hf_model.state_dict()

    model = GPTModel(cfg)

    model.pos_emb.weight = _assign(model.pos_emb.weight, sd["transformer.wpe.weight"])
    model.tok_emb.weight = _assign(model.tok_emb.weight, sd["transformer.wte.weight"])

    for b in range(cfg["n_layers"]):
        prefix = f"transformer.h.{b}."

        q_w, k_w, v_w = sd[prefix + "attn.c_attn.weight"].chunk(3, dim=-1)
        q_b, k_b, v_b = sd[prefix + "attn.c_attn.bias"].chunk(3, dim=-1)

        block = model.trf_blocks[b]
        block.att.W_query.weight = _assign(block.att.W_query.weight, q_w.T)
        block.att.W_key.weight = _assign(block.att.W_key.weight, k_w.T)
        block.att.W_value.weight = _assign(block.att.W_value.weight, v_w.T)
        block.att.W_query.bias = _assign(block.att.W_query.bias, q_b)
        block.att.W_key.bias = _assign(block.att.W_key.bias, k_b)
        block.att.W_value.bias = _assign(block.att.W_value.bias, v_b)

        block.att.out_proj.weight = _assign(
            block.att.out_proj.weight, sd[prefix + "attn.c_proj.weight"].T
        )
        block.att.out_proj.bias = _assign(
            block.att.out_proj.bias, sd[prefix + "attn.c_proj.bias"]
        )

        block.ff.layers[0].weight = _assign(
            block.ff.layers[0].weight, sd[prefix + "mlp.c_fc.weight"].T
        )
        block.ff.layers[0].bias = _assign(
            block.ff.layers[0].bias, sd[prefix + "mlp.c_fc.bias"]
        )
        block.ff.layers[2].weight = _assign(
            block.ff.layers[2].weight, sd[prefix + "mlp.c_proj.weight"].T
        )
        block.ff.layers[2].bias = _assign(
            block.ff.layers[2].bias, sd[prefix + "mlp.c_proj.bias"]
        )

        block.norm1.scale = _assign(block.norm1.scale, sd[prefix + "ln_1.weight"])
        block.norm1.shift = _assign(block.norm1.shift, sd[prefix + "ln_1.bias"])
        block.norm2.scale = _assign(block.norm2.scale, sd[prefix + "ln_2.weight"])
        block.norm2.shift = _assign(block.norm2.shift, sd[prefix + "ln_2.bias"])

    model.final_norm.scale = _assign(model.final_norm.scale, sd["transformer.ln_f.weight"])
    model.final_norm.shift = _assign(model.final_norm.shift, sd["transformer.ln_f.bias"])
    model.out_head.weight = _assign(model.out_head.weight, sd["transformer.wte.weight"])

    model.eval()
    return model, cfg
