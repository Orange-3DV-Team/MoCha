import torch


def set_stand_in(pipe, train=False, model_path=None):
    for block in pipe.dit.blocks:
        block.self_attn.init_lora(train)
    if model_path is not None:
        print(f"Loading Stand-In weights from: {model_path}")
        load_lora_weights_into_pipe(pipe, model_path)


def load_lora_weights_into_pipe(pipe, ckpt_path, strict=True):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    state_dict = ckpt.get("state_dict", ckpt)

    model = {}
    for i, block in enumerate(pipe.dit.blocks):
        prefix = f"blocks.{i}.self_attn."
        attn = block.self_attn
        for name in ["q_loras", "k_loras", "v_loras"]:
            for sub in ["down", "up"]:
                key = f"{prefix}{name}.{sub}.weight"
                if hasattr(getattr(attn, name), sub):
                    model[key] = getattr(getattr(attn, name), sub).weight
                else:
                    if strict:
                        raise KeyError(f"Missing module: {key}")

    for k, param in state_dict.items():
        if k in model:
            if model[k].shape != param.shape:
                if strict:
                    raise ValueError(
                        f"Shape mismatch: {k} | {model[k].shape} vs {param.shape}"
                    )
                else:
                    continue
            model[k].data.copy_(param)
        else:
            if strict:
                raise KeyError(f"Unexpected key in ckpt: {k}")