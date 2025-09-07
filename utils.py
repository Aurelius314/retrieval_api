import torch
from cn_clip.clip import load_from_name
import os
import shutil
from datetime import datetime
from fastapi import FastAPI, HTTPException, Body
import numpy as np

def load_surf_checkpoint_model_from_base(
    ckpt_path: str,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    download_root: str = "./"
):
    # 优先尝试使用本地已有的多种常见命名，避免重复下载
    candidate_names = [
        os.path.join(download_root, 'ViT-H-14.pt'),
        os.path.join(download_root, 'clip_cn_vit-h-14.pt'),
    ]
    local_base = next((p for p in candidate_names if os.path.isfile(p)), None)

    if local_base and not os.path.isfile(expected_pkg_path):
        try:
            os.makedirs(download_root, exist_ok=True)
            shutil.copyfile(local_base, expected_pkg_path)
        except Exception:
            pass

    # 通过名称加载（若 expected_pkg_path 存在将不会重新下载）
    model, preprocess = load_from_name("ViT-H-14", device=device, download_root=download_root)

    checkpoint = torch.load(ckpt_path, map_location='cpu')
    state_dict = checkpoint["state_dict"]

    # 去掉module.前缀
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith("module."):
            new_state_dict[k[len("module."):]] = v
        else:
            new_state_dict[k] = v

    missing_keys, unexpected_keys = model.load_state_dict(new_state_dict, strict=False)
    print("[DEBUG] Missing keys:", missing_keys)
    print("[DEBUG] Unexpected keys:", unexpected_keys)

    model.to(device).eval()
    return model, preprocess

def verify_api_key(api_key: str):
    API_KEYS = {"demo": "surf_demo_api_key"}
    if api_key not in API_KEYS.values():
        raise HTTPException(status_code=403, detail="Unauthorized")

def fix_base64_padding(b64_string: str) -> str:
    return b64_string + '=' * ((4 - len(b64_string) % 4) % 4)


