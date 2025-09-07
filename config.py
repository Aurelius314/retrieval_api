# -*- coding: utf-8 -*-
import os
import psycopg2
import torch
from typing import Dict
from utils import load_surf_checkpoint_model_from_base
from cn_clip.clip import load_from_name
import shutil

# 直接在此处配置路径与参数（基于项目结构自动推导绝对路径）
_ROOT_DIR = os.path.dirname(__file__)
_MODELS_DIR = os.path.join(_ROOT_DIR, 'models')
_HOME_MODELS_DIR = os.path.expanduser(os.path.join('~', 'surf', 'api', 'models', 'cn_clip'))

_CONFIG: Dict[str, Dict] = {
    'paths': {
        # base 底座下载目录（使用 HOME 目录下路径）
        'model_download_root': _HOME_MODELS_DIR,
        # checkpoint 路径（你的微调权重）
        'checkpoint_path': os.path.join(_MODELS_DIR, 'epoch_latest.pt'),
        # base 底座权重文件名（用于校验下载结果）
        'base_model_path': os.path.join(_HOME_MODELS_DIR, 'ViT-H-14.pt'),
    },
    'database': {
        'host': 'localhost',
        'port': '5432',
        'database': 'retrieval_db',
    },
    'retrieval': {
        'prefetch_limit': 200,
        'cache_ttl_seconds': 300,
        'default_limit': 20,
        'max_limit': 100,
    },
    'ivfflat': {
        'text_to_image': {
            'lists': 170,
            'probes': 18,
        },
        'image_to_text': {
            'lists': 180,
            'probes': 18,
        },
    },
}

# 暴露给其它模块使用的只读配置
RETRIEVAL_CONFIG = _CONFIG['retrieval']

def get_ivfflat_config(task_type: str) -> dict:
    if task_type not in _CONFIG['ivfflat']:
        raise ValueError(f"Unknown task type: {task_type}")
    return _CONFIG['ivfflat'][task_type]

def get_probes_for_task(task_type: str) -> int:
    return get_ivfflat_config(task_type)['probes']

def get_model_config() -> dict:
    # 与现有代码兼容的返回结构
    return {
        'checkpoint_path': _CONFIG['paths']['checkpoint_path'],
        'download_root': _CONFIG['paths']['model_download_root'],
    }

def get_db_config() -> dict:
    return _CONFIG['database']

def _assert_file_exists(path: str, hint: str) -> None:
    if not os.path.isfile(path):
        raise FileNotFoundError(f"{hint} not found: {path}")

def validate_config() -> None:
    paths = _CONFIG['paths']
    db = _CONFIG['database']

    base_ok = False
    ckpt_ok = False
    db_ok = False
    messages: list[str] = []

    # 1) base check: exists + load
    base_model_path = paths['base_model_path']
    download_root = paths['model_download_root']
    try:
        # 先尝试就地适配已存在的不同命名文件，避免重复下载
        expected_pkg_name = 'clip_cn_vit-h-14.pt'  # site-packages 版本使用的文件名
        expected_pkg_path = os.path.join(download_root, expected_pkg_name)
        candidate_names = ['ViT-H-14.pt', expected_pkg_name]
        found_path = None
        for name in candidate_names:
            p = os.path.join(download_root, name)
            if os.path.isfile(p):
                found_path = p
                break

        if found_path and not os.path.isfile(expected_pkg_path):
            try:
                os.makedirs(download_root, exist_ok=True)
                shutil.copyfile(found_path, expected_pkg_path)
            except Exception:
                # 复制失败不致命，继续按正常流程加载（可能会触发下载）
                pass

        # 尝试加载（会自动下载到 download_root）
        load_from_name("ViT-H-14", device="cpu", download_root=download_root)

        # 校验最终落盘路径（我们偏好的命名）
        if os.path.isfile(base_model_path):
            base_ok = True
            messages.append(f"✓ Base model: {base_model_path}")
        else:
            # 如若未按偏好命名落盘，回退检查 site-packages 的命名
            if os.path.isfile(expected_pkg_path):
                base_ok = True
                messages.append(f"✓ Base model: {expected_pkg_path}")
            else:
                messages.append(f"[BUG] Base model present check failed at: {base_model_path}")
    except Exception as e:
        messages.append(f"[BUG] Base model load/download failed: {e}")

    # 2) checkpoints check: exists + load
    ckpt_path = paths['checkpoint_path']
    try:
        _assert_file_exists(ckpt_path, 'Checkpoint file')
        try:
            checkpoint = torch.load(ckpt_path, map_location='cpu')
            state_dict = checkpoint.get('state_dict', None)
            if not isinstance(state_dict, dict):
                raise RuntimeError('Invalid checkpoint: missing or invalid state_dict')
            # 轻量校验通过
            ckpt_ok = True
            messages.append(f"✓ Checkpoint: {ckpt_path}")
        except Exception as e:
            messages.append(f"[BUG] Checkpoint parse failed ({ckpt_path}): {e}")
    except Exception as e:
        messages.append(f"[BUG] Checkpoint missing: {e}")

    # 3) database check: connect + simple query
    try:
        conn = psycopg2.connect(
            dbname=db['database'], host=db['host'], port=db['port']
        )
        with conn.cursor() as cur:
            cur.execute('SELECT 1')
            cur.fetchone()
        conn.close()
        db_ok = True
        messages.append(f"✓ Database: {db['host']}:{db['port']}/{db['database']}")
    except Exception as e:
        messages.append(
            f"[BUG] Database failed: {db['host']}:{db['port']}/{db['database']} — {e}"
        )

    # Print summary
    for m in messages:
        print(m)

    # Raise if any failed
    if not (base_ok and ckpt_ok and db_ok):
        failed = []
        if not base_ok:
            failed.append('base')
        if not ckpt_ok:
            failed.append('checkpoint')
        if not db_ok:
            failed.append('database')
        raise RuntimeError(f"validate_config failed: {', '.join(failed)}")
