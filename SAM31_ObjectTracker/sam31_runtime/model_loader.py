"""Build the SAM 3.1 Object-Multiplex tracker once, from local assets only.

Builder: `sam3.model_builder.build_sam3_multiplex_video_model` (pinned upstream
commit in SAM31_VERSION.json). The checkpoint `sam3.1_multiplex.pt` stores the
tracker under `tracker.model.*` and the shared ViT trunk under
`detector.backbone.vision_backbone.*`; the tracker's own `backbone` module is
the same `Sam3TriViTDetNeck` architecture, so the trunk weights are remapped
into it. Verified: 0 missing / 0 unexpected keys.

No network access: `load_from_HF=False` and the checkpoint path is mandatory.
"""

from __future__ import annotations

import os
import time

import torch

from .logging_utils import get_logger

# Diagnostics: how many times a model has been built in this process.
LOAD_COUNT = 0
_MODEL_CACHE = {}

MULTIPLEX_COUNT_DEFAULT = 16


class ModelLoadError(RuntimeError):
    pass


def _remap_checkpoint(ckpt: dict) -> dict:
    remapped = {}
    for k, v in ckpt.items():
        if k.startswith("tracker.model."):
            remapped[k[len("tracker.model."):]] = v
        elif k.startswith("detector.backbone.vision_backbone."):
            remapped["backbone.vision_backbone." + k[len("detector.backbone.vision_backbone."):]] = v
        # everything else (detector transformer/decoder, text encoder) is not
        # needed by the box-initialised tracker and is dropped here
    return remapped


def build_tracker(checkpoint_path, device="cuda", multiplex_count=MULTIPLEX_COUNT_DEFAULT,
                  compile_model=False, strict=True, logger=None):
    """Instantiate the multiplex tracker and load the checkpoint. Always builds."""
    global LOAD_COUNT
    logger = logger or get_logger()
    if not os.path.isfile(checkpoint_path):
        raise ModelLoadError(f"SAM 3.1 checkpoint not found: {checkpoint_path}")
    if not str(device).startswith("cuda"):
        raise ModelLoadError(
            "SAM 3.1 Object Multiplex tracking requires a CUDA GPU "
            "(upstream inference state is CUDA-only); got device=%r" % (device,)
        )
    if not torch.cuda.is_available():
        raise ModelLoadError("torch.cuda.is_available() is False; a CUDA-capable GPU and driver are required")

    from sam3.model_builder import build_sam3_multiplex_video_model

    t0 = time.time()
    model = build_sam3_multiplex_video_model(
        checkpoint_path=None,
        load_from_HF=False,
        multiplex_count=int(multiplex_count),
        use_fa3=False,          # FlashAttention-3 is unavailable on Windows; SDPA path
        use_rope_real=True,
        strict_state_dict_loading=False,
        device="cpu",
        compile=bool(compile_model),
    )
    t_build = time.time() - t0

    t0 = time.time()
    try:
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=True, mmap=True)
    except TypeError:  # older torch without mmap kwarg
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if isinstance(ckpt, dict) and "model" in ckpt and isinstance(ckpt["model"], dict):
        ckpt = ckpt["model"]
    remapped = _remap_checkpoint(ckpt)
    missing, unexpected = model.load_state_dict(remapped, strict=False)
    if strict and (missing or unexpected):
        raise ModelLoadError(
            f"checkpoint/model mismatch: {len(missing)} missing, {len(unexpected)} unexpected keys. "
            f"First missing: {missing[:5]}; first unexpected: {unexpected[:5]}"
        )
    del ckpt, remapped
    t_load = time.time() - t0

    t0 = time.time()
    model = model.to(device=device).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    t_dev = time.time() - t0

    LOAD_COUNT += 1
    logger.info(
        "SAM 3.1 multiplex tracker built (load #%d): build %.1fs, checkpoint %.1fs, to(%s) %.1fs; "
        "multiplex_count=%d, image_size=%d, num_maskmem=%d, max_obj_ptrs=%d, compile=%s, fa3=False",
        LOAD_COUNT, t_build, t_load, device, t_dev, model.multiplex_controller.multiplex_count,
        model.image_size, model.num_maskmem, model.max_obj_ptrs_in_encoder, bool(compile_model),
    )
    return model


def get_tracker(checkpoint_path, device="cuda", multiplex_count=MULTIPLEX_COUNT_DEFAULT,
                compile_model=False, logger=None):
    """Process-wide cached tracker: repeated ArcGIS sessions reuse the model."""
    key = (os.path.abspath(checkpoint_path), str(device), int(multiplex_count), bool(compile_model))
    model = _MODEL_CACHE.get(key)
    if model is None:
        model = build_tracker(checkpoint_path, device=device, multiplex_count=multiplex_count,
                              compile_model=compile_model, logger=logger)
        _MODEL_CACHE[key] = model
    else:
        (logger or get_logger()).info("SAM 3.1 tracker reused from process cache (no reload)")
    return model


def clear_cache():
    _MODEL_CACHE.clear()
    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
