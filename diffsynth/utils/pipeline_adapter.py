# diffsynth/utils/pipeline_adapter.py
"""
Auto-adapter to find encode/decode helpers in Wan/DiffSynth pipelines.

This advanced version intelligently inspects function signatures to provide
common arguments like `device`, handles dtype/device mismatches automatically
by retrying with casted tensors, and reduces warnings for a cleaner log.
"""
# [FIX] Import Callable, Optional, and Any from the typing module.
from typing import Optional, Callable, Any
import torch
import torchvision.utils as vutils
import os
import warnings
import inspect
import torch.nn as nn

# --- Metrics & Dependencies ---
try:
    import lpips
    LPIPS_AVAILABLE = True
    _LPIPS_NET = None # Lazy initialization
except ImportError:
    LPIPS_AVAILABLE = False
    _LPIPS_NET = None

# --- Candidate API Names ---
_ENCODE_CANDIDATES = [
    "encode_frames", "encode_videos", "encode", "vae.encode", "vae.encode_frames"
]
_DECODE_CANDIDATES = [
    "decode_latents", "decode", "decode_frames", "vae.decode", "vae.decode_latents"
]

def _resolve_attr(obj: object, name: str) -> Optional[Callable]:
    """Resolves a potentially dotted attribute name (e.g., 'vae.encode') on an object."""
    try:
        for part in name.split('.'):
            obj = getattr(obj, part)
        return obj if callable(obj) else None
    except AttributeError:
        return None

def _get_module_dtype_device(module: nn.Module) -> tuple[torch.dtype, torch.device]:
    """Returns the dtype and device of the first parameter found in a module."""
    try:
        first_param = next(module.parameters())
        return first_param.dtype, first_param.device
    except StopIteration: # No parameters in the module
        return torch.float32, torch.device("cpu")

def _call_with_adaptive_args(obj: object, name: str, *args: Any, **kwargs: Any) -> Any:
    """
    Intelligently calls a method by name, adapting arguments to its signature.
    - Automatically supplies `device` if the function signature requires it.
    - Retries with type-casted tensors upon a `dtype` or `device` mismatch error.
    """
    fn = _resolve_attr(obj, name)
    if fn is None:
        raise AttributeError(f"Attribute '{name}' not found or not callable on object.")

    # Determine the target dtype and device from the relevant module
    target_module = getattr(fn, '__self__', obj) # The object the method belongs to
    dtype, device = _get_module_dtype_device(target_module)
    
    # Prepare potential keyword arguments based on function signature
    try:
        sig = inspect.signature(fn)
        param_names = sig.parameters.keys()
        
        if 'device' in param_names and 'device' not in kwargs:
            kwargs['device'] = device
        
    except (ValueError, TypeError):
        pass

    try:
        return fn(*args, **kwargs)
    except Exception as e:
        error_msg = str(e).lower()
        is_mismatch_error = any(s in error_msg for s in ["dtype", "type", "half", "float", "cpu", "cuda"])
        
        if is_mismatch_error:
            warnings.warn(f"Caught a mismatch error for '{name}': '{e}'. Retrying with casted tensors.")
            
            casted_args = tuple(
                arg.to(device=device, dtype=dtype, non_blocking=True) if isinstance(arg, torch.Tensor) else arg
                for arg in args
            )
            casted_kwargs = {
                k: v.to(device=device, dtype=dtype, non_blocking=True) if isinstance(v, torch.Tensor) else v
                for k, v in kwargs.items()
            }
            
            try:
                return fn(*casted_args, **casted_kwargs)
            except Exception as e2:
                raise e2 from e
        
        raise e

def _try_call(obj: object, name: str, *args: Any, **kwargs: Any) -> Optional[Any]:
    """A wrapper for _call_with_adaptive_args that returns None on failure."""
    try:
        return _call_with_adaptive_args(obj, name, *args, **kwargs)
    except Exception as e:
        warnings.warn(f"Adapter trial for '{name}' failed: {e}")
        return None

def encode_frames_auto(pipe: object, frames: torch.Tensor) -> Optional[torch.Tensor]:
    """
    Automatically finds and calls the correct video encoding function on a pipeline object.
    Normalizes input/output tensor shapes to the expected [B, T, C, H, W] format.
    """
    if frames.dim() == 5 and frames.shape[2] == 3: # Likely [B, T, C, H, W]
        frames_for_vae = frames.permute(0, 2, 1, 3, 4).contiguous()
    else:
        frames_for_vae = frames

    for name in _ENCODE_CANDIDATES:
        latents = _try_call(pipe, name, frames_for_vae)
        if isinstance(latents, torch.Tensor):
            if latents.dim() == 5 and latents.shape[1] < latents.shape[2]: # Likely [B, C, T, H', W']
                return latents.permute(0, 2, 1, 3, 4).contiguous()
            return latents
            
    return None

def decode_latents_auto(pipe: object, latents: torch.Tensor) -> Optional[torch.Tensor]:
    """
    Automatically finds and calls the correct latent decoding function.
    Normalizes input/output tensor shapes to the expected [B, T, C, H, W] format.
    """
    if latents.dim() == 5 and latents.shape[2] < latents.shape[1]: # Likely [B, T, C, H', W']
        latents_for_vae = latents.permute(0, 2, 1, 3, 4).contiguous()
    else:
        latents_for_vae = latents

    for name in _DECODE_CANDIDATES:
        frames = _try_call(pipe, name, latents_for_vae)
        if isinstance(frames, torch.Tensor):
            if frames.dim() == 5 and frames.shape[1] == 3: # Likely [B, C, T, H, W]
                return frames.permute(0, 2, 1, 3, 4).contiguous()
            return frames
            
    return None