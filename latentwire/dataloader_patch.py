# latentwire/dataloader_patch.py
import os
import torch.utils.data

def patch_dataloader_defaults(num_workers=None, prefetch_factor=None, pin_memory=None):
    try:
        _Orig = torch.utils.data.DataLoader
    except Exception:
        return

    if num_workers is None:
        try:
            ngpu = len(os.environ.get("CUDA_VISIBLE_DEVICES","").split(",")) if os.environ.get("CUDA_VISIBLE_DEVICES") else 1
            cpu = os.cpu_count() or 8
            num_workers = min(cpu, max(4, 4*ngpu))
        except Exception:
            num_workers = 4

    if prefetch_factor is None:
        prefetch_factor = 4

    if pin_memory is None:
        pin_memory = True

    class _Patched(_Orig):
        def __init__(self, *args, **kwargs):
            kwargs.setdefault("pin_memory", pin_memory)
            kwargs.setdefault("num_workers", num_workers)
            if kwargs.get("num_workers", 0) > 0:
                kwargs.setdefault("prefetch_factor", prefetch_factor)
            super().__init__(*args, **kwargs)

    torch.utils.data.DataLoader = _Patched