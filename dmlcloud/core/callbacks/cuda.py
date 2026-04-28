import json

import pynvml
import torch.cuda

import dmlcloud.core.logging as dml_logging
from dmlcloud.core.distributed import all_gather_object, is_root
from .common import Callback


def _call_pynvml(method, *args, **kwargs):
    try:
        return method(*args, **kwargs)
    except pynvml.NVMLError:
        return None


def _get_pynvml_handler(device):
    try:
        return torch.cuda._get_pynvml_handler(device)
    except pynvml.NVMLError:
        return None


def _gather_cuda_info(handler):
    info = {
        'name': _call_pynvml(pynvml.nvmlDeviceGetName, handler),
        'uuid': _call_pynvml(pynvml.nvmlDeviceGetUUID, handler),
        'serial': _call_pynvml(pynvml.nvmlDeviceGetSerial, handler),
        'minor_number': _call_pynvml(pynvml.nvmlDeviceGetMinorNumber, handler),
        'architecture': _call_pynvml(pynvml.nvmlDeviceGetArchitecture, handler),
        'brand': _call_pynvml(pynvml.nvmlDeviceGetBrand, handler),
        'vbios_version': _call_pynvml(pynvml.nvmlDeviceGetVbiosVersion, handler),
        'driver_version': _call_pynvml(pynvml.nvmlSystemGetDriverVersion),
        'cuda_driver_version': _call_pynvml(pynvml.nvmlSystemGetCudaDriverVersion_v2),
        'nvml_version': _call_pynvml(pynvml.nvmlSystemGetNVMLVersion),
        'total_memory': _call_pynvml(pynvml.nvmlDeviceGetMemoryInfo, handler, pynvml.nvmlMemory_v2).total,
        'reserved_memory': _call_pynvml(pynvml.nvmlDeviceGetMemoryInfo, handler, pynvml.nvmlMemory_v2).reserved,
        'num_gpu_cores': _call_pynvml(pynvml.nvmlDeviceGetNumGpuCores, handler),
        'power_managment_limit': _call_pynvml(pynvml.nvmlDeviceGetPowerManagementLimit, handler),
        'power_managment_default_limit': _call_pynvml(pynvml.nvmlDeviceGetPowerManagementDefaultLimit, handler),
        'cuda_compute_capability': _call_pynvml(pynvml.nvmlDeviceGetCudaComputeCapability, handler),
    }
    return info


class CudaCallback(Callback):
    """
    Logs various properties pertaining to CUDA devices.
    """

    def pre_run(self, pipe):
        handler = _get_pynvml_handler(pipe.device)
        info = _gather_cuda_info(handler) if handler is not None else {}
        info['torch_device'] = str(pipe.device)

        all_infos = all_gather_object(info)

        msg = '* CUDA-DEVICES:\n'
        info_strings = []
        for info in all_infos:
            if 'minor_number' in info and 'name' in info and 'uuid' in info:
                info_strings.append(
                    f'{info["torch_device"]} -> /dev/nvidia{info["minor_number"]} -> {info["name"]} (UUID: {info["uuid"]}) (VRAM: {info["total_memory"] / 1000 ** 2:.0f} MB)'
                )
        msg += '\n'.join(f'    - [{i}] {info_str}' for i, info_str in enumerate(info_strings))
        dml_logging.info(msg)

        if pipe.run_dir and is_root():
            self._save(pipe.run_dir / 'diagnostics' / 'cuda_devices.json', all_infos)

    def post_step(self, stage):
        # ``torch.cuda.memory_stats`` queries the CUDA driver and returns an
        # OrderedDict of ~50 entries; called every step it adds measurable
        # per-step overhead. The recorded metric is ``peak`` memory which
        # only changes when allocator high-watermark moves, so polling at
        # 25-step granularity captures the same peak with 1/25th the cost.
        # ``cuda_log_every_n_steps`` (default 25) on stage.config overrides.
        period = int(getattr(stage.config, 'get', lambda *_: None)('cuda_log_every_n_steps', 25) or 1) \
            if hasattr(stage, 'config') else 25
        if not is_root():
            return
        if (stage.global_step % period) != 0:
            return
        stats = torch.cuda.memory_stats(stage.device)
        stage.log(
            'misc/cuda/allocated_bytes_peak',
            stats['allocated_bytes.all.peak'],
            prefixed=False,
            synchronize=False,
            reduction='max',
        )
        stage.log(
            'misc/cuda/reserved_bytes_peak',
            stats['reserved_bytes.all.peak'],
            prefixed=False,
            synchronize=False,
            reduction='max',
        )
        stage.log(
            'misc/cuda/active_bytes_peak',
            stats['active_bytes.all.peak'],
            prefixed=False,
            synchronize=False,
            reduction='max',
        )
        stage.log(
            'misc/cuda/requested_bytes_peak',
            stats['requested_bytes.all.peak'],
            prefixed=False,
            synchronize=False,
            reduction='max',
        )
        stage.log(
            'misc/cuda/num_alloc_retries',
            stats['num_alloc_retries'],
            prefixed=False,
            synchronize=False,
            reduction='max',
        )
        stage.log(
            'misc/cuda/num_device_alloc',
            stats['num_device_alloc'],
            prefixed=False,
            synchronize=False,
            reduction='max',
        )
        stage.log(
            'misc/cuda/num_device_free',
            stats['num_device_free'],
            prefixed=False,
            synchronize=False,
            reduction='max',
        )
        torch.cuda.reset_peak_memory_stats(stage.device)

    def _save(self, path, all_infos):
        with open(path, 'w') as f:
            dct = {f'rank_{i}': info for i, info in enumerate(all_infos)}
            obj = {'devices': dct}
            json.dump(obj, f, indent=4)
