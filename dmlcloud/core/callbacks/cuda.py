import json

import pynvml
import torch.cuda

import dmlcloud.core.logging as dml_logging
from dmlcloud.core.distributed import all_gather_object, is_root
from .common import Callback


class CudaCallback(Callback):
    """
    Logs various properties pertaining to CUDA devices.
    """

    @staticmethod
    def _call_pynvml(method, *args, **kwargs):
        try:
            return method(*args, **kwargs)
        except pynvml.NVMLError:
            return None

    def pre_run(self, pipe):
        handle = torch.cuda._get_pynvml_handler(pipe.device)

        info = {
            'name': self._call_pynvml(pynvml.nvmlDeviceGetName, handle),
            'uuid': self._call_pynvml(pynvml.nvmlDeviceGetUUID, handle),
            'serial': self._call_pynvml(pynvml.nvmlDeviceGetSerial, handle),
            'torch_device': str(pipe.device),
            'minor_number': self._call_pynvml(pynvml.nvmlDeviceGetMinorNumber, handle),
            'architecture': self._call_pynvml(pynvml.nvmlDeviceGetArchitecture, handle),
            'brand': self._call_pynvml(pynvml.nvmlDeviceGetBrand, handle),
            'vbios_version': self._call_pynvml(pynvml.nvmlDeviceGetVbiosVersion, handle),
            'driver_version': self._call_pynvml(pynvml.nvmlSystemGetDriverVersion),
            'cuda_driver_version': self._call_pynvml(pynvml.nvmlSystemGetCudaDriverVersion_v2),
            'nvml_version': self._call_pynvml(pynvml.nvmlSystemGetNVMLVersion),
            'total_memory': self._call_pynvml(pynvml.nvmlDeviceGetMemoryInfo, handle, pynvml.nvmlMemory_v2).total,
            'reserved_memory': self._call_pynvml(pynvml.nvmlDeviceGetMemoryInfo, handle, pynvml.nvmlMemory_v2).reserved,
            'num_gpu_cores': self._call_pynvml(pynvml.nvmlDeviceGetNumGpuCores, handle),
            'power_managment_limit': self._call_pynvml(pynvml.nvmlDeviceGetPowerManagementLimit, handle),
            'power_managment_default_limit': self._call_pynvml(pynvml.nvmlDeviceGetPowerManagementDefaultLimit, handle),
            'cuda_compute_capability': self._call_pynvml(pynvml.nvmlDeviceGetCudaComputeCapability, handle),
        }
        all_devices = all_gather_object(info)

        msg = '* CUDA-DEVICES:\n'
        info_strings = [
            f'{info["torch_device"]} -> /dev/nvidia{info["minor_number"]} -> {info["name"]} (UUID: {info["uuid"]}) (VRAM: {info["total_memory"] / 1000 ** 2:.0f} MB)'
            for info in all_devices
        ]
        msg += '\n'.join(f'    - [{i}] {info_str}' for i, info_str in enumerate(info_strings))
        dml_logging.info(msg)

        if pipe.run_dir and is_root():
            self._save(pipe.run_dir / 'cuda_devices.json', all_devices)

    def _save(self, path, all_devices):
        with open(path, 'w') as f:
            devices = {f'rank_{i}': device for i, device in enumerate(all_devices)}
            obj = {'devices': devices}
            json.dump(obj, f, indent=4)
