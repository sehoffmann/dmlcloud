![dmlcloud logo](./misc/logo/dmlcloud_color.png)
---------------
[![](https://img.shields.io/pypi/v/dmlcloud)](https://pypi.org/project/dmlcloud/)
[![](https://img.shields.io/github/actions/workflow/status/sehoffmann/dmlcloud/run_tests.yml?label=tests&logo=github)](https://github.com/sehoffmann/dmlcloud/actions/workflows/run_tests.yml)
[![](https://img.shields.io/github/actions/workflow/status/sehoffmann/dmlcloud/run_linting.yml?label=lint&logo=github)](https://github.com/sehoffmann/dmlcloud/actions/workflows/run_linting.yml)

*Flexibel, easy-to-use, opinionated*

*dmlcloud* is a library for **distributed training** of deep learning models with *torch*. Unlike other similar frameworks, dmcloud adds as little additional complexity and abstraction as possible. It is tailored towards a carefully selected set of libraries and workflows.

## Installation
```
pip install dmlcloud
```

## Why dmlcloud?
- Easy initialization of `torch.distributed` (supports *slurm* and *MPI*).
- Simple, yet powerful, API. No unnecessary abstractions and complications.
- Checkpointing and metric tracking (distributed)
- Extensive logging and diagnostics out-of-the-box. Greatly improve reproducability and traceability.
- A wealth of useful utility functions required for distributed training (e.g. for data set sharding)
