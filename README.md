# dmlcloud
[![](https://img.shields.io/pypi/v/dmlcloud)](https://pypi.org/project/dmlcloud/)
[![](https://img.shields.io/github/actions/workflow/status/sehoffmann/dmlcloud/run_tests.yml?logo=github)](https://github.com/sehoffmann/dmlcloud/actions/workflows/run_tests.yml)
[![](https://img.shields.io/github/actions/workflow/status/sehoffmann/dmlcloud/run_linting.yml?label=lint&logo=github)](https://github.com/sehoffmann/dmlcloud/actions/workflows/run_linting.yml)

Flexibel, easy-to-use, opinionated

**dmlcloud** is a library for distributed training of deep learning models with torch. Its main aim is to do all these tiny little tedious things that everybody just copy pastes over and over again, while still giving you full control over the training loop and maximum flexibility.

Unlike other similar frameworks, such as *lightning*, dmcloud tries to add as little additional complexity and abstraction as possible. Instead, it is tailored towards a careful selected set of libraries and workflows and sticks with them.
