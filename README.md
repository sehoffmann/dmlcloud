# Python Project Template

This is a quickstart project template for Python that already comes attached with the following features:

* Packaging and metadata support
* Formatting and linting via *pre-commit*, *black*, *usort*, and *flake8*
* Testing via *pytest*
* CI via github-actions


## Configuration

To tailor this template to your needs, the following steps must be taken:

1. Rename the *myproject* package folder to your project name
2. Change metadata and project name in *setup.cfg*.
3. Do not forget to change the version attribute to point to your new package name as well.
4. Add dependencies to *requirements.txt*
5. Adjust the *LICENSE* file to your liking.
6. Adjust this *README.md* file to your liking.

### Formatting and linting

Install *pre-commit* and *pytest* via
```
pip install -r ci_requirements.txt
```

To format and lint the entire codebase run:
```
pre-commit run --all-files
```

To perform this step automatically during each commit (and fail on errors) run:
```
pre-commit install
```

### Testing
To run the tests execute:
```
pytest
```
in the top-level directory.
Tests can also be executed individually by running them as regular python script. This requires you to add a small main function to them, c.f. *test/test_myproject.py*.

### Github Actions
This project defines the following workflows:
1. *run_linting.yml* will run `pre-commit run --all-files` on every push to develop and pull request
2. *run_tests.yml* will run `pytest` on Windows, Ubuntu, and MacOS on every push to develop and pull_request
3. *release_public.yml* and *release_test.yml* can be triggered manually to build a wheel distribution and publish it to PyPI or TestPyPI respectively

For the publising to work, you need to add the PyPI API token as Github secrets:
* *PYPI_TOKEN* for the official PyPI index
* *TEST_PYPI_TOKEN* for the TestPyPI index
