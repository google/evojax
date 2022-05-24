# Copyright 2022 The EvoJAX Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from setuptools import find_packages, setup

_dct = {}
with open("evojax/version.py") as f:
    exec(f.read(), _dct)
__version__ = _dct["__version__"]

JAX_URL = "https://storage.googleapis.com/jax-releases/jax_releases.html"

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="evojax",
    version=__version__,
    author="Google",
    author_email="evojax-dev@google.com",
    description="EvoJAX: Hardware-accelerated Neuroevolution.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/google/evojax",
    license="Apache 2.0",
    packages=[
        package for package in find_packages() if package.startswith("evojax")
    ],
    zip_safe=False,
    install_requires=[
        "flax",
        "jax>=0.2.17",
        "jaxlib>=0.1.65",
        "Pillow",
        "cma",
    ],
    extras_require={
        "extra": ['evosax', 'torchvision', 'pandas', 'procgen', 'brax'],
    },
    dependency_links=[JAX_URL],
    python_requires=">=3.6",
    classifiers=[
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
)
