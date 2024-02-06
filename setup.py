# MIT License

# Copyright (c) 2024 Akhmed Rakhmati

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from setuptools import setup
from setuptools import find_packages

# read the contents of your README file
from pathlib import Path

this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

setup(
    name="torchtrail",
    version="0.0.7",
    author="Akhmed Rakhmati",
    author_email="akhmed.rakhmati@gmail.com",
    description="A library for tracing the execution of Pytorch operations and modules",
    packages=find_packages(),
    install_requires=[
        "torch>=1.3.0",
        "networkx==3.1",
        "pyrsistent==0.20.0",
        "graphviz==0.20.1",
    ],
    extras_require={
        "dev": [
            "black==24.1.1",
            "pytest==8.0.0",
            "transformers==4.37.2",
            "pillow==10.2.0",
            "torchvision==0.17.0",
        ]
    },
    long_description=long_description,
    long_description_content_type="text/markdown",
)
