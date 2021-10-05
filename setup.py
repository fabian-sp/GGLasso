import io
import os
from setuptools import setup, find_packages

# Package meta-data.
NAME = 'gglasso'
DESCRIPTION = 'Algorithms for Single and Multiple Graphical Lasso problems.'
URL = 'https://github.com/fabian-sp/GGLasso'
EMAIL = 'fabian.schaipp@tum.de'
AUTHOR = 'Fabian Schaipp'
REQUIRES_PYTHON = '>=3.7.0'
VERSION = "0.1.4"

# What packages are required for this module to be executed?
REQUIRED = [
    "numpy>=1.17.3", "scipy>=0.11.0", "numba>=0.46.0", "pandas",
    "matplotlib", "seaborn", "networkx", "regain", "decorator==4.4.2"]

# What packages are optional?
EXTRAS = {
        "tests": ["pytest", "pytest-cov"],
        "docs": [
            "sphinx",
            "sphinx-gallery",
            "sphinx_rtd_theme",
        ],
    }

here = os.path.abspath(os.path.dirname(__file__))

# Import the README and use it as the long-description.
# Note: this will only work if 'README.md' is present in your MANIFEST.in file!
try:
    with io.open(os.path.join(here, 'README.md'), encoding='utf-8') as f:
        long_description = '\n' + f.read()
except FileNotFoundError:
    long_description = DESCRIPTION


CLASSIFIERS = """\
Development Status :: 5 - Production/Stable
Intended Audience :: Science/Research
Intended Audience :: Developers
License :: OSI Approved
Programming Language :: Python
Programming Language :: Python :: 3
Topic :: Software Development
Operating System :: Unix
"""


# Where the magic happens:
setup(
    name=NAME,
    version=VERSION,
    description=DESCRIPTION,
    long_description=long_description,
    long_description_content_type='text/markdown',
    author=AUTHOR,
    author_email=EMAIL,
    python_requires=REQUIRES_PYTHON,
    url=URL,
    packages=find_packages(exclude=["benchmarks.*", "benchmarks"]),
    install_requires=REQUIRED,
    extras_require=EXTRAS,
    include_package_data=True,
    license='MIT',
    keywords=[
        "network inference",
        "graphcial models",
        "graphical lasso",
        "optimization"
    ],
    classifiers=[_f for _f in CLASSIFIERS.split("\n") if _f]
)
