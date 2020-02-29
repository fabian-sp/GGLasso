from setuptools import setup, find_packages


with open("README.md", "r") as fh:
    long_description = fh.read()


setup(
    name="gglasso",
    author="Fabian Schaipp",
    author_email= "fabian.schaipp@tum.de",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=find_packages()
)
