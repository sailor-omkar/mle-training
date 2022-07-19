from os import path

import setuptools

# The directory containing this file
HERE = path.abspath(path.dirname(__file__))

# Get the long description from the README file
with open(path.join(HERE, "README.md"), encoding="utf-8") as f:
    long_description = f.read()

setuptools.setup(
    name="mletraining",
    version="0.1",
    author="Omkar Chavan",
    author_email="omkar.chavan@tigeranalytics.com",
    long_description=long_description,
    long_description_content_type="text/markdown",
    description="Practical Python Code",
    packages=setuptools.find_packages(),
    license="MIT",
    include_package_data=True
)
