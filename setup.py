import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="datasci",  # Replace with your own username
    version="0.0.1",
    author="Eric Kehoe",
    author_email="ekehoe32@gmail.com",
    description="A package for automating pre-processing, visualization, and classification of generic data sets",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/ekehoe32/DataSci",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU General Public License v3.0",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.7',
)