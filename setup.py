import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r") as reqs_file:
    lines = reqs_file.readlines()
    reqs = [line.rstrip('\n') for line in lines]

setuptools.setup(
    name="orthrus",  # Replace with your own username
    version="0.0.3",
    author="Eric Kehoe, Kartikay Sharma",
    author_email="ekehoe32@gmail.com",
    description="A package for automating pre-processing, visualization, and classification of generic data sets",
    long_description=long_description,
    long_description_content_type="text/markdown",
    download_url="https://github.com/ekehoe32/orthrus/archive/refs/tags/v1.4-alpha.tar.gz",
    url="https://github.com/ekehoe32/orthrus",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.7',
    install_requirements=reqs,
    )
