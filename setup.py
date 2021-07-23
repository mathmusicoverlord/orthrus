import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="datasci",  # Replace with your own username
    version="0.0.1",
    author="Eric Kehoe, Kartikay Sharma",
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
    install_requirements=['pickle==4.0'
                          'dash>=1.19.0',
                          'tqdm>=4.61.2',
                          'matplotlib>=3.3.4',
                          'ray>=1.4.0',
                          'plotly>=4.14.3',
                          'Flask>=1.1.2',
                          'numpy>=1.19.2',
                          'torch>=1.7.1',
                          'umap_learn>=0.5.1',
                          'harmonypy>=0.0.5',
                          'pandas>=1.1.3',
                          'dash_core_components>=1.15.0',
                          'opencv_python>=4.5.1.48',
                          'calcom.egg>=info',
                          'seaborn>=0.11.1',
                          'dash_html_components>=1.1.2',
                          'GEOparse>=2.0.3',
                          'scikit_learn>=0.24.2',
                          'umap>=0.1.1'],
    )