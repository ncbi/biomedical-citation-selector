import setuptools
from pathlib import Path

with open("README.md", "r") as f:
    long_description = f.read()
setuptools.setup(
        name="BmCS",
        version="4.0.2",
        author="Max Savery, Alastair Rae",
        author_email="alastair.rae@nih.gov",
        description="Biomedical Citation Selector for classification of MEDLINE citations",
        long_description=long_description,
        long_description_content_type="text/markdown",
        url="https://github.com/ncbi/biomedical-citation-selector",
        test_suite="BmCS.BmCS_tests.test_BmCS_cli.test_BmCS_cli",
        packages=["BmCS", "BmCS.BmCS_tests"],
        entry_points={
            'console_scripts': [
                "BmCS=BmCS.BmCS:main"
                ]},
        classifiers=[
            "Programming Language :: Python :: 3.11",
            "Operating System :: Unix"
            ],
        install_requires=[
            "scikit-learn==1.1.3",
            "tensorflow==2.15.0.post1",
            "python-dateutil==2.8.1",
            "nltk==3.6.1",
            "h5py==3.10.0",
            "numpy==1.26.4",
            "six==1.16.0",
            "protobuf==4.23.4",
            ],
        package_data={
            'BmCS': [
                "config/*",
                "models/*",
                "BmCS_tests/datasets/*"
                ]},
        )
