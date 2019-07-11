import setuptools
from pathlib import Path

with open("README.md", "r") as f:
    long_description = f.read()
setuptools.setup(
        name="BmCS",
        version="1.0.0",
        author="Max Savery, Alistair Rae",
        author_email="savermax@gmail.com",
        description="Biomedical Citation Selector for classification of MEDLINE citations",
        long_description=long_description,
        long_description_content_type="text/markdown",
        url="https://github.com/saverymax/selective_indexing",
        packages=["BmCS", "BmCS.BmCS_tests"],
        entry_points={
            'console_scripts': [
                "BmCS=BmCS.BmCS:main"
                ]},
        classifiers=[
            "Programming Language :: Python :: 3.6",
            "Operating System :: Unix"
            ],
        install_requires=[
            "numpy>=1.14.3",
            "scipy>=1.1.0",
            "scikit-learn==0.20.2",
            "tensorflow==1.11.0",
            "Keras==2.2.4",
            "python-dateutil",
            "nltk"
            ],
        package_data={
            'BmCS': [
                "config/*",
                "models/*",
                "BmCS_tests/datasets/*"
                ]},
        )

