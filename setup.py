import setuptools
from pathlib import Path

with open("README.md", "r") as f:
    long_description = f.read()
setuptools.setup(
        name="BmCS",
        version="1.0.9",
        author="Max Savery, Alastair Rae",
        author_email="savermax@gmail.com",
        description="Biomedical Citation Selector for classification of MEDLINE citations",
        long_description=long_description,
        long_description_content_type="text/markdown",
        url="https://github.com/saverymax/selective_indexing",
        test_suite='BmCS.BmCS_tests.test_BmCS_cli.test_BmCS_cli',
        packages=["BmCS", "BmCS.BmCS_tests"],
        entry_points={
            'console_scripts': [
                "BmCS=BmCS.BmCS:main"
                ]},
        classifiers=[
            "Programming Language :: Python :: 3.7",
            "Operating System :: Unix"
            ],
        install_requires=[
            "scikit-learn==0.20.2",
            "tensorflow==2.0.0",
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

