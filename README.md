# Biomedical Citation Selector (BmCS)

This repository contains the source code for the BmCS system, to be used for the prediction of citations requiring MeSH indexing
This README includes the following sections

Installation
Usage
Testing


## Installation

Installation has been tested with Ubuntu 16.04. While this installation will likely work with Windows, this cannot be guaranteed. 

Is anaconda or miniconda installed? Is python 3.6 installed? If so, skip to section ii.
If you do not have anaconda, or miniconda, and python installed, follow the instructions in i.

Note that this package requires python 3.6.

### i
Included in this section are instructions to install miniconda, a lightweight version of anaconda. In installing miniconda, python and 
standard libraries are included.

First, download the miniconda installer:
```
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
```

Then run the bash installer you just downloaded. 
```
bash Miniconda3-latest-Linux-x86_64.sh
```
Maneuver through its direction. Make sure to enter yes when it asks to add the miniconda path to the .bashrc


Unfortunately python 3.7 is not currently compatible with tensorflow 1.11.0, the version used in this system.
Since the latest miniconda comes with python 3.7 (as of March 2019), we will have to downgrade.
Fortunately, this is easy:
```
conda install python=3.6
```

Finally, check your python version. 
```
python --version
```

If python == 3.6, continue on to section ii. If not, return to go, and maybe email someone for help.

### ii

There are two options for installation of BmCS. You can either download the .whl file in the releases section of the repository, or you can clone this repository
and generate the .whl file yourself. This section first describes how to generate the .whl, and then describes how to install it.  

#### Build .whl
Clone this repository 
```
git clone https://github.com/saverymax/selective_indexing.git
```
Once the repository is cloned, navigate into the BmCS directory where the setup.py file lives. 

From the command line run
```
python setup.py bdist_wheel
```
This will create a dist directory and create a .whl file inside. The .whl file is the compressed package. 

#### Download wheel and installation
The .whl file can be found in the releases section of this repository: https://github.com/saverymax/selective_indexing/releases
Under Assets, click on the BmCS-1.0.0-py3-none-any.whl link to download. You should also download the ensemble.joblib
and model_CNN_weights.hdf5 files, whether or not built the wheel yourself or downloaded it. 

Assuming you have downloaded the .whl file, navigate to the directory where it lives and run
```
pip install BmCS-1.0.0-py3-none-any.whl
```
If all goes well, you have now installed the Biomedical Citation Selector (BmCS). Congratulations!
BmCS has been added to PATH, and is executable from the command line. 

The python dependencies have also been installed with the package. Dependencies installed can be found in the setup.py file.

However, before it can be used, there is one more step. Currently the text is tokenized with the NLTK tokenizer. 
NLTK requires you install it separately. Once you do this once,
you don't have to worry about it again, even if you uninstall and 
reinstall BmCS. To install:
``` 
python -m nltk.downloader punkt
```

To uninstall
```
pip uninstall BmCS
```


## Usage

The models are not included in the package in this version of the system. However, they are provided in the release. 

Once downloaded, the paths to the models should be provided via the command line.

### For NCBI usage
To run BmCS
```
BmCS /path/to/cnn/weights.hdf5 /path/to/model.joblib --path path/to/some_citations.xml 
```
For example, if sample_citations.xml is in your current directory and the models are in a models diretory 
```
BmCS ./models/model_CNN_weights.hdf5 ./models/ensemble.joblib --path sample_citations.xml 
```
This will generate a set of predictions for the citations in sample_citations.xml.

The prediction results can be found in the citation_predictions_YYYY-DD-MM.txt output file, which will 
be saved in your current directory, unless otherwise specified. 

Each prediction is printed on a line, in the format 
pmid|prediction|probability|NLM journal ID 

In the prediction field, 4 labels are possible:
0: Out-of-scope for indexing, 99.5% confidence
1: In-scope for indexing, 97% confidence
2: Citation should be human-reviewed. 
3: Citation marked as one of the publication types specified in the publication_types file. This label is off by default and is controlled by --pubtype-filter

### Alternative implementation
If the --filter option is provided, there are a few ways to filter and adjust the predictions.
The flags shown here are explained in more detail in the section below. 

To make predictions solely for selectively indexed journals with statuses not MEDLINE or PubMed-not-MEDLINE
```
BmCS /path/to/cnn/weights.hdf5 /path/to/model.joblib --path path/to/some_citations.xml --filter
```

To mark the citations mentioned above that are of publication types such as comments or erratum with a 3 in the predictions 
```
BmCS /path/to/cnn/weights.hdf5 /path/to/model.joblib --path path/to/some_citations.xml --filter --pubtype-filter
```

To make predictions for citations only of status MEDLINE
```
BmCS /path/to/cnn/weights.hdf5 /path/to/model.joblib --path path/to/some_citations.xml --filter --predict-medline
```

### Command line options:

**CNN_path** 
    Required. Positional. Path to CNN weights.  

**ensemble_path** 
    Required. Positional. Path to ensemble model

**--path /path/to/citations.xml** 
    Path to XML of citations for the system to classify. Include the file.xml in the path. 
    Do not include with --test or --validation

**--dest dir/for/results/** 
    Optional. Destination for predictions, or test results if --test or --validation are used. Defaults to 
    current directory. File names for predictions or test results are hardcoded, for now: 
    citation_predictions_YYYY-DD-MM.txt if running system on a batch of citations; BmCS_test_results.txt 
    if running on test or validation datasets.   

**--filter**
    Optional. By default, the system will make predictions for all citations in XML provided, regardless of MEDLINE status or selective indexing status of a given journal. 
    To turn this behvaior off, this option can be used. The system will switch to behavior intended for use outside of NCBI pipeline, i.e., the system will make predictions for selectively indexed journals with statuses not MEDLINE or Pubmed-not-MEDLINE.
    This option can also be used in conjunction with the other filtering options below for more fine-grained control.  

**--pubtype-filter**
    Optional. Modified from version 0.2.1. If included, the system will mark citations with publication types specified in publication_type file with a 3 in the output file. 
    By default this behavior is off, and though it can be used in conjunction with or without --filter, it should be kept off for NCBI usage.
 
**--group-thresh**
    Optional. If included, the system will use the unique, 
    predetermined thresholds for citations from journals in the science or jurisprudence category. 
    Originally, this was added to improve performance; however, it was shown to be difficult to apply a threshold chosen on the validation set to the test set.

**--journal-drop**
    Optional. By default the system makes predictions for a small set of journals previously misindexed. Include this option to not include those predictions in the output. Doing so has been shown to improve system precision.
    --filter must be included to use this option, and when --predict-medline is included, this option has no effect. 

**--predict-medline**
    Optional. If included, the system will make predictions for 
    ONLY non-selectively indexed journals, with ONLY MEDLINE statuses. 
    The reason for this switch is to be able to test the performance of the system 
    on citations labeled MEDLINE.
    --filter must be included to use this option.

**--validation** 
    Optional. Include to run system on 2018 validation dataset. Do not include --path if
    --validation included.  

**--test**
    Optional. Include to run system on 2018 test dataset. Do not include --path if
    --test included. 

If you forget your options, input
```
BmCS --help
```
and you'll get a little help.

Usage of --validation and --test will be explained below


## Testing
To measure performance of the system, validation and test datasets are included with the BmCS
package, as well as in the repository. To run the models on these datasets, include the --validation or --test option,
as shown in the example below. Input one or the other, not both. These tests are not affected
by the predict-medline, pubtype-filter and filter options, but journal-drop and group-thresh do change system performance for these tests.

For example:
```
BmCS /path/to/cnn/weights.hdf5 /path/to/model.joblib --validation --dest path/to/output/
```
This will output a performance report into the file BmCS_test_results.txt in the output directory. 
It is not necessary to include the --path option when running these tests. 
For a given test the following information is appended to the BmCS_test_results.txt file:

Date
All command line keywords and values
BmCS recall
BmCS precision
Voting recall
Voting precision
CNN recall
CNN precision

Once the program is installed run both of the following commands: 
```
BmCS /path/to/cnn/weights.hdf5 /path/to/model.joblib --validation
```
and
```
BmCS /path/to/cnn/weights.hdf5 /path/to/model.joblib --test
```
If no options are included, a set of assertions 
will be tested on the model's performance. If the assertions are passed,
you can be fairly confident that BmCS has been installed correctly and is ready for 
further use. If options such as journal-drop are included, no assertions will be run,
but the system's performance can still be observed in the test results.

Important to note is that the system will NOT pass the assertions if the models are retrained. The values used were hand-selected for these models and the values will have to be recalculated or the tests redesigned if new models are provided. 

Further functional testing can be performed using the pytest package.
To successfully run these tests, the 2 models must be in a directory named models in your current directory.
```
pip install pytest
pytest --pyargs BmCS
```

Happy indexing.
