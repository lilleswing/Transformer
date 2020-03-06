#!/bin/bash

INSTALL_MODE=${1:-install}

export DATE=`date +%Y-%m-%d`
export ENV_NAME=${ENV_NAME:-transformer}
export CPU_ONLY=${CPU_ONLY:-0}
export CONDA_URL=https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh

unamestr=`uname`
if [[ "$unamestr" == 'Darwin' ]];
then
   echo "Using OSX Conda"
   export CONDA_URL=https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-x86_64.sh
   export CPU_ONLY=1
fi


export CONDA_EXISTS=`which conda`
if [[ "$CONDA_EXISTS" = "" ]];
then
    wget ${CONDA_URL} -O anaconda.sh;
    bash anaconda.sh -b -p `pwd`/anaconda
    export PATH=`pwd`/anaconda/bin:$PATH
else
    echo "Using Existing Conda"
fi

# Install Libraries
conda config --add channels conda-forge
if [[ $NO_ENV -eq 0 ]]; then
    conda create -y --name $ENV_NAME delegator
    source activate $ENV_NAME
else
    conda install -y -q delegator
fi
python devtools/conda_install_from_json.py devtools/requirements.json
python -m spacy download en_core_web_sm
python -m spacy download fr_core_news_sm

python train.py -src_data data/english.txt -trg_data data/french.txt -src_lang en_core_web_sm -trg_lang fr_core_news_sm

