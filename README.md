<p align="center">
  <img src="https://raw.githubusercontent.com/Jason-Young-AI/YoungNMT/master/docs/youngnmt_logo.png" width="150">
  <br />
  <br />
  <a href="https://github.com/Jason-Young-AI/YoungNMT/blob/master/LICENSE"><img alt="Apache License 2.0" src="https://img.shields.io/badge/License-Apache%202.0-brightgreen" /></a>
  <a href="https://github.com/Jason-Young-AI/YoungNMT/releases"><img alt="Latest Release" src="https://img.shields.io/badge/Release-Latest-blue" /></a>
  <a href="https://jason-young.me/YoungNMT/"><img alt="Documentation" src="https://img.shields.io/badge/Docs-Latest-yellowgreen" /></a>
</p>

--------------------------------------------------------------------------------

**YoungNMT** is a young but low coupling, flexible and scalable neural machine translation system.
The system is designed for researchers and developers to realize their ideas quickly without changing the original system.

--------------------------------------------------------------------------------

<a href="#"><img alt="Documentation" src="https://img.shields.io/badge/Notifications-Warning-red" /></a>

------ 2020.11.17 ------

*Version 0.1.1b0 is released.*

------ 2020.10.21 ------

*Version 0.1.1a0 is released, although 0.1.1a0 is an alpha version, it is more stable than version 0.1.0.*

------ 2020.10.10 ------

*Version 0.1.0 has some bugs (but these bugs do not affect normal use of YoungNMT):*
  * *loading exception of user define hocon files;*
  * *logging exception of BLEU scorer.*

--------------------------------------------------------------------------------

## Table of Contents

* [Full Documentation](https://jason-young.me/YoungNMT/)
* [Requirements](#dependencies)
* [Installation](#installation)
* [Arguments](#arguments)
* [Quickstart](#quickstart)
* [Models and Configurations](#models-and-configurations)
* [Citation](#citation)

## Dependencies

**Required**

It's better to configure and install the following dependency packages by the user:
* Python version >= 3.6
* [PyTorch](http://pytorch.org/) version >= 1.4.0

The following dependency packages will be installed automatically during system installation. If there are errors, please configure them manually.
* [YoungToolkit](https://github.com/Jason-Young-AI/YoungToolkit.git) is a Toolkit for a series of Young projects.

**Optional**

* [NCCL](https://github.com/NVIDIA/nccl) is used to train models on NVIDIA GPU.
* [apex](https://github.com/NVIDIA/apex) is used to train models with mixed precision.

## Installation

Three different installation methods are shown bellow:

1. Install `YoungNMT` from PyPI:
``` bash
pip install YoungNMT
```

2. Install `YoungNMT` from sources:
```bash
git clone https://github.com/Jason-Young-AI/YoungNMT.git
cd YoungNMT
python setup.py install
```

3. Develop `YoungNMT` locally:
```bash
git clone https://github.com/Jason-Young-AI/YoungNMT.git
cd YoungNMT
python setup.py build develop
```
After installation, run `ynmt`, if you get the following information, it proves that the installation is successful.

```bash
                >   Welcome to use YoungNMT!   <                
----------------------------------------------------------------

Please use the following command to make the most of the system:
0. ynmt --help
1. ynmt-preprocess --help
2. ynmt-train --help
3. ynmt-test --help
```

## Arguments

In YoungNMT, we built a module, which is a encapsulation of [pyhocon](https://github.com/chimpler/pyhocon),
that parses files which are wrote in a HOCON style to obtain arguments of system. 
HOCON (Human-Optimized Config Object Notation) is a superset of JSON.
So YoungNMT can load arguments from `*.json` or pure HOCON files.

After installation, the commonds `ynmt` `ynmt-preprocess`, `ynmt-train` and `ynmt-test` can be excuted directly and system arguments will be loaded from default HOCON files.

**Save Default Arguments** 
The following command will save all default arguments of modules of YoungNMT.
```bash
ynmt -s {path to save args} -t {json|yaml|properties|hocon}
```
If you want to save default arguments of a specified type of preprocess, train or test. The following commands will help.
Let's take command `ynmt-train` as an example:
```bash
ynmt-train --name transformer -s transformer.json -t json
```
This command will save the default **train**ing arguments of **transformer** to file `transformer.json`.

## Quickstart

See [Full Documentation](https://jason-young.me/YoungNMT/) for more details.

Here is an example of the WMT16 English to German Transformer experiment.

**Step 0. preliminaries**

 * Download corpora from [OneDrive](http://storage.live.com/items/F4F499EA04FAAA42\!1846:/WMT16_English-Romania.zip);
 * Download configuration file from [YoungNMT-configs](https://github.com/Jason-Young-AI/YoungNMT-configs);
 ```bash
 git clone https://github.com/Jason-Young-AI/YoungNMT-configs.git
 ```

**Step 1. Dataset preparation**

```bash
unzip -d Corpora WMT16_English-German.zip
mkdir -p Datasets
ynmt-preprocess \
    --name bilingual \
    -l YoungNMT-configs/Transformer/wmt16_en-de/preprocess.hocon
```

**Step 2. Train the model on 4 GPU**
```bash
mkdir -p Outputs
mkdir -p Checkpoints
CUDA_VISIBLE_DEVICES=0,1,2,3 ynmt-train \
    --name transformer \
    -l YoungNMT-configs/Transformer/wmt16_en-de/train.hocon
```

**Step 3. Test the model using 4 GPU**
```bash
mkdir -p Outputs
CUDA_VISIBLE_DEVICES=0,1,2,3 ynmt-test \
    --name transformer \
    -l YoungNMT-configs/Transformer/wmt16_en-de/test.hocon
```

## Models and Configurations

We provide pre-trained models and its configurations for several tasks. Please refer to [YoungNMT-configs](https://github.com/Jason-Young-AI/YoungNMT-configs).

## Citation
