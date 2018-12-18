# E2E-TBSA
Source code of our AAAI paper on End-to-End Target/Aspect-Based Sentiment Analysis.

## Requirements
* Python 3.6
* [DyNet 2.0.2](https://github.com/clab/dynet) (For building DyNet and enabling the python bindings, please follow the instructions in this [link](http://dynet.readthedocs.io/en/latest/python.html#manual-installation))
* nltk 3.2.2
* numpy 1.13.3

## Data
* **rest_total** consist of the reviews from the SemEval-2014, SemEval-2015, SemEval-2016 restaurant datasets
* **laptop14** is identical to the SemEval-2014 laptop dataset
* **twitter** is built by [Mitchell et al.](https://www.aclweb.org/anthology/D13-1171) (EMNLP 2013). 
* We also provide the data in the format of conll03 NER dataset.

## Citation
If the code is used in your research, please star this repo and cite our paper as follows:
```
@article{li2018unified,
  title={A Unified Model for Opinion Target Extraction and Target Sentiment Prediction},
  author={Li, Xin and Bing, Lidong and Li, Piji and Lam, Wai},
  journal={arXiv preprint arXiv:1811.05082},
  year={2018}
}
```


