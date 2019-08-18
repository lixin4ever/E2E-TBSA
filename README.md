# E2E-TBSA
Source code of our AAAI paper on End-to-End Target/Aspect-Based Sentiment Analysis.

## Requirements
* Python 3.6
* [DyNet 2.0.2](https://github.com/clab/dynet) (For building DyNet and enabling the python bindings, please follow the instructions in this [link](http://dynet.readthedocs.io/en/latest/python.html#manual-installation))
* nltk 3.2.2
* numpy 1.13.3

## Data
* **rest_total** consist of the reviews from the SemEval-2014, SemEval-2015, SemEval-2016 restaurant datasets.
* **laptop14** is identical to the SemEval-2014 laptop dataset.
* **twitter** is built by [Mitchell et al.](https://www.aclweb.org/anthology/D13-1171) (EMNLP 2013). 
* We also provide the data in the format of conll03 NER dataset.

## Parameter Settings
* To reproduce the results, please refer to the settings in **config.py**.

## Environment
* OS: REHL Server 6.4 (Santiago)
* CPU: Intel Xeon CPU E5-2620 (Yes, we do not use GPU to gurantee the deterministic outputs)

## Citation
If the code is used in your research, please star this repo and cite our paper as follows:
```
@inproceedings{li2019unified,
  title={A unified model for opinion target extraction and target sentiment prediction},
  author={Li, Xin and Bing, Lidong and Li, Piji and Lam, Wai},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  pages={6714--6721},
  year={2019}
}
```


