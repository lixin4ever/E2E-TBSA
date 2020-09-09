# E2E-TBSA
Source code of our AAAI paper on End-to-End Target/Aspect-Based Sentiment Analysis.

## Requirements
* Python 3.6
* [DyNet 2.0.2](https://github.com/clab/dynet) (For building DyNet and enabling the python bindings, please follow the instructions in this [link](http://dynet.readthedocs.io/en/latest/python.html#manual-installation))
* nltk 3.2.2
* numpy 1.13.3

## Data
* ~~**rest_total** consist of the reviews from the SemEval-2014, SemEval-2015, SemEval-2016 restaurant datasets.~~
* (**Important**) **rest14**, **rest15**, **rest16**: restaurant reviews from SemEval 2014 (task 4), SemEval 2015 (task 12) and SemEval 2016 (task 5) respectively. We have prepared data files with train/dev/test split in our another [project](https://github.com/lixin4ever/BERT-E2E-ABSA/tree/master/data), check it out if needed.
* (**Important**) **DO NOT** use the ```rest_total``` dataset built by ourselves again, more details can be found in [Updated Results](https://github.com/lixin4ever/E2E-TBSA/blob/master/README.md#updated-results-important). 
* **laptop14** is identical to the SemEval-2014 laptop dataset.
* **twitter** is built by [Mitchell et al.](https://www.aclweb.org/anthology/D13-1171) (EMNLP 2013). 
* We also provide the data in the format of conll03 NER dataset.

## Parameter Settings
* To reproduce the results, please refer to the settings in **config.py**.

## Environment
* OS: REHL Server 6.4 (Santiago)
* CPU: Intel Xeon CPU E5-2620 (Yes, we do not use GPU to gurantee the deterministic outputs)

## Updated results (IMPORTANT)
* The data files of the ```rest_total``` dataset are created by concatenating the train/test counterparts from ```rest14```, ```rest15``` and ```rest16``` and our motivation is to build a larger training/testing dataset to stabilize the training/faithfully reflect the capability of the ABSA model. However, we recently found that the SemEval organizers directly treat the union set of ```rest15.train``` and ```rest15.test``` as the training set of rest16 (i.e., ```rest16.train```), and thus, there exists overlap between ```rest_total_train.txt``` and ```rest_total_test.txt```, which makes this dataset invalid. When you follow our works on this E2E-ABSA task, we hope you **DO NOT** use this ```rest_total``` dataset any more but change to the officially released ```rest14```, ```rest15``` and ```rest16```. We have prepared data files with train/dev/test split in our another [project](https://github.com/lixin4ever/BERT-E2E-ABSA), check it out if needed.
* To facilitate the comparison in the future, we re-run our models following the settings in **config.py** and report the results on ```rest14```, ```rest15``` and ```rest16```:  

    | Model | rest14 | rest15 | rest16 |
    | --- | --- | --- | --- |
    | E2E-ABSA (OURS) | 67.10 | 57.27 | 64.31 |
    | [He et al. (2019)](https://arxiv.org/pdf/1906.06906.pdf) | 69.54 | 59.18 | - |
    | [Liu et al. (2020)](https://arxiv.org/pdf/2004.06427.pdf) | 68.91 | 58.37 | - |
    | BERT-Linear (OURS) | 72.61 | 59.47 | 69.84 |
    | BERT-GRU (OURS) | 73.17 | 59.54 | 69.53 |
    | BERT-SAN (OURS) | 73.51 | 59.88 | 70.23 |
    | [Chen and Qian (2019)](https://www.aclweb.org/anthology/2020.acl-main.340.pdf)| 75.42 | 66.05 | - |
    | [Liang et al. (2020)](https://arxiv.org/pdf/2004.01951.pdf)| 72.60 | 62.37 | - |



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


