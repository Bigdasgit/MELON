# MELON: Learning Multi-Aspect Modality Preferences for Accurate Multimedia Recommendation

This repository provides the official PyTorch implementation of LETTER, as introduced in the following paper:

> MELON: Learning Multi-Aspect Modality Preferences for Accurate Multimedia Recommendation  
> Dongho Jeong, Taeri Kim, Donghyun Cho and Sang-Wook Kim  
> In Proceedings of the 48th International ACM SIGIR Conference on Research and Development in Information Retrieval (SIGIR’25)  


### Requirements
The code has been tested running under Python 3.6.13. The required packages are as follows:
- ```gensim==3.8.3```
- ```pytorch==1.10.2+cu113```
- ```torch_geometric=2.0.3```
- ```sentence_transformers=2.2.0```
- ```pandas```
- ```numpy```

### Dataset Preparation
#### Dataset Download
*Men Clothing and Women Clothing*: Download Amazon product dataset provided by [MAML](https://github.com/liufancs/MAML). Put data folder into the directory data/.

*Sports and Toys & Games*: Download 5-core reviews data, meta data, and image features from [Amazon product dataset](http://jmcauley.ucsd.edu/data/amazon/links.html). Put data into the directory data/{folder}/meta-data/.

#### Dataset Preprocessing
Run ```python build_data.py --name={Dataset}```

### Run
```
python main.py --alpha=0.3 --beta=0.6 --gamma=0.4 --delta=0.9 --dataset=WomenClothing --model_name=MELON_3_6_4_9
```
### Acknowledgement
The structure of this code is largely based on [MONET](https://github.com/Kimyungi/MONET). Thank for their work.
