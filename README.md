# QW-loss
* A Quasi-Wasserstein Loss for Learning Graph Neural Networks (WWW 2024) [https://dl.acm.org/doi/10.1145/3589334.3645586]. 

## Training & Evaluation


* Bregman Divergence-based Solver.

```
python training.py  --lr 0.01 --weight_decay 0.0 --dataset Chameleon --net GAT --Original_ot ot --lambda_ 1.0 --q_linear_lr 0.002 --q_linear_delay 0.0005 
```


* Bregman ADMM-based Solver

```
python training_badmm.py  --lr 0.002 --weight_decay 0.0 --dataset Photo --net GIN --Original_ot ot --lambda_ 100.0 --q_linear_lr 0.001 --q_linear_delay 0.0005 --iter_num_theta 5 --iter_num_q 5
```

* Traditional GNN

```
python training.py  --lr 0.01 --weight_decay 0.0 --dataset Chameleon --net GAT --Original_ot Original
```

* The main GNN framework is based on BernNet[https://github.com/ivam-he/BernNet] and ChebNetII[https://github.com/ivam-he/ChebNetII]. 

## Parameters


```Original_ot``` indicates whether to choose QW-loss improved GNNs (Original_ot='ot') or traditional GNNs (Original_ot='Original').

```lambda_``` corresponds to the wight of Bergman Divergence.

```q_linear_lr``` corresponds to the learning rate for label transport matrix F. 

```q_linear_delay``` corresponds to the  weight decay for label transport matrix F. 

## Citation

If our work can help you, please cite it
```
@inproceedings{10.1145/3589334.3645586,
author = {Cheng, Minjie and Xu, Hongteng},
title = {A Quasi-Wasserstein Loss for Learning Graph Neural Networks},
year = {2024},
isbn = {9798400701719},
publisher = {Association for Computing Machinery},
url = {https://doi.org/10.1145/3589334.3645586},
doi = {10.1145/3589334.3645586},
booktitle = {Proceedings of the ACM on Web Conference 2024},
pages = {815–826},
numpages = {12},
series = {WWW '24}
}
```
