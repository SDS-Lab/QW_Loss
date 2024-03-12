# QW-loss
* A Quasi-Wasserstein Loss for Learning Graph Neural Networks (WWW 2024) [https://arxiv.org/abs/2310.11762]. 

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
@article{cheng2023quasi,
  title={A Quasi-Wasserstein Loss for Learning Graph Neural Networks},
  author={Cheng, Minjie and Xu, Hongteng},
  journal={arXiv preprint arXiv:2310.11762},
  year={2023}
}
```
