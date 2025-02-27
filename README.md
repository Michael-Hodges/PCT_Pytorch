## PCT: Point Cloud Transformer
This is a Pytorch implementation of PCT: Point Cloud Transformer.

Paper link: https://arxiv.org/pdf/2012.09688.pdf

## Changes made to the original repository
Implemented the simple PCT version for classification and part segmentation
Utilised simple PCT part segmentation network to pre train the network on self-supervised data generated using the method from this paper: https://papers.nips.cc/paper/2019/file/993edc98ca87f7e08494eec37fa836f7-Paper.pdf and implemented in this repository: https://github.com/Michael-Hodges/Self-Supervised_Point_Cloud

Our pretrained model achieves accuracy of 90.56% and the non pretrained model achieves an accuracy of 88.69% showing that the pretraining helped improve our accuracy.

### Requirements
python >= 3.7

pytorch >= 1.6

h5py

scikit-learn

and

```shell script
pip install pointnet2_ops_lib/.
```
The code is from https://github.com/erikwijmans/Pointnet2_PyTorch https://github.com/WangYueFt/dgcnn and https://github.com/MenghaoGuo/PCT

### Models
We get an accuracy of 93.2% on the ModelNet40(http://modelnet.cs.princeton.edu/) validation dataset

The path of the model is in ./checkpoints/train/models/model.t7

### Example training and testing
```shell script
# train
python main.py --exp_name=train --num_points=1024 --use_sgd=True --batch_size 32 --epochs 250 --lr 0.0001

# test
python main.py --exp_name=test --num_points=1024 --use_sgd=True --eval=True --model_path=checkpoints/best/models/model.t7 --test_batch_size 8

```

### Citation
If it is helpful for your work, please cite this paper:
```latex
@misc{guo2020pct,
      title={PCT: Point Cloud Transformer}, 
      author={Meng-Hao Guo and Jun-Xiong Cai and Zheng-Ning Liu and Tai-Jiang Mu and Ralph R. Martin and Shi-Min Hu},
      year={2020},
      eprint={2012.09688},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```
