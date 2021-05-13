# Multi-class Classification without Multi-class Labels




## Introduction
This repository provides reproduction of  [Multi-class Classification without Multi-class Labels
](https://arxiv.org/abs/1901.00544) 

The open-source code can be found at: [https://github.com/GT-RIPL/L2C]()

## Team Member (Team Relu)
Ruihuang Ding (<rd3n20@soton.ac.uk>)

Bin Zhang (<bz1u20@soton.ac.uk>)

Yefan Zhuang (<yz5e20@soton.ac.uk>)

## Requirement

```bash
PyTorch 1.0, 
python 2.7, 3.6, and 3.7
torch>=0.4.1
torchvision>=0.2.1
argparse
scipy
sklearn

```
Or you can just use the following command:

```bash
pip install -r requirements.txt
```

## Our own implementation for evaluation
### Supervised scenario
```bash
# The test files we used to test the MCL strategy

# generate the data in Table 1
python generate_table_test.py

# generate the accuracy and loss curve
python supervised_learning_compare_test.py
```
### Unsupervised scenario
#### Learn the Similarity Prediction Network (SPN) with Omniglot_background and then transfer to the 20 alphabets in Omniglot_evaluation.
```bash
python demo_omniglot_transfer.py
```

## Demo

### Supervised Classification/Clustering with only pairwise similarity
```bash
# A quick trial:
python demo.py  # Default Dataset:MNIST, Network:LeNet, Loss:MCL
python demo.py --loss KCL

# Lookup available options:
python demo.py -h

# For more examples:
./scripts/exp_supervised_MCL_vs_KCL.sh
```
### Unsupervised Clustering (Cross-task Transfer Learning)
```bash
# Learn the Similarity Prediction Network (SPN) with Omniglot_background and then transfer to the 20 alphabets in Omniglot_evaluation.
# Default loss is MCL with an unknown number of clusters (Set a large cluster number, i.e., k=100)
# It takes about half an hour to finish.
python demo_omniglot_transfer.py

# An example of using KCL and set k=gt_#cluster
python demo_omniglot_transfer.py --loss KCL --num_cluster -1

# Lookup available options:
python demo_omniglot_transfer.py -h

# Other examples:
./scripts/exp_unsupervised_transfer_Omniglot.sh
```


