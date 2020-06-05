# Cross-Entropy Guided Adversarial Hypernetwork for Modeling Policy Distribution
This repository is the official implementation of CEM-AH.
This work is owned by the authors of the paper submitted to NeurIPS 2020 (under review).

## Usage

```Train and Evaluation (maze)
python CEM_AH_discreteAction.py --lr 1e-4 --layer 2 --batch_size 100
```

```Train and Evaluation (reacher)
python CEM_AH_continuousAction.py --lr 1e-4 --layer 2 --batch_size 100
```

> The experiments are done with the following hyper-parameters. Learning rate (--lr) {0.001, 0.005,0.0001\}, batch size (--batch_size) {50, 100, 1000\}, and number of neural network layers (--layer) {2, 3\}. 

The train and evaluation are done within the same execution. Evaluation is done at each training iteration to measure the performance gain over the guiding CEM at any step (if we were to stop the training). 
It signifies faster convergence and confirm that we can use our algorithm with relatively few episodes of training. 
Similarly, other RL algorithm has exploration and exploitation strategies, so we compare the performance of algorithms which are typically using greedy exploitation during testing at any stage.

> Log data is generated at 'CEM-AH' folder. Which could be visualized by running tensorboard i.e.
```tensorboard
tensorboard --logdir=CEM-AH
```
