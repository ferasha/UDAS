# UDAS

This is the code for the paper **Unsupervised Domain Adaptation for Medical Image Segmentation vis Self-Training of Early Features** accepted for MIDL 2022, that you can find here:
https://openreview.net/pdf?id=wc9qnxw35tS

Given a U-Net model that was trained on a source domain, our goal is to adapt it to another target domain without the use of target ground-truth labels. We accomplish this by first adding a second segmentation head that we place just before the first downsampling layer and train it on the source domain. To perform adaptation on the target domain, we use the probabilistic predictions at the end of the network as pseudo-labels for the first segmentation head and refine the early features of the network.

To use the code, you can follow these steps:

1) Create a PyTorch dataset object. An example can be found in ``src\calgary_campinas_dataset.py``.
2) Train a U-Net model on the source domain using ``src\train_unet.py``.
3) Train a second segmentation head on the source domain using ``src\train_features_segmenter.py``.
4) Refine features on the target domain using ``src\refine_features.py``.
