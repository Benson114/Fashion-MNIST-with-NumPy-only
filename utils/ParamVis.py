"""
对模型网络初始化和训练后各层参数的可视化
1. 绘制各层参数的直方图
2. 绘制各层参数的热力图
"""
import os
import sys
import json

sys.path.append("..")

import matplotlib.pyplot as plt

import src.Layers as L
from src.MLPModel import MLPModel

if not os.path.exists("images"):
    os.makedirs("images")

ckpt_path = "../models/model_epoch_100.pkl"
nn_architecture = json.load(open(ckpt_path.replace(".pkl", ".json"), "r"))

model = MLPModel(nn_architecture)
for i, layer in enumerate(model.layers):
    if isinstance(layer, L.Linear):
        plt.figure()
        plt.hist(layer.W.flatten(), bins=100)
        plt.title(f"Layer {i + 1} Weight Distribution")
        plt.savefig(f"images/layer_{i + 1}_weight_distribution_init.png")

        plt.figure()
        plt.imshow(layer.W, cmap="hot", interpolation="nearest")
        plt.title(f"Layer {i + 1} Weight Matrix")
        plt.colorbar()
        plt.savefig(f"images/layer_{i + 1}_weight_matrix_init.png")

model.load_model_dict(path=ckpt_path)
for i, layer in enumerate(model.layers):
    if isinstance(layer, L.Linear):
        plt.figure()
        plt.hist(layer.W.flatten(), bins=100)
        plt.title(f"Layer {i + 1} Weight Distribution")
        plt.savefig(f"images/layer_{i + 1}_weight_distribution.png")

        plt.figure()
        plt.imshow(layer.W, cmap="hot", interpolation="nearest")
        plt.title(f"Layer {i + 1} Weight Matrix")
        plt.colorbar()
        plt.savefig(f"images/layer_{i + 1}_weight_matrix.png")
