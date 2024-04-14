import matplotlib.pyplot as plt

from src.Dataloader import FashionMNISTDataloader
from src.Loss import CrossEntropyLoss
from src.MLPModel import MLPModel
from src.Optimizer import SGDOptimizer
from src.Trainer import Trainer

nn_architecture = [
    {"input_dim": 784, "output_dim": 128, "activation": "relu"},
    {"input_dim": 128, "output_dim": 32, "activation": "relu"},
    {"input_dim": 32, "output_dim": 10, "activation": "softmax"},
]  # 神经网络结构参数（包括层数、隐藏层大小、激活函数）

dataloader_kwargs = {
    "path_dir": "fashion-mnist",
    "n_valid": 2000,
    "batch_size": 16,
}  # 数据加载器参数（包括数据集路径、验证集大小、batch_size）

optimizer_kwargs = {
    "lr": 0.05,
    "ld": 0.001,
    "decay_rate": 0.95,
    "decay_step": 6000,
}  # 优化器参数（包括学习率、L2正则化系数、学习率衰减率、学习率衰减步数）

trainer_kwargs = {
    "n_epochs": 100,
    "eval_step": 10,
}  # 训练器参数（包括训练轮数、评估步数）


def main():
    dataloader = FashionMNISTDataloader(**dataloader_kwargs)  # 数据加载器
    model = MLPModel(nn_architecture)  # MLP模型
    optimizer = SGDOptimizer(**optimizer_kwargs)  # SGD优化器
    loss = CrossEntropyLoss()  # 交叉熵损失函数

    trainer = Trainer(model, optimizer, loss, dataloader, **trainer_kwargs)  # 训练器
    trainer.train(save_ckpt=True, verbose=True)  # 训练模型
    trainer.save_log("logs/")  # 保存训练日志
    trainer.save_best_model("models/", metric="loss", n=3, keep_last=True)  # 保存最优模型
    trainer.clear_cache()  # 清空缓存

    plt.show(block=True)


if __name__ == "__main__":
    main()
