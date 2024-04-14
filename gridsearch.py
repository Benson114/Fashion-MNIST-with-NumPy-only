from src.GridSearch import GridSearcher

hyper_param_defaults = {
    "input_dim": 784,
    "hidden_size_1": 128,
    "activation_1": "relu",
    "hidden_size_2": 32,
    "activation_2": "relu",
    "output_dim": 10,
    "activation_3": "softmax",
    "lr": 0.05,
    "ld": 0.001,
    "decay_rate": 0.95,
    "decay_step": 6000,
}  # 超参数默认值（主要是神经网络结构和优化器参数）

dataloader_kwargs = {
    "path_dir": "fashion-mnist",
    "n_valid": 2000,
    "batch_size": 16,
}  # 数据加载器参数（包括数据集路径、验证集大小、batch_size）

trainer_kwargs = {
    "n_epochs": 10,
    "eval_step": 10,
}  # 训练器参数（包括训练轮数、评估步数）


def main():
    hyper_param_opts = {
        "hidden_size_1": [128, 256],
        "hidden_size_2": [64, 32],
        "lr": [0.05, 0.01],
        "ld": [0.001, 0.005],
    }
    searcher = GridSearcher(hyper_param_opts, hyper_param_defaults)
    results = searcher.search(dataloader_kwargs, trainer_kwargs, metric="loss")
    print(results)


if __name__ == "__main__":
    main()
