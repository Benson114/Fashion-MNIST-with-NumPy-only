import json

from src.GridSearcher import GridSearcher

hyper_param_defaults = {
    "input_dim": 784,
    "hidden_size_1": 128,
    "hidden_size_2": 32,
    "output_dim": 10,
    "activation_1": "relu",
    "activation_2": "relu",
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
    "eval_step": 100,  # 搜索超参数组合不需要在训练过程中评估
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
    with open("gridsearch_results.json", "w") as f:
        json.dump(results, f, indent=4)


if __name__ == "__main__":
    main()
