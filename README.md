# Fashion-MNIST with NumPy only

## Instruction

本项目为复旦大学研究生课程DATA620004——神经网络和深度学习作业《从零开始构建三层神经网络分类器，实现图像分类》的代码仓库

* 作业：从零开始构建三层神经网络分类器，实现图像分类

* 任务描述：
  手工搭建三层神经网络分类器，在数据集[Fashion-MNIST](https://github.com/zalandoresearch/fashion-mnist)上进行训练以实现图像分类

* 基本要求：
  （1）本次作业要求自主实现反向传播，**不允许使用pytorch，tensorflow**等现成的支持自动微分的深度学习框架，可以使用numpy
  （2）最终提交的代码中应至少包含**模型**、**训练**、**测试**和**参数查找**四个部分，鼓励进行模块化设计
  （3）其中模型部分应允许自定义隐藏层大小、激活函数类型，支持通过反向传播计算给定损失的梯度；训练部分应实现SGD优化器、学习率下降、交叉熵损失和L2正则化，并能根据验证集指标自动保存最优的模型权重；参数查找环节要求调节学习率、隐藏层大小、正则化强度等超参数，观察并记录模型在不同超参数下的性能；测试部分需支持导入训练好的模型，输出在测试集上的分类准确率（Accuracy）

## Requirements

```bash
# NumPy is all you need ^^
pip install numpy

# Not exactly...
pip install matplotlib
pip install tqdm
```

## How to Run

### 数据下载

从[Fashion-MNIST](https://github.com/zalandoresearch/fashion-mnist)仓库可以下载Fashion-MNIST数据集，将`fashion-mnist/data/fashion`下的四个`.gz`文件移至本项目的`fashion-mnist`目录下即可

### 模型训练

* 进入 [`train.py`](train.py) 修改以下部分（可选）：

  MLP神经网络结构参数：

  ```python
  # 你可以自定义若干个线性层并添加到nn_architecture中
  # 在这里指定每个线性层的输入维度、输出维度和后接的激活函数
  nn_architecture = [
      {"input_dim": 784, "output_dim": 128, "activation": "relu"},
      {"input_dim": 128, "output_dim": 32, "activation": "relu"},
      {"input_dim": 32, "output_dim": 10, "activation": "softmax"},
  ]
  ```

  数据加载器参数：

  ```python
  # 在这里指定数据集所在路径、验证集大小、批量大小
  dataloader_kwargs = {
      "path_dir": "fashion-mnist",
      "n_valid": 2000,  # 其实这里改成验证集比例valid_size更好一点
      "batch_size": 16,
  }
  ```

  SGD优化器参数：

  ```python
  # 在这里指定学习率、L2正则项系数、学习率衰减系数、学习率衰减步数
  optimizer_kwargs = {
      "lr": 0.05,
      "ld": 0.001,
      "decay_rate": 0.95,
      "decay_step": 6000,
  }
  ```

  训练器参数：

  ```python
  # 在这里指定训练轮次、验证步数
  trainer_kwargs = {
      "n_epochs": 100,
      "eval_step": 10,
  }
  ```

* 进入仓库根目录，运行：

  ```bash
  python train.py
  ```

### 模型测试

* 将模型权重文件（一定要包括`.pkl`和`.json`文件）放至某一目录，例如`models/`

* 进入 [`test.py`](test.py) 修改以下部分（可选）：

  数据加载器参数：

  ```python
  # 在这里指定数据集所在路径、批量大小
  dataloader_kwargs = {
      "path_dir": "fashion-mnist",
      "batch_size": 16,
  }
  ```

  模型权重文件的路径（指定`.pkl`的路径即可，`.json`文件会自动读取）：

  ```python
  ckpt_path = "models/model_epoch_100.pkl"
  ```

* 进入仓库根目录，运行：

  ```python
  python test.py
  ```

## Extra

### GridSearch

在这里进行超参数组合的搜索！

你可以在 [`gridsearch.py`](gridsearch.py) 中设置部分超参数的默认值和选项表（其中超参数名要严格和`nn_architecture`对应，譬如有多少个`hidden_size`和`activation`，并且`hidden_size`和`activation`后面必须跟上数字）

随后设置网格搜索超参数组合的基准（`loss`或者`acc`）以及其他参数（和前面模型训练、测试的部分的类似）

然后：

```python
python gridsearch.py
```

超参数组合的搜索结果会自动保存在 [`gridsearch_results.json`](gridsearch_results.json)

### ParamVis

[`utils/ParamVis.py`](utils/ParamVis.py)提供了对模型网络初始化和训练后各层参数的可视化代码（包括直方图和热力图）

### 更多代码细节的说明详见报告