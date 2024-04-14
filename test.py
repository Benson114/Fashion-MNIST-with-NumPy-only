import numpy as np

from src.Dataloader import FashionMNISTDataloader
from src.Loss import CrossEntropyLoss
from src.MLPModel import MLPModel

dataloaders_kwargs = {
    "path_dir": "fashion-mnist",
    "batch_size": 32,
}

ckpt_path = "models/model_epoch_100.pkl"


def main():
    dataloader = FashionMNISTDataloader(**dataloaders_kwargs)
    model = MLPModel()
    model.load_model_dict(ckpt_path)  # 从已经训练好的权重加载模型
    loss = CrossEntropyLoss()

    total_loss = 0
    total_acc = 0

    for x_batch, y_batch in dataloader.generate_test_batch():
        y_pred = model.forward(x_batch)
        total_acc += np.sum(np.argmax(y_pred, axis=1) == np.argmax(y_batch, axis=1))
        ce_loss = loss.forward(y_pred, y_batch)
        total_loss += ce_loss * len(x_batch)

    test_loss = total_loss / len(dataloader.y_test)
    test_acc = total_acc / len(dataloader.y_test)

    print(f"Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.4f} | Checkpoint: {ckpt_path} | ")


if __name__ == "__main__":
    main()
