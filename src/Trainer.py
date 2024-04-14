import os
import time

import matplotlib.pyplot as plt
import numpy as np


class Trainer:
    def __init__(self, model, optimizer, loss, dataloader, n_epochs, eval_step):
        self.model = model
        self.optimizer = optimizer
        self.loss = loss
        self.dataloader = dataloader
        self.n_epochs = n_epochs
        self.eval_step = eval_step

        self.train_loss = []
        self.train_acc = []
        self.valid_loss = []
        self.valid_acc = []

        self.model_cache = {}
        self.train_log = []

    def train(self, save_ckpt=True, verbose=True):
        for epoch in range(1, self.n_epochs + 1):
            start = time.time()
            total_loss = 0
            total_acc = 0
            for x_batch, y_batch in self.dataloader.generate_train_batch():
                y_pred = self.model.forward(x_batch)
                total_acc += np.sum(np.argmax(y_pred, axis=1) == np.argmax(y_batch, axis=1))
                loss = self.loss.forward(y_pred, y_batch)
                total_loss += loss * len(x_batch)
                grad = self.loss.backward()
                self.model.backward(grad)
                self.optimizer.step(self.model)
            end = time.time()

            train_loss = total_loss / len(self.dataloader.y_train)
            train_acc = total_acc / len(self.dataloader.y_train)
            self.train_loss.append(train_loss)
            self.train_acc.append(train_acc)

            if epoch % self.eval_step == 0:
                valid_loss, valid_acc = self.evaluate()
                self.valid_loss.append(valid_loss)
                self.valid_acc.append(valid_acc)

                if save_ckpt:
                    self.cache_epoch(epoch)

                f = len(str(self.n_epochs))
                log = (f"Epoch {epoch:0>{f}} | "
                       f"learning rate: {self.optimizer.lr:.4f} | "
                       f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
                       f"Valid Loss: {valid_loss:.4f} | Valid Acc: {valid_acc:.4f} | "
                       f"Epoch Time: {end - start:.2f} s")
                self.train_log.append(log)

                if verbose:
                    self.plot_train(epoch)
                    self.plot_valid(epoch)
                    print(log)
            else:
                f = len(str(self.n_epochs))
                log = (f"Epoch {epoch:0>{f}} | "
                       f"learning rate: {self.optimizer.lr:.4f} | "
                       f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
                       f"Epoch Time: {end - start:.2f} s")
                self.train_log.append(log)

                if verbose:
                    self.plot_train(epoch)
                    print(log)

        if save_ckpt:
            self.cache_epoch(self.n_epochs)

    def evaluate(self):
        total_loss = 0
        total_acc = 0
        for x_batch, y_batch in self.dataloader.generate_valid_batch():
            y_pred = self.model.forward(x_batch)
            total_acc += np.sum(np.argmax(y_pred, axis=1) == np.argmax(y_batch, axis=1))
            loss = self.loss.forward(y_pred, y_batch)
            total_loss += loss * len(x_batch)
        valid_loss = total_loss / len(self.dataloader.y_valid)
        valid_acc = total_acc / len(self.dataloader.y_valid)
        return valid_loss, valid_acc

    def save_log(self, save_dir):
        """
        保存训练日志
        """
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        with open(os.path.join(save_dir, "train_log.txt"), "w") as f:
            for log in self.train_log:
                f.write(log + "\n")

    def cache_epoch(self, epoch):
        """
        缓存模型参数，epoch 为 key，模型为 value
        """
        self.model_cache[epoch] = self.model.deep_copy()

    def clear_cache(self):
        """
        清除模型缓存和训练日志
        """
        self.model_cache = {}
        self.train_log = []

    def save_best_model(self, save_dir, metric="loss", n=3, keep_last=True):
        """
        根据训练器记录的验证集指标，保存最好的 n 个权重，以及最后的权重（可选）
        :param save_dir: 保存路径（文件夹）
        :param metric: 选择最好的权重的指标，可选 "loss" 或 "acc"
        :param n: 保存最好的 n 个权重
        :param keep_last: 是否保存最后一个权重
        """
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        if metric == "loss":
            best_idxs = np.argsort(self.valid_loss)[:n]
        elif metric == "acc":
            best_idxs = np.argsort(self.valid_acc)[::-1][:n]
        else:
            raise ValueError("metric should be 'loss' or 'acc'")

        best_epochs = [(index + 1) * self.eval_step for index in best_idxs]
        for epoch in best_epochs + [self.n_epochs] * keep_last:
            if epoch not in self.model_cache:
                continue
            model = self.model_cache[epoch]
            model.save_model_dict(os.path.join(save_dir, f"model_epoch_{epoch}.pkl"))

    def plot_train(self, epoch):
        """
        绘制训练过程中训练集上的 loss 和 acc，每个 epoch 一个点
        """
        plt.figure(1)
        plt.clf()
        ax1 = plt.gca()

        color = 'tab:red'
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Train Loss', color=color)
        line1, = ax1.plot(range(1, epoch + 1), self.train_loss, color=color, label='Train Loss')
        ax1.tick_params(axis='y', labelcolor=color)

        ax2 = ax1.twinx()
        color = 'tab:blue'
        ax2.set_ylabel('Train Acc', color=color)
        line2, = ax2.plot(range(1, epoch + 1), self.train_acc, color=color, label='Train Acc')
        ax2.tick_params(axis='y', labelcolor=color)

        lines = [line1, line2]
        labels = [line.get_label() for line in lines]
        ax1.legend(lines, labels, loc='upper left')

        plt.tight_layout()
        plt.draw()
        plt.pause(0.001)

    def plot_valid(self, epoch):
        """
        绘制训练过程中验证集上的 loss 和 acc，每 eval_step 个 epoch 一个点
        """
        plt.figure(2)
        plt.clf()
        ax1 = plt.gca()

        color = 'tab:red'
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Valid Loss', color=color)
        line1, = ax1.plot(range(self.eval_step, epoch + 1, self.eval_step), self.valid_loss, color=color,
                          label='Valid Loss')
        ax1.tick_params(axis='y', labelcolor=color)

        ax2 = ax1.twinx()
        color = 'tab:blue'
        ax2.set_ylabel('Valid Acc', color=color)
        line2, = ax2.plot(range(self.eval_step, epoch + 1, self.eval_step), self.valid_acc, color=color,
                          label='Valid Acc')
        ax2.tick_params(axis='y', labelcolor=color)

        lines = [line1, line2]
        labels = [line.get_label() for line in lines]
        ax1.legend(lines, labels, loc='upper left')

        plt.tight_layout()
        plt.draw()
        plt.pause(0.001)
