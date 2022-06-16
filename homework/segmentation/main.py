import wandb
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from torch.optim.lr_scheduler import *
from tqdm import tqdm
from sklearn.metrics import f1_score
import numpy as np
from utils import getCorpus, getWord2Id
from model import CONFIG, RNN_LSTM, RNN_GRU, TextCNN, MLP
import argparse

config = CONFIG()  # 配置模型参数


def get_data(max_length, batch_size):
    word2id = getWord2Id()
    train_contents, train_labels = getCorpus(
        "./Dataset/train.txt", word2id, maxLength=max_length
    )
    val_contents, val_labels = getCorpus(
        "./Dataset/validation.txt", word2id, maxLength=max_length
    )
    test_contents, test_labels = getCorpus(
        "./Dataset/test.txt", word2id, maxLength=max_length
    )
    train_dataset = TensorDataset(
        torch.from_numpy(train_contents).type(torch.float),
        torch.from_numpy(train_labels).type(torch.long),
    )
    train_dataloader = DataLoader(
        dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=2
    )

    val_dataset = TensorDataset(
        torch.from_numpy(val_contents).type(torch.float),
        torch.from_numpy(val_labels).type(torch.long),
    )
    val_dataloader = DataLoader(
        dataset=val_dataset, batch_size=batch_size, shuffle=True, num_workers=2
    )

    test_dataset = TensorDataset(
        torch.from_numpy(test_contents).type(torch.float),
        torch.from_numpy(test_labels).type(torch.long),
    )
    test_dataloader = DataLoader(
        dataset=test_dataset, batch_size=batch_size, shuffle=True, num_workers=2
    )
    return train_dataloader, val_dataloader, test_dataloader


def parser_data():
    parser = argparse.ArgumentParser(
        prog="IAI 2022_spring", description="Segement me if U can", allow_abbrev=True
    )
    parser.add_argument(
        "-l",
        "--start-learning-rate",
        dest="learning_rate",
        type=float,
        default=1e-3,
        help="The starting learning rate, which will decay exponentially",
    )
    parser.add_argument(
        "-e", "--epoch", dest="epoch", type=int, default=10, help="epoch"
    )
    parser.add_argument(
        "-m",
        "--max_length",
        dest="max_length",
        type=int,
        default=120,
        help="max sentence length",
    )
    parser.add_argument(
        "-b", "--batch-size", dest="batch_size", type=int, default=50, help="batch size"
    )
    parser.add_argument(
        "-n",
        "--neural network",
        dest="neural_network",
        type=str,
        default="TextCNN",
        help="neural network, default is TextCNN, you can choose RNN",
    )
    args = parser.parse_args()
    learning_rate = args.learning_rate
    epoch = args.epoch
    max_length = args.max_length
    batch_size = args.batch_size
    neural_network = args.neural_network
    if neural_network == "RNN_LSTM":
        model = RNN_LSTM(config).to(DEVICE)
    elif neural_network == "RNN_GRU":
        model = RNN_GRU(config).to(DEVICE)
    elif neural_network == "TextCNN":
        model = TextCNN(config).to(DEVICE)
    elif neural_network == "MLP":
        model = MLP(config).to(DEVICE)
    else:
        print("please choose neural network type")
        exit(1)
    return learning_rate, epoch, max_length, batch_size, neural_network, model


def train(dataloader):
    # 定义训练过程
    model.train()
    train_loss, train_acc = 0.0, 0.0
    count, correct = 0, 0
    full_true = []
    full_pred = []
    for _, (x, y) in enumerate(dataloader):
        x, y = x.to(DEVICE), y.to(DEVICE)
        optimizer.zero_grad()
        output = model(x)
        loss = criterion(output, y)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        correct += (output.argmax(1) == y).float().sum().item()
        count += len(x)
        full_true.extend(y.cpu().numpy().tolist())
        full_pred.extend(output.argmax(1).cpu().numpy().tolist())
    train_loss *= batch_size
    train_loss /= len(dataloader.dataset)
    train_acc = correct / count
    scheduler.step()
    f1 = f1_score(np.array(full_true), np.array(full_pred), average="binary")
    return train_loss, train_acc, f1


def valid_and_test(dataloader):
    model.eval()
    # 验证过程
    val_loss, val_acc = 0.0, 0.0
    count, correct = 0, 0
    full_true = []
    full_pred = []
    for _, (x, y) in enumerate(dataloader):
        x, y = x.to(DEVICE), y.to(DEVICE)
        output = model(x)
        loss = criterion(output, y)
        val_loss += loss.item()
        correct += (output.argmax(1) == y).float().sum().item()
        count += len(x)
        full_true.extend(y.cpu().numpy().tolist())
        full_pred.extend(output.argmax(1).cpu().numpy().tolist())
    val_loss *= batch_size
    val_loss /= len(dataloader.dataset)
    val_acc = correct / count
    f1 = f1_score(np.array(full_true), np.array(full_pred), average="binary")
    return val_loss, val_acc, f1


if __name__ == "__main__":
    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    learning_rate, epoch, max_length, batch_size, neural_network, model = parser_data()
    train_dataloader, val_dataloader, test_dataloader = get_data(max_length, batch_size)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()
    scheduler = StepLR(optimizer, step_size=5)
    wandb.init(project=f"Valid", name=f"{model.__name__}", entity="eren-zhao")
    wandb.config = {"learning_rate": 0.001, "epochs": 100, "batch_size": 50}
    for each in tqdm(range(1, epoch + 1)):
        tr_loss, tr_acc, tr_f1 = train(train_dataloader)
        val_loss, val_acc, val_f1 = valid_and_test(val_dataloader)
        test_loss, test_acc, test_f1 = valid_and_test(test_dataloader)
        wandb.log(
            {
                "train_loss": tr_loss,
                "train_acc": tr_acc,
                 "train_f1": tr_f1,
                "val_loss": val_loss,
                "val_acc": val_acc,
                "val_f1": val_f1,
                "test_loss": test_loss,
                "test_acc": test_acc,
                "test_f1": test_f1,
            }
        )
        print(
            f"for epoch {each}, train_loss: {tr_loss:.4f}, train_acc: {tr_acc:.4f}, val_loss: {val_loss:.4f}, val_acc: {val_acc:.4f}, test_loss: {test_loss:.4f}, test_acc: {test_acc:.4f}"
        )
