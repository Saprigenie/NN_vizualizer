import torch
from torch.utils.data import TensorDataset
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split


def load_digits_dataset():
    digits_data = load_digits()
    x = digits_data.data
    y = digits_data.target

    X_train, _, y_train, _ = train_test_split(x, y, random_state=42)

    X_train = torch.from_numpy(X_train).to(torch.float32)
    y_train = torch.from_numpy(y_train).to(torch.int64)

    train_dataset = TensorDataset(X_train, y_train)

    # Возвращаем тренировочный датасет
    return train_dataset


def load_xor_dataset():
    X_train, y_train = [
        torch.tensor([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]]),
        torch.tensor([0, 1, 1, 0]),
    ]
    train_dataset = TensorDataset(X_train, y_train)

    # Возвращаем тренировочный датасет
    return train_dataset
