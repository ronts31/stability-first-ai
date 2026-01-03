import random
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset, TensorDataset

def get_mnist(root="./data"):
    tfm = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    train_ds = datasets.MNIST(root, train=True, download=True, transform=tfm)
    test_ds  = datasets.MNIST(root, train=False, download=True, transform=tfm)
    return train_ds, test_ds

def indices_for_classes(dataset, classes):
    return [i for i, t in enumerate(dataset.targets) if int(t) in classes]

def make_loader(dataset, indices, batch_size=64, shuffle=True, num_workers=0, pin_memory=False):
    return DataLoader(
        Subset(dataset, indices),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

def split_mnist_loaders(task_a_classes, task_b_classes, batch_size=128, root="./data"):
    train_ds, test_ds = get_mnist(root=root)
    idx_a_train = indices_for_classes(train_ds, task_a_classes)
    idx_b_train = indices_for_classes(train_ds, task_b_classes)
    idx_a_test  = indices_for_classes(test_ds,  task_a_classes)
    idx_b_test  = indices_for_classes(test_ds,  task_b_classes)

    train_a = make_loader(train_ds, idx_a_train, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=False)
    test_a  = make_loader(test_ds,  idx_a_test,  batch_size=1000, shuffle=False, num_workers=0, pin_memory=False)
    train_b = make_loader(train_ds, idx_b_train, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=False)
    test_b  = make_loader(test_ds,  idx_b_test,  batch_size=1000, shuffle=False, num_workers=0, pin_memory=False)
    return train_a, test_a, train_b, test_b, idx_a_train, train_ds

def build_replay_buffer(train_ds, indices, k, seed=42):
    rng = random.Random(seed)
    chosen = rng.sample(indices, k=min(k, len(indices)))
    xs, ys = [], []
    for i in chosen:
        x, y = train_ds[i]
        xs.append(x)
        ys.append(y)
    X = torch.stack(xs, dim=0)
    Y = torch.tensor(ys, dtype=torch.long)
    return X, Y

def tensor_loader(X, Y, batch_size=10, shuffle=True):
    ds = TensorDataset(X, Y)
    idx = list(range(len(ds)))
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)


def sample_replay_batch(X, Y, n):
    idx = torch.randint(0, X.size(0), (n,))
    return X[idx], Y[idx]
