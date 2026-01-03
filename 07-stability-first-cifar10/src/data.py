import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset

def _get_indices(dataset, labels):
    """Get indices for given classes"""
    t = dataset.targets
    return [i for i in range(len(dataset)) if int(t[i]) in labels]

def get_split_cifar10_loaders(batch_size=128):
    """Loads CIFAR-10 and splits into Task A (0-4) and Task B (5-9)"""
    # Normalization for CIFAR-10
    tfm_train = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    
    tfm_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    
    train = datasets.CIFAR10("./data", train=True, download=True, transform=tfm_train)
    test = datasets.CIFAR10("./data", train=False, download=True, transform=tfm_test)

    # Task A: classes 0-4, Task B: classes 5-9
    A = [0, 1, 2, 3, 4]
    B = [5, 6, 7, 8, 9]

    train_a = DataLoader(
        Subset(train, _get_indices(train, A)), 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=0, 
        pin_memory=False
    )
    test_a = DataLoader(
        Subset(test, _get_indices(test, A)), 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=0, 
        pin_memory=False
    )

    train_b = DataLoader(
        Subset(train, _get_indices(train, B)), 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=0, 
        pin_memory=False
    )
    test_b = DataLoader(
        Subset(test, _get_indices(test, B)), 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=0, 
        pin_memory=False
    )

    return train_a, test_a, train_b, test_b

def build_replay_buffer_from_loader(loader, k=500):
    """Creates replay buffer from data loader"""
    xs, ys, seen = [], [], 0
    for xb, yb in loader:
        xs.append(xb.cpu())
        ys.append(yb.cpu())
        seen += xb.size(0)
        if seen >= k:
            break
    X = torch.cat(xs, dim=0)[:k]
    Y = torch.cat(ys, dim=0)[:k]
    return X, Y

def sample_replay_batch(X, Y, n):
    """Samples random batch from replay buffer"""
    idx = torch.randint(0, X.size(0), (n,))
    return X[idx], Y[idx]

