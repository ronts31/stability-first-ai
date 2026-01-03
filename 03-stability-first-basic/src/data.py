import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset

def _get_indices(dataset, labels):
    t = dataset.targets
    return [i for i in range(len(dataset)) if int(t[i]) in labels]

def get_split_mnist_loaders(batch_size=128):
    tfm = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    train = datasets.MNIST("./data", train=True, download=True, transform=tfm)
    test  = datasets.MNIST("./data", train=False, download=True, transform=tfm)

    A = [0,1,2,3,4]
    B = [5,6,7,8,9]

    train_a = DataLoader(Subset(train, _get_indices(train, A)), batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=False)
    test_a  = DataLoader(Subset(test,  _get_indices(test,  A)), batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=False)

    train_b = DataLoader(Subset(train, _get_indices(train, B)), batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=False)
    test_b  = DataLoader(Subset(test,  _get_indices(test,  B)), batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=False)

    return train_a, test_a, train_b, test_b

def build_replay_buffer_from_loader(loader, k=500):
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
    idx = torch.randint(0, X.size(0), (n,))
    return X[idx], Y[idx]
