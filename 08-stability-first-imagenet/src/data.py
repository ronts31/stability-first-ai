import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset

def _get_indices(dataset, labels):
    """Получает индексы для заданных классов"""
    # Для ImageNet используем targets напрямую
    if hasattr(dataset, 'targets'):
        t = dataset.targets
    elif hasattr(dataset, 'samples'):
        # Для ImageFolder
        t = [s[1] for s in dataset.samples]
    else:
        raise ValueError("Не удалось определить targets для датасета")
    
    return [i for i in range(len(dataset)) if int(t[i]) in labels]

def get_split_imagenet_loaders(data_root="./data", batch_size=64, image_size=224):
    """
    Загружает ImageNet и разделяет на Task A (0-499) и Task B (500-999)
    
    Примечание: Для полного ImageNet требуется загрузка датасета.
    Здесь используется упрощенная версия с ImageNet-100 или подмножеством.
    """
    # Нормализация для ImageNet
    tfm_train = transforms.Compose([
        transforms.RandomResizedCrop(image_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    tfm_test = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Примечание: ImageNet не доступен напрямую через torchvision.datasets.ImageNet
    # Для полного ImageNet требуется ручная загрузка датасета и использование ImageFolder
    # Здесь используем CIFAR-100 как замену для тестирования
    # Для реального ImageNet используйте:
    # train = datasets.ImageFolder(root=f"{data_root}/ImageNet/train", transform=tfm_train)
    # test = datasets.ImageFolder(root=f"{data_root}/ImageNet/val", transform=tfm_test)
    
    try:
        # Пробуем загрузить ImageNet через ImageFolder (если доступен)
        from torchvision.datasets import ImageFolder
        import os
        imagenet_train_path = os.path.join(data_root, "ImageNet", "train")
        imagenet_val_path = os.path.join(data_root, "ImageNet", "val")
        
        if os.path.exists(imagenet_train_path) and os.path.exists(imagenet_val_path):
            train = ImageFolder(root=imagenet_train_path, transform=tfm_train)
            test = ImageFolder(root=imagenet_val_path, transform=tfm_test)
            
            # Для ImageNet используем первые 500 классов как Task A, остальные как Task B
            A = list(range(0, 500))
            B = list(range(500, 1000))
            print("[INFO] Загружен ImageNet из ImageFolder")
        else:
            raise FileNotFoundError("ImageNet не найден")
        
    except (FileNotFoundError, RuntimeError, ImportError):
        # Если ImageNet недоступен, используем CIFAR-100 как замену для тестирования
        print("[WARNING] ImageNet не найден. Используется CIFAR-100 как замена для тестирования.")
        print("[INFO] Для полного ImageNet требуется загрузить датасет вручную.")
        
        # Используем CIFAR-100 с увеличенным размером изображения
        tfm_train_cifar = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        tfm_test_cifar = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        train = datasets.CIFAR100(
            root=data_root, 
            train=True, 
            download=True, 
            transform=tfm_train_cifar
        )
        test = datasets.CIFAR100(
            root=data_root, 
            train=False, 
            download=True, 
            transform=tfm_test_cifar
        )
        
        # Для CIFAR-100: Task A (0-49), Task B (50-99)
        A = list(range(0, 50))
        B = list(range(50, 100))

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
    """Создает replay buffer из загрузчика данных"""
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
    """Выбирает случайный батч из replay buffer"""
    idx = torch.randint(0, X.size(0), (n,))
    return X[idx], Y[idx]

