import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
from torchvision.transforms import v2

class TwoViewsTransform:
    """
    A wrapper that applies a given transformation twice 
    to generate two different views of the same image.
    """
    def __init__(self, base_transform):
        self.base_transform = base_transform

    def __call__(self, x):
        # Applies the stochastic transforms twice to create two distinct views
        return self.base_transform(x), self.base_transform(x)


def get_dataloaders(data_dir='./data', debug_subset_size = None, batch_size=64, val_split=0.15, num_workers=2, apply_augmentation=True, rotation_angle=15, translate_fraction=0.1, blur_kernel_size=3, blur_possibility=0.5):
   # 1. Define preprocessing transforms
    train_transform = transforms.Compose([
        transforms.RandomRotation(degrees=rotation_angle),
        transforms.RandomAffine(degrees=0, translate=(translate_fraction, translate_fraction)),
        transforms.RandomApply([transforms.GaussianBlur(kernel_size=blur_kernel_size)], p=blur_possibility),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    base_eval_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    # Wrap both transforms so they return two views
    two_views_train_transform = TwoViewsTransform(train_transform)
    two_views_eval_transform = TwoViewsTransform(base_eval_transform)

    # 2. Instantiate datasets independently for train and eval transforms
    actual_train_transform = two_views_train_transform if apply_augmentation else two_views_eval_transform
    
    train_mnist = datasets.MNIST(root=data_dir, train=True, download=True, transform=actual_train_transform)
    val_mnist = datasets.MNIST(root=data_dir, train=True, download=True, transform=two_views_eval_transform)
    test_dataset = datasets.MNIST(root=data_dir, train=False, download=True, transform=base_eval_transform)

    # 3. Calculate split sizes
    num_train = len(train_mnist)
    train_size = int((1 - val_split) * num_train)
    val_size = num_train - train_size
    
   # 4. Generate shuffled indices for train and validation using random_split and a fixed seed
    generator = torch.Generator().manual_seed(42)
    train_subset, val_subset = random_split(
        range(num_train), 
        [train_size, val_size], 
        generator=generator
    )
    
    # Extract the actual lists of indices from the Subset objects
    train_indices = train_subset.indices
    val_indices = val_subset.indices
    
    # Slice the indices if we are in debug mode for a smaller dataset
    if debug_subset_size is not None:
        debug_train = int(debug_subset_size * (1 - val_split))
        debug_val = debug_subset_size - debug_train
        
        train_indices = train_indices[:debug_train]
        val_indices = val_indices[:debug_val]
    
    # 5. Create final Subsets pointing to the correct underlying dataset
    train_dataset = torch.utils.data.Subset(train_mnist, train_indices)
    val_dataset = torch.utils.data.Subset(val_mnist, val_indices)

    # 6. Wrap the datasets in DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    
    return train_loader, val_loader, test_loader