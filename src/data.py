import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
from torchvision.transforms import v2

def get_dataloaders(data_dir='./data', batch_size=64, val_split=0.15, num_workers=2, apply_augmentation=True, rotation_angle=15, translate_fraction=0.1,blur_kernel_size=3, blur_possibility=0.5):
    """
    Downloads MNIST data, splits it into Train, Validation, and Test sets,
    and returns DataLoaders for each.
    """
    # 1. Define preprocessing transforms
    train_transform = transforms.Compose([
        transforms.RandomRotation(degrees=rotation_angle),
        transforms.RandomAffine(degrees=0, translate=(translate_fraction, translate_fraction)),
        transforms.RandomApply([transforms.GaussianBlur(kernel_size=blur_kernel_size)], p=blur_possibility),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    # Define evaluation transforms (no augmentation), for both validation and test sets
    eval_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

   # 2. Instantiate datasets independently for train and eval transforms
    actual_train_transform = train_transform if apply_augmentation else eval_transform
    train_mnist = datasets.MNIST(root=data_dir, train=True, download=True, transform=actual_train_transform)
    val_mnist = datasets.MNIST(root=data_dir, train=True, download=True, transform=eval_transform)
    test_dataset = datasets.MNIST(root=data_dir, train=False, download=True, transform=eval_transform)

    # 3. Calculate split sizes
    num_train = len(train_mnist)
    train_size = int((1 - val_split) * num_train)
    val_size = num_train - train_size
    
    # 4. Generate shuffled indices using random_split and a fixed seed
    generator = torch.Generator().manual_seed(42)
    train_indices, val_indices = random_split(
        range(num_train), 
        [train_size, val_size], 
        generator=generator
    )
    
    # 5. Create final Subsets pointing to the correct underlying dataset
    train_dataset = torch.utils.data.Subset(train_mnist, train_indices)
    val_dataset = torch.utils.data.Subset(val_mnist, val_indices)

    # 6. Wrap the datasets in DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    return train_loader, val_loader, test_loader



# Test block - runs only if the file is executed directly
if __name__ == "__main__":
    print("Downloading and splitting the dataset...")
    train_dl, val_dl, test_dl = get_dataloaders()
    
    print(f"Train dataset size: {len(train_dl.dataset)} images")
    print(f"Validation dataset size: {len(val_dl.dataset)} images")
    print(f"Test dataset size: {len(test_dl.dataset)} images")
    print("Success! Data is ready in the './data' directory.")