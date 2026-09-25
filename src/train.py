import torch
import hydra
from src.data import get_dataloaders
from src.models import CNNModel, ViTModel
from src.utils import VISLoss, plot_loss_curves



def one_training_epoch(model ,train_loader, optimizer, criterion, device):
    """
    Performs one epoch of training for the given model.

    Args:
        model (torch.nn.Module): The model to be trained.
        train_loader (DataLoader): DataLoader for the training dataset.
        optimizer (torch.optim.Optimizer): Optimizer for updating model parameters.
        criterion (callable): Loss function to compute the loss.
        device (torch.device): Device to perform computations on (CPU or GPU).
    Returns:
        float: Average training loss for the epoch.
    """
    model.train()
    current_loss = 0.0
    for batch_idx, (data, _) in enumerate(train_loader):
        view1, view2 = data  # Unpack the two views
        view1 = view1.to(device)
        view2 = view2.to(device)
        optimizer.zero_grad()
        z1 = model(view1)
        z2 = model(view2)
        loss = criterion(z1, z2)  # Using the model's output as both inputs for the loss
        loss.backward()
        optimizer.step()
        current_loss += loss.item()
    return current_loss / len(train_loader)

def one_validation_epoch(model, val_loader, criterion, device):
    """
    Performs one epoch of validation for the given model.

    Args:
        model (torch.nn.Module): The model to be validated.
        val_loader (DataLoader): DataLoader for the validation dataset.
        criterion (callable): Loss function to compute the loss.
        device (torch.device): Device to perform computations on (CPU or GPU).
    Returns:
        float: Average validation loss for the epoch.
    """
    model.eval()
    current_loss = 0.0
    with torch.no_grad(): #Disables gradient computation for validation
        for batch_idx, (data, _) in enumerate(val_loader):
            view1, view2 = data  # Unpack the two views
            view1 = view1.to(device)
            view2 = view2.to(device)
            z1 = model(view1)
            z2 = model(view2)
            loss = criterion(z1, z2)  # Using the model's output as both inputs for the loss
            current_loss += loss.item()
    return current_loss / len(val_loader)

def train_model(model, train_loader, val_loader, optimizer, scheduler, criterion, device, num_epochs=100):
    """
    Trains the model for a specified number of epochs, performing validation after each epoch.

    Args:
        model (torch.nn.Module): The model to be trained.
        train_loader (DataLoader): DataLoader for the training dataset.
        val_loader (DataLoader): DataLoader for the validation dataset.
        optimizer (torch.optim.Optimizer): Optimizer for updating model parameters.
        scheduler (torch.optim.lr_scheduler.LambdaLR): Learning rate scheduler.
        criterion (callable): Loss function to compute the loss.
        device (torch.device): Device to perform computations on (CPU or GPU).
        num_epochs (int): Number of epochs to train the model.
    """
    loss_train_list = []
    loss_val_list = []
    best_val_loss = float('inf')  # Initialize best validation loss to infinity
    learning_rate = []
    for epoch in range(num_epochs):
        train_loss = one_training_epoch(model, train_loader, optimizer, criterion, device)
        val_loss = one_validation_epoch(model, val_loader, criterion, device)
        current_lr = scheduler.get_last_lr()[0]
        learning_rate.append(current_lr)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            checkpoint = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_val_loss': best_val_loss
            }
            # Save in the current Hydra run directory
            torch.save(checkpoint, "best_checkpoint.pth")
            print(f"Best checkpoint saved at epoch {epoch+1} with validation loss: {best_val_loss:.4f}")
        latest_checkpoint = {
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'best_val_loss': best_val_loss
        }
        torch.save(latest_checkpoint, "latest_checkpoint.pth")
        loss_train_list.append(train_loss)
        loss_val_list.append(val_loss)
        scheduler.step()  # Update the learning rate based on the scheduler
        print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}")
    return loss_train_list, loss_val_list, learning_rate


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg):
    """
    Main function to set up and run the training process.

    Args:
        cfg (DictConfig): Configuration object containing all parameters.
    """
    # Set random seed for reproducibility
    torch.manual_seed(cfg.seed)
    
    # Determine device to use (GPU if available, else CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Get data loaders for training and validation
    train_loader, val_loader, _ = get_dataloaders(
        data_dir=cfg.data_dir,
        debug_subset_size=cfg.debug_subset_size,
        batch_size=cfg.batch_size,
        apply_augmentation=cfg.augmentation.apply_augmentation,
        rotation_angle=cfg.augmentation.rotation_angle,
        translate_fraction=cfg.augmentation.translate_fraction,
        blur_kernel_size=cfg.augmentation.blur_kernel_size,
        blur_possibility=cfg.augmentation.blur_possibility
    )
    
    # Initialize the model and move it to the appropriate device
    if cfg.model.name == "CNNModel":
        model = CNNModel(
            base_channels=cfg.model.base_channels, 
            expander_dim=cfg.model.expander_dim
        ).to(device)
    elif cfg.model.name == "ViTModel":
        model = ViTModel(
            patch_size=cfg.model.patch_size,
            projection_dim=cfg.model.projection_dim,
            number_of_heads=cfg.model.number_of_heads,
            number_of_layers=cfg.model.number_of_layers,
            expander_dim=cfg.model.expander_dim
        ).to(device)
    else:
        raise ValueError(f"Unknown model name: {cfg.model.name}")
    
    if cfg.optimizer.name == "adamW":
        optimizer = torch.optim.AdamW(
            model.parameters(), 
            lr=cfg.optimizer.learning_rate, 
            weight_decay=cfg.optimizer.weight_decay
        )
    elif cfg.optimizer.name == "sgd":
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=cfg.optimizer.learning_rate
        )
    else:
        raise ValueError(f"Unknown optimizer name: {cfg.optimizer.name}")
    
    if cfg.scheduler.name == "stepLR":
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, 
            step_size=cfg.scheduler.step_size, 
            gamma=cfg.scheduler.gamma
        )
    elif cfg.scheduler.name == "cosine_warmup":
        warmup_epochs = 5
        warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer, 
            start_factor=0.01, 
            total_iters=warmup_epochs
        )
        cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, 
            T_max=cfg.num_epochs - warmup_epochs
        )
        scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer, 
            schedulers=[warmup_scheduler, cosine_scheduler], 
            milestones=[warmup_epochs]
        )
    else:
        raise ValueError(f"Unknown scheduler name: {cfg.scheduler.name}")

    criterion = VISLoss(
        lambda_inv=cfg.loss.lambda_inv, 
        lambda_var=cfg.loss.lambda_var, 
        lambda_shape=cfg.loss.lambda_shape
    )

    # Start the training process
    train_loss, val_loss, learning_rate = train_model(
        model=model, 
        train_loader=train_loader, 
        val_loader=val_loader, 
        optimizer=optimizer, 
        scheduler=scheduler, 
        criterion=criterion, 
        device=device, 
        num_epochs=cfg.num_epochs
    )

    # Plot the loss curves after training
    plot_loss_curves(train_loss, val_loss, learning_rate, save_path="loss_curves.png")


if __name__ == "__main__":
    main()