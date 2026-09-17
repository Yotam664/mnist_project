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

def train_model(model, train_loader, val_loader, optimizer, criterion, device, num_epochs=100):
    """
    Trains the model for a specified number of epochs, performing validation after each epoch.

    Args:
        model (torch.nn.Module): The model to be trained.
        train_loader (DataLoader): DataLoader for the training dataset.
        val_loader (DataLoader): DataLoader for the validation dataset.
        optimizer (torch.optim.Optimizer): Optimizer for updating model parameters.
        criterion (callable): Loss function to compute the loss.
        device (torch.device): Device to perform computations on (CPU or GPU).
        num_epochs (int): Number of epochs to train the model.
    """
    for epoch in range(num_epochs):
        train_loss = one_training_epoch(model, train_loader, optimizer, criterion, device)
        val_loss = one_validation_epoch(model, val_loader, criterion, device)
        
        print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}")