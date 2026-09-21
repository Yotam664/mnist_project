import matplotlib.pyplot as plt
import numpy as np
import torch
from src.data import get_dataloaders

def plot_images(images, labels, predictions=None, num_images=10 , save_path="../results/sample_batch.png"):
    """
    Plots a grid of images with their corresponding labels and optional predictions.
    
    Parameters:
    - images: A batch of images (tensor or numpy array).
    - labels: True labels corresponding to the images.
    - predictions: Optional predicted labels for the images.
    - num_images: Number of images to display (default is 10).
    """
    plt.figure(figsize=(15, 5))
    
    for i in range(num_images):
        plt.subplot(2, num_images // 2, i + 1)
        img = images[i].numpy().squeeze()
        img =(img*0.3081) + 0.1307  # Unnormalize the image
        img = np.clip(img, 0, 1)  # Ensure pixel values are in [0, 1]
        plt.imshow(img, cmap='gray')
        title = f"Label: {labels[i]}"
        if predictions is not None:
            title += f"\nPred: {predictions[i]}"
        plt.title(title)
        plt.axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"Plot saved successfully to {save_path}")

if __name__ == "__main__":
    print("Fetching a batch to plot...")
    train_loader, _, _ = get_dataloaders(batch_size=10)
    
    (view1, view2), labels = next(iter(train_loader))
    
    plot_images(view1, labels, save_path="../results/sample_batch_view1.png")
    plot_images(view2, labels, save_path="../results/sample_batch_view2.png")

class VISLoss(torch.nn.Module):

    def __init__(self,lambda_inv,lambda_var, lambda_shape):
        super(VISLoss, self).__init__()
        self.lambda_inv = lambda_inv
        self.lambda_var = lambda_var
        self.lambda_shape = lambda_shape

    def computeInvarianceLoss(self, z1, z2):
        """
        Computes the invariance loss between two sets of features.
        """
        return torch.nn.functional.mse_loss(z1, z2)
    
    def computeVarianceLoss(self, z):
        """
        Computes the variance loss to ensure that the features have sufficient variance.
        """
        std = torch.sqrt(z.var(dim=0) + 1e-4)
        return torch.mean(torch.relu(1 - std))

    def computeShapeLoss(self, z):
        """
        Computes the shape loss to ensure that the features are not degenerate.
        """
        z_centered = z - torch.mean(z, dim=0)
        std_of_z = torch.std(z_centered, dim=0) + 1e-4
        z_norm = z_centered / (std_of_z.detach() + 1e-4)
        projection_matrix = torch.rand(2048,64)
        projection_matrix = torch.nn.functional.normalize(projection_matrix, p=2, dim=0, eps=1e-4)
        projection_matrix = projection_matrix.to(z_norm.device)
        projected_z = torch.matmul(z_norm, projection_matrix)
        sorted_projected_z, _ = torch.sort(projected_z, dim=0)
        batch_size = z.size(0)
        quantiles = (torch.arange(1, batch_size + 1, device=z.device) - 0.5) / batch_size
        target = torch.distributions.Normal(0, 1).icdf(quantiles)
        target = target.unsqueeze(1).expand_as(sorted_projected_z)
        return torch.mean((sorted_projected_z - target) ** 2)

    def forward(self, z1, z2):
        """
        Computes the total loss as a weighted sum of invariance, variance, and shape losses.
        """
        invariance_loss = self.computeInvarianceLoss(z1, z2)
        variance_loss = self.computeVarianceLoss(z1) + self.computeVarianceLoss(z2)
        shape_loss = self.computeShapeLoss(z1) + self.computeShapeLoss(z2)
        
        total_loss = (self.lambda_inv * invariance_loss +
                      self.lambda_var * variance_loss +
                      self.lambda_shape * shape_loss)
        
        return total_loss



def plot_loss_curves(train_loss, val_loss, learning_rate, save_path="../results/loss_curves.png"):
    """
    Plots the training and validation loss curves over epochs.
    
    Parameters:
    - train_loss: List of training loss values.
    - val_loss: List of validation loss values.
    - save_path: Path to save the plot image.
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    axes[0].plot(train_loss, label='Training Loss', color='blue')
    axes[0].plot(val_loss, label='Validation Loss', color='orange')
    axes[0].set_title('Loss Curves')
    axes[0].set_xlabel('Epochs')
    axes[0].set_ylabel('Loss')
    axes[0].legend()
    axes[0].grid(True)
    
    axes[1].plot(learning_rate, label='Learning Rate', color='green', linestyle='--')
    axes[1].set_title('Learning Rate Schedule')
    axes[1].set_xlabel('Epochs')
    axes[1].set_ylabel('Learning Rate')
    axes[1].legend()
    axes[1].grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"Metrics saved successfully to {save_path}")
    plt.close()