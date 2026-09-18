import torch

###################################################
##################Convolution Model################
###################################################

class ConvBlock(torch.nn.Module):
    """
    A convolutional block consisting of a convolutional layer, layer normalization, and GELU activation.
    """

    def __init__(self, dim , kernel_size=3, stride=1, padding=1):
        super(ConvBlock, self).__init__()
        self.conv = torch.nn.Conv2d(dim, dim, kernel_size=kernel_size, stride=stride, padding=padding)
        self.layerNorm = torch.nn.LayerNorm(dim)

    def forward(self, x):
        y = self.conv(x)
        y = y.permute(0, 2, 3, 1)
        y = self.layerNorm(y)
        y = y.permute(0, 3, 1, 2)
        y = torch.nn.functional.gelu(y)
        return y+x


class Encoder(torch.nn.Module):
    """
    Encoder model for MNIST images. 
    It consists of convolutional layers, layer normalization, and global average pooling.
    """

    def __init__(self, base_channels):
        super(Encoder, self).__init__()
        self.stem = torch.nn.Conv2d(1, base_channels, kernel_size=3, stride=1, padding=1) #First layer of CNN  
        self.layerNorm = torch.nn.LayerNorm(base_channels) #Layer normalization for the first layer
        self.block1 = ConvBlock(dim=base_channels) #First convolutional block
        self.downsample = torch.nn.Conv2d(base_channels, base_channels * 2, kernel_size=3, stride=2, padding=1) #Downsampling layer to reduce spatial dimensions
        self.layerNorm2 = torch.nn.LayerNorm(base_channels * 2) #Layer normalization for the second layer
        self.block2 = ConvBlock(dim=base_channels * 2) #Second convolutional block
        self.globalAvgPool = torch.nn.AdaptiveAvgPool2d((1, 1)) #Global average pooling layer - 1D vector output

    def forward(self, x):
        """
        Forward pass through the encoder.
        """
        x = self.stem(x)
        x = x.permute(0, 2, 3, 1) # Permute for LayerNorm
        x = self.layerNorm(x)
        x = x.permute(0, 3, 1, 2) # Permute back
        x = torch.nn.functional.gelu(x)
        
        # --- First Block ---
        x = self.block1(x)
        
        # --- Downsample Phase ---
        x = self.downsample(x)
        x = x.permute(0, 2, 3, 1) # Permute for LayerNorm2
        x = self.layerNorm2(x)
        x = x.permute(0, 3, 1, 2) # Permute back
        x = torch.nn.functional.gelu(x)
    
        # --- Second Block ---
        x = self.block2(x)

        # --- Output Phase ---
        x = self.globalAvgPool(x)
        x = torch.flatten(x, 1)
        return x

class Expander(torch.nn.Module):
    """
    Expander model that takes the output of the encoder and expands it to a 10-class output.
    """

    def __init__(self,encoder_output_dim, expander_output_dim):
        super(Expander, self).__init__()
        self.fc1 = torch.nn.Linear(encoder_output_dim, expander_output_dim)
        self.layerNorm = torch.nn.LayerNorm(expander_output_dim)
        self.activation = torch.nn.GELU()
        self.fc2 = torch.nn.Linear(expander_output_dim, expander_output_dim)
        

    def forward(self, x):
        """
        Forward pass through the expander.
        """
        x = self.fc1(x)
        x = self.layerNorm(x)
        x = self.activation(x)
        x = self.fc2(x)
        return x

class CNNModel(torch.nn.Module):
    """
    A wrapper model for Self-Supervised Learning that combines an Encoder 
    and an MLPExpander into a single unified architecture.
    """
    def __init__(self, base_channels, expander_dim):
        """
        Initializes the CNNModel with an encoder and an expander.
        """
        super(CNNModel, self).__init__()
        self.encoder = Encoder(base_channels)
        self.expander = Expander(base_channels * 2, expander_dim)

    def forward(self, x):
        """
        Passes the raw image through the encoder to get the 64-dim representation,
        then passes it through the expander to get the final 2048-dim embedding.
        """
        representation = self.encoder(x)
        embedding = self.expander(representation)
        return embedding




##########################################################
##################Visual Transformer Model################
##########################################################

class ViTModel(torch.nn.Module):
    """
    A Vision Transformer (ViT) model for image classification.
    """

    def __init__(self, patch_size, projection_dim, number_of_heads, number_of_layers, expander_dim):
        super(ViTModel, self).__init__()
        self.patch_size = patch_size
        self.projection_dim = projection_dim
        self.position_embedding = torch.nn.Parameter(torch.randn(1, (28 // patch_size) ** 2, projection_dim))  # Assuming MNIST images of size 28x28
        self.patches_embs = torch.nn.Conv2d(1, projection_dim, kernel_size=patch_size, stride=patch_size)  # Assuming grayscale images
        self.number_of_heads = number_of_heads
        self.number_of_layers = number_of_layers
        self.expander_dim = expander_dim
        self.transformer_layers = torch.nn.ModuleList([
            torch.nn.TransformerEncoderLayer(d_model=projection_dim, nhead=number_of_heads, dim_feedforward=expander_dim,batch_first=True, activation='gelu')
            for _ in range(number_of_layers)
        ])
        self.expander = Expander(projection_dim, expander_dim)

    def forward(self, x):
        """
        Forward pass through the ViT model.
        """
        x = self.patches_embs(x)  # Convert image to patches and project to embedding dimension
        x = x.flatten(2).transpose(1, 2)  # Flatten patches and prepare for transformer input
        x = x + self.position_embedding  # Add position embedding
        for layer in self.transformer_layers:
            x = layer(x)  # Pass through each transformer layer
        x = torch.mean(x, dim=1)  # Global average pooling over the sequence dimension
        x = self.expander(x)  # Pass through the expander to get the final embedding
        return x  # Replace with actual output after processing through ViT layers