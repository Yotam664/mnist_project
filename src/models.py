import torch

class ConvBlock(torch.nn.Module):
    """
    A convolutional block consisting of a convolutional layer, layer normalization, and GELU activation.
    """

    def __init__(self, , dim , kernel_size=3, stride=1, padding=1):
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

    def __init__(self):
        super(Encoder, self).__init__()
        self.stem = torch.nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1) #First layer of CNN  
        self.layerNorm = torch.nn.LayerNorm(32) #Layer normalization for the first layer
        self.block1 = ConvBlock(dim=32) #First convolutional block
        self.downsample = torch.nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1) #Downsampling layer to reduce spatial dimensions
        self.layerNorm2 = torch.nn.LayerNorm(64) #Layer normalization for the second layer
        self.block2 = ConvBlock(dim=64)
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

    def __init__(self):
        super(Expander, self).__init__()
        self.fc1 = torch.nn.Linear(64, 2048)
        self.layerNorm = torch.nn.LayerNorm(2048)
        self.activation = torch.nn.GELU()
        self.fc2 = torch.nn.Linear(2048, 2048)
        

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
    def __init__(self):
        super(CNNModel, self).__init__()
        self.encoder = Encoder()
        self.expander = Expander()

    def forward(self, x):
        """
        Passes the raw image through the encoder to get the 64-dim representation,
        then passes it through the expander to get the final 2048-dim embedding.
        """
        representation = self.encoder(x)
        embedding = self.expander(representation)
        return embedding