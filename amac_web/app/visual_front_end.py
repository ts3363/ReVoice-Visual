import torch
import torch.nn as nn

class VisualFrontend(nn.Module):
    def __init__(self, output_dim=512):
        super(VisualFrontend, self).__init__()
        
        # Input shape: [Batch, Channel(1), Time, Height(64), Width(64)]
        # We use a simplified LipNet-style architecture
        
        self.frontend3D = nn.Sequential(
            # Layer 1: Learn rudimentary features (edges, curves)
            nn.Conv3d(1, 32, kernel_size=(3, 5, 5), stride=(1, 2, 2), padding=(1, 2, 2)),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2)),
            
            # Layer 2: Learn complex lip shapes
            nn.Conv3d(32, 64, kernel_size=(3, 5, 5), stride=(1, 1, 1), padding=(1, 2, 2)),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2)),
            
            # Layer 3: Deep temporal features
            nn.Conv3d(64, 96, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1)),
            nn.BatchNorm3d(96),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))
        )
        
        # The 3D CNN reduces the 64x64 image to a small feature map.
        # We need to flatten it to feed into the Conformer/Transformer.
        # Calculated size after layers: 96 channels * 2 height * 2 width = 384
        self.fc = nn.Linear(96 * 4 * 4, output_dim) 
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        # x input: [Batch, Time, 1, 64, 64]
        # Conv3d expects: [Batch, Channel, Depth(Time), Height, Width]
        
        x = x.transpose(1, 2) # Swap Time and Channel -> [B, 1, T, 64, 64]
        
        x = self.frontend3D(x) # -> [B, 96, T, 4, 4]
        
        # We need to get back to [Batch, Time, Features] for the Transformer
        x = x.permute(0, 2, 1, 3, 4).contiguous() # -> [B, T, 96, 4, 4]
        
        batch, time, channels, h, w = x.size()
        x = x.view(batch, time, -1) # Flatten spatial dims -> [B, T, 96*4*4]
        
        x = self.fc(x) # Project to model dimension (e.g., 512)
        x = self.dropout(x)
        
        return x

if __name__ == "__main__":
    # Test with a fake video batch
    # Batch=2, Time=75 frames, Channel=1, H=64, W=64
    dummy_video = torch.randn(2, 75, 1, 64, 64)
    model = VisualFrontend(output_dim=256) # Matching standard Conformer dim
    output = model(dummy_video)
    print(f"Input: {dummy_video.shape}")
    print(f"Output: {output.shape} (Should be [2, 75, 256])")