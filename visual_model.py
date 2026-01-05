import torch
import torch.nn as nn

class VisualEncoder(nn.Module):
    def __init__(self, hidden_dim=512):
        super(VisualEncoder, self).__init__()
        
        # Front-end: 3D CNN (Spatiotemporal features)
        # Input shape: (Batch, Channel=1, Depth/Time, Height, Width)
        self.conv3d = nn.Sequential(
            nn.Conv3d(1, 32, kernel_size=(3, 5, 5), stride=(1, 2, 2), padding=(1, 2, 2)),
            nn.BatchNorm3d(32),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2)),
            
            nn.Conv3d(32, 64, kernel_size=(3, 5, 5), stride=(1, 1, 1), padding=(1, 2, 2)),
            nn.BatchNorm3d(64),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2)),
            
            nn.Conv3d(64, 96, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1)),
            nn.BatchNorm3d(96),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))
        )
        
        # Back-end: Bi-directional LSTM (Temporal modeling)
        # We assume the flattened feature size after Conv3D is 96*H*W roughly map to 512
        self.lstm_input_size = 96 * 3 * 6 # Approximate depending on crop size (usually 50x100)
        self.lstm = nn.LSTM(self.lstm_input_size, 256, bidirectional=True, batch_first=True)
        
        # Projector: Ensures output is exactly the same size as hers (512)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        # Input x: (Batch, Channel, Time, Height, Width)
        b = x.size(0)
        
        # Pass through 3D CNN
        x = self.conv3d(x) # Output: (B, C, T, H, W)
        
        # Reshape for LSTM: (Batch, Time, Features)
        x = x.permute(0, 2, 1, 3, 4).contiguous()
        x = x.view(b, x.size(1), -1) 
        
        # Pass through LSTM
        self.lstm.flatten_parameters()
        x, _ = self.lstm(x) # Output: (Batch, Time, 512)
        
        x = self.dropout(x)
        return x