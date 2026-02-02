import torch
import torch.nn as nn
from audio_model import AudioEncoder  # Her file
from visual_model import VisualEncoder # Your file

class AVFusionModel(nn.Module):
    def __init__(self, num_classes=30):
        super(AVFusionModel, self).__init__()
        
        self.audio_net = AudioEncoder()
        self.visual_net = VisualEncoder()
        
        # Remove her classifier (we only want the features)
        self.audio_net.classifier = nn.Identity()
        
        # Fusion Layer: Attention Mechanism
        # Why? Audio might be faster than Video. We need to align them.
        self.fusion_layer = nn.MultiheadAttention(embed_dim=512, num_heads=4)
        
        self.final_classifier = nn.Linear(512, num_classes)

    def forward(self, audio_input, video_input):
        # 1. Get Audio Features
        # Output: (Batch, Audio_Time, 512)
        audio_emb = self.audio_net(audio_input)
        
        # 2. Get Visual Features
        # Output: (Batch, Video_Time, 512)
        video_emb = self.visual_net(video_input)
        
        # 3. Fusion (Cross-Modal Attention)
        # We treat Audio as "Query" and Video as "Key/Value"
        # This helps the model use video to "fill in the gaps" of audio
        
        # Note: Permute for MultiheadAttention which expects (Time, Batch, Feat)
        audio_emb = audio_emb.permute(1, 0, 2)
        video_emb = video_emb.permute(1, 0, 2)
        
        # Attention: Q=Audio, K=Video, V=Video
        fused_features, _ = self.fusion_layer(audio_emb, video_emb, video_emb)
        
        # Permute back to (Batch, Time, Feat)
        fused_features = fused_features.permute(1, 0, 2)
        
        # 4. Final Prediction
        output = self.final_classifier(fused_features)
        return output