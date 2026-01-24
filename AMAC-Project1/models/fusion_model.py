import torch
import torch.nn as nn

# --- IMPORTS ---
from .video_model import LipNet 
from .audio_model import AudioEncoder 

class AVFusionModel(nn.Module):
    def __init__(self, num_classes=28, hidden_dim=512):
        super(AVFusionModel, self).__init__()
        
        # --- A. VIDEO ENCODER (LipNet) ---
        self.video_encoder = LipNet()
        # Bypass final layer to get 512 features
        # (LipNet.fc was Linear(512, 28))
        self.video_encoder.fc = nn.Identity()
        
        # --- B. AUDIO ENCODER ---
        self.audio_encoder = AudioEncoder()
        # Bypass final layer to get 512 features
        # (AudioEncoder.classifier was Linear(512, 30))
        self.audio_encoder.classifier = nn.Identity()

        # --- C. ATTENTION FUSION ---
        # Input: 512 (Audio) + 512 (Video) = 1024
        self.attention = nn.MultiheadAttention(embed_dim=1024, num_heads=4, batch_first=True)
        
        # --- D. FUSION LSTM ---
        self.fusion_lstm = nn.LSTM(input_size=1024, hidden_size=hidden_dim, 
                                   num_layers=2, batch_first=True, bidirectional=True)
        
        # --- E. FINAL CLASSIFIER ---
        self.classifier = nn.Linear(hidden_dim * 2, num_classes)
        self.dropout = nn.Dropout(0.3)

    def forward(self, audio_input, video_input):
        # audio_input expected: (Batch, Time, Freq)
        # video_input expected: (Batch, Time, Channel, H, W)
        
        # --- 1. PREPARE AUDIO ---
        # AudioEncoder expects (Batch, 1, Freq, Time)
        # We need to reshape: (B, T, F) -> (B, 1, F, T)
        if audio_input.dim() == 3:
            # Transpose Time and Freq, then add Channel dim
            audio_reshaped = audio_input.transpose(1, 2).unsqueeze(1)
        else:
            audio_reshaped = audio_input

        # Get Features (Batch, Time, 512)
        a_features = self.audio_encoder(audio_reshaped)
        
        # --- 2. PREPARE VIDEO ---
        # Get Features (Batch, Time, 512)
        v_features = self.video_encoder(video_input)
        
        # --- 3. ALIGN TIME STEPS ---
        # Pooling in CNNs might shrink time dimensions differently
        min_len = min(a_features.size(1), v_features.size(1))
        a_features = a_features[:, :min_len, :]
        v_features = v_features[:, :min_len, :]
        
        # --- 4. FUSION & ATTENTION ---
        # Concatenate (Batch, Time, 1024)
        combined = torch.cat((a_features, v_features), dim=2)
        
        # Attention Mechanism
        attn_out, _ = self.attention(combined, combined, combined)
        fused = combined + attn_out # Residual connection
        
        # --- 5. CLASSIFICATION ---
        lstm_out, _ = self.fusion_lstm(fused)
        output = self.classifier(self.dropout(lstm_out))
        
        return output

# --- TEST BLOCK ---
if __name__ == "__main__":
    print("--- Testing Fusion with Corrected Input ---")
    try:
        model = AVFusionModel()
        
        # Dummy Video: (Batch, Time, Channel, H, W)
        # Standard LipNet shape
        dummy_video = torch.randn(2, 50, 1, 64, 64) 
        
        # Dummy Audio: (Batch, Time, Freq)
        # Standard Mel Spectrogram shape
        dummy_audio = torch.randn(2, 50, 80)       
        
        out = model(dummy_audio, dummy_video)
        print(f"✅ Success! Fusion Output Shape: {out.shape}")
        print("(Expected: [2, Time, 28])")
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()