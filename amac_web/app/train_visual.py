import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

# --- 1. IMPORTS ---
try:
    from grid_dataset import GridDataset
except ImportError:
    print("🚨 Error: grid_dataset.py not found.")
    exit()

try:
    from visual_front_end import VisualFrontend
except ImportError:
    from model import VideoModel as VisualFrontend 

# --- CONFIG ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 2
LEARNING_RATE = 3e-4
EPOCHS = 50
TARGET_FRAMES = 75  # The strict rule: Every video MUST be this long
SAVE_DIR = "checkpoints_visual"
MANIFEST_PATH = os.path.join("data", "s1", "grid_manifest.csv")
VOCAB_PATH = "vocab.json" 

os.makedirs(SAVE_DIR, exist_ok=True)

# --- 2. LOAD VOCABULARY ---
if os.path.exists(VOCAB_PATH):
    with open(VOCAB_PATH, "r") as f:
        vocab_list = json.load(f)
        word2id = {word: idx for idx, word in enumerate(vocab_list)}
    
    NUM_CLASSES = len(vocab_list)
    print(f"[+] Loaded Vocabulary: {NUM_CLASSES} words.")
else:
    print("🚨 Critical Error: vocab.json not found! Run create_vocab.py first.")
    exit()

# --- 3. SMART COLLATE FUNCTION (The Fix) ---
def ensure_fixed_length(video, target_frames=75):
    """
    Forces the video tensor to have exactly 'target_frames' length.
    Shape Input: [Time, Channel, H, W]
    """
    current_frames = video.shape[0]
    
    if current_frames == target_frames:
        return video
    
    if current_frames > target_frames:
        # Too long? Crop the end.
        return video[:target_frames]
    
    if current_frames < target_frames:
        # Too short? Pad by repeating the last frame.
        diff = target_frames - current_frames
        last_frame = video[-1].unsqueeze(0) 
        padding = last_frame.repeat(diff, 1, 1, 1)
        return torch.cat([video, padding], dim=0)

def visual_collate_fn(batch):
    videos = []
    labels = []
    
    for item in batch:
        video_item = None
        label_raw = None
        
        # Unpack
        if len(item) == 3: video_item, label_raw = item[0], item[2]
        elif len(item) == 2: video_item, label_raw = item[0], item[1]
            
        if video_item is not None and label_raw is not None:
            # 1. FIX THE VIDEO LENGTH
            video_item = ensure_fixed_length(video_item, TARGET_FRAMES)

            # 2. FIX THE LABEL
            if isinstance(label_raw, torch.Tensor) or isinstance(label_raw, int):
                label_id = label_raw
            elif isinstance(label_raw, str):
                if label_raw in word2id:
                    label_id = word2id[label_raw]
                else:
                    first_word = label_raw.split(" ")[0]
                    label_id = word2id.get(first_word, 0)
            
            videos.append(video_item)
            if isinstance(label_id, torch.Tensor):
                labels.append(label_id)
            else:
                labels.append(torch.tensor(label_id, dtype=torch.long))
            
    return torch.stack(videos), torch.stack(labels)

# --- 4. THE WRAPPER MODEL ---
class VisualClassifier(nn.Module):
    def __init__(self, num_classes):
        super(VisualClassifier, self).__init__()
        try:
            self.frontend = VisualFrontend() 
        except:
            self.frontend = VisualFrontend(relu_type='prelu') 

        self.flat_dim = 512 
        # Auto-detect shape logic
        with torch.no_grad():
            dummy = torch.randn(1, 75, 1, 64, 64)
            try:
                features = self.frontend(dummy)
                self.flat_dim = features.view(1, -1).shape[1]
                self.input_needs_permute = False
            except:
                dummy_perm = torch.randn(1, 1, 75, 64, 64)
                features = self.frontend(dummy_perm)
                self.flat_dim = features.view(1, -1).shape[1]
                self.input_needs_permute = True

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(self.flat_dim, num_classes) 
        )

    def forward(self, x):
        if getattr(self, 'input_needs_permute', False):
             if x.shape[2] == 1: 
                 x = x.permute(0, 2, 1, 3, 4) 
        features = self.frontend(x)
        out = self.classifier(features)
        return out

def train():
    print(f"[*] Training on: {DEVICE}")
    
    if not os.path.exists(MANIFEST_PATH):
        print(f"🚨 Manifest not found.")
        return

    dataset = GridDataset(manifest_path=MANIFEST_PATH)
    
    loader = DataLoader(
        dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=True, 
        drop_last=True,
        collate_fn=visual_collate_fn # Use the "Fixing" function
    )
    
    print(f"[+] Data Loaded: {len(dataset)} samples")
    print(f"[+] Target Video Length: {TARGET_FRAMES} frames")

    model = VisualClassifier(num_classes=NUM_CLASSES).to(DEVICE)
    model.train()

    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()

    print(f"[*] Starting Training for {EPOCHS} epochs...")
    for epoch in range(EPOCHS):
        loop = tqdm(loader, desc=f"Epoch {epoch+1}/{EPOCHS}")
        total_loss = 0
        
        for video, label in loop:
            video = video.to(DEVICE)
            label = label.to(DEVICE)

            optimizer.zero_grad()
            predictions = model(video)
            loss = criterion(predictions, label)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            loop.set_postfix(loss=f"{loss.item():.4f}")

        avg_loss = total_loss / len(loader)
        print(f"Epoch {epoch+1} Complete. Avg Loss: {avg_loss:.4f}")
        torch.save(model.state_dict(), f"{SAVE_DIR}/visual_epoch_{epoch+1}.pt")
        print(f"[+] Saved checkpoint.")

if __name__ == "__main__":
    train()