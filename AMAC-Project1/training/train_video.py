import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from utils.lip_dataset import LipDataset
from models.video_model import LipNet

if __name__ == "__main__":

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = LipDataset()
    loader = DataLoader(
    dataset,
    batch_size=64,
    shuffle=True,
    num_workers=0,   # ← IMPORTANT FOR WINDOWS
    pin_memory=True,
    drop_last=True
)


    model = LipNet().to(device)
    criterion = nn.CTCLoss(blank=0)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    scaler = torch.amp.GradScaler("cuda")

    print("🔥 Training Lip Reading Model on RTX 4060...")

    for epoch in range(80):
        model.train()
        total_loss = 0

        for videos, texts in loader:

            videos = videos.to(device, non_blocking=True)

            # ----- BUILD CTC TARGETS -----
            labels = []
            target_lengths = []

            for txt in texts:
                txt = str(txt).lower()
                seq = [(ord(c)-96 if c.isalpha() else 27) for c in txt if c.isalpha() or c==" "]
                labels.append(torch.tensor(seq, dtype=torch.long))
                target_lengths.append(len(seq))

            target_lengths = torch.tensor(target_lengths, dtype=torch.long, device=device)
            targets = torch.cat(labels).to(device)

            # ----- FORWARD + LOSS -----
            with torch.autocast("cuda"):
                outputs = model(videos)           # [B,T,C]
                outputs = outputs.permute(1,0,2)  # → [T,B,C]

                input_lengths = torch.full(
                    size=(outputs.size(1),),
                    fill_value=outputs.size(0),
                    dtype=torch.long,
                    device=device
                )

                loss = criterion(outputs.log_softmax(2), targets, input_lengths, target_lengths)

            # ----- BACKPROP -----
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()

        print(f"Epoch {epoch+1} Loss: {total_loss/len(loader):.4f}")

    torch.save(model.state_dict(), "models/video_model.pt")
    print("✅ Lip Model Saved")
