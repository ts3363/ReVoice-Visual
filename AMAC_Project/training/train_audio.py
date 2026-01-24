import torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import DataLoader
from models.audio_model import AudioEncoder
from training.audio_dataset import AudioDataset, collate_fn
from jiwer import wer

device = "cuda" if torch.cuda.is_available() else "cpu"

chars = " abcdefghijklmnopqrstuvwxyz"
char2idx = {c:i+1 for i,c in enumerate(chars)}   # 0 is blank
idx2char = {i:c for c,i in char2idx.items()}

def encode(text):
    return torch.tensor([char2idx[c] for c in text if c in char2idx], dtype=torch.long)

dataset = AudioDataset()
loader = DataLoader(dataset, batch_size=2, shuffle=True, collate_fn=collate_fn, num_workers=0)


model = AudioEncoder().to(device)
criterion = nn.CTCLoss(blank=0, zero_infinity=True)
optimizer = optim.Adam(model.parameters(), lr=1e-4)

print("Starting Training...")

for epoch in range(15):
    total = 0
    for x, y in loader:
        x = x.to(device)
        y_enc = [encode(t) for t in y if len(t) > 0]
        if len(y_enc) == 0:
            continue

        y_pad = nn.utils.rnn.pad_sequence(y_enc, batch_first=True)
        y_lens = torch.tensor([len(t) for t in y_enc])

        logits = model(x)
        log_probs = logits.log_softmax(2).permute(1,0,2)
        input_lens = torch.full(size=(logits.size(0),), fill_value=logits.size(1), dtype=torch.long)

        loss = criterion(log_probs, y_pad, input_lens, y_lens)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total += loss.item()

    print(f"Epoch {epoch+1} Loss:", total/len(loader))

torch.save(model.state_dict(),"models/audio_model.pt")
print("✅ Model Saved")
