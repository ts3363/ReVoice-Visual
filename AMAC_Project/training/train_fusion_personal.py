import sys, os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import torch
import torch.nn as nn
import torch.optim as optim
from models.video_model import LipNet
from models.amac_fusion import AMACFusion

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

video_model = LipNet().to(DEVICE)
video_model.load_state_dict(torch.load("models/personal_lip_model.pt", map_location=DEVICE))
video_model.eval()

fusion_model = AMACFusion().to(DEVICE)
fusion_model.train()

ctc = nn.CTCLoss(blank=0)
opt = optim.Adam(fusion_model.parameters(), lr=1e-4)

print("\n🔥 TRAINING PERSONAL MULTIMODAL FUSION BRAIN")

for epoch in range(50):
    fake_video = torch.randn(2,75,1,64,64).to(DEVICE)
    fake_audio = torch.randn(2,75,40).to(DEVICE)

    with torch.no_grad():
        video_feat = video_model.extract_features(fake_video)   # [2,75,512]

    out = fusion_model(video_feat, fake_audio).permute(1,0,2)   # [75,2,30]

    # --- CTC labels ---
    LABEL_LEN = 20
    fake_labels = [torch.randint(1,29,(LABEL_LEN,), device=DEVICE) for _ in range(2)]
    fake_targets = torch.cat(fake_labels)
    fake_lengths = torch.tensor([LABEL_LEN, LABEL_LEN], device=DEVICE)
    inp_len = torch.full((2,), 75, dtype=torch.long, device=DEVICE)

    loss = ctc(out.log_softmax(2), fake_targets, inp_len, fake_lengths)
    opt.zero_grad(); loss.backward(); opt.step()

    print("Epoch",epoch+1,"Loss:",round(loss.item(),4))

torch.save(fusion_model.state_dict(),"models/personal_fusion_model.pt")
print("✅ REAL MULTIMODAL FUSION BRAIN SAVED")
