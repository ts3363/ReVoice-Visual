import torch

def extract_video_feat(video_model, x):
    with torch.no_grad():
        x = video_model.conv(x)
        b,c,t,h,w = x.size()
        x = x.permute(0,2,1,3,4).contiguous().view(b,t,-1)
        x,_ = video_model.rnn(x)
        return x.mean(1)

def extract_audio_feat(audio_model, x):
    with torch.no_grad():
        x = audio_model.cnn(x)
        b,c,f,t = x.shape
        x = x.permute(0,3,1,2).contiguous().view(b,t,-1)
        return x.mean(1)
