import torch

def greedy_decode(output):
    output = torch.argmax(output, dim=-1)
    decoded = []
    prev = -1
    for c in output:
        if c != prev and c != 0:
            decoded.append(chr(c + 96))
        prev = c
    return "".join(decoded)
