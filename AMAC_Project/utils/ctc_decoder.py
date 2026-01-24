import math

LETTERS = " abcdefghijklmnopqrstuvwxyz"

def ctc_decode(logits):
    pred = logits.argmax(1)
    last = -1
    txt = ""
    for p in pred:
        p = p.item()
        if p != last and p != 0:
            txt += LETTERS[p]
        last = p
    return txt


def ctc_beam_decode(logits, beam_width=10):
    # logits: [T, C]
    T, C = logits.shape

    beams = [((), 0.0)]  # (sequence, log_prob)

    for t in range(T):
        new_beams = {}
        for prefix, score in beams:
            for c in range(C):
                p = logits[t, c].item()
                new_seq = prefix + (c,)
                new_score = score + math.log(p + 1e-9)
                if new_seq not in new_beams or new_score > new_beams[new_seq]:
                    new_beams[new_seq] = new_score

        beams = sorted(new_beams.items(), key=lambda x: x[1], reverse=True)[:beam_width]
        beams = [(k, v) for k, v in beams]

    best = beams[0][0]

    # CTC collapse
    res = []
    prev = None
    for b in best:
        if b != prev and b != 0:
            res.append(LETTERS[b])
        prev = b

    return "".join(res)
