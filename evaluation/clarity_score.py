from jiwer import wer
import numpy as np

def clarity_score(ref, hyp, phoneme_conf=None):
    w = wer(ref, hyp)
    base = max(0, 100 - w * 100)

    if phoneme_conf:
        base = base * (sum(phoneme_conf) / len(phoneme_conf))

    return round(base, 2)
