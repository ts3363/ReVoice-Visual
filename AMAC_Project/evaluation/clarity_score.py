import numpy as np
from jiwer import wer

def speaking_rate(text, duration):
    words = len(text.split())
    return words / (duration/60 + 1e-5)

def pause_ratio(pauses, duration):
    return sum(pauses) / duration

def clarity_score(ref, hyp, phoneme_conf):
    w = wer(ref, hyp)
    conf = np.mean(phoneme_conf)
    clarity = max(0, 100 - w*100) * conf
    return round(clarity, 2)
