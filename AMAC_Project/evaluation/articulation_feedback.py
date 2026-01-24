from jiwer import wer
from rapidfuzz.distance import Levenshtein

def articulation_feedback(reference, hypothesis):

    w = wer(reference, hypothesis)

    dist = Levenshtein.distance(reference, hypothesis)
    ref_len = max(1, len(reference))

    feedback = []

    if dist / ref_len > 0.25:
        feedback.append("Slow down speech")

    if w > 0.35:
        feedback.append("Over-articulate consonants")

    if not feedback:
        feedback.append("Excellent articulation")

    return feedback
