import os
import pyarrow.parquet as pq
import soundfile as sf
import numpy as np

os.makedirs("data/TORGO/audio", exist_ok=True)
os.makedirs("data/TORGO/transcripts", exist_ok=True)

base = os.path.expanduser("~/.cache/huggingface/hub")

parquets = []
for root, dirs, files in os.walk(base):
    for f in files:
        if "TORGO-database" in root and f.endswith(".parquet"):
            parquets.append(os.path.join(root, f))

print("Found", len(parquets), "parquet files")

counter = 0
for pfile in parquets:
    table = pq.read_table(pfile)
    cols = table.column_names
    data = table.to_pydict()

    audio_col = [c for c in cols if "audio" in c.lower()][0]
    text_col  = [c for c in cols if "text" in c.lower() or "transcript" in c.lower()][0]

    audios = data[audio_col]
    texts  = data[text_col]

    for i in range(len(texts)):
        audio_bytes = audios[i]["bytes"]
        wav = np.frombuffer(audio_bytes, dtype=np.int16)

        sf.write(f"data/TORGO/audio/{counter}.wav", wav, 16000)

        with open(f"data/TORGO/transcripts/{counter}.txt", "w", encoding="utf-8") as f:
            f.write(texts[i])

        counter += 1

print("✅ TORGO EXTRACTION COMPLETE:", counter, "FILES")
