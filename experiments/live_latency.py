"""Offline (never microphone) VAD -> Parakeet latency evidence."""

import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from live import FRAME_SAMPLES, ParakeetEngine, SAMPLE_RATE, SileroVad  # noqa: E402
from store import decode_wav, resample_linear  # noqa: E402


def files():
    found = []
    if (ROOT / "last.wav").exists():
        found.append(ROOT / "last.wav")
    found.extend((ROOT / "data" / "dataset").glob("**/*.wav"))
    found.extend((ROOT / "experiments" / "vad_run" / "wavs").glob("*.wav"))
    return found[:3]


def main():
    wavs = files()
    if not wavs:
        raise SystemExit("No last.wav or dataset WAV samples found")
    engine = ParakeetEngine("nemo-parakeet-tdt-0.6b-v3",
                            ROOT / "data" / "models" / "parakeet")
    print(f"provider={engine.provider} samples={len(wavs)}")
    for path in wavs:
        audio, sr = decode_wav(path.read_bytes())
        audio = resample_linear(audio, sr, SAMPLE_RATE)
        vad = SileroVad()
        t0 = time.perf_counter()
        probs = []
        for i in range(0, len(audio) - FRAME_SAMPLES + 1, FRAME_SAMPLES):
            probs.append(vad.probability(audio[i:i + FRAME_SAMPLES]))
        vad_ms = (time.perf_counter() - t0) * 1000
        speech = [i for i, p in enumerate(probs) if p >= 0.5]
        if speech:
            start = max(0, speech[0] * FRAME_SAMPLES - int(.15 * SAMPLE_RATE))
            end = min(len(audio), (speech[-1] + 1) * FRAME_SAMPLES + int(.15 * SAMPLE_RATE))
            utterance = audio[start:end]
        else:
            utterance = audio
        text, decode_s = engine.decode(utterance)
        endpoint_ms = 280
        # Streaming pays one frame's VAD cost at EOS, not the whole offline
        # file scan printed above.
        eos_ms = endpoint_ms + (vad_ms / max(1, len(probs))) + decode_s * 1000
        print(f"{path.name}: audio={len(audio)/SAMPLE_RATE:.2f}s "
              f"vad={vad_ms:.1f}ms decode={decode_s*1000:.1f}ms "
              f"eos->text~{eos_ms:.1f}ms provider={engine.provider} text={text!r}")


if __name__ == "__main__":
    main()
