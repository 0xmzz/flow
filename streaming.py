"""Network streaming sessions: remote clients (phone, thin client) upload audio
chunks WHILE recording; the same pause-boundary commit algorithm the PC hotkey
uses (StreamWorker) runs against the growing buffer, so clients can show/type
committed text live and get a near-instant final result on finish.

Wire format: PCM16 little-endian mono @16kHz (what AudioRecord produces; the
python client converts float32 -> int16 before upload).
"""

import secrets
import threading
import time

import numpy as np


class NetSource:
    """Recorder-shaped adapter over chunks arriving from the network."""

    def __init__(self):
        self._chunks = []
        self._lock = threading.Lock()

    def append_pcm16(self, raw):
        a = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
        with self._lock:
            self._chunks.append(a)

    def snapshot(self):
        with self._lock:
            if not self._chunks:
                return np.zeros(0, dtype=np.float32)
            return np.concatenate(self._chunks)


class StreamSession:
    def __init__(self, flow, device):
        from flow import StreamWorker  # late import: avoid cycle at module load
        self.sid = secrets.token_urlsafe(8)
        self.device = device
        self.source = NetSource()
        self.flow = flow
        self.last_seen = time.time()
        self.partial_served = False
        self.worker = StreamWorker(flow, live=False, source=self.source)
        self.worker.start()

    def add_audio(self, raw):
        self.last_seen = time.time()
        self.source.append_pcm16(raw)
        return self.worker.committed_text

    def finish(self):
        """Final decode of the uncommitted tail; returns the full text."""
        committed = self.worker.stop()
        audio = self.source.snapshot()
        # If the client already displayed/typed partials, cleanup must not
        # restyle the committed prefix out from under them.
        text, _meta = self.flow.transcribe_and_store(
            audio, source="remote", window=self.device,
            committed=committed, skip_cleanup=self.partial_served)
        return text or ""

    def cancel(self):
        self.worker.stop()


class SessionManager:
    TTL = 120  # orphaned sessions (client died mid-stream) get reaped

    def __init__(self, flow):
        self.flow = flow
        self.sessions = {}
        self._lock = threading.Lock()
        t = threading.Thread(target=self._reaper, daemon=True)
        t.start()

    def start(self, device):
        s = StreamSession(self.flow, device)
        with self._lock:
            self.sessions[s.sid] = s
        return s

    def get(self, sid):
        with self._lock:
            return self.sessions.get(sid)

    def drop(self, sid):
        with self._lock:
            return self.sessions.pop(sid, None)

    def _reaper(self):
        while True:
            time.sleep(30)
            now = time.time()
            with self._lock:
                dead = [k for k, s in self.sessions.items()
                        if now - s.last_seen > self.TTL]
                for k in dead:
                    s = self.sessions.pop(k)
                    try:
                        s.cancel()
                    except Exception:
                        pass
                    print(f"[stream] reaped orphan session {k} ({s.device})")
