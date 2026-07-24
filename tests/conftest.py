"""Keep the test suite out of the user's real data directory.

The intent log is a training corpus for a future fine-tune, so a test run that
appends synthetic utterances to it is not merely untidy — it poisons the
dataset. 190 fabricated rows landed in the real log before this existed. The
fixture is autouse so a new test cannot forget to opt in.
"""

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


@pytest.fixture(autouse=True)
def _isolate_user_data(tmp_path, monkeypatch):
    import commands

    sandbox = tmp_path / "data"
    sandbox.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(commands, "DATA_DIR", sandbox, raising=False)

    # Routers built without an explicit path resolve it at construction time,
    # so patching the module constant is not enough on its own.
    original_init = commands.Router.__init__

    def init(self, *args, **kwargs):
        kwargs.setdefault("intent_log_path", sandbox / "intent_log.jsonl")
        original_init(self, *args, **kwargs)

    monkeypatch.setattr(commands.Router, "__init__", init)

    import agents

    original_agent_init = agents.AgentRunner.__init__

    def agent_init(self, *args, **kwargs):
        kwargs.setdefault("log_path", sandbox / "agent_runs.jsonl")
        original_agent_init(self, *args, **kwargs)

    monkeypatch.setattr(agents.AgentRunner, "__init__", agent_init)
    yield
