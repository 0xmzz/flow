# Flow HUD — voice assistant panel + workflows + agent lanes

F9 currently toggles live listening with a dot/pill. This spec upgrades F9 into
a small assistant surface ("mini app") with visible state, a clarify loop,
replayable workflows, and pluggable agent lanes. Builds ON TOP of the router
fixes (normalization, close/type skills, structured intent fallback).

## Phase A — HUD panel + dialog loop

A borderless always-on-top Tk card (managed by the existing Overlay thread,
same styling family as the pill). **Placement: right edge of the work area,
vertically centered — deliberately AWAY from the bottom-center dictation pill
(user feedback: sharing that spot made the two modes indistinguishable).
Size: ~420x260px — big enough to read its thinking, not a status pill.**
Layout, stacked:

1. **heard** — last transcript (live-updating; dim when ignored/no wake word,
   so silent drops are visible instead of confusing)
2. **thinking** — scrolling area (3-6 lines) showing resolution steps: intent
   chosen, skill/workflow/agent selected, questions it wants to ask, workflow
   step progress. This is the "show me its thinking" surface — append-only
   log within the current interaction, cleared on the next utterance.
3. **status** — result / spoken response / open question, visually distinct
   (question = accent color, awaiting the user's spoken answer)

Behavior:
- Panel appears when live mode is ON (F9) and collapses to the small dot when
  idle >8s; any utterance re-expands it. F9 off = everything hidden.
- Never takes keyboard focus (Tk overrideredirect, no focus steal).
- **Dialog state**: when a skill/workflow/intent needs input, the router
  returns a question instead of an answer. HUD shows + speaks it, and the
  LiveWorker enters `awaiting_answer`: the NEXT utterance routes as the answer
  WITHOUT the wake word (30s timeout back to normal; "cancel"/"never mind"
  aborts). Wake-gated commands still work mid-dialog (wake word wins).

## Phase B — workflow engine

`data/workflows/*.json` (gitignored dir, like history):

```json
{"name": "morning setup",
 "triggers": ["morning setup", "start my morning"],
 "steps": [
   {"do": "open", "arg": "chrome"},
   {"do": "wait_window", "arg": "chrome", "timeout_s": 10},
   {"do": "open", "arg": "discord"},
   {"do": "ask", "arg": "Open the trading terminal too?", "yes": [
     {"do": "open", "arg": "https://folionomics.com/trade"}]}
 ]}
```

- Primitives = existing router skills + `keys` (hotkey send), `type`, `wait_s`,
  `wait_window` (poll title), `ask` (yes/no via dialog loop), `say`, `agent`
  (Phase C). Runner executes on a worker thread, HUD shows step progress,
  "flow stop"/"flow cancel" aborts between steps.
- Matching: after normalization, fuzzy-match utterance against all triggers
  (difflib >=0.85) BEFORE the intent LLM; the intent LLM also gets the recipe
  name list so "run my morning thing" can resolve to a workflow.
- **Voice creation**: "flow, create a workflow called X" → dialog loop asks
  for steps one at a time; each spoken step goes through the intent parser to
  a primitive; "done" saves and confirms. "flow, delete workflow X" (confirm),
  "flow, list workflows".

## Phase C — agent lanes

`agents.py`: ordered lane registry, llm.py-style; each lane = name, health
check, run(task) -> generator of progress lines + final text.

- Lane 1 **codex**: `codex exec --sandbox workspace-write "<task>"` cwd
  C:\Calude (subprocess, CREATE_NO_WINDOW); stream stdout tail into HUD status.
- Lane 2 **claude**: `claude -p "<task>"` headless, same shape.
- Lane 3 **hermes** (stub for now): NotImplemented placeholder with TODO
  Tailscale endpoint — do not build, keep the seam.
- Routing: intent action `agent` (added to the intent JSON schema: use when
  the request is multi-step/open-ended beyond skills). HUD asks ONE confirm
  before dispatch ("Send to codex: <task>? "), speaks completion, keeps last
  agent output in the panel until dismissed. Timeout 10min, cancellable.
- **No auto-distillation yet** (the solve-once-replicate loop): log every
  agent task + outcome to data/agent_runs.jsonl so distillation has a corpus;
  the distiller itself is a later phase.

## Constraints

- Local-only invariants unchanged: lanes run local binaries; nothing new
  leaves the box (Hermes stub inert). New tests: dialog-state machine
  (awaiting_answer routing, timeout, cancel), workflow trigger matching +
  runner order with fake skills, workflow voice-creation dialog, agent lane
  registry with a fake lane. pytest green.
- ASR/VAD/engine code untouched. PTT precedence rules unchanged.
- Keep flow.py diffs surgical; new logic in hud.py / workflows.py / agents.py.
