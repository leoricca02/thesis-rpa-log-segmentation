# Demo UI

A browser front-end for the segmentation pipeline, built for the thesis defence.
It runs the **unmodified** pipeline in `src/` and shows, phase by phase, what the
model was asked, what it thought, what it answered and what the pipeline did
with the answer.

```
Input log  →  Oracle (human-in-the-loop)  →  Pipeline (live)  →  Result
```

## Run it

```bash
pip install -r ui/requirements.txt
python ui/server.py                 # → http://127.0.0.1:8000
```

A live run needs `GEMINI_API_KEY` in `.env`, exactly like the CLI. Replaying a
recording needs no key and no network.

## What each screen does

| Screen | Content |
| :-- | :-- |
| **Input log** | Upload a SmartRPA CSV, or pick one of the logs under `data/`. Preview: event count, populated columns, applications, time span, and the raw interleaved sequence. |
| **Oracle** | Declare routine names and execution counts — the only supervision the method needs. Options: human review of the topology, relevance tagging (0A-2), Gemini thought summaries, cache reuse, model id. |
| **Pipeline** | A timeline of the phases (0A → 0B → 0B-2 → serialisation → 0A-2 → 1 → review → 2). For each phase: what it does and its failure policy, the model call (cache hit or live, thinking level, tokens, latency, full prompt), the model's thought summary, and the phase's own result. The pipeline's console output streams alongside. |
| **Human review** | The inferred sharing topology as a matrix *shared action × routine*. Toggle cells to edit a subset, or turn an action into an ordinary step, then approve. Same semantics as the CLI's `--review`. |
| **Result** | The "untangle" animation: every event leaves the interleaved line for the lane of its execution, shared actions are copied into every trace their subset allows, noise drops to its own lane. Then per-execution traces with the model's reasoning, the validation checks (P1–P3, S1–S3), the Noise sheet, the final topology, and downloads of the XLSX workbook and the JSON audit trail. |

## How "what the model thinks" is obtained

Two sources, both shown:

1. **The reasoning the pipeline already asks for.** Phase 1 writes its reasoning
   before the topology and a justification for each shared action; Phase 2 writes
   one reasoning note per execution. These are part of the schema-constrained
   JSON answer.
2. **Gemini thought summaries.** With the option on, the UI's client
   (`instrumented_client.py`, a subclass of `SmartLLMClient`) sets
   `includeThoughts` on the phases that are given a thinking budget (1 and 2),
   and separates the thought parts from the JSON answer. Phases 0A, 0B and 0A-2
   run with no thinking budget, so they have no thoughts to show.

The cache key of `SmartLLMClient` does not cover `includeThoughts`, so cached
answers — including the committed thesis cache, which the UI reuses by default —
are still hits. A cached answer replays without new thoughts unless the UI stored
them when the call was first made; turn off **Reuse cached responses** to watch
the model think on every call.

## Recordings and replay

Every live run is journalled to `ui/runs/<id>/events.jsonl`. On the result
screen, **Save as recording** copies it to `ui/recordings/<name>/`, where it
appears in the **Recordings** list. A replay re-emits the events with their
original pacing (0.5×–4×) and pauses at the review step until you continue, so a
presentation never depends on network, quota or the stochasticity of Phase 2.

`ui/recordings/case-study-thesis-reference-run/` is built offline from the
committed artefacts — the cached Phase 0–1 responses and the committed Phase 2
routing — by:

```bash
python ui/build_reference_recording.py
```

It is the thesis's reference run seen through the UI, not a live inference, and
it carries no thought summaries. For the defence, record a live run.

## Recording the demo video — suggested flow

1. Light or dark theme (toggle in the top bar); light usually reads better on a projector.
2. **Input** → `caso_studio_trasferte_2exec.csv`; point at the interleaved ribbon.
3. **Oracle** → "Use the case-study routines"; leave review and thought summaries on,
   turn **Reuse cached responses** off so every phase is a real inference.
4. **Pipeline** → follow the timeline; click a finished phase to go back to it.
   In Phase 1 the thought summary and the reasoning appear above the topology.
5. **Review** → explain the 4/4, 3/4, 2/4 and 1/4 subsets (assumption A1′); approve.
6. **Result** → "Untangle again" replays the animation.
7. **Save as recording**, so the same run can be replayed during the discussion.

## Files

```
ui/
├── server.py                     FastAPI app: REST + server-sent events, serves static/
├── pipeline_runner.py            LiveRun (runs src/ and narrates it), ReplayRun, recordings
├── instrumented_client.py        SmartLLMClient subclass: call events, thought summaries
├── build_reference_recording.py  Offline reference recording from the committed artefacts
├── static/                       index.html, styles.css, app.js — no build step
└── recordings/                   Saved runs available for replay
```

The categorical palette used for routines is validated for colour-vision
deficiency on both themes; every lane and matrix column is also labelled, so
identity never rests on colour alone. It intentionally differs from the
exported workbook's palette, which belongs to the pipeline.
