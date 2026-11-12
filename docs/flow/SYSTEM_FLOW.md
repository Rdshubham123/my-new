# Full System Flow

Problem statement: *Long-context underwater video understanding: tracking objects and identifying temporal patterns across long underwater video sequences.*

Diagram: `system-flow.png` (made from `flow.html`).

## Which model does which job

| # | Step | Model / tool | Input | Output | Runs on | Type |
|---|---|---|---|---|---|---|
| 0A | Label data | SAM 2.1 / SAM 3 | raw videos | boxes, masks, IDs (person checks) | GPU, offline | ready-made |
| 0B | Train detector | YOLO26 / YOLO11 | SFISHTRACK, MFT25, BrackishMOT, FishTrack23 | detector weights | GPU, offline | ready-made |
| 0C | Re-ID network | V-JEPA 2.1 ViT-L or small ReID CNN | fish crops with IDs | embedding model | GPU, offline | adapt |
| 1 | Decode video | FFmpeg / PyAV | MP4 or live stream | frames + frame number + timestamp | CPU | ready-made |
| 2 | Water quality | OpenCV (no AI) | frame | visibility score | CPU | ready-made |
| 3 | Detect | YOLO (fine-tuned) | frame batches | boxes, class, confidence | GPU | ready-made |
| 4 | Short-term track | BoT-SORT | boxes in order | track IDs, tracklets | CPU/GPU | ready-made |
| 5 | Hard moments | SAM 2.1 tiny/small | boxes before a crossing / 'uncertain' | precise masks, fewer ID swaps | GPU, on demand | adapt |
| 6 | Appearance fingerprint | SAM 2 mask + V-JEPA 2.1 / ReID CNN | masked fish crop | embedding | GPU | adapt |
| 7 | Self-taught identity (STIM) | our contrastive head | tracklets of this video | embeddings tuned to these fish | GPU | **ours** |
| 8 | Identity memory | our linker (`prototype/identity.py`) | embedding + place + time gap | global ID: confirmed / probable / uncertain | CPU | **ours** |
| 9 | Events | Python rules | tracks + visibility | arrive, leave, return, group, visibility drop | CPU | ready-made |
| 10 | Temporal patterns | pandas, SciPy FFT, ruptures, HDBSCAN + DTW | time series from memory B | cycles, trends, change points, routes, revisits | CPU | **ours** (uncertainty-aware) |
| 11 | Alerts | robust statistics + health gate | patterns, visibility, tracker health | trustworthy alerts | CPU | **ours** |
| 12 | Keyframe picker | event triggers + duplicate filter | events / alerts + 1 frame per 30-60 s | few useful keyframes | CPU | adapt |
| 13 | Describe | Qwen3-VL 4B/8B (quantised) | keyframes with IDs drawn + tracker facts | plain-language description (IDs validated) | GPU, async | adapt |
| 14 | Summaries | Qwen3-VL (text) | descriptions + events | clip, 10-min and video summaries | GPU, async | adapt |
| 15 | Store & search | SQLite + FAISS | everything above | search by ID / time / event / text | CPU | ready-made |
| 16 | Annotated video | OpenCV + FFmpeg | original video + final IDs | same-length video with boxes + IDs | CPU | ready-made |
| 17 | Dashboard | FastAPI + Streamlit | database + video | player, timeline, patterns, alerts | CPU | ready-made |

## The three memories
- **A, short-term (RAM):** live tracks, speed, recent appearance (steps 4-5).
- **B, objects & events (SQLite):** global IDs, paths, first/last seen, events (steps 8-11).
- **C, meaning (SQLite + FAISS):** descriptions, summaries, search vectors (steps 13-15).

Clips of about 10 s are only work units for saving and checkpoints. The tracker and the memories never reset at a clip boundary.

## One fish, start to finish (example)
1. **00:02:10** YOLO detects a grouper. BoT-SORT gives it track 14. The identity memory creates global ID **G7**.
2. **00:02:40** It crosses another grouper. BoT-SORT is unsure, so SAM 2 follows both masks through the crossing and the IDs stay correct.
3. **00:03:05** It swims behind coral. Track 14 ends, and event *G7 left* is stored.
4. **00:07:50** A grouper appears near the same coral as new track 31. The masked crop gives a STIM embedding. Its best match is G7 and the second-best is far lower, so it is linked as **G7, probable**, and event *G7 returned after 4 m 45 s* is stored.
5. **Over 2 hours** the pattern step finds that G7 returns about every 5 minutes. Total group activity also peaks twice, and a change point appears when visibility drops at 01:12:00. Alerts during that drop are suppressed.
6. Qwen3-VL gets the keyframe at 00:07:50 plus the facts *"G7 returned, 4 groupers present"*. It writes: *"A grouper (G7) returns from behind the coral and joins three others near the reef edge."*
7. Summaries roll up: clip, then 10 minutes, then the whole video. Everything is searchable: *"when did G7 come back?"*
8. The renderer re-reads the original video and draws **G7** on every frame of tracks 14 and 31.

## Simple rule
YOLO **finds**, BoT-SORT **follows**, SAM 2 **fixes hard moments**, STIM and the identity memory **remember who is who**, statistics **find patterns**, and Qwen3-VL **explains in words**.
