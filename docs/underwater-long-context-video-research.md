# Long-Context Underwater Video Understanding and Tracking: Research Assessment and Architecture

Date: 2026-11-09

**Evidence labels used throughout**
- **[V]** verified by web search in this session (links given; license or numbers still need a check on the official page before you rely on them).
- **[K]** from my background knowledge, not re-verified in this session. Check the repo before you build on it.
- **[E]** established technique. **[A]** needs adaptation. **[P]** my proposal. **[R]** research idea that needs experimental validation.

Where I could not confirm something, I say so. I give no performance numbers beyond those in the cited sources.

---

## 1. Executive assessment

The idea is sound, but one framing in it is wrong and will cost you the main research claim if it stays.

**"Divide into clips, process each, send as packets" is the wrong unit of work for tracking.** Tracking is a stateful, sequential computation. If clips are processed independently (your Experiment B), identity continuity is destroyed by construction. The research claim is "clips are a scheduling and I/O unit, not a state boundary." That means:

- Decode and track **one continuous stream** with persistent tracker state.
- Use "clips" only for batching, checkpointing, VLM keyframe windows, and database commits.
- Parallelise across *pipeline stages* (decode, detect, track, VLM), not across clips, unless you have to. If you must, use overlapping chunks plus tracklet stitching (§8.4).

**The research contribution is not the detector.** It is a three-tier memory with an uncertainty-aware global identity layer, grounded in tracker output and used by a VLM. The system must report "I don't know whether this is the same fish."

**A VLM cannot maintain identities.** It should consume tracker-grounded structured context and produce semantics. A dedicated module owns identity (§13, Q7).

**Biggest risks**
1. Re-identification of near-identical fish over minutes. Often not solvable from appearance (§8.5).
2. Data. Few public datasets have long sequences with persistent IDs (§5).
3. Evaluating cross-clip identity. Standard HOTA/IDF1 on short sequences won't show your effect; you need long sequences with deliberate gaps.
4. VLM hallucination in turbid footage.

---

## 2. Critical analysis of the proposed approach

| Your step | Verdict | Change |
|---|---|---|
| 1 Input | Fine | Probe with `ffprobe`. Record PTS, not just FPS. |
| 2 Fixed 10 s clips | Weak | Fixed cuts are fine for I/O, but cut mid-event. Use clips only as scheduling units. Make VLM windows event-driven. |
| 3 "Packets" | Unnecessary | No network packetization on one machine. Use in-memory frame batches plus a metadata record `(video_id, frame_idx, pts)`. A queue/RPC is justified only for edge→cloud, and there you send metadata and keyframe crops, not frames. |
| 4 Detection | Fine | Fine-tune on underwater data. Detection is per-frame and the only embarrassingly parallel stage. |
| 5 Tracking | Core | Stateful across the whole stream. Distinguish *track ID* from *global identity* (§8.1). |
| 6 Reconstruction | Correct and important | Never rebuild from sampled frames. Overlay on original decoded frames (§6, Problem 6). |
| 7 VLM | Right idea | Event-triggered keyframes plus structured tracker context. No per-frame VLM. |
| 8 Memory | The contribution | Three tiers (§7). |
| 9 Outputs | Good | Add an uncertainty field to every identity link. |

**Unnecessary complexity to cut for the prototype:** SAM3 everywhere, a vector DB server, PostgreSQL, TensorRT, a streaming edge deployment, enhancement networks. All can come later (§15).

---

## 3. Recommended architecture

### 3.1 Diagram

```
                ┌───────────────────────────────────────────────┐
 video / RTSP → │ 1 Decoder (PyAV/FFmpeg, keeps pts, frame_idx)  │  CPU
                └───────────────┬───────────────────────────────┘
                                │ frames (ring buffer) + FrameMeta
        ┌───────────────────────┼───────────────────────────┐
        ▼                       ▼                           ▼
┌──────────────┐     ┌────────────────────┐      ┌──────────────────────┐
│ 2 Quality    │     │ 3 Detector (batch) │      │ Raw frame store /    │
│ monitor      │     │ YOLO / RT-DETR     │ GPU  │ re-decode by pts     │
│ (visibility) │     └─────────┬──────────┘      └──────────────────────┘
└──────┬───────┘               ▼
       │             ┌────────────────────┐
       │             │ 4 Tracker (stateful│  sequential
       │             │ BoT-SORT/ByteTrack)│  state = Memory A
       │             └─────────┬──────────┘
       │                       ▼ tracklets (closed + live)
       │             ┌────────────────────┐   embeddings from crops
       │             │ 5 Global Identity  │◄──(small ReID model)
       │             │ Manager            │   state = Memory B (SQLite)
       │             └─────────┬──────────┘
       │                       ▼ events (appear/disappear/reappear/
       ▼                         uncertain-link/crowding)
┌────────────────────────────────────────┐
│ 6 Event + Anomaly engine               │──► alerts
└──────────────┬─────────────────────────┘
               ▼ keyframe triggers
┌────────────────────────────────────────┐
│ 7 Keyframe selector → 8 VLM (async)    │ GPU, separate process
│   input: frames + tracker JSON + recent│
│   summaries → output: caption/summary  │
└──────────────┬─────────────────────────┘
               ▼
┌────────────────────────────────────────┐
│ 9 Semantic memory (Memory C):          │
│   SQLite + FAISS; hierarchical summaries│
└──────────────┬─────────────────────────┘
               ▼
 10 Renderer: re-decode original, overlay final IDs by frame_idx → mp4
 11 Search API (FastAPI) → 12 Dashboard (Streamlit/Gradio)
```

### 3.2 Data flow, concurrency, and failure handling

| Component | Input | Output | Execution |
|---|---|---|---|
| Decoder | file/stream | `FrameMeta(video_id, frame_idx, pts_us, wh)` + frame | Sequential, CPU, producer |
| Detector | frame batch (e.g. 8–16) | boxes, cls, conf | GPU, batched; can run ahead of the tracker |
| Tracker | detections per frame, in order | local tracks (ID, box, velocity, embedding) | **Strictly sequential**, one instance per video/camera |
| Identity manager | closed tracklets + embeddings | global ID + link confidence | Sequential on tracklet-close events; cheap |
| Event engine | track/identity stream | typed events | Streaming, CPU |
| Keyframe selector | events, scene-change score | frame refs for the VLM | Streaming |
| VLM | keyframes + JSON context | text + structured fields | Async worker, GPU, queue |
| Renderer | original video + per-frame result table | annotated mp4 | After the pass, or lagging stream |

**Memory sharing:** one writer process owns Memory A (in-process) and B (SQLite, WAL mode). Other processes read B. Workers do not share tracker state. If you scale to many cameras, shard by camera, not by clip.

**Failure handling:** each stage reads and writes keyed by `(video_id, frame_idx)`. Checkpoint every N frames: tracker state (pickled), the identity table (already in SQLite), the last committed frame. On a crash, resume from the last checkpoint. VLM failures are isolated: the queue retries and the pipeline is not blocked, because VLM output enriches but never gates tracking. Mark frames with `status ∈ {ok, detector_failed, skipped}`.

**Synchronisation:** the output video is produced by re-decoding the original and joining per-frame results on `frame_idx`/`pts`. It never depends on what the AI stages sampled.

---

## 4. Existing research and model comparison

### 4.1 Detection

| Model | Status | Notes |
|---|---|---|
| Ultralytics YOLO26 [V] | Released; docs at <https://docs.ultralytics.com/>; blog <https://www.ultralytics.com/blog/ultralytics-yolo26-the-new-standard-for-edge-first-vision-ai> | NMS-free end-to-end, edge-oriented. **Licence: AGPL-3.0 or paid Enterprise** [V]. Matters if you ship a product. Release-date reports conflict across sources, so check the repo. |
| YOLOv8/11 | [K] | Mature, well documented. Fallback if YOLO26 tooling is unstable. |
| RT-DETR [K] | Ultralytics and original repo | Transformer detector; heavier. Try only if YOLO struggles with occlusion crowding. |
| Underwater-specific detectors | [K] mostly small architecture tweaks on YOLO | Fine-tuning a general detector on in-domain data usually beats exotic architectures. |

**Recommendation:** fine-tune a YOLO small/medium model at 960–1280 px. No evidence here that underwater-specific architectures beat in-domain fine-tuning; treat that as an ablation, not a premise.

### 4.2 Tracking

| Tracker | Role | Notes |
|---|---|---|
| ByteTrack [K] | Motion/IoU, uses low-confidence boxes | Fast baseline. No appearance, so weak on re-id. |
| BoT-SORT [K] | Adds camera-motion compensation + optional ReID | Built into Ultralytics (`tracker=botsort.yaml`, `persist=True`). **Use as default.** Camera-motion compensation matters for ROV/AUV footage. |
| DeepSORT [K] | Appearance + Kalman | Older; superseded by the above for this use. |
| SU-T [V] | Scale-aware Unscented Kalman Filter + FishIoU + optional ReID, from the MFT25 paper (arXiv 2507.06400). Reported 34.1 HOTA / 44.6 IDF1 on MFT25. | Fish-specific motion. Relevant as a baseline for erratic swimming. Project: <https://vranlee.github.io/SU-T/> |
| TIDE [V, partial] | "Tracking Identities with Dual-branch Elasticity" for schooling fish; project <https://vranlee.github.io/TIDE> | I found only an aggregator listing, no arXiv ID. Read the paper before you depend on it. |
| SAM2 / SAM2Long [K] | Mask propagation with memory bank; SAM2Long adds tree-based memory | Prompt-driven VOS, not a multi-object detector-tracker. Good for a few targets, expensive for tens of fish. |
| SAM 3 [V] | Concept-prompted detection + tracking ("fish" text prompt), official repo <https://github.com/facebookresearch/sam3>, Transformers integration (`Sam3VideoProcessor`, weights `facebook/sam3`). | Detector is identity-agnostic; tracker separates identities. Checkpoint access/licence gating not verified. Use for **auto-labelling and for ambiguous cases**, not as the per-frame workhorse. |

**Underwater suitability:** none of these was designed for long gaps. They hold identity for seconds to tens of seconds. Minutes-long recovery is your global identity layer's job.

### 4.3 VLMs

| Model | Status | Fit |
|---|---|---|
| Qwen3-VL [V] | Repo <https://github.com/QwenLM/Qwen3-VL>; tech report arXiv 2511.21631. Dense 2B/4B/8B/32B plus MoE 30B-A3B/235B-A22B; native 256K context; timestamp alignment for video. | **Primary choice.** The 2B–8B sizes fit your budget. Check the model cards for exact sizes (one secondary source mentions 14B, which conflicts with the report). |
| InternVideo2.5 [V] | OpenGVLab, 8B chat model `OpenGVLab/InternVideo2_5_Chat_8B`; repo <https://github.com/OpenGVLab/InternVideo> | Long-video MLLM. Alternative backbone. |
| InternVideo3 [V, partial] | arXiv 2606.12195 (June 2026), "Agentify Foundation Models…"; folder `InternVideo/tree/main/InternVideo3`. A mirror README claims an 8B model + video-agent code. | **I could not verify the release on the official GitHub.** Check before planning around it. |
| VideoBrain [V] | arXiv 2602.04094, ICML 2026; adaptive frame sampling with CLIP-retrieval + uniform agents; code <https://github.com/junbo-zou/VideoBrain>. Reported +3.5–9.0% with 30–40% fewer frames on LongVideoBench/LVBench/Video-MME-Long/MLVU. | Idea relevant to your keyframe selection. Benchmarks are terrestrial; it needs a trained policy. Borrow the *idea* (retrieve then densify), not the training. |

**None of these has been validated on underwater footage in anything I verified.** Expect domain gaps (species names, turbidity). Plan for zero-shot first, then prompt grounding with tracker classes, then optional LoRA.

### 4.4 Closest existing research (from what I verified)
- MFT25 / SU-T (underwater multi-fish tracking benchmark + tracker) [V]
- FishTrack23 (WACV 2024; ~850k boxes in ~26k tracks, per the search summary) [V]
- SFISHTRACK (segmentation + tracking, temporally consistent IDs) [V]
- SAM2Long / SAM 3 (long-horizon memory for segmentation) [K/V]
- VideoBrain, InternVideo3 (agentic long-video reasoning, terrestrial) [V]

I found **no work that combines** a persistent tracker, an uncertainty-aware global identity layer, and a VLM semantic memory for underwater footage. That is not proof none exists. Do a proper literature search (Google Scholar, arXiv, CVPR/ICRA/OCEANS workshops) before claiming novelty.

---

## 5. Datasets

Facts marked [V] come from the search results; confirm licence and counts on the official page.

| Dataset | Content | Track IDs | Masks | Licence | Notes |
|---|---|---|---|---|---|
| **MFT25** [V] | 15 sequences, 48,066 frames, 408,578 boxes; groupers and koi, occlusion/erratic motion. arXiv 2507.06400; <https://vranlee.github.io/SU-T/> | Yes | Boxes | Not verified. One aggregator says the Baidu mirror is password-protected. | Good for tracker/ReID training. Aquaculture tank; **not open-ocean, not long-gap.** |
| **SFISHTRACK** [V] | Zenodo DOI 10.5281/zenodo.20617668, v1.0, CC BY 4.0; Balearic Sea. Secondary (unconfirmed) summary: 54 videos, 24,084 frames at 1920×1080, 5–40 m depth. | Yes (temporally consistent) | Yes (pixel-level) | CC BY 4.0 [V] | Closest to your target environment. Verify counts. |
| **BrackishMOT** [V] | 98 sequences, brackish water, small fish schools; arXiv 2302.10645; <https://vap.aau.dk/brackishmot> | Yes | Boxes | CC BY 4.0 per a mirror, **unverified** | Murky, real turbidity. Short sequences. |
| **FishTrack23** [V] | ~850k boxes, ~26k tracks; WACV 2024 paper | Yes | Boxes | Not verified | Ensemble dataset. Good for pre-training. |
| **FathomNet** [K/unverified] | Large annotated marine image archive, mostly deep-sea | **No tracks** (images) | Boxes | I could not verify | Great for classification/detection, **useless for tracking evaluation.** |

**Gap:** none gives hour-long footage with annotated identities *across multi-minute disappearances* plus event labels. That is exactly what you need to prove the contribution.

**Custom evaluation set (recommended, small):**
1. Take 3–5 long stationary-camera or ROV recordings (30–60 min). Use your own footage or openly licensed footage.
2. Auto-propose tracks with SAM 3 or your tracker, then correct them manually in CVAT, only for a *selected set of target individuals* (e.g. 20–40). Don't annotate everything.
3. Annotate: identity across gaps for those targets, a small set of event labels (appear, leave, congregate, equipment intrusion) with timestamps, and "same/different/unknown" for gap pairs.
4. Hold out whole videos for testing.

Annotating only chosen targets with deliberate long gaps is feasible; annotating every fish for an hour is not.

---

## 6. Technical challenges and solutions

**Problem 1 – cross-clip identity.** Persistent tracker state across the stream removes the *clip* boundary problem entirely. What remains is *occlusion/exit gaps*, handled by tracklet linking (§8).

**Problem 2 – occlusion.** Three time scales:
- ≤ ~1–2 s: Kalman prediction plus BoT-SORT's lost-track buffer (raise `track_buffer`). [E]
- seconds to ~1 min: appearance + spatial gating in the identity manager. [A]
- minutes: only feasible for visually distinctive individuals or constrained scenes. [R]

**Problem 3 – image quality.** Evidence is mixed; [K] enhancement (UWCNN, WaterNet, newer) improves *human-perceived* quality but often does not improve detector mAP and can introduce artefacts. **Do not apply by default.** Treat as an ablation: raw vs. enhanced (e.g. simple CLAHE/white balance, then a learned model) on your validation set. Instead, fine-tune on diverse turbid data (cheaper and more reliable). Compute a per-frame visibility score (contrast, dark-channel/laplacian variance) and use it to gate anomaly alerts (§Problem 8) and to widen association gates.

**Problem 4 – long-context VLM.** Do not feed raw frames over time. Use:
1. Tracker/event-triggered keyframes (§9).
2. Hierarchical text summarisation (clip → 10-min → video).
3. Retrieval from Memory C for questions; the VLM sees only the relevant summaries.
Qwen3-VL's long window exists, but video tokens are expensive; sparse keyframes + structured text is far cheaper.

**Problem 5 – look-alikes.** See §8.5. Marking identity as *uncertain* is a feature.

**Problem 6 – sync and reconstruction.**
- Decode with PyAV or `ffmpeg` and keep `frame.pts` and a monotonically increasing `frame_idx`.
- Store results as `(video_id, frame_idx) → tracks`. Never key on wall-clock or sampled index.
- Variable frame rate: keep original PTS; render with `-vsync passthrough` (or normalise once to constant FPS *before* the pipeline and state it).
- Overlap: only matters if you cut for parallel workers. Then give each chunk a fixed overlap (e.g. 2–5 s), and the *later* chunk's results win in the overlap or you reconcile by IoU; dedupe by `frame_idx`.
- Render by re-decoding the original, drawing the **final** (post-linking) global IDs. Optionally show both local and global ID.

**Problem 7 – compute.** See §10.

**Problem 8 – anomalies.** Compute features on *identity-level* and *scene-level* streams: counts per class per window, speed distribution, nearest-neighbour distance (congregation), time-since-last-seen, visibility score. Baselines: robust z-score / EWMA / seasonal baseline per time-of-day. **Gate:** suppress an anomaly when visibility is low, detector confidence dropped globally, or the signal coincides with a spike in track fragmentation (ID switches/new IDs per minute). A "disappearance anomaly" is only raised if visibility was good and the tracker was healthy.

---

## 7. Long-term memory design

| Tier | Contents | Store | Lifetime |
|---|---|---|---|
| A short-term | live tracks: recent boxes, velocity, embedding EMA, lost counters | in-process Python objects | seconds–minutes |
| B persistent | global identities, tracklets, trajectories (downsampled), events, link evidence | SQLite (WAL) | whole video, resumable |
| C semantic | keyframe captions, clip/segment/video summaries, text embeddings | SQLite rows + FAISS index | whole archive |

**Why this and not a fancier store:** SQLite handles tens of millions of rows for one stream; FAISS (flat or HNSW) handles up to millions of vectors in RAM. Postgres/pgvector or a vector DB server is a scale-out change, not a prototype need. A **temporal graph** (identity ↔ event ↔ clip nodes) is a good *query model*; implement it as relational tables with foreign keys first. Move to a graph DB only if multi-hop queries prove necessary. [P]

Schema sketch:

```sql
identities(gid, cls, first_pts, last_pts, n_tracklets, status)      -- status: confirmed|probable|uncertain
tracklets(tid, gid, video_id, start_idx, end_idx, mean_emb BLOB, n_obs, quality)
links(tid_a, tid_b, score, motion_score, appearance_score, decision, reason)  -- audit trail
traj(gid, frame_idx, x, y, w, h)
events(eid, type, gid, start_pts, end_pts, severity, visibility, source)
segments(sid, level, start_pts, end_pts, text, emb BLOB)            -- level 0 keyframe, 1 clip, 2 10-min, 3 video
```

**Hierarchical summarisation** [E for the pattern, A for this domain]: caption keyframes → merge into clip summaries → merge into segment summaries → video summary. Each level is generated from the level below plus the structured event list, not from raw frames, so context stays bounded.

---

## 8. Tracking and re-identification strategy

### 8.1 Track ID versus verified identity
A **track ID** is a tracker's local label for a continuous association chain; it resets or switches freely. A **global identity** is a hypothesis that several tracklets belong to the same animal. **Verified identity** requires evidence strong enough to rule out alternatives (for example, a unique scar or tag, or a confirmed continuous trajectory). Most links in this system are *hypotheses with a confidence*, and the output should say so: `confirmed / probable / uncertain`.

### 8.2 Pipeline
1. Tracker emits tracklets with per-observation embeddings (crop → small ReID CNN, e.g. OSNet-class or your own fine-tuned model). [A]
2. On tracklet close, compute a robust embedding (mean of top-quality crops, filtering blur/small crops).
3. Candidate matching against *open* global identities within gate: same class; elapsed-time ≤ T_max; feasible displacement (max speed × gap, using the camera's coverage zone); for exits, **location-aware gating**: an object that left via the left edge should reappear near the left edge or from an occluder zone.
4. Score = w_a·appearance + w_m·motion/plausibility + w_t·time prior. Solve by Hungarian with a "new identity" dummy cost.
5. **Decision with margin:** link only if best score ≥ τ_link **and** best − second-best ≥ τ_margin; otherwise create a new identity and record `candidate_links` as *uncertain*.
6. Keep a non-destructive audit trail (`links` table) so decisions can be revisited when more evidence arrives. [P]

### 8.3 Overlapping clip boundaries
If you keep a single tracker stream, there are no clip-boundary identity breaks. For checkpoint/resume, restore pickled tracker state. For parallel chunk workers (only if required), use overlap and stitch by IoU/embedding on the overlap frames (§6 Problem 6).

### 8.4 Re-id after minutes
- Stationary camera, short-lived animals, schooling species: appearance rarely suffices. Report *counts and unique-visit estimates with uncertainty*, not individual IDs.
- Distinctive individuals (turtles with scute/face patterns, individually marked animals, equipment): feasible, if resolution is good. [R]
- Honest framing: your claim should be "recover identity **when evidence allows**", measured as precision/recall of links versus abstentions, not "always recovers".

### 8.5 Limits of visual ReID
Fish of one species at the same size look nearly identical, and body pose changes embeddings more than identity does. Embeddings trained on classes (not individuals) cluster by species/pose. If the top two candidates score within margin, **abstain**. Report the abstention rate; low abstention with high wrong-link rate is worse than high abstention.

---

## 9. VLM integration strategy

**Keyframe triggers (any of):** new global identity, identity exit/re-entry, count change beyond threshold, crowding event, scene-change score, visibility drop/recovery, anomaly candidate, plus a low-rate uniform heartbeat (e.g. one per 30–60 s) so quiet periods still get coverage. De-duplicate with perceptual hash/CLIP similarity so redundant frames are skipped. Pick the sharpest frame in a ±1 s window.

**Prompt = structured grounding + frames.** Example input to the VLM:

```
Camera: stationary, depth unknown. Time 00:12:41–00:12:51.
Tracker facts (do not contradict): fish#7 (probable re-entry, last seen 00:08:03),
fish#12 new, count=9 (prev window 4), visibility=0.62.
Recent summary: "<previous clip summary>"
Task: describe activity in ≤3 sentences; state uncertainty; output JSON
{activity, notable, confidence}.
```

Overlay IDs on the keyframe (boxes + labels) so the VLM can refer to them. The VLM **describes**; it does not assign identity. After generation, validate: reject outputs naming IDs that are not in the tracker facts.

**Model choice:** start with Qwen3-VL 4B or 8B (quantised) running in its own process with a request queue. Smaller is sufficient for captions on tracker-grounded keyframes; larger helps with free-form Q&A. Test 2B/4B first.

---

## 10. Computational feasibility (≈20 GB A100 slice)

Figures below are rough planning estimates from parameter counts, not measurements. **Measure on your hardware** before committing.

| Component | Rough GPU memory | Notes |
|---|---|---|
| YOLO small/medium, 1280 px, batch 8, FP16 | ~1–3 GB | fits easily |
| Small ReID CNN | <1 GB | |
| Qwen3-VL-2B/4B bf16 | ~5–10 GB weights + activations/KV | fits alongside detector |
| Qwen3-VL-8B bf16 | ~16 GB weights alone | **does not co-reside**; use 4/8-bit (AWQ/GPTQ/FP8) or run in a separate time slice |
| SAM 3 / SAM2 | several GB, rises with video memory | run on demand, not continuously |

**Plan:** the detector, tracker, and ReID share the GPU continuously; the VLM runs as a separate async worker on a quantised small model; SAM 3 is invoked only on demand for ambiguous cases or offline auto-labelling. Use mixed precision for everything. Sample frames adaptively for the VLM (not detection). For detection, process every frame (or every 2nd frame with interpolation if throughput-bound, noting the tracking cost). Stream frames through a bounded queue (backpressure) so memory is constant regardless of video length. A small VLM is generally sufficient for caption-level summaries; do not assume it handles open-ended reasoning.

**Edge:** export the detector to ONNX/TensorRT; keep tracking and identity on-device; send only events, crops, and keyframes to the cloud for VLM/search. Postpone until the pipeline works.

---

## 11. Implementation roadmap

Install baseline (check versions against each project's docs):

```bash
python -m venv .venv && source .venv/bin/activate
pip install ultralytics av opencv-python numpy scipy faiss-cpu fastapi uvicorn streamlit
pip install trackeval  # or clone https://github.com/JonathonLuiten/TrackEval  [K]
pip install transformers accelerate  # for Qwen3-VL; see https://github.com/QwenLM/Qwen3-VL
```

Suggested folder structure is in §12.

### Phase 1 – detection
1. Convert a dataset (SFISHTRACK / BrackishMOT / MFT25) to YOLO format.
2. `yolo detect train model=<yolo26 or yolo11 small weights> data=data.yaml imgsz=960 epochs=100`
3. Validate: mAP50, mAP50-95, precision, recall on held-out *videos* (not frames from the same video as training).
4. Output: weights + per-frame JSONL detections.

### Phase 2 – tracking with persistent state
`prototype/run_tracking.py` (included) decodes once, tracks continuously with `persist=True`, stores per-frame results, and renders the annotated output from the original frames. Test: track IDs never reset at 10 s boundaries; output frame count and duration equal the input.

### Phase 3 – cross-clip memory and global identity
`prototype/identity.py` (included) implements the tracklet→global identity linker with margin-based abstention. Test: synthetic gap test (below) first, then a real long video with a planted gap.

### Phase 4 – VLM
Keyframe selection from events, Qwen3-VL captioning in a worker, hierarchical summaries stored in `segments`. Test: spot-check timestamps against a hand-written event list; measure hallucinated IDs (target: zero after validation).

### Phase 5 – anomalies
Windowed features + robust baselines + health gating (§6 Problem 8). Test: inject known events (e.g. a spliced clip) and measure precision/recall/false-positive rate.

### Phase 6 – dashboard
FastAPI endpoints (`/identities`, `/events`, `/search?q=`), Streamlit with video player, timeline, ID filter, summaries.

**Executable code in this repo:** see `prototype/` (core identity manager, runnable on synthetic data, and the Phase-2 tracking/rendering script).

---

## 12. Software architecture and folder structure

```
my-new/
├── docs/underwater-long-context-video-research.md
├── prototype/
│   ├── identity.py          # global identity manager (Memory B logic)
│   ├── test_identity.py     # synthetic gap/similar-fish tests
│   └── run_tracking.py      # Phase 2: continuous tracking + faithful rendering
├── src/uwvid/               # (future) package layout
│   ├── io/        decoder.py, renderer.py, frame_index.py
│   ├── detect/    detector.py
│   ├── track/     tracker.py, reid.py, tracklets.py
│   ├── memory/    short_term.py, store.py (SQLite), semantic.py (FAISS)
│   ├── events/    events.py, anomaly.py, health.py
│   ├── vlm/       keyframes.py, worker.py, summarize.py
│   └── api/       main.py (FastAPI)
├── dashboard/app.py
├── configs/
├── eval/          # TrackEval wrappers, reid-gap metric, event matching
└── data/          # raw/, processed/, db/, outputs/
```

---

## 13. Experimental design and evaluation

### 13.1 The three experiments
- **A – Conventional:** one tracker over the whole video. This is the *upper bound* for continuity without gaps.
- **B – Independent clips:** same detector and tracker per clip, IDs reset, naive stitching by clip-order. Lower bound.
- **C – Memory-aware:** stateful tracker plus global identity manager plus (optionally) VLM memory.

Because A already preserves state, C will not beat A on short continuous occlusions. The informative comparisons are **C vs B** (value of memory when you must chunk) and **C vs A on long gaps** (value of the identity layer when the tracker's own buffer expires). State that explicitly; otherwise a reviewer will ask why A isn't the whole answer.

### 13.2 Fair design
- Split by *video/site*, never by frame; keep identical detector weights across A/B/C so differences come only from the tracking/memory layer.
- Choose clip lengths in advance (e.g. 10 s, 60 s) and report sensitivity.
- Use TrackEval for HOTA, IDF1, IDSW, AssA, DetA.
- Include sequences with planted long gaps (exits, occluders) and measure link precision/recall separately.

### 13.3 Metrics

| Task | Metrics |
|---|---|
| Detection | mAP50, mAP50-95, P, R |
| Tracking | HOTA, IDF1, IDSW, fragmentation |
| Long-gap re-id | link precision/recall, wrong-link rate, **abstention rate**, identity-purity of final global IDs |
| Understanding | event-timestamp error (matched within ±Δs), event P/R, caption factuality via human or LLM-judge with tracker-fact check |
| Anomaly | P, R, false-positive rate per hour, share of alerts traced to tracker faults |
| Performance | FPS, end-to-end latency, peak GPU memory, VLM calls per hour |

### 13.4 Additional experiments that would show novelty
1. Gap-length curve: link accuracy versus disappearance duration, with and without abstention.
2. Abstention sweep: precision/coverage trade-off.
3. VLM grounding ablation: with vs. without tracker facts in the prompt; hallucinated-ID rate.
4. Keyframe policy ablation: uniform vs. event-triggered vs. retrieval-densify (VideoBrain-style) at equal VLM budget.
5. Enhancement ablation: raw vs. enhanced for mAP and IDF1.
6. Health-gated vs. ungated anomaly alerts.

---

## 14. Research novelty and potential improvements

| Claim | Status |
|---|---|
| Stateful tracking across chunked processing | [E] mostly engineering; not novel alone |
| Tracklet-to-global identity linking with margin-based abstention and audit trail | [A] components exist; the underwater long-gap evaluation is the contribution |
| Tracker-grounded VLM with ID validation and hierarchical text memory | [P] plausible contribution; needs experiments |
| Tracker-health-gated anomaly detection | [P] plausible, needs validation |
| Evaluation protocol for gap-aware identity with abstention on underwater video | [P] likely the most defensible novelty, because public data has this gap |
| Individual re-id of schooling fish over minutes | [R] may simply not be feasible; report it honestly if so |

**Reduce risk:** frame the paper around the *protocol and the uncertainty-aware identity layer*, not around claims of perfect long-term re-id.

---

## 15. Final recommended stack

| Component | Prototype | Postpone |
|---|---|---|
| Decode/render | PyAV or FFmpeg pipe + OpenCV drawing | |
| Detector | Fine-tuned YOLO (small/medium) | RT-DETR, TensorRT |
| Tracker | Ultralytics BoT-SORT (persist), ByteTrack as baseline | SU-T/TIDE comparisons |
| ReID | Small CNN embeddings of crops, fine-tuned if data allows | SAM-based appearance |
| Identity | `prototype/identity.py` | graph DB |
| Segmentation | none | SAM 3 for auto-labelling/ambiguity |
| VLM | Qwen3-VL 4B (or 2B), quantised, async worker | InternVideo3 (verify release), larger models |
| Storage | SQLite (+ FAISS-CPU) | Postgres/pgvector |
| Anomaly | rolling robust statistics with health gating | learned models |
| API/UI | FastAPI + Streamlit | Gradio, auth |
| Edge | none | ONNX/TensorRT |

### Smallest prototype that demonstrates the contribution (Q15)
1. One fine-tuned detector. 2. BoT-SORT with persistent state. 3. The identity manager with abstention. 4. One 30–60 min video with a handful of annotated targets and planted gaps. 5. Experiments A/B/C with HOTA/IDF1 and gap-link precision/recall. The VLM and anomaly modules can follow once this works.

---

## Direct answers to the 15 questions

1. **Is clipping a good approach?** For I/O and scheduling, yes. As a *state boundary*, no. Keep state across clips.
2. **Individual images, batches, or encoded segments?** Decode once; pass in-memory batches to the detector. Pass tracker-grounded keyframes (images) to the VLM. Encoded segments only if the model needs native video input.
3. **Is packetization necessary?** No locally. Use typed metadata records and queues. Only edge→cloud transport needs a protocol, and it should carry metadata and crops.
4. **Sequential or parallel?** Tracking: sequential per camera. Detection batches, VLM, and rendering: parallel/async.
5. **Identity across clip boundaries?** Persistent tracker state plus checkpointing; overlap and stitching only for parallel chunk workers.
6. **Re-id after minutes?** Only through the global identity layer, and only where appearance, location and time plausibility give a clear margin; otherwise abstain.
7. **Can a VLM hold identities?** Not reliably. Hand identity to the tracker/identity manager, and feed the VLM verified facts.
8. **Combining tracker metadata and VLM text?** Tracker JSON goes into the prompt; VLM output is validated against tracker IDs and stored in `segments` linked to identities/events by timestamp.
9. **Most efficient temporal memory?** The three-tier design (§7): structured tables for facts, text summaries plus embeddings for semantics.
10. **Avoiding redundant frames?** Event-triggered keyframes, perceptual-hash dedup, scene-change gating, bounded queues; detect every frame, VLM only on keyframes.
11. **Cloud/edge split?** Detect/track/identity at the edge; VLM and search in the cloud, fed only with events and keyframes.
12. **Closest research?** §4.4: MFT25/SU-T, FishTrack23, SFISHTRACK, SAM2Long/SAM 3, VideoBrain/InternVideo3. None verified as covering your full combination.
13. **What's novel?** §14: the uncertainty-aware identity layer plus the gap-aware evaluation protocol and health-gated anomalies, pending a literature check.
14. **Biggest risks?** Re-id feasibility, lack of long-gap ground truth, VLM hallucination, turbidity.
15. **Smallest prototype?** §15.

---

## Sources
- SAM 3 official repo: <https://github.com/facebookresearch/sam3>; Transformers docs: <https://huggingface.co/docs/transformers/model_doc/sam3_video>
- MFT25 / SU-T: <https://arxiv.org/html/2507.06400v3>, <https://vranlee.github.io/SU-T/>
- SFISHTRACK (UIB record): <https://portalinvestigacio.uib.es/documentos/6a29a0822b7a0802a525921a>
- FishTrack23: <https://openaccess.thecvf.com/content/WACV2024/html/Dawkins_FishTrack23_An_Ensemble_Underwater_Dataset_for_Multi-Object_Tracking_WACV_2024_paper.html>
- BrackishMOT: <https://arxiv.org/pdf/2302.10645>
- Qwen3-VL: <https://arxiv.org/pdf/2511.21631v2>, <https://github.com/QwenLM/Qwen3-VL>
- InternVideo2.5: <https://arxiv.org/html/2501.12386v2>; InternVideo3: <https://arxiv.org/pdf/2606.12195> (release unverified)
- YOLO26: <https://www.ultralytics.com/blog/ultralytics-yolo26-the-new-standard-for-edge-first-vision-ai>, <https://docs.ultralytics.com/>
- VideoBrain: <https://arxiv.org/abs/2602.04094>
- Not re-verified this session: BoT-SORT, ByteTrack, DeepSORT, SAM2/SAM2Long, RT-DETR, TrackEval, FathomNet, WaterNet/UWCNN. Confirm each repo and licence directly.
