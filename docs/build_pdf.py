"""Builds docs/underwater-video-project-plan.pdf. Run: python docs/build_pdf.py"""
from pathlib import Path

from reportlab.graphics.shapes import Drawing, Line, Polygon, Rect, String
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import (KeepTogether, PageBreak, Paragraph, SimpleDocTemplate,
                                Spacer, Table, TableStyle)

OUT = Path(__file__).with_name("underwater-video-project-plan.pdf")
NAVY = colors.HexColor("#0B3954")
TEAL = colors.HexColor("#087E8B")
LIGHT = colors.HexColor("#E8F1F2")
AMBER = colors.HexColor("#FFF4D6")
RED = colors.HexColor("#FBE3E3")
GREEN = colors.HexColor("#E3F4E8")

ss = getSampleStyleSheet()
H1 = ParagraphStyle("H1", parent=ss["Heading1"], textColor=NAVY, fontSize=17, spaceBefore=6, spaceAfter=8)
H2 = ParagraphStyle("H2", parent=ss["Heading2"], textColor=TEAL, fontSize=12.5, spaceBefore=10, spaceAfter=4)
B = ParagraphStyle("B", parent=ss["BodyText"], fontSize=9.6, leading=13.2, spaceAfter=5)
SM = ParagraphStyle("SM", parent=B, fontSize=8.4, leading=10.8, spaceAfter=0)
SMB = ParagraphStyle("SMB", parent=SM, fontName="Helvetica-Bold", textColor=colors.white)
BUL = ParagraphStyle("BUL", parent=B, leftIndent=12, bulletIndent=2, spaceAfter=2)
NOTE = ParagraphStyle("NOTE", parent=B, fontSize=8.8, leading=11.5, textColor=colors.HexColor("#444444"))
TITLE = ParagraphStyle("T", parent=ss["Title"], textColor=NAVY, fontSize=24, leading=29, alignment=TA_CENTER)
SUB = ParagraphStyle("S", parent=B, fontSize=12, leading=16, alignment=TA_CENTER, textColor=TEAL)


def P(t, s=B):
    return Paragraph(t, s)


def bullets(items):
    return [Paragraph(i, BUL, bulletText="•") for i in items]


def link(url, text=None):
    return f'<link href="{url}" color="#087E8B"><u>{text or url}</u></link>'


def table(rows, widths, head_bg=NAVY, zebra=LIGHT, row_bg=None):
    data = [[P(c, SMB) for c in rows[0]]] + [[P(c, SM) for c in r] for r in rows[1:]]
    t = Table(data, colWidths=[w * mm for w in widths], repeatRows=1)
    st = [("BACKGROUND", (0, 0), (-1, 0), head_bg),
          ("VALIGN", (0, 0), (-1, -1), "TOP"),
          ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#B8C7CC")),
          ("LEFTPADDING", (0, 0), (-1, -1), 4), ("RIGHTPADDING", (0, 0), (-1, -1), 4),
          ("TOPPADDING", (0, 0), (-1, -1), 3), ("BOTTOMPADDING", (0, 0), (-1, -1), 3)]
    for i in range(1, len(data)):
        bg = (row_bg[i - 1] if row_bg else (zebra if i % 2 == 0 else colors.white))
        st.append(("BACKGROUND", (0, i), (-1, i), bg))
    t.setStyle(TableStyle(st))
    return t


def box(text, colour=LIGHT, border=TEAL):
    t = Table([[P(text, B)]], colWidths=[170 * mm])
    t.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, -1), colour),
                           ("BOX", (0, 0), (-1, -1), 0.8, border),
                           ("LEFTPADDING", (0, 0), (-1, -1), 8), ("RIGHTPADDING", (0, 0), (-1, -1), 8),
                           ("TOPPADDING", (0, 0), (-1, -1), 6), ("BOTTOMPADDING", (0, 0), (-1, -1), 6)]))
    return t


# ---------- architecture diagram ----------
def arrow(d, x1, y1, x2, y2):
    d.add(Line(x1, y1, x2, y2, strokeColor=NAVY, strokeWidth=1))
    import math
    a = math.atan2(y2 - y1, x2 - x1)
    s = 5
    p1 = (x2 - s * math.cos(a - 0.4), y2 - s * math.sin(a - 0.4))
    p2 = (x2 - s * math.cos(a + 0.4), y2 - s * math.sin(a + 0.4))
    d.add(Polygon([x2, y2, *p1, *p2], fillColor=NAVY, strokeColor=NAVY))


def node(d, x, y, w, h, title, sub, fill):
    d.add(Rect(x, y, w, h, rx=4, ry=4, fillColor=fill, strokeColor=TEAL, strokeWidth=0.8))
    d.add(String(x + w / 2, y + h - 12, title, textAnchor="middle", fontName="Helvetica-Bold", fontSize=8, fillColor=NAVY))
    for k, line in enumerate(sub.split("|")):
        d.add(String(x + w / 2, y + h - 23 - 9 * k, line, textAnchor="middle", fontName="Helvetica", fontSize=6.6))


def diagram():
    W, H = 480, 330
    d = Drawing(W, H)
    w, h = 108, 44
    # row 1
    node(d, 5, 270, w, h, "1 Video input", "file / camera / ROV|keep every timestamp", GREEN)
    node(d, 128, 270, w, h, "2 Detector", "fine-tuned YOLO|boxes per frame", GREEN)
    node(d, 251, 270, w, h, "3 Tracker", "BoT-SORT, never reset|short-term IDs", GREEN)
    node(d, 374, 270, w, h, "4 Identity memory", "re-link after gaps|or say 'uncertain'", AMBER)
    arrow(d, 113, 292, 128, 292); arrow(d, 236, 292, 251, 292); arrow(d, 359, 292, 374, 292)
    # row 2
    node(d, 374, 190, w, h, "5 Self-taught ReID", "learns THESE fish|from the video itself", RED)
    node(d, 251, 190, w, h, "6 Event + alert", "arrive / leave / crowd|ignore bad visibility", AMBER)
    node(d, 128, 190, w, h, "7 Keyframe picker", "only important frames|no duplicates", AMBER)
    node(d, 5, 190, w, h, "8 VLM (Qwen3-VL)", "describes what happens|only confirmed IDs", AMBER)
    arrow(d, 428, 270, 428, 234); arrow(d, 400, 234, 400, 270)
    arrow(d, 374, 212, 359, 212); arrow(d, 251, 212, 236, 212); arrow(d, 128, 212, 113, 212)
    # row 3
    node(d, 5, 110, w, h, "9 Summaries", "clip -> 10 min -> video|stored as text", GREEN)
    node(d, 128, 110, w, h, "10 Database", "SQLite + FAISS|tracks, events, text", GREEN)
    node(d, 251, 110, w, h, "11 Video renderer", "original frames|+ boxes + final IDs", GREEN)
    node(d, 374, 110, w, h, "12 Dashboard", "play, search, timeline|alerts", GREEN)
    arrow(d, 59, 190, 59, 154); arrow(d, 113, 132, 128, 132); arrow(d, 236, 132, 251, 132); arrow(d, 359, 132, 374, 132)
    arrow(d, 305, 190, 182, 154)
    # legend
    for i, (c, t) in enumerate([(GREEN, "Ready to build with existing tools"),
                                (AMBER, "Exists, but we must adapt it"),
                                (RED, "New - our own research part")]):
        d.add(Rect(10 + i * 158, 70, 12, 10, fillColor=c, strokeColor=TEAL, strokeWidth=0.6))
        d.add(String(27 + i * 158, 72, t, fontName="Helvetica", fontSize=7.5))
    d.add(String(W / 2, 45, "Memory A (live tracks, in RAM)  ->  Memory B (identities & events, SQLite)  ->  Memory C (summaries, FAISS)",
                 textAnchor="middle", fontName="Helvetica-Oblique", fontSize=7.6, fillColor=NAVY))
    d.add(String(W / 2, 30, "Clips are only work units for saving and the VLM. The tracker and memory never restart at a clip boundary.",
                 textAnchor="middle", fontName="Helvetica-Oblique", fontSize=7.6, fillColor=NAVY))
    return d


def on_page(c, doc):
    c.saveState()
    c.setFont("Helvetica", 7.5)
    c.setFillColor(colors.grey)
    c.drawString(20 * mm, 10 * mm, "Long-Context Underwater Video Understanding - Project Plan")
    c.drawRightString(190 * mm, 10 * mm, f"Page {doc.page}")
    c.restoreState()


story = []
# ---------- cover ----------
story += [Spacer(1, 45 * mm),
          P("Long-Context Underwater Video Understanding and Object Tracking", TITLE),
          Spacer(1, 6 * mm),
          P("Project plan: the problem, what we are building, what already exists,<br/>and what we must solve to make it our own", SUB),
          Spacer(1, 25 * mm),
          box("<b>One-line idea:</b> break long underwater videos into manageable pieces of work, "
              "<b>without breaking</b> the memory of what happened or which fish is which.", LIGHT),
          Spacer(1, 10 * mm),
          box("<b>How to read the labels in this document</b><br/>"
              "<b>[Verified]</b> - checked by web search in November 2026 (link given). Still confirm licence and numbers on the official page.<br/>"
              "<b>[Not re-checked]</b> - well-known tool, but not re-verified for this document.<br/>"
              "<b>[Our idea]</b> - a proposal that has not been tested yet.", AMBER, colors.HexColor("#C9A227")),
          Spacer(1, 30 * mm),
          P("Date: 12 November 2026", ParagraphStyle("c", parent=B, alignment=TA_CENTER)),
          PageBreak()]

# ---------- contents ----------
story += [P("Contents", H1)]
for i, t in enumerate(["The problem", "What we are building", "System design (diagram)",
                       "What is available on the internet right now", "What we can build today",
                       "Problems we must solve (this makes it ours)", "Our new merged approach",
                       "How we prove it works", "Build plan, step by step", "Hardware and costs",
                       "Risks and honest limits", "Sources"], 1):
    story.append(P(f"{i}.  {t}", B))
story.append(PageBreak())

# ---------- 1 problem ----------
story += [P("1. The problem", H1),
          P("Underwater cameras (fixed cameras, ROVs, AUVs) record for hours. Today most AI systems look at one frame "
            "or a short clip at a time. That causes real failures:"),
          *bullets(["A fish hides behind coral or in murky water and comes back with a <b>new ID</b>, so one fish is counted many times.",
                    "When a long video is cut into clips, every clip <b>forgets</b> the previous one.",
                    "Many fish of the same species <b>look almost identical</b>, so it is easy to swap their identities.",
                    "Water quality changes all the time: green/blue colour, low contrast, floating particles, poor light, motion blur.",
                    "There is no easy way to ask <i>\"what happened in this 3-hour video?\"</i> or <i>\"when did the fish gather?\"</i>.",
                    "Simple alert systems cannot tell a <b>real unusual event</b> from a <b>tracking mistake</b> or bad visibility."]),
          P("Why it matters", H2),
          P("Marine biologists, fish farms and inspection teams need correct counts, movement histories and searchable archives. "
            "Watching hours of footage by hand does not scale."),
          P("The core research question", H2),
          box("<b>How can we keep context and object identities across a long underwater video, "
              "while still processing it efficiently in smaller pieces?</b>"),
          PageBreak()]

# ---------- 2 what we build ----------
story += [P("2. What we are building", H1),
          P("You upload a long underwater video (or connect a camera). The system returns:"),
          table([["#", "Output", "What it means for the user"],
                 ["1", "Annotated video", "The same video, same length and frame rate, with a box and ID on every animal or object."],
                 ["2", "Track table", "For each object: class, when it appeared, its path, when it left."],
                 ["3", "Persistent IDs", "The same fish keeps its ID after short disappearances. If the system is not sure, it says <b>uncertain</b> instead of guessing."],
                 ["4", "Event timeline", "Timestamped events: arrived, left, came back, group formed, visibility dropped."],
                 ["5", "Summaries", "Short text for each part of the video and for the whole video, written by a vision-language model."],
                 ["6", "Alerts", "Unusual crowding, sudden emptiness, odd movement, unexpected objects."],
                 ["7", "Search", "Search by fish ID, time, event type or plain text (\"when did many fish gather?\")."],
                 ["8", "Dashboard", "A web page to play the video, see the timeline and read summaries and alerts."]],
                [8, 35, 127]),
          Spacer(1, 6),
          P("Three kinds of memory", H2),
          table([["Memory", "Holds", "Lives for"],
                 ["A. Short-term (tracking)", "Recent boxes, speed, appearance, current IDs, lost tracks", "Seconds to minutes (in RAM)"],
                 ["B. Persistent (objects and events)", "Global identities, paths, first/last seen, re-identification history, events", "The whole video (SQLite database)"],
                 ["C. Semantic (meaning)", "Clip summaries, activity descriptions, video summary, search vectors", "The whole archive (SQLite + FAISS)"]],
                [48, 82, 40]),
          Spacer(1, 6),
          box("<b>Important design rule:</b> clips are only a unit of work (saving, checkpoints, VLM windows). "
              "The tracker and the memory run continuously and <b>never restart at a clip boundary</b>. "
              "Sending frames as network \"packets\" is not needed on one machine; in-memory batches with a frame number and timestamp are enough.",
              GREEN, TEAL),
          PageBreak()]

# ---------- 3 diagram ----------
story += [P("3. System design", H1), diagram(),
          P("How data flows", H2),
          *bullets(["<b>Sequential:</b> the tracker and identity memory must see frames in order (one per camera).",
                    "<b>Parallel:</b> detection in batches, the VLM in a separate worker, and video rendering.",
                    "<b>Failures:</b> everything is saved by frame number. After a crash we resume from the last checkpoint. "
                    "If the VLM fails, tracking keeps going.",
                    "<b>Correct output video:</b> we re-read the original video and draw the final IDs on every original frame, "
                    "matched by frame number - never rebuilt from sampled frames."]),
          PageBreak()]

# ---------- 4 available ----------
story += [P("4. What is available on the internet right now", H1),
          P("Models and tools", H2),
          table([["Job", "Tool", "Status / notes", "Link"],
                 ["Read / write video", "FFmpeg, PyAV, OpenCV", "Free, open source. [Not re-checked]", "ffmpeg.org"],
                 ["Detect objects", "Ultralytics YOLO26 (or YOLO11)", "[Verified] NMS-free, edge-friendly. Licence AGPL-3.0 or paid Enterprise.", link("https://docs.ultralytics.com/", "docs.ultralytics.com")],
                 ["Detect (alternative)", "RT-DETR", "Transformer detector, heavier. [Not re-checked]", "-"],
                 ["Short-term tracking", "BoT-SORT, ByteTrack", "Built into Ultralytics. BoT-SORT handles camera motion (ROV). [Not re-checked]", "-"],
                 ["Fish-specific tracker", "SU-T", "[Verified] Unscented Kalman filter + fish-shaped IoU. 34.1 HOTA / 44.6 IDF1 on MFT25.", link("https://vranlee.github.io/SU-T/", "vranlee.github.io/SU-T")],
                 ["Schooling-fish tracker", "TIDE", "[Verified, partly] Only an aggregator listing found; read the paper first.", link("https://vranlee.github.io/TIDE", "vranlee.github.io/TIDE")],
                 ["Segment / track by text", "SAM 3 (Meta)", "[Verified] Text prompt like \"fish\" finds and tracks all fish. Heavy; use for labelling and hard cases.", link("https://github.com/facebookresearch/sam3", "github.com/facebookresearch/sam3")],
                 ["Video features", "V-JEPA 2 / 2.1 (Meta)", "[Verified] Self-supervised video encoder. 2.1 has dense features; small 80M/300M versions.", link("https://github.com/facebookresearch/vjepa2", "github.com/facebookresearch/vjepa2")],
                 ["Vision-language model", "Qwen3-VL (2B/4B/8B/32B + MoE)", "[Verified] Long context, timestamp alignment. Main choice for summaries.", link("https://github.com/QwenLM/Qwen3-VL", "github.com/QwenLM/Qwen3-VL")],
                 ["Vision-language (alt.)", "InternVideo2.5 / InternVideo3", "[Verified] 2.5 is released. InternVideo3 paper exists; code release not confirmed.", link("https://github.com/OpenGVLab/InternVideo", "github.com/OpenGVLab/InternVideo")],
                 ["Smart frame selection", "VideoBrain", "[Verified] ICML 2026. Uses 30-40% fewer frames. We borrow the idea.", link("https://arxiv.org/abs/2602.04094", "arXiv 2602.04094")],
                 ["Self-taught identity", "new idtracker.ai", "[Verified] Learns individual identities from the video itself with contrastive learning (lab videos).", link("https://elifesciences.org/reviewed-preprints/107602v1.pdf", "eLife preprint")],
                 ["Global track linking", "Min-cost flow, GMCP, CoNo-Link", "[Verified] Classic and newer methods to link track pieces over a whole video.", link("https://arxiv.org/html/2312.08951v1", "arXiv 2312.08951")],
                 ["Storage / search", "SQLite, FAISS", "Free. Enough for a prototype. [Not re-checked]", "-"],
                 ["Accuracy scoring", "TrackEval", "HOTA, IDF1, ID switches. [Not re-checked]", "-"],
                 ["App", "FastAPI, Streamlit", "Free. [Not re-checked]", "-"]],
                [28, 33, 72, 37]),
          Spacer(1, 6),
          P("Datasets", H2),
          table([["Dataset", "What is inside", "Track IDs?", "Licence", "Good for"],
                 ["SFISHTRACK", "Balearic Sea videos. Secondary source: 54 videos, 24,084 frames, 1080p, 5-40 m deep.", "Yes + masks", "CC BY 4.0 [Verified]", "Closest to real open water. Start here."],
                 ["MFT25", "15 videos, 48,066 frames, 408,578 boxes (groupers, koi).", "Yes", "Check (download may need password)", "Tracker / ReID training. Fish tanks, not open sea."],
                 ["BrackishMOT", "98 sequences, murky brackish water, small fish schools.", "Yes", "CC BY 4.0 (unconfirmed)", "Turbid water testing. Short clips."],
                 ["FishTrack23", "About 850k boxes in about 26k tracks (WACV 2024).", "Yes", "Check", "Pre-training the tracker."],
                 ["Melops", "24,578 images of 9,861 individual wrasse, PIT-tag ground truth (2026).", "Individual IDs (images)", "Check", "Testing individual fish re-ID."],
                 ["FathomNet", "Large marine image archive.", "No (images only)", "Not checked", "Detection only, not tracking."]],
                [24, 60, 22, 30, 34]),
          Spacer(1, 4),
          box("<b>Missing on the internet:</b> no public dataset we found has <b>hour-long</b> underwater video with the same fish "
              "labelled across <b>multi-minute disappearances</b> plus event labels. We will have to make a small one ourselves.",
              RED, colors.HexColor("#C0392B")),
          PageBreak()]

# ---------- 5 buildable today ----------
story += [P("5. What we can build today", H1),
          P("Using only the tools above, without new research, we can build a working system:"),
          table([["Part", "Built with", "Status"],
                 ["Video reading with exact timestamps", "PyAV / FFmpeg", "Ready"],
                 ["Fish detector", "YOLO fine-tuned on SFISHTRACK", "Ready (needs a GPU for training)"],
                 ["Continuous tracking across the whole video", "BoT-SORT with persistent state", "Ready - script in repo: prototype/run_tracking.py (not yet run)"],
                 ["Annotated output video", "Re-draw on original frames", "Ready (same script)"],
                 ["ID re-linking after short gaps, with 'uncertain'", "prototype/identity.py", "Prototype done, tests pass on synthetic data"],
                 ["Database of tracks and events", "SQLite", "Ready"],
                 ["Event rules and simple alerts", "Counts, speed, crowding per time window", "Ready"],
                 ["Summaries", "Qwen3-VL 4B/8B on chosen keyframes", "Ready"],
                 ["Text search", "FAISS over summaries", "Ready"],
                 ["Dashboard", "FastAPI + Streamlit", "Ready"]],
                [62, 60, 48], row_bg=[GREEN] * 10),
          Spacer(1, 8),
          box("This alone is a <b>useful product</b>, but it is mostly <b>engineering</b> - joining existing tools. "
              "To make it a research project that is truly ours, we must solve the problems in the next section."),
          PageBreak()]

# ---------- 6 problems to solve ----------
story += [P("6. Problems we must solve (this makes it ours)", H1),
          P("These are the parts that nobody has fully solved for long underwater video. Solving even one of them well is the "
            "project's own contribution."),
          table([["#", "Problem", "Why existing tools fail", "Our direction"],
                 ["P1", "<b>Recognise the same fish after minutes away</b>, even when others look identical",
                  "Trackers remember for seconds. Normal ReID learns 'species', not 'this fish'.",
                  "Self-taught identity learning from the video itself + global linking + 'uncertain' when not sure (section 7)."],
                 ["P2", "<b>Know when NOT to guess</b>",
                  "Most trackers always pick a match, so mistakes look like facts.",
                  "Link only when the best match is clearly better than the second best. Report the 'uncertain' rate."],
                 ["P3", "<b>Chunked processing without forgetting</b>",
                  "Independent clips reset IDs at every boundary.",
                  "Clips are work units only; tracker state, identity memory and summaries carry across."],
                 ["P4", "<b>VLM that does not invent things</b> in murky water",
                  "VLMs are not tested underwater and can name fish or events that are not there.",
                  "Give the VLM only tracker-confirmed facts; reject any output that names unknown IDs."],
                 ["P5", "<b>Long video understanding on a small GPU</b>",
                  "Sending every frame to a VLM is far too expensive.",
                  "Event-triggered keyframes + duplicate removal + summaries in levels (clip, 10 min, video)."],
                 ["P6", "<b>Real anomaly vs tracking error</b>",
                  "A 'fish disappeared' alert is often just the tracker losing it.",
                  "Health-gated alerts: ignore alerts when visibility is low or the tracker is creating many new IDs."],
                 ["P7", "<b>Underwater image quality</b>",
                  "Enhancement looks nicer to people but may not help the detector.",
                  "Do not enhance by default; test raw vs enhanced. Use a visibility score instead."],
                 ["P8", "<b>No long-gap test data</b>",
                  "Public datasets are short or have no identities across long gaps.",
                  "Create a small labelled test set (20-40 fish, 3-5 long videos) and a new scoring method."]],
                [9, 42, 55, 64]),
          PageBreak()]

# ---------- 7 merged approach ----------
story += [P("7. Our new merged approach  [Our idea]", H1),
          P("Working name: <b>Self-Taught Identity Memory (STIM)</b>. It combines four ideas that exist separately on the internet "
            "into one system for long underwater video. As far as our searches went, we did not find this combination published, "
            "but a full literature search is still needed before claiming it is new."),
          table([["Idea we borrow", "From", "What it gives us"],
                 ["Learn identities from the video itself, no labels", "new idtracker.ai (eLife 2025)", "Free training pairs: same track piece = same fish; two pieces in the same frame = different fish."],
                 ["Contrastive re-ID for wild fish", "Corkwing wrasse study (arXiv 2301.00596)", "Shows contrastive learning works on real wild fish."],
                 ["Link all track pieces together, in rounds", "Min-cost flow, GMCP-Tracker, CoNo-Link", "Avoids chains of greedy mistakes; uses the whole video."],
                 ["Strong video features", "V-JEPA 2.1", "A good starting network for the appearance embedding."],
                 ["Smart frame selection", "VideoBrain", "Fewer frames to the VLM: search first, then look closer."]],
                [52, 50, 68]),
          P("How STIM works", H2),
          *bullets(["<b>Step 1 - Track:</b> BoT-SORT makes short, reliable track pieces (tracklets).",
                    "<b>Step 2 - Self-train:</b> from those pieces, build positive and negative pairs automatically and fine-tune a small "
                    "embedding head (on V-JEPA 2.1 or a CNN) for this video or site. It learns <i>these particular fish</i>.",
                    "<b>Step 3 - Link in rounds:</b> join the most certain pieces first, retrain on the bigger groups, link again.",
                    "<b>Step 4 - Refuse when unsure:</b> if the best match is not clearly better than the second best, mark 'uncertain'.",
                    "<b>Step 5 - Ground the VLM:</b> the VLM describes only confirmed IDs and events; anything else is rejected.",
                    "<b>Step 6 - Health-gated alerts:</b> alerts are trusted only when visibility and tracker health are good."]),
          P("Why this fits our problem", H2),
          *bullets(["No identity labels needed - the video teaches itself.",
                    "It gets <b>better as the video gets longer</b> (more training pairs) - the opposite of normal trackers.",
                    "The 'uncertain' rule gives an honest, measurable result instead of hidden mistakes."]),
          P("Known weak points", H2),
          *bullets(["idtracker.ai was made for clear lab tanks; open water, turbidity and fish leaving the view will add noise.",
                    "If the tracker merges two fish into one piece, the training pairs become wrong - use only high-confidence pieces.",
                    "For truly identical schooling fish it may still fail; then the right answer is 'uncertain' plus a count estimate."]),
          PageBreak()]

# ---------- 8 proof ----------
story += [P("8. How we prove it works", H1),
          table([["Experiment", "What it does", "What it shows"],
                 ["A - Normal tracking", "One tracker over the whole video", "Baseline without long-term memory"],
                 ["B - Independent clips", "Cut into clips, IDs reset each clip", "What breaks when you cut naively"],
                 ["C - Memory-aware", "Continuous tracker + identity memory", "Value of the memory layer"],
                 ["D - STIM", "C + self-taught identity learning", "Our new contribution"]],
                [40, 65, 65]),
          Spacer(1, 6),
          P("Fair testing rules", H2),
          *bullets(["Split by whole video or site, never by frames from the same video.",
                    "Same detector weights in every experiment, so only the tracking/memory part changes.",
                    "Test at gap lengths of 10 seconds, 1 minute and 5 minutes.",
                    "Note: A already keeps state, so C and D will not beat A on very short occlusions. The interesting results are "
                    "C/D vs B, and C/D vs A on long gaps."]),
          P("What we measure", H2),
          table([["Area", "Metrics"],
                 ["Detection", "mAP50, mAP50-95, precision, recall"],
                 ["Tracking", "HOTA, IDF1, ID switches, fragmentation"],
                 ["Long-gap re-ID (new)", "Correct links, wrong links, 'uncertain' rate, identity purity, accuracy vs gap length"],
                 ["Understanding", "Event time error, event precision/recall, invented-ID rate in VLM text"],
                 ["Alerts", "Precision, recall, false alarms per hour, share of alerts caused by tracker errors"],
                 ["Speed", "Frames per second, delay, peak GPU memory, VLM calls per hour"]],
                [40, 130]),
          P("Extra experiments", H2),
          *bullets(["Backbone swap: normal CNN vs V-JEPA 2.1 features for identity.",
                    "Keyframe policy: every N seconds vs event-triggered vs search-then-look-closer, at equal VLM budget.",
                    "VLM with vs without tracker facts in the prompt (invented-ID rate).",
                    "Raw vs enhanced images for detection and tracking.",
                    "Alerts with vs without the health gate."]),
          PageBreak()]

# ---------- 9 build plan ----------
story += [P("9. Build plan, step by step", H1),
          table([["Phase", "Goal", "Main tools", "Done when"],
                 ["1 Detect", "Fine-tune detector, run on video", "Ultralytics YOLO, SFISHTRACK", "mAP measured on held-out videos"],
                 ["2 Track", "Continuous IDs, annotated video", "BoT-SORT, run_tracking.py", "Output video same length and frame count; IDs do not reset at clip edges"],
                 ["3 Memory", "Re-link IDs, store everything", "identity.py, SQLite", "Planted-gap test works; 'uncertain' cases logged"],
                 ["4 STIM", "Self-taught identity learning", "PyTorch, V-JEPA 2.1 or CNN", "Better long-gap link accuracy than phase 3"],
                 ["5 VLM", "Keyframes, summaries in levels", "Qwen3-VL 4B/8B quantised", "Event times match a hand-made list; zero invented IDs"],
                 ["6 Alerts", "Health-gated anomaly alerts", "Python statistics", "Measured precision / false alarms"],
                 ["7 App", "Upload, play, search, timeline", "FastAPI, Streamlit, FAISS", "End-to-end demo on a long video"],
                 ["8 Paper", "Experiments A-D, write-up", "TrackEval, own long-gap metric", "Results tables and figures"]],
                [20, 45, 50, 55]),
          Spacer(1, 6),
          P("Smallest demo that proves the core idea", H2),
          box("Phases 1-3 on one 30-60 minute video with a few labelled fish and planted gaps, compared in experiments A, B and C. "
              "This already shows that chunked processing can keep identities. STIM, the VLM and alerts come next."),
          P("Folder structure", H2),
          P("<font face='Courier' size='8'>docs/ (this plan, research report)<br/>"
            "prototype/ identity.py, test_identity.py, run_tracking.py<br/>"
            "src/uwvid/ io/  detect/  track/  memory/  events/  vlm/  api/<br/>"
            "dashboard/  configs/  eval/  data/ (raw, processed, db, outputs)</font>", B),
          PageBreak()]

# ---------- 10 hardware ----------
story += [P("10. Hardware and costs", H1),
          P("Target: one NVIDIA GPU slice with about 20 GB of memory. The numbers below are <b>rough planning estimates</b> "
            "from model sizes, not measurements."),
          table([["Component", "Rough GPU memory", "Plan"],
                 ["YOLO small/medium, FP16", "about 1-3 GB", "Runs all the time"],
                 ["Small ReID / STIM head", "under 1 GB", "Runs all the time"],
                 ["V-JEPA 2.1 ViT-L (optional)", "about 1-2 GB", "Only for identity features"],
                 ["Qwen3-VL 2B/4B", "about 5-10 GB", "Separate worker, fits alongside detection"],
                 ["Qwen3-VL 8B full precision", "about 16 GB just for weights", "Does not fit with the rest; use a 4/8-bit version"],
                 ["SAM 3", "several GB", "Only on demand or offline for labelling"]],
                [55, 45, 70]),
          Spacer(1, 6),
          *bullets(["Use mixed precision and a bounded frame queue, so memory stays flat no matter how long the video is.",
                    "Edge (on the ROV/camera): detector + tracker + identity. Cloud: VLM + search. Send only events and keyframes.",
                    "All software listed is free; check the YOLO AGPL licence if this becomes a commercial product (RT-DETR is an alternative)."]),
          PageBreak()]

# ---------- 11 risks ----------
story += [P("11. Risks and honest limits", H1),
          table([["Risk", "How likely", "What we do about it"],
                 ["Identical schooling fish cannot be told apart after long gaps", "High", "Report 'uncertain' and counts; measure the 'uncertain' rate honestly"],
                 ["No ready long-gap test data", "Certain", "Label a small custom set with SAM 3 help in CVAT"],
                 ["VLM makes up details in murky water", "Medium-High", "Tracker-grounded prompts and ID validation"],
                 ["Self-training learns wrong pairs from tracker mistakes", "Medium", "Use only high-confidence track pieces; check pair quality"],
                 ["Someone already published the same combination", "Unknown", "Full literature search before writing the paper"],
                 ["Not enough GPU time", "Medium", "Small models first, quantisation, run VLM separately"],
                 ["Dataset licences block some uses", "Medium", "Prefer CC BY data (SFISHTRACK); check each licence"]],
                [62, 25, 83]),
          Spacer(1, 8),
          box("<b>Bottom line:</b> most of this system can be built today from existing open tools. "
              "The part that makes it <b>our own</b> is: long-gap identity with honest uncertainty (STIM), a tracker-grounded VLM memory, "
              "health-gated alerts, and a new way to test all of this on long underwater video.", GREEN, TEAL),
          PageBreak()]

# ---------- 12 sources ----------
src = [("Ultralytics YOLO26 docs", "https://docs.ultralytics.com/"),
       ("YOLO26 announcement", "https://www.ultralytics.com/blog/ultralytics-yolo26-the-new-standard-for-edge-first-vision-ai"),
       ("MFT25 and SU-T paper (arXiv 2507.06400)", "https://arxiv.org/html/2507.06400v3"),
       ("SU-T project page", "https://vranlee.github.io/SU-T/"),
       ("SFISHTRACK record (UIB)", "https://portalinvestigacio.uib.es/documentos/6a29a0822b7a0802a525921a"),
       ("FishTrack23 (WACV 2024)", "https://openaccess.thecvf.com/content/WACV2024/html/Dawkins_FishTrack23_An_Ensemble_Underwater_Dataset_for_Multi-Object_Tracking_WACV_2024_paper.html"),
       ("BrackishMOT (arXiv 2302.10645)", "https://arxiv.org/pdf/2302.10645"),
       ("Melops dataset (Scientific Data 2026)", "https://iris.univr.it/retrieve/fe930e5b-2ab4-4e29-a004-01f0efc87f87/s41597-026-07045-1.pdf"),
       ("SAM 3 official repository", "https://github.com/facebookresearch/sam3"),
       ("SAM 3 in Transformers", "https://huggingface.co/docs/transformers/model_doc/sam3_video"),
       ("V-JEPA 2 / 2.1 repository", "https://github.com/facebookresearch/vjepa2"),
       ("V-JEPA 2.1 paper (arXiv 2603.14482)", "https://arxiv.org/abs/2603.14482"),
       ("Qwen3-VL technical report", "https://arxiv.org/pdf/2511.21631v2"),
       ("Qwen3-VL repository", "https://github.com/QwenLM/Qwen3-VL"),
       ("InternVideo2.5 paper", "https://arxiv.org/html/2501.12386v2"),
       ("InternVideo3 paper (code release unconfirmed)", "https://arxiv.org/pdf/2606.12195"),
       ("VideoBrain (arXiv 2602.04094)", "https://arxiv.org/abs/2602.04094"),
       ("New idtracker.ai (eLife reviewed preprint)", "https://elifesciences.org/reviewed-preprints/107602v1.pdf"),
       ("Contrastive re-ID of wild fish (arXiv 2301.00596)", "https://arxiv.org/abs/2301.00596v1"),
       ("Zhang et al. 2008, network flow with occlusion model", "https://www.cvlibs.net/projects/autonomous_vision_survey/literature/Zhang2008CVPR.pdf"),
       ("GMCP-Tracker (ECCV 2012)", "https://vilab.epfl.ch/zamir/index_files/GMCP-Tracker_ECCV12.pdf"),
       ("CoNo-Link (arXiv 2312.08951)", "https://arxiv.org/html/2312.08951v1")]
story += [P("12. Sources", H1),
          P("Checked by web search in November 2026. Licences and numbers should be confirmed on each official page before use.", NOTE)]
for i, (t, u) in enumerate(src, 1):
    story.append(P(f"{i}. {t} - {link(u)}", SM))
    story.append(Spacer(1, 2))
story += [Spacer(1, 6),
          P("Not re-checked for this document: BoT-SORT, ByteTrack, DeepSORT, SAM2 / SAM2Long, RT-DETR, TrackEval, FathomNet, "
            "WaterNet / UWCNN, FFmpeg, FAISS, FastAPI, Streamlit.", NOTE)]

doc = SimpleDocTemplate(str(OUT), pagesize=A4, leftMargin=20 * mm, rightMargin=20 * mm,
                        topMargin=18 * mm, bottomMargin=18 * mm,
                        title="Long-Context Underwater Video Understanding - Project Plan",
                        author="Project team")
doc.build(story, onFirstPage=on_page, onLaterPages=on_page)
print(OUT)
