#!/usr/bin/env python3
"""Generate the SVG assets for Volume II. Output dir is argv[1]."""
import os, sys, html

OUT = sys.argv[1]
os.makedirs(OUT, exist_ok=True)

# Palette (dataviz reference, dark surface, validated)
SURF = "#1a1a19"; SURF2 = "#22221f"; LINE = "#33332f"; GRID = "#2b2b28"
T1 = "#ffffff"; T2 = "#c3c2b7"; T3 = "#8b8a82"
BLUE = "#3987e5"; ORANGE = "#d95926"; AQUA = "#199e70"; YELLOW = "#c98500"; MAGENTA = "#d55181"; VIOLET = "#9085e9"
CRIT = "#d03b3b"; GOOD = "#0ca30c"
FONT = "Inter, 'Segoe UI', Helvetica, Arial, sans-serif"
MONO = "'JetBrains Mono', 'SFMono-Regular', Consolas, Menlo, monospace"

def esc(s): return html.escape(str(s), quote=True)

def svg(w, h, body, title):
    return (f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}" '
            f'font-family="{FONT}" role="img" aria-label="{esc(title)}">\n'
            f'<title>{esc(title)}</title>\n'
            f'<defs><linearGradient id="acc" x1="0" x2="1"><stop offset="0" stop-color="{BLUE}"/>'
            f'<stop offset="0.5" stop-color="{AQUA}"/><stop offset="1" stop-color="{ORANGE}"/></linearGradient>'
            f'<linearGradient id="bg" x1="0" y1="0" x2="1" y2="1"><stop offset="0" stop-color="#1c1c1b"/>'
            f'<stop offset="1" stop-color="#141413"/></linearGradient></defs>\n'
            f'<rect x="0.5" y="0.5" width="{w-1}" height="{h-1}" rx="14" fill="url(#bg)" stroke="{LINE}"/>\n'
            f'<rect x="24" y="0" width="{w-48}" height="3" rx="1.5" fill="url(#acc)"/>\n'
            + body + '\n</svg>\n')

def text(x, y, s, size=14, fill=T2, anchor="start", weight="normal", family=None, extra=""):
    fam = f' font-family="{family}"' if family else ""
    return (f'<text x="{x}" y="{y}" font-size="{size}" fill="{fill}" text-anchor="{anchor}" '
            f'font-weight="{weight}"{fam} {extra}>{esc(s)}</text>')

def header(w, title, sub, chapter=None):
    b = ""
    if chapter:
        b += f'<rect x="32" y="26" width="{14 + 8*len(chapter)}" height="24" rx="6" fill="{SURF2}" stroke="{LINE}"/>'
        b += text(39 + 4*len(chapter), 43, chapter, 12, T2, "middle", "600", MONO)
        b += text(32 + 14 + 8*len(chapter) + 14, 44, title, 20, T1, "start", "700")
    else:
        b += text(32, 44, title, 20, T1, "start", "700")
    b += text(32, 68, sub, 13, T3)
    return b

def footer(w, h, s):
    return text(w - 32, h - 16, s, 11, T3, "end")

def write(name, content):
    with open(os.path.join(OUT, name), "w", encoding="utf-8") as f:
        f.write(content)

# ---------- generic charts ----------
def hbar_chart(w, h, title, sub, chapter, rows, unit="", vmax=None, color=BLUE, top=92, label_w=260, highlight=None, foot=""):
    """rows: list of (label, value, color_override or None, note)"""
    vmax = vmax or max(r[1] for r in rows) * 1.08
    n = len(rows); area_h = h - top - 66; bh = min(26, (area_h / n) * 0.62); gap = area_h / n
    x0 = 32 + label_w; x1 = w - 120; sx = (x1 - x0) / vmax
    b = header(w, title, sub, chapter)
    # grid
    for t in range(5):
        gx = x0 + (x1 - x0) * t / 4
        b += f'<line x1="{gx:.1f}" y1="{top-6}" x2="{gx:.1f}" y2="{top+area_h}" stroke="{GRID}" stroke-width="1"/>'
        b += text(gx, top + area_h + 16, f"{vmax*t/4:.0f}{unit}" if vmax >= 20 else f"{vmax*t/4:.1f}{unit}", 10, T3, "middle")
    for i, (label, val, col, note) in enumerate(rows):
        y = top + i * gap + (gap - bh) / 2
        c = col or color
        b += text(x0 - 12, y + bh/2 + 5, label, 13, T2 if (highlight is None or label != highlight) else T1, "end", "400" if (highlight is None or label != highlight) else "700")
        wv = max(4, val * sx)
        b += f'<rect x="{x0}" y="{y:.1f}" width="{wv:.1f}" height="{bh:.1f}" rx="4" fill="{c}" opacity="0.92"/>'
        b += f'<rect x="{x0}" y="{y:.1f}" width="4" height="{bh:.1f}" fill="{c}"/>'
        vs = f"{val:g}{unit}" if not note else f"{val:g}{unit}  {note}"
        b += text(x0 + wv + 10, y + bh/2 + 5, vs, 12, T1, "start", "600", MONO)
    if foot: b += footer(w, h, foot)
    return svg(w, h, b, title)

def grouped_bars(w, h, title, sub, chapter, groups, series, unit="", vmax=None, top=100, foot=""):
    """groups: list of (label, [v1, v2,...]); series: list of (name, color)"""
    vmax = vmax or max(max(g[1]) for g in groups) * 1.15
    n = len(groups); k = len(series)
    x0 = 70; x1 = w - 40; area_h = h - top - 60; y0 = top + area_h
    gw = (x1 - x0) / n; bw = min(44, gw * 0.7 / k); sy = area_h / vmax
    b = header(w, title, sub, chapter)
    for t in range(5):
        gy = y0 - area_h * t / 4
        b += f'<line x1="{x0}" y1="{gy:.1f}" x2="{x1}" y2="{gy:.1f}" stroke="{GRID}"/>'
        b += text(x0 - 8, gy + 4, f"{vmax*t/4:.0f}{unit}" if vmax >= 20 else f"{vmax*t/4:.1f}{unit}", 10, T3, "end")
    for gi, (gl, vals) in enumerate(groups):
        cx = x0 + gw * gi + gw / 2
        total = k * bw + (k - 1) * 2
        for si, v in enumerate(vals):
            x = cx - total / 2 + si * (bw + 2)
            hgt = v * sy
            b += f'<rect x="{x:.1f}" y="{y0-hgt:.1f}" width="{bw:.1f}" height="{hgt:.1f}" rx="4" fill="{series[si][1]}" opacity="0.92"/>'
            b += text(x + bw/2, y0 - hgt - 6, f"{v:g}{unit}", 11, T1, "middle", "600", MONO)
        b += text(cx, y0 + 18, gl, 12, T2, "middle")
    # legend
    lx = w - 40
    for si in (range(k - 1, -1, -1) if k > 1 else []):
        name, col = series[si]
        tw = 7 * len(name) + 26
        lx -= tw
        b += f'<rect x="{lx}" y="{top-24}" width="12" height="12" rx="3" fill="{col}"/>'
        b += text(lx + 18, top - 14, name, 12, T2)
        lx -= 16
    if foot: b += footer(w, h, foot)
    return svg(w, h, b, title)

def stacked_bar(w, h, title, sub, chapter, segs, total_label, foot=""):
    """segs: list of (label, value, color)"""
    total = sum(s[1] for s in segs)
    x0 = 32; x1 = w - 32; y = 110; bh = 46
    b = header(w, title, sub, chapter)
    x = x0
    for i, (label, v, col) in enumerate(segs):
        wv = (x1 - x0) * v / total
        b += f'<rect x="{x:.1f}" y="{y}" width="{max(wv-2,2):.1f}" height="{bh}" rx="5" fill="{col}" opacity="0.92"/>'
        b += text(x + wv/2, y + bh/2 + 5, f"{v/total*100:.0f}%", 14, T1, "middle", "700", MONO)
        # label below with connector
        ly = y + bh + 26 + (i % 2) * 22
        b += f'<line x1="{x+wv/2:.1f}" y1="{y+bh+4}" x2="{x+wv/2:.1f}" y2="{ly-12}" stroke="{LINE}"/>'
        b += text(x + wv/2, ly, f"{label}  {v:g} s", 12, T2, "middle")
        x += wv
    b += text(x1, y - 14, total_label, 12, T3, "end")
    if foot: b += footer(w, h, foot)
    return svg(w, h, b, title)

# ---------- banner and parts ----------
def banner():
    w, h = 1200, 420
    b = ""
    # decorative grid of dots
    for i in range(0, w, 40):
        for j in range(0, h, 40):
            b += f'<circle cx="{i+20}" cy="{j+20}" r="1" fill="#2a2a27"/>'
    b += text(60, 92, "THE LLM INFERENCE HANDBOOK", 14, AQUA, "start", "700", extra='letter-spacing="4"')
    b += text(60, 160, "Volume II", 64, T1, "start", "800")
    b += text(60, 206, "Production Failure Modes", 30, T2, "start", "600")
    b += text(60, 236, "What breaks after the benchmark passes", 18, T3)
    chips = [("I", "Numerics", BLUE), ("II", "Scheduler and Cache", AQUA), ("III", "Host and Hardware", ORANGE),
             ("IV", "Fleet Realities", YELLOW), ("V", "Arabic Inference", MAGENTA)]
    x = 60
    for num, name, col in chips:
        tw = 8 * len(name) + 70
        b += f'<rect x="{x}" y="336" width="{tw}" height="34" rx="17" fill="{SURF2}" stroke="{LINE}"/>'
        b += f'<circle cx="{x+18}" cy="353" r="6" fill="{col}"/>'
        b += text(x + 32, 358, f"Part {num}", 12, T3, "start", "600")
        b += text(x + 32 + 50, 358, name, 12, T1, "start", "600")
        x += tw + 12
    b += text(60, 290, "22 chapters   5 parts   2 appendices   every number sourced or measured", 13, T3)
    # right side: abstract KV block art
    bx, by = 820, 90
    cols = [BLUE, BLUE, BLUE, AQUA, AQUA, ORANGE, CRIT, "#3a3a37", "#3a3a37"]
    for r in range(6):
        for c in range(9):
            col = cols[c] if r < 4 else ("#3a3a37" if c > 4 else cols[c])
            op = 0.95 - r * 0.1
            b += f'<rect x="{bx + c*34}" y="{by + r*34}" width="28" height="28" rx="5" fill="{col}" opacity="{op:.2f}"/>'
    b += text(bx, by + 6*34 + 20, "prefix hit", 11, T3)
    b += f'<rect x="{bx+60}" y="{by + 6*34 + 10}" width="10" height="10" rx="2" fill="{BLUE}"/>'
    b += text(bx + 76, by + 6*34 + 20, "reused", 11, T3)
    b += f'<rect x="{bx+130}" y="{by + 6*34 + 10}" width="10" height="10" rx="2" fill="{CRIT}"/>'
    b += text(bx + 146, by + 6*34 + 20, "first divergent byte", 11, T3)
    b += f'<rect x="{bx+270}" y="{by + 6*34 + 10}" width="10" height="10" rx="2" fill="#3a3a37"/>'
    b += text(bx + 286, by + 6*34 + 20, "cold", 11, T3)
    return svg(w, h, b, "The LLM Inference Handbook, Volume II: Production Failure Modes")

def part_banner(num, name, chapters, col, blurb):
    w, h = 1200, 150
    b = f'<rect x="0" y="0" width="10" height="{h}" rx="5" fill="{col}"/>'
    b += text(48, 56, f"PART {num}", 13, col, "start", "700", extra='letter-spacing="3"')
    b += text(48, 96, name, 32, T1, "start", "800")
    b += text(48, 124, blurb, 13, T3)
    b += f'<rect x="{w-200}" y="52" width="168" height="30" rx="15" fill="{SURF2}" stroke="{LINE}"/>'
    b += text(w - 116, 72, f"Chapters {chapters}", 12, T2, "middle", "600")
    return svg(w, h, b, f"Part {num}: {name}")

# ---------- chapter diagrams ----------
def ch01():
    w, h = 1200, 330
    b = header(w, "Same request, two batches, two answers", "The kernel is deterministic for a fixed shape. The shape is set by whoever else arrived in the same step.", "Ch 1")
    for i, (lab, n, col, path) in enumerate([("Quiet server", 1, BLUE, "one row per core"), ("Busy server", 32, ORANGE, "split-K, partial sums")]):
        y = 110 + i * 100
        b += text(32, y + 24, lab, 14, T1, "start", "700")
        b += text(32, y + 44, f"batch = {n}", 12, T3, "start", "400", MONO)
        # batch rows
        for r in range(min(n, 12)):
            fill = col if r == 0 else "#3a3a37"
            b += f'<rect x="{170 + r*22}" y="{y+12}" width="18" height="34" rx="4" fill="{fill}"/>'
        if n > 12: b += text(170 + 12*22 + 6, y + 36, "...", 14, T3)
        b += f'<path d="M 470 {y+29} L 540 {y+29}" stroke="{LINE}" stroke-width="2"/><path d="M 534 {y+23} L 544 {y+29} L 534 {y+35}" fill="none" stroke="{LINE}" stroke-width="2"/>'
        b += f'<rect x="552" y="{y+6}" width="250" height="46" rx="8" fill="{SURF2}" stroke="{LINE}"/>'
        b += text(677, y + 26, "reduction strategy", 11, T3, "middle")
        b += text(677, y + 44, path, 13, T1, "middle", "600", MONO)
        b += f'<path d="M 802 {y+29} L 870 {y+29}" stroke="{LINE}" stroke-width="2"/><path d="M 864 {y+23} L 874 {y+29} L 864 {y+35}" fill="none" stroke="{LINE}" stroke-width="2"/>'
        logit = "logit = 12.734 192" if i == 0 else "logit = 12.734 188"
        b += f'<rect x="882" y="{y+6}" width="286" height="46" rx="8" fill="{SURF2}" stroke="{col}"/>'
        b += text(1025, y + 26, "your row, last bits", 11, T3, "middle")
        b += text(1025, y + 44, logit, 13, col, "middle", "700", MONO)
    b += text(32, 310, "Near-tie at argmax, temperature 0: the token flips. 1,000 runs of Qwen3-235B produced about 80 distinct completions.", 12, T2)
    return svg(w, h, b, "Batch invariance mechanism")

def ch02():
    return grouped_bars(1200, 330, "FP8 KV cache: fine until the context is long", "Needle-in-a-haystack accuracy, Hopper, Flash Attention 3 FP8 path before the accumulator fix", "Ch 2",
                        [("Short-context reasoning", [100, 99.3]), ("128k needle retrieval", [91, 13]), ("128k after FP32 partial-sum fix", [91, 89])],
                        [("BF16 KV", BLUE), ("FP8 KV", ORANGE)], unit="%", vmax=110,
                        foot="Short-context deltas were at most 0.7 points. Source: vLLM FP8 KV-cache study, April 2026")

def ch03():
    w, h = 1200, 300
    b = header(w, "E2M1 has no 5", "The sixteen encodings of a 4-bit float, as a share of the block maximum. Everything from 66.6% to 100% lands in one gap.", "Ch 3")
    x0, x1, y = 80, w - 80, 170
    sx = (x1 - x0) / 6
    b += f'<rect x="{x0 + 4*sx:.1f}" y="{y-44}" width="{2*sx:.1f}" height="88" rx="8" fill="{CRIT}" opacity="0.18"/>'
    b += text(x0 + 5*sx, y - 54, "the 4 to 6 gap: 33% of the range, 1 of 8 magnitudes", 12, CRIT, "middle", "700")
    b += f'<line x1="{x0}" y1="{y}" x2="{x1}" y2="{y}" stroke="{T3}" stroke-width="2"/>'
    for v in [0, 0.5, 1, 1.5, 2, 3, 4, 6]:
        x = x0 + v * sx
        b += f'<circle cx="{x:.1f}" cy="{y}" r="9" fill="{BLUE}" stroke="{SURF}" stroke-width="2"/>'
        b += text(x, y + 32, f"{v:g}", 13, T1, "middle", "600", MONO)
        b += text(x, y + 50, f"{v/6*100:.0f}%", 11, T3, "middle")
    # the missing 5
    x5 = x0 + 5 * sx
    b += f'<circle cx="{x5:.1f}" cy="{y}" r="9" fill="none" stroke="{CRIT}" stroke-width="2" stroke-dasharray="3 3"/>'
    b += text(x5, y - 18, "5 (83%)  rounds to 4 or 6: 20% error", 12, CRIT, "middle", "600")
    b += text(32, 272, "NVFP4: 16-element blocks, E4M3 scale.   MXFP4: 32-element blocks, power-of-two scale, about 40% more MSE.", 12, T2)
    return svg(w, h, b, "E2M1 representable values")

def ch04():
    return grouped_bars(1200, 320, "The benchmark barely moves. The humans notice.", "Cohere, 103B model, 4-bit weights: degradation versus FP16", "Ch 4",
                        [("Japanese", [1.7, 16.0]), ("French", [0.3, 16.6])],
                        [("Automatic metrics", BLUE), ("Human raters", ORANGE)], unit="%", vmax=20,
                        foot="Values are percentage-point drops. Source: Marchisio et al., EMNLP 2024 Findings")

def ch05():
    return hbar_chart(1200, 300, "Where a bit flip hurts", "Fault injection on A100s: accuracy cost by phase, and what a NaN check sees", "Ch 5",
                      [("Accuracy points lost, fault during prefill (writes the KV cache)", 4.5, CRIT, "(up to)"),
                       ("Accuracy points lost, fault during decode (one token)", 0.1, BLUE, "(under)"),
                       ("Share of SDC cases that surface as NaN or Inf", 1.01, YELLOW, "% (simulation)")],
                      unit="", vmax=5.5, label_w=420, foot="Single preprint for the phase ranking; gate-level simulation for the NaN share")

def ch06():
    w, h = 1200, 330
    b = header(w, "p50 meets the model. p99 meets the scheduler.", "Two requests, same model, same prompt length. Only the scheduler's decisions differ.", "Ch 6")
    def lane(y, label, col, steps):
        b2 = text(32, y + 20, label, 14, T1, "start", "700")
        x = 150
        for i, (s, wdt, kind) in enumerate(steps):
            fill = col if kind == "work" else ("#3a3a37" if kind == "wait" else CRIT)
            b2 += f'<rect x="{x}" y="{y}" width="{wdt}" height="32" rx="6" fill="{fill}" opacity="0.9"/>'
            b2 += text(x + wdt/2, y + 21, s, 11, T1 if kind != "wait" else T2, "middle", "600")
            x += wdt + 4
        return b2, x
    s1, _ = lane(104, "p50", BLUE, [("admit", 70, "work"), ("prefill", 150, "work"), ("decode", 400, "work")])
    s2, xe = lane(176, "p99", BLUE, [("queue wait", 110, "wait"), ("prefill", 150, "work"), ("decode", 120, "work"), ("preempted: blocks freed", 170, "bad"), ("wait", 90, "wait"), ("prefill again", 200, "work"), ("decode", 140, "work")])
    b += s1 + s2
    b += text(32, 262, "Defaults that make the second lane: long_prefill_token_threshold = 0, watermark = 0.0, admission loop breaks instead of continuing.", 12, T2)
    b += text(32, 284, "Set the threshold: up to 93% more tokens per GPU-second and about 2.3x better p90 interactivity on agentic traffic (AgentX).", 12, T2)
    return svg(w, h, b, "p50 versus p99 request path")

def ch07():
    return hbar_chart(1200, 380, "Prefix cache ceilings from production traces", "Hit rate with an unlimited cache, unless noted. The marketing number is a single agentic session.", "Ch 7",
                      [("Single agentic session, within a turn (Copilot)", 98, AQUA, "median"),
                       ("Alibaba consumer chat", 62, BLUE, ""),
                       ("DeepSeek API, 24h (disk cache)", 56.3, BLUE, ""),
                       ("Alibaba enterprise API", 54, BLUE, ""),
                       ("Copilot, across a user-turn boundary", 55, ORANGE, "TTL-bound"),
                       ("Mooncake consumer chat", 51, BLUE, ""),
                       ("Multi-instance vLLM, per-instance cache, Codex traces", 1.7, CRIT, "load-balanced")],
                      unit="%", vmax=100, label_w=400, foot="Sources: Alibaba ATC '25, DeepSeek Feb 2025, Mooncake, Microsoft Research 2026, vLLM Mooncake Store post")

def ch08():
    w, h = 1200, 300
    b = header(w, "Hashes chain from block 0", "Each block hash includes the previous hash. One changed byte, anywhere, resets everything after it.", "Ch 8")
    blocks = ["system prompt", "few-shot", "turn 1", "turn 2", "tool result", "turn 3", "turn 4", "new message"]
    x = 40; y = 120; bw = 128
    div = 4
    for i, lab in enumerate(blocks):
        if i < div: col, st = BLUE, "hit"
        elif i == div: col, st = CRIT, "changed byte"
        else: col, st = "#3a3a37", "miss"
        b += f'<rect x="{x}" y="{y}" width="{bw}" height="54" rx="8" fill="{col}" opacity="{0.92 if i<=div else 0.8}"/>'
        b += text(x + bw/2, y + 24, lab, 12, T1, "middle", "600")
        b += text(x + bw/2, y + 42, f"h{i} = H(h{i-1}, tokens)" if i else "h0 = H(tokens)", 10, T1 if i <= div else T2, "middle", "400", MONO)
        b += text(x + bw/2, y + 74, st, 11, col if i != div + 1 else T3, "middle", "700")
        if i < len(blocks) - 1:
            b += f'<path d="M {x+bw+2} {y+27} L {x+bw+14} {y+27}" stroke="{LINE}" stroke-width="2"/>'
        x += bw + 16
    b += text(32, 232, "Usual culprits for the changed byte: a date in the template, a think block stripped by turn index, a re-serialized tool JSON,", 12, T2)
    b += text(32, 252, "a different LoRA name in the hash, a 528-token block on a hybrid model, or NFC reordering a shadda.", 12, T2)
    b += text(32, 280, "Diagnostic: POST /tokenize with two consecutive turns, diff the token IDs, read the first differing index.", 12, AQUA, "start", "600")
    return svg(w, h, b, "Prefix cache hash chain")

def ch09():
    return grouped_bars(1200, 320, "Speculative decoding speedup collapses with batch size", "EAGLE on vLLM, measured in the MLSys 2026 study. Batch 1 is the paper number; batch 128 is production.", "Ch 9",
                        [("batch 1", [1.73]), ("batch 128", [1.21])], [("EAGLE speedup, x", BLUE)], unit="x", vmax=2.2,
                        foot="Verification was 42% to 95% of SD time. Break-even acceptance rate near 0.55 at high concurrency (secondary figure).")

def ch10():
    w, h = 1200, 260
    b = header(w, "Two fast features, one serial path", "vLLM structured output manager, V1: what happens when speculation is on", "Ch 10")
    b += f'<rect x="32" y="96" width="540" height="120" rx="10" fill="{SURF2}" stroke="{AQUA}"/>'
    b += text(52, 122, "num_speculative_tokens == 0  and  > 128 structured requests", 12, AQUA, "start", "700", MONO)
    b += text(52, 148, "thread pool fills the bitmask in parallel", 13, T1)
    b += text(52, 170, "cost: amortized across workers", 12, T2)
    b += text(52, 196, "benchmarked, fast", 12, T3)
    b += f'<rect x="628" y="96" width="540" height="120" rx="10" fill="{SURF2}" stroke="{CRIT}"/>'
    b += text(648, 122, "speculation on", 12, CRIT, "start", "700", MONO)
    b += text(648, 148, "for each request: for each draft position: accept, mask, rollback", 13, T1)
    b += text(648, 170, "serial, on the CPU, inside the scheduler step", 12, T2)
    b += text(648, 196, "cost: (structured requests) x (1 + k), blocks the next GPU step", 12, T3)
    b += text(32, 242, "Grammar-rejected drafts become -1 and are subtracted from num_draft_tokens, so the acceptance-rate dashboard stays green while speedup falls.", 12, T2)
    return svg(w, h, b, "Structured output and speculation code path")

def ch11():
    w, h = 1200, 240
    b = header(w, "The bottleneck ladder, V1 architecture", "Where the limit usually sits, from most to least common. Detokenization runs in the API process, not the GPU worker.", "Ch 11")
    rungs = [("KV / VRAM", BLUE), ("GPU compute", BLUE), ("scheduling", AQUA), ("API CPU", ORANGE), ("ZMQ transport", YELLOW), ("WAN", "#3a3a37")]
    x = 32; y = 108; bw = 176
    for i, (lab, col) in enumerate(rungs):
        b += f'<rect x="{x}" y="{y}" width="{bw}" height="48" rx="8" fill="{col}" opacity="0.9"/>'
        b += text(x + bw/2, y + 29, lab, 13, T1, "middle", "700")
        if i < len(rungs) - 1:
            b += f'<path d="M {x+bw+4} {y+24} L {x+bw+12} {y+24}" stroke="{LINE}" stroke-width="2"/>'
        x += bw + 16
    b += text(32, 190, "GPU idle, engine queue empty, request rate rising: look at API CPU. Mitigation: --api-server-count N.", 12, T2)
    b += text(32, 212, "Template handling is part of the model: Kimi K2 tool-call success went from under 20% to 4.4x better with no weight change.", 12, T2)
    return svg(w, h, b, "vLLM bottleneck ordering")

def ch12():
    w, h = 1200, 300
    b = header(w, "CUDA graph capture list: pad up or fall back", "TensorRT-LLM default sizes. A batch between captured sizes is padded to the next one.", "Ch 12")
    sizes = [1, 2, 4] + list(range(8, 129, 8)) + [256, 512, 1024, 2048]
    x0, x1, y = 60, w - 60, 150
    import math
    def px(v): return x0 + (x1 - x0) * (math.log2(v) / math.log2(2048))
    b += f'<line x1="{x0}" y1="{y}" x2="{x1}" y2="{y}" stroke="{T3}" stroke-width="2"/>'
    for v in sizes:
        b += f'<line x1="{px(v):.1f}" y1="{y-10}" x2="{px(v):.1f}" y2="{y+10}" stroke="{BLUE}" stroke-width="2"/>'
    for v in [1, 8, 32, 128, 256, 512, 1024, 2048]:
        b += text(px(v), y + 30, str(v), 11, T2, "middle", "400", MONO)
    b += f'<rect x="{px(256):.1f}" y="{y-36}" width="{px(512)-px(256):.1f}" height="22" rx="4" fill="{CRIT}" opacity="0.25"/>'
    b += f'<circle cx="{px(300):.1f}" cy="{y}" r="7" fill="{ORANGE}"/>'
    b += text(px(300), y - 44, "batch 300 pads to 512: 41% of the step is padding", 12, ORANGE, "middle", "700")
    b += text(32, 222, "23 captured sizes by default (log scale above). Dense below 128, coarse above it, which is where high-concurrency serving lives.", 12, T2)
    b += text(32, 244, "Adding 64-step graphs above 128: up to 1.3x to 1.5x throughput on GB200. Capturing 259 graphs: 2.6 GB, KV pool down 17.2%, throughput down 5%.", 12, T2)
    b += text(32, 272, "The graphs and the KV cache compete for the same HBM.", 12, AQUA, "start", "600")
    return svg(w, h, b, "CUDA graph capture list")

def ch13():
    w, h = 1200, 300
    b = header(w, "Two regimes for the all-reduce", "Message size per layer, 70B model in BF16. The spec sheet number predicts the left column only.", "Ch 13")
    for i, (lab, msg, regime, col, note) in enumerate([
        ("Prefill, 4k-token prompt", "tens of MB", "bandwidth-bound", BLUE, "NVLink GB/s matters. On PCIe L40S, all-reduce reached 65% of prefill latency."),
        ("Decode, batch 32", "512 KB", "latency-bound", ORANGE, "Fixed cost per collective dominates. NCCL 1.5x to 2x slower than MPI here; NVRAR 1.9x to 3.6x faster than NCCL.")]):
        x = 32 + i * 580
        b += f'<rect x="{x}" y="96" width="556" height="150" rx="10" fill="{SURF2}" stroke="{col}"/>'
        b += text(x + 20, 124, lab, 14, T1, "start", "700")
        b += text(x + 20, 156, msg, 26, col, "start", "800", MONO)
        b += text(x + 20, 182, regime, 13, col, "start", "600")
        words = note.split(); l1 = ""
        while words and len(l1) + len(words[0]) < 72: l1 += words.pop(0) + " "
        b += text(x + 20, 210, l1.strip(), 11, T2)
        b += text(x + 20, 228, " ".join(words), 11, T2)
    b += text(32, 278, "Rule: measure all-reduce latency at 128 KB to 1 MB on your topology before choosing TP degree. Replicas beat ranks for decode.", 12, AQUA, "start", "600")
    return svg(w, h, b, "Tensor parallel decode regimes")

def ch14():
    return hbar_chart(1200, 280, "Decode does not draw TDP", "H200, 700 W part. Power draw by phase, and what a clock lock buys (single preprint)", "Ch 14",
                      [("TDP, what the capacity plan assumed", 700, "#3a3a37", ""),
                       ("Decode, measured, upper end", 300, BLUE, ""),
                       ("Decode, measured, lower end", 137, BLUE, ""),
                       ("Saved per GPU by locking clocks to 780 MHz", 90, AQUA, "(up to, under 1% throughput loss)")],
                      unit=" W", vmax=760, label_w=330, foot="Power caps from 280 W to 700 W never engaged during decode. They bite during prefill, the TTFT phase.")

def ch15():
    return stacked_bar(1200, 260, "Cold start is CPU-bound", "vLLM startup, Llama-3.2-3B on an H100, 20.32 s total. Faster storage improved the total by 1.04x.", "Ch 15",
                       [("weight loading", 4.57, BLUE), ("load compiled graphs", 4.86, ORANGE), ("CUDA graph capture", 3.96, YELLOW), ("import, init, profiling, allocation", 6.93, "#4a4a46")],
                       "20.32 s, one CPU core pinned at 100% throughout", foot="Startup varied 4x across nine vLLM releases on the same hardware (7.8 s to 36.41 s)")

def ch16():
    w, h = 1200, 320
    b = header(w, "DeepSeek's 24 hours: the decode pool is 4.5x the prefill pool", "V3 and R1 on H800, February 27 to 28, 2025. Per-user speed was chosen, not optimized.", "Ch 16")
    b += f'<rect x="32" y="96" width="300" height="150" rx="10" fill="{SURF2}" stroke="{BLUE}"/>'
    b += text(52, 124, "Prefill unit", 14, T1, "start", "700")
    b += text(52, 156, "4 nodes, EP32", 22, BLUE, "start", "800", MONO)
    b += text(52, 186, "73.7k input tok/s per node", 12, T2)
    b += text(52, 206, "cache hits included", 11, T3)
    b += f'<rect x="356" y="96" width="812" height="150" rx="10" fill="{SURF2}" stroke="{ORANGE}"/>'
    b += text(376, 124, "Decode unit", 14, T1, "start", "700")
    b += text(376, 156, "18 nodes, EP144", 22, ORANGE, "start", "800", MONO)
    b += text(376, 186, "14.8k output tok/s per node    2 routed experts per GPU    20 to 22 tok/s per user    4,989 KV tokens per output token", 12, T2)
    b += text(376, 206, "throughput per GPU is the economic variable; tokens per second per user is a product variable", 11, T3)
    for i in range(18):
        b += f'<rect x="{376 + i*42}" y="218" width="36" height="16" rx="3" fill="{ORANGE}" opacity="0.6"/>'
    for i in range(4):
        b += f'<rect x="{52 + i*42}" y="218" width="36" height="16" rx="3" fill="{BLUE}" opacity="0.6"/>'
    b += text(32, 282, "226.75 nodes average in service, 278 peak.  $87,072 per day at $2 per GPU-hour against $562,027 theoretical at R1 pricing (545% margin, actual lower).", 12, T2)
    return svg(w, h, b, "DeepSeek inference disclosure")

def ch17():
    w, h = 1200, 320
    b = header(w, "The turn boundary is where the cache dies", "GitHub Copilot production trace: prefix-cache hit rate along one session", "Ch 17")
    # timeline
    x0, x1, y = 60, w - 60, 170
    b += f'<line x1="{x0}" y1="{y}" x2="{x1}" y2="{y}" stroke="{T3}" stroke-width="2"/>'
    events = [(0.03, "call 1", 45, BLUE), (0.11, "call 2", 86, BLUE), (0.19, "call 3", 92, BLUE), (0.27, "call 4", 94, BLUE), (0.35, "call 5", 98, AQUA),
              (0.62, "next user turn", 55, ORANGE), (0.70, "call", 86, BLUE), (0.78, "call", 94, BLUE), (0.92, "after compaction", 32, CRIT)]
    for fx, lab, hit, col in events:
        x = x0 + (x1 - x0) * fx
        bh = hit * 0.8
        b += f'<rect x="{x-14:.1f}" y="{y-bh:.1f}" width="28" height="{bh:.1f}" rx="4" fill="{col}" opacity="0.9"/>'
        b += text(x, y - bh - 6, f"{hit}%", 11, T1, "middle", "600", MONO)
        b += text(x, y + 18, lab, 10, T2, "middle")
    gx = x0 + (x1 - x0) * 0.485
    b += f'<rect x="{x0 + (x1-x0)*0.38:.1f}" y="{y-96}" width="{(x1-x0)*0.21:.1f}" height="90" rx="6" fill="{ORANGE}" opacity="0.12"/>'
    b += text(gx, y - 76, "human think time", 11, ORANGE, "middle", "700")
    b += text(gx, y - 60, "172 s median gap", 11, ORANGE, "middle", "400", MONO)
    b += text(x0 + (x1 - x0) * 0.19, y + 40, "1.2 s between tool calls", 11, T3, "middle")
    b += text(32, 250, "Median within a turn: 98%.  Across a turn boundary: 55%.  Compaction (7.8% of sessions, 44.2% of tokens): median 66.1% drop.  Model switch: 8%.", 12, T2)
    b += text(32, 272, "Median prompt 68k tokens, input to output above 275:1, 33 turns per session. Per-instance caching with load balancing: 1.7%; shared KV store: 92.2%.", 12, T2)
    b += text(32, 300, "Set TTL from the inter-turn gap distribution and route by session, not by load.", 12, AQUA, "start", "600")
    return svg(w, h, b, "Agentic session cache hit rate timeline")

def ch18():
    return hbar_chart(1200, 300, "From the vendor number to what you get", "Meta inference team, QCon SF: effective capacity relative to the headline tokens per second", "Ch 18",
                      [("Vendor headline", 100, "#3a3a37", ""),
                       ("After kernel-level losses (very common)", 50, BLUE, ""),
                       ("After latency bound and operating buffer", 10, ORANGE, "plan here")],
                      unit="%", vmax=110, label_w=330, foot="Anthropic's 2025 postmortem: a routing bug hit 16% of Sonnet 4 requests at peak and an XLA top-k bug dropped the best token. Evals passed.")

def ch19():
    w, h = 1200, 280
    b = header(w, "A prefix hit is a timing signal", "Shared prefix cache across tenants: the attacker extends a guess one block at a time and watches TTFT", "Ch 19")
    y = 110
    guesses = [("The patient", True), ("The patient is", True), ("The patient is John", True), ("The patient is John Smith", True), ("The patient is John Doe", False)]
    for i, (g, hit) in enumerate(guesses):
        x = 32 + i * 232
        col = AQUA if hit else CRIT
        b += f'<rect x="{x}" y="{y}" width="216" height="64" rx="8" fill="{SURF2}" stroke="{col}"/>'
        b += text(x + 12, y + 24, g, 11, T1, "start", "600", MONO)
        b += text(x + 12, y + 48, ("TTFT low: hit, keep going" if hit else "TTFT high: miss, backtrack"), 11, col)
    b += text(32, 212, "Signal strength on live vLLM: Cohen's d 0.78 with no load, 0.21 with two background workers; AUROC 0.650 to 0.531. Weaker, not closed.", 12, T2)
    b += text(32, 234, "Defenses: per-tenant partition (loses the shared-template hits), selective sharing, owner-tracked continuation, block granularity (k = 16), timing padding.", 12, T2)
    b += text(32, 262, "Decide explicitly whether the cache crosses tenants. Every major engine defaults to yes.", 12, AQUA, "start", "600")
    return svg(w, h, b, "KV cache timing side channel")

def ch20():
    return hbar_chart(1200, 440, "ensure_ascii=True inflates Arabic tokens 3.7x to 19x", "50k characters of Arabic Wikipedia inside a JSON string, escaped as \\uXXXX versus raw. Measured for this handbook.", "Ch 20",
                      [("ALLaM 7B", 19.3, MAGENTA, ""), ("Falcon-H1", 17.0, MAGENTA, ""), ("Gemma 3", 14.0, BLUE, ""), ("Fanar", 13.7, MAGENTA, ""),
                       ("Qwen 2.5 / Qwen3", 10.9, BLUE, ""), ("GPT-4o o200k", 8.2, BLUE, ""), ("DeepSeek V3", 7.0, BLUE, ""), ("Llama 3", 6.9, BLUE, ""), ("cl100k", 3.7, BLUE, "")],
                      unit="x", vmax=21, label_w=200, foot="Arabic-first vocabularies (magenta) are hurt most: no merge exists for \\u0627. Fix: json.dumps(obj, ensure_ascii=False)")

def ch21():
    return grouped_bars(1200, 360, "Full diacritization multiplies tokens per word", "Tokens per Arabic word, undiacritized versus fully diacritized, same corpus. Measured for this handbook.", "Ch 21",
                        [("o200k", [1.66, 5.12]), ("Llama 3", [1.93, 4.84]), ("Qwen3", [1.90, 6.61]), ("Gemma 3", [1.73, 4.39]), ("Falcon-H1", [1.57, 4.87]), ("ALLaM", [1.25, 3.08]), ("bge-m3", [1.58, 6.64])],
                        [("Undiacritized", BLUE), ("Fully diacritized", ORANGE)], unit="", vmax=8,
                        foot="A 512-token embedding window holds about 270 words of MSA and about 80 words of diacritized text. Chunk by tokens.")

def ch22():
    return hbar_chart(1200, 300, "The MSA number and the dialect number", "Word error rate of the best model on the Open Universal Arabic ASR Leaderboard, by test set", "Ch 22",
                      [("MSA", 19.23, BLUE, ""), ("Egyptian", 40.97, ORANGE, ""), ("Khaliji (Gulf)", 48.23, CRIT, ""),
                       ("Saudi (SADA), Whisper large-v3", 55.96, CRIT, ""), ("Saudi (SADA), 121M Conformer-CTC on 15k h", 27.71, AQUA, "one tenth the size")],
                      unit="%", vmax=62, label_w=330, foot="Leaderboard scores on normalized output; raw WER is higher. Whisper transcribes silence as a known subtitle credit: VAD-gate it.")

# ---------- write everything ----------
parts = [("I", "Numerics and Determinism", "1 to 5", BLUE, "Batch invariance, FP8 accumulators, the E2M1 gap, evals that lie, silent data corruption"),
         ("II", "Scheduler and Cache Internals", "6 to 10", AQUA, "Preemption and the tail, prefix cache ceilings, invalidation mechanics, speculation under load, structured output"),
         ("III", "The Host and Hardware Path", "11 to 15", ORANGE, "The CPU epilogue, CUDA graph capture lists, TP decode latency, power and clocks, cold start"),
         ("IV", "Fleet Realities", "16 to 19", YELLOW, "MoE imbalance, agentic goodput, benchmark bias and silent bugs, KV timing side channels"),
         ("V", "Arabic Inference Hygiene", "20 to 22", MAGENTA, "JSON escaping, NFC and shadda, diacritics, streaming splits, quantization and speculation, speech")]
for i, (num, name, ch, col, blurb) in enumerate(parts, 1):
    write(f"part-{i}.svg", part_banner(num, name, ch, col, blurb))
gens = {1: ch01, 2: ch02, 3: ch03, 4: ch04, 5: ch05, 6: ch06, 7: ch07, 8: ch08, 9: ch09, 10: ch10, 11: ch11, 12: ch12,
        13: ch13, 14: ch14, 15: ch15, 16: ch16, 17: ch17, 18: ch18, 19: ch19, 20: ch20, 21: ch21, 22: ch22}
for n, fn in gens.items():
    write(f"ch{n:02d}.svg", fn())
print("wrote", len(os.listdir(OUT)), "files to", OUT)

# =====================================================================
# Fancy additions: animated hero, stat strip, failure map, chapter map,
# per-chapter key-number tiles
# =====================================================================

def hero():
    w, h = 1200, 480
    b = ('<defs>'
         f'<radialGradient id="g1" cx="0.2" cy="0.1" r="0.7"><stop offset="0" stop-color="{BLUE}" stop-opacity="0.35"/><stop offset="1" stop-color="{BLUE}" stop-opacity="0"/></radialGradient>'
         f'<radialGradient id="g2" cx="0.9" cy="0.9" r="0.6"><stop offset="0" stop-color="{ORANGE}" stop-opacity="0.30"/><stop offset="1" stop-color="{ORANGE}" stop-opacity="0"/></radialGradient>'
         f'<radialGradient id="g3" cx="0.6" cy="0.3" r="0.5"><stop offset="0" stop-color="{AQUA}" stop-opacity="0.22"/><stop offset="1" stop-color="{AQUA}" stop-opacity="0"/></radialGradient>'
         '<style>'
         '@keyframes drift1{0%{transform:translate(0,0)}50%{transform:translate(40px,20px)}100%{transform:translate(0,0)}}'
         '@keyframes drift2{0%{transform:translate(0,0)}50%{transform:translate(-30px,-25px)}100%{transform:translate(0,0)}}'
         '@keyframes scan{0%,100%{opacity:.18}50%{opacity:1}}'
         '@keyframes pulse{0%,100%{opacity:.55;r:13}50%{opacity:1;r:15}}'
         '@keyframes caret{0%,100%{opacity:1}50%{opacity:0}}'
         '.b1{animation:drift1 14s ease-in-out infinite}.b2{animation:drift2 18s ease-in-out infinite}'
         '.cell{animation:scan 6s ease-in-out infinite}.pulse{animation:pulse 2.4s ease-in-out infinite}.caret{animation:caret 1.1s step-end infinite}'
         '</style></defs>')
    b += f'<rect x="0" y="0" width="{w}" height="{h}" rx="16" fill="url(#g1)" class="b1"/>'
    b += f'<rect x="0" y="0" width="{w}" height="{h}" rx="16" fill="url(#g2)" class="b2"/>'
    b += f'<rect x="0" y="0" width="{w}" height="{h}" rx="16" fill="url(#g3)"/>'
    for i in range(0, w, 40):
        for j in range(0, h, 40):
            b += f'<circle cx="{i+20}" cy="{j+20}" r="1" fill="#2e2e2b"/>'
    b += text(64, 96, "THE LLM INFERENCE HANDBOOK", 13, AQUA, "start", "700", extra='letter-spacing="5"')
    b += text(64, 176, "Volume II", 78, T1, "start", "800")
    b += text(64, 226, "Production Failure Modes", 32, T2, "start", "600")
    b += text(64, 258, "What breaks after the benchmark passes", 18, T3)
    b += f'<rect x="64" y="292" width="560" height="54" rx="10" fill="#101010" stroke="{LINE}"/>'
    b += text(84, 326, "$ vllm serve ...  --prefix-caching  --kv-cache-dtype fp8", 15, AQUA, "start", "400", MONO)
    b += f'<rect x="596" y="311" width="9" height="20" fill="{AQUA}" class="caret"/>'
    chips = [("I", "Numerics", BLUE), ("II", "Scheduler and Cache", AQUA), ("III", "Host and Hardware", ORANGE),
             ("IV", "Fleet Realities", YELLOW), ("V", "Arabic Inference", MAGENTA)]
    x = 64
    for num, name, col in chips:
        tw = 8 * len(name) + 72
        b += f'<rect x="{x}" y="376" width="{tw}" height="36" rx="18" fill="#141413" stroke="{col}" stroke-opacity="0.6"/>'
        b += f'<circle cx="{x+19}" cy="394" r="6" fill="{col}"/>'
        b += text(x + 33, 399, f"Part {num}", 12, T3, "start", "600")
        b += text(x + 33 + 50, 399, name, 12, T1, "start", "600")
        x += tw + 10
    b += text(64, 448, "22 chapters   5 parts   107 sources   4 measurement scripts   every number sourced or measured", 13, T3)
    # animated KV grid, right
    bx, by = 800, 86
    cols = [BLUE, BLUE, BLUE, BLUE, AQUA, AQUA, ORANGE, CRIT, "#3a3a37", "#3a3a37"]
    for r in range(7):
        for c in range(10):
            col = cols[c] if r < 5 else ("#3a3a37" if c > 5 else cols[c])
            delay = (r * 0.25 + c * 0.35)
            cls = ' class="cell"' if col not in ("#3a3a37", CRIT) else ""
            style = f' style="animation-delay:{delay:.2f}s"' if cls else ""
            b += f'<rect x="{bx + c*34}" y="{by + r*34}" width="28" height="28" rx="6" fill="{col}"{cls}{style}/>'
    b += f'<circle cx="{bx + 7*34 + 14}" cy="{by + 2*34 + 14}" r="13" fill="none" stroke="{CRIT}" stroke-width="2" class="pulse"/>'
    ly = by + 7 * 34 + 22
    for i, (col, lab) in enumerate([(BLUE, "reused prefix"), (AQUA, "hit this turn"), (CRIT, "first divergent byte"), ("#3a3a37", "cold")]):
        lx = bx + [0, 100, 190, 322][i]
        b += f'<rect x="{lx}" y="{ly-10}" width="10" height="10" rx="2" fill="{col}"/>'
        b += text(lx + 16, ly, lab, 11, T3)
    return svg(w, h, b, "The LLM Inference Handbook, Volume II: Production Failure Modes")

def stat_tiles(w, h, tiles, chapter=None, title=None):
    """tiles: list of (value, label, color). Hero-number tiles in one strip."""
    n = len(tiles); gap = 16; x0 = 24; tw = (w - 2 * x0 - gap * (n - 1)) / n
    b = ""
    top = 20
    if title:
        b += header(w, title, "", chapter); top = 84
    th = h - top - 20
    for i, (val, lab, col) in enumerate(tiles):
        x = x0 + i * (tw + gap)
        b += f'<rect x="{x:.1f}" y="{top}" width="{tw:.1f}" height="{th}" rx="12" fill="#141413" stroke="{LINE}"/>'
        b += f'<rect x="{x:.1f}" y="{top}" width="5" height="{th}" rx="2.5" fill="{col}"/>'
        size = 34 if len(val) <= 9 else (26 if len(val) <= 14 else 20)
        b += text(x + 24, top + th * 0.52, val, size, T1, "start", "800", MONO)
        b += text(x + 24, top + th * 0.52 + 26, lab, 12, T2)
    return svg(w, h, b, title or "Key numbers")

def failure_map():
    w, h = 1200, 520
    b = header(w, "Where each failure lives", "The path of one request, and the chapter that covers what breaks at each stage", None)
    stages = [("Text", "tokenizer, template, JSON", MAGENTA, [20, 21, 22, 8]),
              ("Gateway", "routing, tenancy, TTL", YELLOW, [17, 19, 18]),
              ("API process", "render, sample, detokenize", ORANGE, [11, 10]),
              ("Scheduler", "admit, chunk, preempt, draft", AQUA, [6, 9, 10]),
              ("KV cache", "blocks, hashes, precision", BLUE, [7, 8, 2]),
              ("GPU kernels", "reductions, formats, faults", BLUE, [1, 3, 4, 5, 12]),
              ("Interconnect", "all-reduce, all-to-all", ORANGE, [13, 16]),
              ("Host and fleet", "power, clocks, cold start", YELLOW, [14, 15, 18])]
    n = len(stages); x0 = 32; gap = 10; sw = (w - 2 * x0 - gap * (n - 1)) / n; y = 110; sh = 92
    for i, (name, sub, col, chs) in enumerate(stages):
        x = x0 + i * (sw + gap)
        b += f'<rect x="{x:.1f}" y="{y}" width="{sw:.1f}" height="{sh}" rx="12" fill="#141413" stroke="{col}" stroke-opacity="0.7"/>'
        b += f'<rect x="{x:.1f}" y="{y}" width="{sw:.1f}" height="4" rx="2" fill="{col}"/>'
        b += text(x + sw/2, y + 36, name, 15, T1, "middle", "700")
        ws = sub.split(); l1 = ""
        while ws and len(l1) + len(ws[0]) < 24: l1 += ws.pop(0) + " "
        b += text(x + sw/2, y + 58, l1.strip(), 10, T3, "middle")
        if ws: b += text(x + sw/2, y + 72, " ".join(ws), 10, T3, "middle")
        if i < n - 1:
            b += f'<path d="M {x+sw+1:.1f} {y+sh/2} L {x+sw+gap-1:.1f} {y+sh/2}" stroke="{T3}" stroke-width="2"/>'
        # chapter chips
        last_cy = y + sh + 22 + (len(chs) - 1) * 52
        b += f'<line x1="{x+sw/2:.1f}" y1="{y+sh}" x2="{x+sw/2:.1f}" y2="{last_cy}" stroke="{LINE}" stroke-dasharray="3 3"/>'
        for j, c in enumerate(chs):
            cy = y + sh + 22 + j * 52
            b += f'<rect x="{x+6:.1f}" y="{cy}" width="{sw-12:.1f}" height="44" rx="8" fill="{SURF2}" stroke="{LINE}"/>'
            b += f'<rect x="{x+14:.1f}" y="{cy+10}" width="40" height="24" rx="5" fill="#101010" stroke="{col}" stroke-opacity="0.7"/>'
            b += text(x + 34, cy + 26, f"Ch {c}", 10, col, "middle", "700", MONO)
            ws = SHORT[c].split(); l1 = ""
            while ws and len(l1) + len(ws[0]) <= 15: l1 += ws.pop(0) + " "
            if not l1: l1 = ws.pop(0)
            b += text(x + 62, cy + 20 if ws else cy + 26, l1.strip(), 10, T1, "start", "600")
            if ws: b += text(x + 62, cy + 33, " ".join(ws), 10, T2, "start", "400")
    b += text(32, h - 20, "Prefill flows left to right. Decode loops between scheduler, KV cache, kernels, and interconnect once per token, which is why those four stages own the most chapters.", 11, T3)
    return svg(w, h, b, "Failure map of the inference request path")

SHORT = {1: "Batch invariance", 2: "FP8 KV accumulator", 3: "NVFP4 gap", 4: "Evals that lie", 5: "Silent corruption",
         6: "Scheduler tail", 7: "Cache ceilings", 8: "Cache invalidation", 9: "Speculation at load", 10: "Grammar + spec",
         11: "CPU epilogue", 12: "CUDA graphs", 13: "TP decode latency", 14: "Power and clocks", 15: "Cold start",
         16: "MoE imbalance", 17: "Agentic goodput", 18: "Benchmarks and bugs", 19: "Timing side-channel",
         20: "Arabic text hygiene", 21: "Arabic streaming, quant", 22: "Arabic speech"}

HEADLINE = {1: "80 of 1,000 differ", 2: "91% to 13% at 128k", 3: "no 5 between 4 and 6", 4: "16 pt human drop", 5: "4.5 pt per prefill fault",
            6: "93% more tok/GPU-s", 7: "50 to 60% ceiling", 8: "1 byte resets all", 9: "1.73x to 1.21x", 10: "serial CPU loop",
            11: "tool calls < 20%", 12: "41% padding at 300", 13: "512 KB, latency-bound", 14: "137 to 300 W of 700", 15: "weights are 22%",
            16: "4 : 18 nodes", 17: "1.7% vs 92.2%", 18: "plan for 1/10", 19: "d 0.78 to 0.21", 20: "3.7x to 19x inflation",
            21: "1 split per 12 words", 22: "19% MSA, 48% Khaliji"}

def chapter_map():
    w, h = 1200, 470
    b = header(w, "Chapter map", "Five parts, twenty-two chapters, one headline number each", None)
    parts = [("I", "Numerics", BLUE, [1, 2, 3, 4, 5]), ("II", "Scheduler and Cache", AQUA, [6, 7, 8, 9, 10]),
             ("III", "Host and Hardware", ORANGE, [11, 12, 13, 14, 15]), ("IV", "Fleet", YELLOW, [16, 17, 18, 19]),
             ("V", "Arabic", MAGENTA, [20, 21, 22])]
    n = len(parts); x0 = 32; gap = 14; cw = (w - 2 * x0 - gap * (n - 1)) / n; y0 = 96
    for i, (num, name, col, chs) in enumerate(parts):
        x = x0 + i * (cw + gap)
        b += f'<rect x="{x:.1f}" y="{y0}" width="{cw:.1f}" height="30" rx="8" fill="{col}" opacity="0.9"/>'
        b += text(x + cw/2, y0 + 20, f"Part {num}: {name}", 13, T1, "middle", "700")
        for j, c in enumerate(chs):
            cy = y0 + 40 + j * 66
            b += f'<rect x="{x:.1f}" y="{cy}" width="{cw:.1f}" height="58" rx="10" fill="#141413" stroke="{LINE}"/>'
            b += f'<rect x="{x+12:.1f}" y="{cy+12}" width="34" height="20" rx="5" fill="{SURF2}" stroke="{col}" stroke-opacity="0.7"/>'
            b += text(x + 29, cy + 26, f"{c:02d}", 11, col, "middle", "700", MONO)
            b += text(x + 54, cy + 26, SHORT[c], 12, T1, "start", "700")
            b += text(x + 14, cy + 47, HEADLINE[c], 11, T3, "start", "400", MONO)
    return svg(w, h, b, "Chapter map")

STATS = {
1: [("80 / 1,000", "distinct completions at temperature 0", BLUE), ("1.6x to 2.1x", "slowdown with batch-invariant kernels", ORANGE), ("500+", "output tokens needed to see it", AQUA)],
2: [("91% to 13%", "128k retrieval, BF16 to FP8 KV", CRIT), ("2x", "KV capacity from FP8", BLUE), ("96%", "ITL slope on gpt-oss-20b: no speedup", YELLOW)],
3: [("16", "encodings in E2M1, no 5", BLUE), ("66.6% to 100%", "of block max lands in one gap", CRIT), ("69% to 75%", "8B recovery under W4A4", ORANGE)],
4: [("+0.0003", "MMLU after quantization, up", YELLOW), ("16 pt", "human-rated drop, metrics saw 1.7", CRIT), ("4.37x", "Arabic perplexity ratio, 3-bit GPTQ", ORANGE)],
5: [("1 / 1,000", "devices with SDC, Meta fleet", BLUE), ("4.5 pt", "accuracy lost per prefill fault, up to", CRIT), ("1.01%", "of SDC visible as NaN or Inf", YELLOW)],
6: [("0", "default long_prefill_token_threshold", CRIT), ("93%", "more tokens per GPU-second with it set", AQUA), ("2.3x", "better p90 interactivity", BLUE)],
7: [("50 to 60%", "fleet hit-rate ceiling, infinite cache", BLUE), ("97%", "of enterprise hits from single-turn", AQUA), ("0.3 s", "until 90% of enterprise blocks are dead", ORANGE)],
8: [("1 byte", "changed resets every later block", CRIT), ("528", "token block size on Qwen3.5 hybrid", ORANGE), ("0%", "prefix reuse across LoRA adapters", YELLOW)],
9: [("1.73x", "EAGLE speedup at batch 1", BLUE), ("1.21x", "EAGLE speedup at batch 128", ORANGE), ("42 to 95%", "of SD time spent in verification", YELLOW)],
10: [("128", "requests before the parallel bitmask path", BLUE), ("1 + k", "serial mask steps per structured request", CRIT), ("-1", "what a grammar-rejected draft becomes", ORANGE)],
11: [("< 20%", "Kimi K2 tool-call success, template bug", CRIT), ("4.4x", "after the template fix, no weight change", AQUA), ("3", "detokenizer bugs filed in 2026", YELLOW)],
12: [("23", "default captured batch sizes", BLUE), ("41%", "padding when batch 300 pads to 512", ORANGE), ("2.6 GB", "device memory for 259 graphs", CRIT)],
13: [("512 KB", "decode all-reduce, batch 32, 70B", ORANGE), ("1.5x to 2x", "NCCL slower than MPI at that size", CRIT), ("136%", "two TP=2 replicas over one TP=4 (hobbyist)", BLUE)],
14: [("137 to 300 W", "decode draw on a 700 W H200", BLUE), ("24 to 32%", "power saved by locking clocks", AQUA), ("< 1%", "throughput lost doing it", YELLOW)],
15: [("22%", "of startup is weight loading", BLUE), ("43%", "is compile plus graph capture", ORANGE), ("1.04x", "gain from faster storage", YELLOW)],
16: [("4 : 18", "prefill nodes to decode nodes", ORANGE), ("20 to 22", "tokens per second per user, by choice", BLUE), ("545%", "theoretical margin, actual lower", AQUA)],
17: [("1.7%", "hit rate, per-instance cache behind an LB", CRIT), ("92.2%", "hit rate with a shared KV store", AQUA), ("58%", "of program delay is queue wait", ORANGE)],
18: [("1 / 10", "of the vendor number is what you get", CRIT), ("16%", "of requests misrouted at the worst hour", ORANGE), ("30%", "of Claude Code users hit at least once", YELLOW)],
19: [("0.78 to 0.21", "Cohen's d of the timing signal under load", BLUE), ("16", "tokens per block: the guess granularity", AQUA), ("3", "named hosted services sharing cache", ORANGE)],
20: [("3.7x to 19x", "token inflation from ensure_ascii", CRIT), ("55%", "of diacritized lines changed by NFC", ORANGE), ("+1.9 to +11.2%", "tokens added by classic normalization", YELLOW)],
21: [("1 / 12", "diacritized words with a split char on Qwen", ORANGE), ("4.37x", "Arabic perplexity ratio, English calibration", CRIT), ("none", "published acceptance rate for Arabic", YELLOW)],
22: [("19% / 48%", "WER, MSA versus Khaliji", CRIT), ("55.96 vs 27.71", "Whisper v3 vs 121M Saudi CTC, WER", ORANGE), ("10x", "Whisper slower than CTC, RTF", BLUE)],
}

write("hero.svg", hero())
write("stats.svg", stat_tiles(1200, 150, [("22", "chapters across five parts", BLUE), ("18.7k", "words, textbook register", AQUA), ("107", "sources, each linked", ORANGE), ("4", "measurement scripts, Appendix B", MAGENTA), ("2", "measurements with no prior source", YELLOW)]))
write("failure-map.svg", failure_map())
write("chapter-map.svg", chapter_map())
for n, tiles in STATS.items():
    write(f"ch{n:02d}-stats.svg", stat_tiles(1200, 120, tiles))
print("fancy assets written")
