from __future__ import annotations
from typing import List, Dict
import plotly.graph_objects as go
from collections import Counter, defaultdict
from .entities import AssignmentRow, Topic, Coach, Department


def color_for_avg_rank(avg: float) -> str:
    if avg <= 1.5: return "rgba(46,125,50,0.7)"
    if avg <= 2.3: return "rgba(255,143,0,0.7)"
    return "rgba(198,40,40,0.7)"


def build_sankey(rows: List[AssignmentRow], topics: Dict[str, Topic], out_html: str, title: str = "Allocation Sankey"):
    # Aggregate counts and average preference rank per topic
    counts = Counter(r.assigned_topic for r in rows)
    avg_rank = {}
    tmp = defaultdict(list)
    for r in rows:
        tmp[r.assigned_topic].append(r.preference_rank if r.preference_rank != 999 else 3.5)
    for t, arr in tmp.items():
        avg_rank[t] = sum(arr)/len(arr) if arr else 3.5

    # Nodes: Students -> Topics -> Departments
    nodes: List[str] = []
    idx = {}

    def add_node(label: str):
        idx[label] = len(nodes)
        nodes.append(label)

    add_node("Students")
    for tid in topics.keys():
        add_node(f"T: {tid}")
    depts = sorted({t.department_id for t in topics.values()})
    for d in depts:
        add_node(f"D: {d}")

    # Links
    src, tgt, val, col, lab = [], [], [], [], []

    # Students -> Topics
    for tid, topic in topics.items():
        c = counts.get(tid, 0)
        if c <= 0: continue
        src.append(idx["Students"])
        tgt.append(idx[f"T: {tid}"])
        val.append(c)
        col.append(color_for_avg_rank(avg_rank.get(tid, 3.5)))
        lab.append(f"{tid}: {c} (avg pref={avg_rank.get(tid,3.5):.2f})")

    # Topics -> Departments
    for tid, topic in topics.items():
        c = counts.get(tid, 0)
        if c <= 0: continue
        src.append(idx[f"T: {tid}"])
        tgt.append(idx[f"D: {topic.department_id}"])
        val.append(c)
        col.append("rgba(120,144,156,0.35)")
        lab.append(f"{tid} → {topic.department_id}: {c}")

    node_colors = []
    for labn in nodes:
        if labn == "Students": node_colors.append("rgba(227,242,253,1)")
        elif labn.startswith("T:"): node_colors.append("rgba(232,245,233,1)")
        elif labn.startswith("D:"): node_colors.append("rgba(243,229,245,1)")
        else: node_colors.append("rgba(255,255,255,1)")

    fig = go.Figure(data=[go.Sankey(
        arrangement="snap",
        node=dict(pad=20, thickness=18, label=nodes, color=node_colors),
        link=dict(source=src, target=tgt, value=val, color=col, label=lab, hovertemplate="%{label}<extra></extra>")
    )])
    fig.update_layout(title_text=title, font_size=11)
    fig.write_html(out_html, auto_open=False)
    return out_html
