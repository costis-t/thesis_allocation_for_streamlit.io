#!/usr/bin/env python3
"""
Network Flow Visualization for Min-Cost Max-Flow Algorithm (hover highlight per student).

Key behavior:
- Each student's full path is plotted as separate edge traces (SOURCE->Student, Student->Topic, Topic->Coach, Coach->SINK).
- Hovering any edge belonging to a student highlights that student's full path.
- Unhover resets highlighting automatically.
"""
import csv
import argparse
from collections import defaultdict
try:
    import networkx as nx
    import plotly.graph_objects as go
    import numpy as np
except ImportError:
    print("Error: networkx or plotly not installed. Run: pip install networkx plotly")
    exit(1)


def load_allocation(path):
    rows = []
    with open(path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


def get_rank_colors():
    """Get the color mapping for preference ranks."""
    return {
        0: "rgba(46, 204, 113, 0.8)",   # Tier1: green (best)
        1: "rgba(52, 152, 219, 0.8)",   # Tier2: blue
        2: "rgba(241, 196, 15, 0.8)",   # Tier3: yellow
        10: "rgba(46, 204, 113, 0.8)",  # 1st choice: green (best)
        11: "rgba(52, 152, 219, 0.8)",  # 2nd choice: blue
        12: "rgba(241, 196, 15, 0.8)",  # 3rd choice: yellow
        13: "rgba(230, 126, 34, 0.8)",  # 4th choice: orange
        14: "rgba(231, 76, 60, 0.8)",  # 5th choice: red
        999: "rgba(149, 165, 166, 0.8)",  # Unranked: gray
    }


def create_multipartite_visualization(rows, output_path):
    """Create a simplified multipartite visualization using bipartite layout."""
    # Create a simple version using create_network_visualization but with different layout
    return create_network_visualization(rows, output_path)


def create_edge_colormap_visualization(rows, output_path):
    """Create an edge colormap visualization - same as main for now."""
    return create_network_visualization(rows, output_path)


def create_bundled_traces(rows, pos, rank_colors, G):
    """Create bundled edge traces with curved paths."""
    import numpy as np
    from math import atan2, pi, cos, sin
    
    # Store allocations and aggregate edges
    student_alloc = defaultdict(list)
    for row in rows:
        s = row["student"]
        t = row["assigned_topic"]
        c = row["assigned_coach"]
        rank = int(row.get("preference_rank", "999"))
        student_alloc[s].append((t, c, rank))
    
    # Aggregate edges
    edge_counts = defaultdict(int)
    edge_colors_map = {}
    
    for student, allocations in student_alloc.items():
        for (topic, coach, rank) in allocations:
            s_node = f"S_{student}"
            t_node = f"T_{topic}"
            c_node = f"C_{coach}"
            
            edge_key_st = (s_node, t_node)
            edge_counts[edge_key_st] += 1
            edge_colors_map[edge_key_st] = rank_colors.get(rank, "rgba(200, 200, 200, 0.8)")
            
            edge_key_tc = (t_node, c_node)
            edge_counts[edge_key_tc] += 1
            edge_colors_map[edge_key_tc] = "rgba(150, 150, 150, 0.6)"
            
            edge_key_cs = (c_node, "SINK")
            edge_counts[edge_key_cs] += 1
            edge_colors_map[edge_key_cs] = "rgba(150, 150, 150, 0.6)"
    
    # Create bundled traces with curves
    bundled_traces = []
    for (src, tgt), count in edge_counts.items():
        if src not in pos or tgt not in pos:
            continue
        
        x_src, y_src = pos[src]
        x_tgt, y_tgt = pos[tgt]
        
        # Create curved path for bundling
        mid_x = (x_src + x_tgt) / 2
        mid_y = (y_src + y_tgt) / 2
        
        angle = atan2(y_tgt - y_src, x_tgt - x_src)
        perp_angle = angle + pi/2
        offset = 0.05 * min(count, 10)  # Cap offset
        
        ctrl_x = mid_x + offset * cos(perp_angle)
        ctrl_y = mid_y + offset * sin(perp_angle)
        
        # Bezier curve
        t = np.linspace(0, 1, 50)
        x_curve = (1-t)**2 * x_src + 2*(1-t)*t * ctrl_x + t**2 * x_tgt
        y_curve = (1-t)**2 * y_src + 2*(1-t)*t * ctrl_y + t**2 * y_tgt
        
        hover_text = f"{src} → {tgt}<br>Flow: {count}"
        color = edge_colors_map.get((src, tgt), "rgba(180, 180, 180, 0.6)")
        
        edge_trace = go.Scatter(
            x=x_curve, y=y_curve,
            mode='lines',
            line=dict(width=max(1.5, min(count * 0.5, 5)), color=color),
            hoverinfo='text',
            text=hover_text,
            showlegend=False,
        )
        bundled_traces.append(edge_trace)
    
    # Add nodes
    x_nodes = [pos[node][0] for node in G.nodes()]
    y_nodes = [pos[node][1] for node in G.nodes()]
    node_colors = [G.nodes[node].get('color', 'rgba(100, 100, 100, 0.8)') for node in G.nodes()]
    node_labels = [G.nodes[node].get('label', node) for node in G.nodes()]
    
    node_trace = go.Scatter(
        x=x_nodes, y=y_nodes,
        mode='markers+text',
        marker=dict(size=15, color=node_colors, line=dict(width=2, color="white")),
        text=node_labels,
        textposition="top center",
        hoverinfo='text',
        showlegend=False
    )
    
    return bundled_traces + [node_trace]


def create_network_visualization(rows, output_path="network_flow.html"):
    """Create interactive network flow visualization with per-student traces for hover behavior."""
    from plotly.subplots import make_subplots
    # Collect unique entities
    students = list({row["student"] for row in rows})
    topics = list({row["assigned_topic"] for row in rows})
    coaches = list({row["assigned_coach"] for row in rows})
    
    students.sort()
    
    # Natural sort for topics
    import re
    def natural_sort_key(text):
        return [int(c) if c.isdigit() else c for c in re.split(r'(\d+)', text)]
    topics.sort(key=natural_sort_key)
    coaches.sort()
    
    # Build flow graph (structure only for layout)
    G = nx.DiGraph()
    
    # Add nodes
    for s in students:
        G.add_node(f"S_{s}", node_type="student", label=f"👤 {s}", color="rgba(52, 152, 219, 0.8)")
    
    for t in topics:
        G.add_node(f"T_{t}", node_type="topic", label=f"📚 {t}", color="rgba(46, 204, 113, 0.8)")
    
    for c in coaches:
        G.add_node(f"C_{c}", node_type="coach", label=f"🏛️ {c}", color="rgba(155, 89, 182, 0.8)")
    
    G.add_node("SOURCE", node_type="source", label="SOURCE", color="rgba(39, 174, 96, 0.9)")
    G.add_node("SINK", node_type="sink", label="SINK", color="rgba(230, 126, 34, 0.9)")
    
    # Store allocations per student
    student_alloc = defaultdict(list)
    for row in rows:
        s = row["student"]
        t = row["assigned_topic"]
        c = row["assigned_coach"]
        rank = int(row.get("preference_rank", "999"))
        student_alloc[s].append((t, c, rank))

    # Use spring layout for nodes
    pos = nx.spring_layout(G, k=2.5, iterations=60, seed=42)

    # Node scatter
    x_nodes = [pos[node][0] for node in G.nodes()]
    y_nodes = [pos[node][1] for node in G.nodes()]
    node_colors = [G.nodes[node].get('color', 'rgba(100, 100, 100, 0.8)') for node in G.nodes()]
    node_labels = [G.nodes[node].get('label', node) for node in G.nodes()]
    
    # Colors for preference ranks
    rank_colors = {
        0: "rgba(46, 204, 113, 0.8)",   # Tier1: green (best)
        1: "rgba(52, 152, 219, 0.8)",   # Tier2: blue
        2: "rgba(241, 196, 15, 0.8)",   # Tier3: yellow
        10: "rgba(46, 204, 113, 0.8)",  # 1st choice: green (best)
        11: "rgba(52, 152, 219, 0.8)",  # 2nd choice: blue
        12: "rgba(241, 196, 15, 0.8)",  # 3rd choice: yellow
        13: "rgba(230, 126, 34, 0.8)",  # 4th choice: orange
        14: "rgba(231, 76, 60, 0.8)",  # 5th choice: red
        999: "rgba(149, 165, 166, 0.8)",  # Unranked: gray
    }

    # Build edge traces per student
    edge_traces = []
    for student, allocations in student_alloc.items():
        for (topic, coach, rank) in allocations:
            # Node identifiers
            s_node = f"S_{student}"
            t_node = f"T_{topic}"
            c_node = f"C_{coach}"

            # Color for student's S->T edge based on preference rank
            col = rank_colors.get(rank, "rgba(200, 200, 200, 0.8)")

            # Four edges for the student path: SOURCE->S, S->T, T->C, C->SINK
            path_edges = [
                ("SOURCE", s_node, "rgba(180, 180, 180, 0.6)"),
                (s_node, t_node, col),
                (t_node, c_node, "rgba(180, 180, 180, 0.6)"),
                (c_node, "SINK", "rgba(180, 180, 180, 0.6)"),
            ]
            
            for (src, tgt, color) in path_edges:
                if src not in pos or tgt not in pos:
                    continue
                    
                x = [pos[src][0], pos[tgt][0]]
                y = [pos[src][1], pos[tgt][1]]
                hover_text = f"{src} → {tgt}<br>Student: {student}<br>Preference Rank: {rank}<br><i>Hover to highlight student's full path</i>"

                # Use customdata to store student ID (array with 2 values for 2 points of the line)
        edge_trace = go.Scatter(
            x=x, y=y,
            mode='lines',
                    line=dict(width=2, color=color),
            hoverinfo='text',
            text=hover_text,
                    customdata=[student, student],  # Store student ID for JS to access (2 points)
                    name=f"edge_{student}_{src}_{tgt}",
            showlegend=False,
        )
        edge_traces.append(edge_trace)
    
    # Node trace
    node_trace = go.Scatter(
        x=x_nodes, y=y_nodes,
        mode='markers+text',
        marker=dict(size=18, color=node_colors, line=dict(width=2, color="white")),
        text=node_labels,
        textposition="top center",
        hoverinfo='text',
        hovertext=[f"{label}" for label in node_labels],
        showlegend=False
    )
    
    # Create subplots: top = detailed, bottom = bundled
    from plotly.subplots import make_subplots
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=("Detailed View (Hover to highlight paths)", "Bundled View (Aggregated edges)"),
        vertical_spacing=0.08,
        row_heights=[0.6, 0.4]
    )
    
    # Add traces to the first subplot (detailed view)
    for trace in edge_traces + [node_trace]:
        fig.add_trace(trace, row=1, col=1)
    
    # Create bundled traces for second subplot
    # First, let's create a simpler aggregated version
    bundled_traces = create_bundled_traces(rows, pos, rank_colors, G)
    
    # Add bundled traces to second subplot
    for trace in bundled_traces:
        fig.add_trace(trace, row=2, col=1)
    
    # Update layout for subplots
    fig.update_layout(
        title={
            "text": "Min-Cost Max-Flow Network Visualization",
            "x": 0.5, "xanchor": "center",
            "font": {"size": 24, "color": "#333"}
        },
        showlegend=False,
        hovermode='closest',
        margin=dict(b=0, l=0, r=0, t=80),
        plot_bgcolor='rgba(240, 240, 240, 1)',
        paper_bgcolor='white',
        height=1200,
    )
    
    # Update axes for both subplots
    for i in [1, 2]:
        fig.update_xaxes(showgrid=False, zeroline=False, showticklabels=False, row=i, col=1)
        fig.update_yaxes(showgrid=False, zeroline=False, showticklabels=False, scaleanchor="x", row=i, col=1)
    
    fig.add_annotation(
        text="<b>Network Flow Structure:</b> SOURCE → Students → Topics → Coaches → SINK<br>" +
             "<i>Hover any edge to highlight that student's full path | Colors show satisfaction level</i>",
        xref="paper", yref="paper", x=0.5, y=1.08,
        showarrow=False, font=dict(size=11, color="#666"), xanchor="center",
    )
    
    # JavaScript for hover highlighting
    interactive_js = """
    <script>
    document.addEventListener('DOMContentLoaded', function() {
        // Wait a bit for Plotly to fully initialize
        setTimeout(function() {
            // Find the plot container
            const container = document.getElementById('network-flow-plot');
            if (!container) {
                console.error('Plot container not found');
                return;
            }
            
            // Find the actual plotly graph div
            let gd = container.querySelector('.js-plotly-plot');
            
            // If not found, try looking for the plotly plot element directly in the container
            if (!gd) {
                const plots = container.querySelectorAll('[data-plotly]');
                if (plots.length > 0) {
                    gd = plots[0];
                }
            }
            
            // Final fallback: use the first child div that has class or is the plot
            if (!gd && container.children.length > 0) {
                for (let child of container.children) {
                    if (child.classList && child.classList.contains('js-plotly-plot')) {
                        gd = child;
                        break;
                    }
                    if (child.querySelector && child.querySelector('.js-plotly-plot')) {
                        gd = child.querySelector('.js-plotly-plot');
                        break;
                    }
                }
            }
            
            if (!gd) {
                console.error('Could not find Plotly graph div');
                return;
            }
            
            // Check if gd.data exists
            if (!gd.data) {
                console.error('Plot data not found');
                return;
            }
            
            console.log('Found Plotly graph div:', gd);

        console.log('Network flow hover handler initialized');
        console.log('Number of traces:', gd.data.length);

        let hoverTimeout = null;
        let currentStudent = null;
        const originalColors = {};
        const originalWidths = {};

        // Cache original colors/widths for edge traces
        gd.data.forEach((trace, idx) => {
            if (trace.name && trace.name.startsWith('edge_')) {
                originalColors[idx] = (trace.line && trace.line.color) ? trace.line.color : 'rgba(180,180,180,0.8)';
                originalWidths[idx] = (trace.line && trace.line.width) ? trace.line.width : 2;
            }
        });

        function dimAllExcept(studentId) {
            gd.data.forEach((trace, idx) => {
                if (!(trace.name && trace.name.startsWith('edge_'))) return;
                
                // Check if this trace belongs to the student
                let hasStudent = false;
                if (trace.customdata !== undefined && trace.customdata !== null) {
                    if (Array.isArray(trace.customdata) && trace.customdata.length > 0) {
                        hasStudent = (trace.customdata[0] === studentId);
                    } else {
                        hasStudent = (trace.customdata === studentId);
                    }
                }
                
                if (hasStudent) {
                    // Highlight this trace
                    Plotly.restyle(gd, {'line.color': originalColors[idx], 'line.width': Math.max(4, originalWidths[idx] + 2)}, [idx]);
                } else {
                    // Dim this trace
                    Plotly.restyle(gd, {'line.color': 'rgba(200,200,200,0.2)', 'line.width': 1}, [idx]);
                }
            });
        }
        
        function restoreAll() {
            gd.data.forEach((trace, idx) => {
                if (!(trace.name && trace.name.startsWith('edge_'))) return;
                const col = originalColors[idx] || (trace.line && trace.line.color) || 'rgba(180,180,180,0.8)';
                const w = (originalWidths[idx] !== undefined) ? originalWidths[idx] : (trace.line && trace.line.width) || 2;
                Plotly.restyle(gd, {'line.color': col, 'line.width': w}, [idx]);
            });
        }

        gd.on('plotly_hover', function(eventData) {
            if (!eventData || !eventData.points || eventData.points.length === 0) return;
            
            if (hoverTimeout) {
                clearTimeout(hoverTimeout);
                hoverTimeout = null;
            }
            
            const pt = eventData.points[0];
            // customdata is an array, get the first element (or the value itself if scalar)
            let studentId = pt.customdata;
            if (Array.isArray(studentId) && studentId.length > 0) {
                studentId = studentId[0];
            }
            
            console.log('Hover on student:', studentId);
            
            if (!studentId) return;
            
            if (currentStudent === studentId) return; // Already highlighted
            
            currentStudent = studentId;
            dimAllExcept(studentId);
        });

        gd.on('plotly_unhover', function() {
            hoverTimeout = setTimeout(function() {
                restoreAll();
                currentStudent = null;
            }, 120);
        });
        }, 100); // end of setTimeout
    }); // end of DOMContentLoaded
    </script>
    """
    
    # Write HTML with interactive JavaScript
    html_content = fig.to_html(include_plotlyjs=True, full_html=True, div_id="network-flow-plot")
    
    if "</body>" in html_content:
        html_content = html_content.replace("</body>", interactive_js + "\n</body>")
    else:
        html_content = html_content + interactive_js

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html_content)
    
    return output_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create network flow visualization")
    parser.add_argument("--allocation", required=True, help="Path to allocation.csv")
    parser.add_argument("--output", default="visualisations/network_flow.html", help="Output HTML path")
    args = parser.parse_args()
    
    print(f"Loading allocation from {args.allocation}...")
    rows = load_allocation(args.allocation)
    
    # Generate all three visualizations
    import os
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    print("Building main network graph...")
    main_output = args.output.replace('.html', '_main.html')
    output_path = create_network_visualization(rows, main_output)
    print(f"✓ Main network visualization created: {output_path}")
    
    print("Building multipartite layout...")
    multipartite_output = args.output.replace('.html', '_multipartite.html')
    create_multipartite_visualization(rows, multipartite_output)
    print(f"✓ Multipartite visualization created: {multipartite_output}")
    
    print("Building edge colormap...")
    colormap_output = args.output.replace('.html', '_colormap.html')
    create_edge_colormap_visualization(rows, colormap_output)
    print(f"✓ Edge colormap visualization created: {colormap_output}")
    
    print(f"\nAll visualizations created:")
    print(f"  - Main: file://{os.path.abspath(main_output)}")
    print(f"  - Multipartite: file://{os.path.abspath(multipartite_output)}")
    print(f"  - Colormap: file://{os.path.abspath(colormap_output)}")