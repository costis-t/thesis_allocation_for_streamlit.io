#!/usr/bin/env python3
"""
Chord Diagram Visualization for Thesis Allocation.
Shows flow from Students → Topics → Coaches with interactive D3.js-based visualization.
"""
from __future__ import annotations
import csv
import json
from collections import defaultdict
from typing import Dict, List, Tuple
import argparse


def load_allocation(path: str) -> List[Dict]:
    """Load allocation CSV."""
    rows = []
    with open(path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


def build_chord_data(rows: List[Dict]) -> Tuple[List[str], List[List[int]], List[str]]:
    """
    Build chord diagram data.
    Returns: (labels, matrix, colors)
    
    Structure: Students | Topics | Coaches
    Labels: S01, S02, ... | T01, T02, ... | C01, C02, ...
    Matrix: connections between each pair
    Colors: satisfaction levels
    """
    students = []
    topics = []
    coaches = []
    
    # Collect unique values
    for row in rows:
        s = row["student"]
        t = row["assigned_topic"]
        c = row["assigned_coach"]
        
        if s not in students:
            students.append(s)
        if t not in topics:
            topics.append(t)
        if c not in coaches:
            coaches.append(c)
    
    students.sort()
    topics.sort()
    coaches.sort()
    
    # Create labels: Students | Topics | Coaches
    labels = students + topics + coaches
    n = len(labels)
    
    # Create adjacency matrix (n x n)
    matrix = [[0] * n for _ in range(n)]
    
    # Satisfaction levels
    satisfaction_map = {}  # (from_idx, to_idx) -> satisfaction_level
    
    # Fill matrix with connections
    for row in rows:
        s_idx = students.index(row["student"])
        t_idx = len(students) + topics.index(row["assigned_topic"])
        c_idx = len(students) + len(topics) + coaches.index(row["assigned_coach"])
        
        # Student → Topic
        matrix[s_idx][t_idx] = 1
        matrix[t_idx][s_idx] = 1
        
        # Topic → Coach
        matrix[t_idx][c_idx] = 1
        matrix[c_idx][t_idx] = 1
        
        # Track satisfaction for coloring
        rank = int(row["preference_rank"])
        satisfaction_map[(s_idx, t_idx)] = rank
    
    # Create colors based on satisfaction
    colors = []
    rank_to_color = {
        0: "#2ecc71",      # Tier1 - green
        1: "#27ae60",      # Tier2 - dark green
        2: "#16a085",      # Tier3 - teal
        10: "#3498db",     # 1st choice - blue
        11: "#f39c12",     # 2nd choice - orange
        12: "#e67e22",     # 3rd choice - dark orange
        13: "#e74c3c",     # 4th choice - red
        14: "#c0392b",     # 5th choice - dark red
        999: "#95a5a6",    # Unranked - gray
    }
    
    for i in range(n):
        if i < len(students):
            colors.append("#3498db")  # Students - blue
        elif i < len(students) + len(topics):
            colors.append("#2ecc71")  # Topics - green
        else:
            colors.append("#9b59b6")  # Coaches - purple
    
    return labels, matrix, colors, students, topics, coaches


def create_chord_html(labels: List[str], matrix: List[List[int]], colors: List[str], 
                     students: List[str], topics: List[str], coaches: List[str],
                     output_path: str = "chord_diagram.html") -> str:
    """Create interactive HTML chord diagram using D3.js."""
    
    # Convert matrix to chord data format
    # We need to aggregate flows: Student→Topic, Topic→Coach
    
    n_students = len(students)
    n_topics = len(topics)
    n_coaches = len(coaches)
    
    # Create simplified matrix: rows are students, columns are coaches
    # (topics are intermediate, so we aggregate through them)
    simple_matrix = [[0] * n_coaches for _ in range(n_students)]
    
    # This isn't a pure chord diagram anymore, but a "flow" visualization
    # Let me create a better D3-based visualization
    
    html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <style>
        body {{
            font-family: Arial, sans-serif;
            display: flex;
            flex-direction: column;
            align-items: center;
            background: #f5f5f5;
            margin: 0;
            padding: 20px;
        }}
        
        h1 {{
            color: #333;
            margin-bottom: 10px;
        }}
        
        .info {{
            background: #ecf0f1;
            padding: 15px;
            border-radius: 5px;
            margin-bottom: 20px;
            max-width: 800px;
            text-align: center;
        }}
        
        #chart {{
            background: white;
            border-radius: 8px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
            padding: 20px;
        }}
        
        .tooltip {{
            position: fixed;
            padding: 8px 12px;
            background: rgba(0, 0, 0, 0.9);
            color: #fff;
            border-radius: 4px;
            font-size: 12px;
            pointer-events: none;
            z-index: 1000;
            display: none;
            border: 1px solid #666;
        }}
        
        .chord-group {{
            cursor: pointer;
        }}
        
        .chord-group path {{
            transition: stroke-width 0.2s;
        }}
        
        .chord-group path:hover {{
            stroke-width: 2px !important;
        }}
        
        .chord {{
            fill-opacity: 0.4;
            stroke-width: 0.5px;
            transition: fill-opacity 0.2s;
            cursor: pointer;
        }}
        
        .chord:hover {{
            fill-opacity: 0.8 !important;
            stroke-width: 1.5px !important;
        }}
        
        .legend {{
            margin-top: 30px;
            background: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
            max-width: 600px;
        }}
        
        .legend-item {{
            display: inline-block;
            margin-right: 20px;
            margin-bottom: 10px;
        }}
        
        .legend-color {{
            display: inline-block;
            width: 20px;
            height: 20px;
            margin-right: 8px;
            vertical-align: middle;
            border-radius: 3px;
        }}
    </style>
    <script src="https://d3js.org/d3.v7.min.js"></script>
</head>
<body>
    <h1>Thesis Allocation Chord Diagram</h1>
    
    <div class="info">
        <p><strong>Interactive Visualization:</strong> Hover over chords and nodes to see connections.</p>
        <p>Shows flow from Students (blue) → Topics (green) → Coaches (purple)</p>
    </div>
    
    <div id="chart"></div>
    
    <div class="legend">
        <strong>Legend:</strong><br>
        <div class="legend-item">
            <div class="legend-color" style="background: #3498db;"></div>
            <span>Students</span>
        </div>
        <div class="legend-item">
            <div class="legend-color" style="background: #2ecc71;"></div>
            <span>Topics</span>
        </div>
        <div class="legend-item">
            <div class="legend-color" style="background: #9b59b6;"></div>
            <span>Coaches</span>
        </div>
        <br><br>
        <strong>Satisfaction Levels (chord colors):</strong><br>
        <div class="legend-item">
            <div class="legend-color" style="background: #3498db;"></div>
            <span>1st Choice</span>
        </div>
        <div class="legend-item">
            <div class="legend-color" style="background: #f39c12;"></div>
            <span>2nd Choice</span>
        </div>
        <div class="legend-item">
            <div class="legend-color" style="background: #e74c3c;"></div>
            <span>Unranked</span>
        </div>
    </div>
    
    <div class="tooltip" id="tooltip"></div>
    
    <script>
        // Data from allocation
        const data = {json.dumps({
            'students': students,
            'topics': topics,
            'coaches': coaches,
            'matrix': matrix,
            'colors': colors
        })};
        
        const width = 900;
        const height = 900;
        const innerRadius = Math.min(width, height) * 0.3;
        const outerRadius = innerRadius + 30;
        
        const svg = d3.select("#chart")
            .append("svg")
            .attr("width", width)
            .attr("height", height)
            .attr("viewBox", [-width / 2, -height / 2, width, height]);
        
        // Create chord layout
        const chord = d3.chord()
            .padAngle(0.05)
            .sortSubgroups(d3.descending);
        
        const arc = d3.arc()
            .innerRadius(innerRadius)
            .outerRadius(outerRadius);
        
        const ribbon = d3.ribbon()
            .source(d => d.source)
            .target(d => d.target);
        
        // Convert matrix to chord format
        const chords = chord(data.matrix);
        
        // Color scale for nodes
        const color = d3.scaleOrdinal()
            .domain(d3.range(data.matrix.length))
            .range(data.colors);
        
        // Tooltip functions
        function showTooltip(event, text) {{
            const tooltip = document.getElementById('tooltip');
            tooltip.textContent = text;
            tooltip.style.display = 'block';
            tooltip.style.left = (event.pageX + 10) + 'px';
            tooltip.style.top = (event.pageY - 10) + 'px';
        }}
        
        function hideTooltip() {{
            const tooltip = document.getElementById('tooltip');
            tooltip.style.display = 'none';
        }}
        
        // Add groups
        const group = svg.append("g")
            .selectAll("g")
            .data(chords.groups)
            .join("g")
            .attr("class", "chord-group");
        
        group.append("path")
            .attr("fill", d => color(d.index))
            .attr("stroke", d => d3.rgb(color(d.index)).darker())
            .attr("d", arc)
            .on("mouseover", function(event, d) {{
                const idx = d.index;
                let nodeName = '';
                if (idx < data.students.length) {{
                    nodeName = data.students[idx];
                }} else if (idx < data.students.length + data.topics.length) {{
                    nodeName = data.topics[idx - data.students.length];
                }} else {{
                    nodeName = data.coaches[idx - data.students.length - data.topics.length];
                }}
                showTooltip(event, nodeName);
                d3.select(this).attr("stroke-width", 2.5);
            }})
            .on("mouseout", function() {{
                hideTooltip();
                d3.select(this).attr("stroke-width", 1);
            }});
        
        // Add labels
        group.append("text")
            .each(d => {{ d.angle = (d.startAngle + d.endAngle) / 2; }})
            .attr("dy", "0.35em")
            .attr("transform", d => {{
                const angle = d.angle * 180 / Math.PI - 90;
                const rotate1 = angle;
                const translate = outerRadius + 30;
                const rotate2 = d.angle > Math.PI ? 180 : 0;
                return 'rotate(' + rotate1 + ')translate(' + translate + ')rotate(' + rotate2 + ')';
            }})
            .attr("text-anchor", d => d.angle > Math.PI ? "end" : null)
            .text(d => {{
                const idx = d.index;
                if (idx < data.students.length) return data.students[idx];
                else if (idx < data.students.length + data.topics.length) 
                    return data.topics[idx - data.students.length];
                else 
                    return data.coaches[idx - data.students.length - data.topics.length];
            }})
            .style("font-size", "11px")
            .style("font-weight", "bold")
            .style("pointer-events", "none");
        
        // Add ribbons (chords)
        svg.append("g")
            .attr("fill-opacity", 0.67)
            .selectAll("path")
            .data(chords)
            .join("path")
            .attr("class", "chord")
            .attr("d", ribbon)
            .attr("fill", d => {{
                const source_idx = d.source.index;
                const target_idx = d.target.index;
                
                if (source_idx < data.students.length && 
                    target_idx >= data.students.length && 
                    target_idx < data.students.length + data.topics.length) {{
                    return "#3498db";
                }}
                return "#95a5a6";
            }})
            .attr("stroke", d => {{
                return d3.rgb(d.source.startAngle > d.target.startAngle ? 
                    color(d.source.index) :
                    color(d.target.index)).darker();
            }})
            .on("mouseover", function(event, d) {{
                const source_idx = d.source.index;
                const target_idx = d.target.index;
                
                let source_name = '';
                let target_name = '';
                
                if (source_idx < data.students.length) {{
                    source_name = data.students[source_idx];
                }} else if (source_idx < data.students.length + data.topics.length) {{
                    source_name = data.topics[source_idx - data.students.length];
                }} else {{
                    source_name = data.coaches[source_idx - data.students.length - data.topics.length];
                }}
                
                if (target_idx < data.students.length) {{
                    target_name = data.students[target_idx];
                }} else if (target_idx < data.students.length + data.topics.length) {{
                    target_name = data.topics[target_idx - data.students.length];
                }} else {{
                    target_name = data.coaches[target_idx - data.students.length - data.topics.length];
                }}
                
                showTooltip(event, source_name + ' → ' + target_name);
                d3.select(this).attr("stroke-width", 2);
            }})
            .on("mouseout", function() {{
                hideTooltip();
                d3.select(this).attr("stroke-width", 0.5);
            }});
    </script>
</body>
</html>
"""
    
    with open(output_path, "w") as f:
        f.write(html_content)
    
    return output_path


def main():
    parser = argparse.ArgumentParser(description="Create chord diagram from allocation")
    parser.add_argument("--allocation", required=True, help="Path to allocation.csv")
    parser.add_argument("--output", default="chord_diagram.html", help="Output HTML path")
    
    args = parser.parse_args()
    
    print(f"Loading allocation from {args.allocation}...")
    rows = load_allocation(args.allocation)
    
    print(f"Building chord data ({len(rows)} allocations)...")
    labels, matrix, colors, students, topics, coaches = build_chord_data(rows)
    
    print(f"Creating chord diagram ({len(students)} students, {len(topics)} topics, {len(coaches)} coaches)...")
    output_path = create_chord_html(labels, matrix, colors, students, topics, coaches, args.output)
    
    print(f"✓ Chord diagram created: {output_path}")
    print(f"\nOpen in browser: file://{output_path}")


if __name__ == "__main__":
    main()
