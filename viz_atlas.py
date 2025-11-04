#!/usr/bin/env python3
"""
Atlas Visualization for Thesis Allocation Network
Shows different graph patterns and structures in the allocation network.
"""
import csv
import argparse
from collections import defaultdict
try:
    import networkx as nx
    import plotly.graph_objects as go
    import numpy as np
    import matplotlib.pyplot as plt
    import random
except ImportError:
    print("Error: networkx or plotly not installed. Run: pip install networkx plotly matplotlib")
    exit(1)


def load_allocation(path):
    rows = []
    with open(path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


def create_atlas_visualization(rows, output_path="visualisations/atlas.html"):
    """Create an atlas visualization showing graph patterns in the allocation."""
    
    # Create sample graph patterns instead of using atlas
    traces = []
    
    # Create some example graph patterns with allocation-specific descriptions
    patterns_info = [
        ("Star Pattern", "Popular Topic - One topic assigned to many students<br><i>Example: Topic 'Machine Learning' → 5 students</i>", nx.star_graph(5)),
        ("Path Pattern", "Sequential Allocation - Chain of student-topic assignments<br><i>Example: S1→T1→S2→T2→S3→T3</i>", nx.path_graph(6)),
        ("Cycle Pattern", "Circular Assignment - Closed allocation loop<br><i>Rare pattern in thesis allocation</i>", nx.cycle_graph(6)),
        ("Complete Pattern", "Fully Connected - Every student has access to every topic<br><i>Ideal but unlikely in constrained allocation</i>", nx.complete_graph(4)),
        ("Hub Pattern", "Coach Hub - One coach supervising multiple student-topic pairs<br><i>Example: Coach supervises 5 different projects</i>", nx.wheel_graph(5)),
        ("Parallel Pattern", "Side-by-Side Allocations - Independent allocation paths<br><i>Example: Multiple students with different topics</i>", nx.ladder_graph(5)),
        ("Ring Pattern", "Circular Supervision - Ring of coach-student-topic connections<br><i>Complex allocation structure</i>", nx.circular_ladder_graph(4)),
    ]
    
    # Position offset for each pattern
    current_x = 0
    current_y = 0
    spacing = 4
    patterns_per_row = 3
    
    for idx, (pattern_name, pattern_desc, G) in enumerate(patterns_info):
        row = idx // patterns_per_row
        col = idx % patterns_per_row
        
        x_offset = col * spacing
        y_offset = -row * spacing
        
        # Get layout for this pattern
        try:
            pos = nx.spring_layout(G, k=1.5, seed=42 + idx)
        except:
            pos = {n: (0, 0) for n in G.nodes()}
        
        # Scale and offset positions
        for node in pos:
            pos[node] = (pos[node][0] + x_offset, pos[node][1] + y_offset)
        
        # Calculate center position for label
        if G.number_of_nodes() > 0:
            center_x = x_offset
            center_y = y_offset - 2.5  # Position label below the pattern
        else:
            center_x = x_offset
            center_y = y_offset
        
        # Color for this pattern
        r = 0.3 + random.random() * 0.4
        g = 0.3 + random.random() * 0.4
        b = 0.3 + random.random() * 0.4
        color = f"rgba({int(r*255)}, {int(g*255)}, {int(b*255)}, 0.7)"
        
        # Add edges
        edge_x = []
        edge_y = []
        for (src, tgt) in G.edges():
            edge_x += [pos[src][0], pos[tgt][0], None]
            edge_y += [pos[src][1], pos[tgt][1], None]
        
        if edge_x:
            edge_trace = go.Scatter(
                x=edge_x, y=edge_y,
                mode='lines',
                line=dict(width=2, color=color),
                hoverinfo='skip',
                showlegend=False
            )
            traces.append(edge_trace)
        
        # Add nodes with better hover info
        node_x = [pos[n][0] for n in G.nodes()]
        node_y = [pos[n][1] for n in G.nodes()]
        
        # Create informative hover text for each node
        hover_texts = []
        for node in G.nodes():
            degree = G.degree(node)
            # Interpret the node based on its degree and pattern
            if pattern_name == "Star Pattern":
                if degree == (G.number_of_nodes() - 1):
                    node_role = "🏛️ Popular Topic"
                else:
                    node_role = "👤 Student assigned to topic"
            elif pattern_name == "Hub Pattern":
                if degree > 2:
                    node_role = "🏛️ Coach Hub"
                else:
                    node_role = "👤 Student-Topic pair"
            elif pattern_name in ["Path Pattern", "Parallel Pattern"]:
                node_role = "🔗 Allocation connection"
            elif pattern_name == "Complete Pattern":
                node_role = "🌐 Any entity (Student/Topic/Coach)"
            else:
                node_role = "📋 Allocation entity"
            
            hover_texts.append(
                f"<b>{pattern_name}</b><br>"
                f"<b>{node_role}</b><br>"
                f"{pattern_desc}<br>"
                f"<br>Node ID: {node}<br>"
                f"Degree: {degree} connection(s)<br>"
                f"Total nodes in pattern: {G.number_of_nodes()}"
            )
        
        node_trace = go.Scatter(
            x=node_x, y=node_y,
            mode='markers',
            marker=dict(size=25, color=color, line=dict(width=2, color="white")),
            text=hover_texts,
            hoverinfo='text',
            showlegend=False,
            name=pattern_name  # For legend
        )
        traces.append(node_trace)
        
        # Add label for this pattern
        label_trace = go.Scatter(
            x=[center_x], y=[center_y],
            mode='text',
            text=[pattern_name],
            textfont=dict(size=14, color="#333", family="Arial Black"),
            hoverinfo='skip',
            showlegend=False
        )
        traces.append(label_trace)
    
    # Create figure
    fig = go.Figure(data=traces)
    
    fig.update_layout(
        title={
            "text": f"Allocation Pattern Reference - {len(patterns_info)} Common Structures<br><sub>Example patterns that can appear in your student-topic-coach allocation network</sub>",
            "x": 0.5,
            "xanchor": "center",
            "font": {"size": 20}
        },
        showlegend=False,
        hovermode='closest',
        height=1000,
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        plot_bgcolor='rgba(240, 240, 240, 1)',
        paper_bgcolor='white'
    )
    
    # Write HTML
    html_content = fig.to_html(include_plotlyjs=True, full_html=True)
    
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html_content)
    
    return output_path


def create_analytical_atlas(rows, output_path="visualisations/analytical_atlas.html"):
    """Create an atlas from actual allocation data analyzing real patterns."""
    
    # Build the allocation graph
    G_full = nx.DiGraph()
    
    students = list({row["student"] for row in rows})
    topics = list({row["assigned_topic"] for row in rows})
    coaches = list({row["assigned_coach"] for row in rows})
    
    # Add nodes
    for s in sorted(students):
        G_full.add_node(f"S_{s}", node_type="student")
    for t in sorted(topics):
        G_full.add_node(f"T_{t}", node_type="topic")
    for c in sorted(coaches):
        G_full.add_node(f"C_{c}", node_type="coach")
    
    G_full.add_node("SOURCE", node_type="source")
    G_full.add_node("SINK", node_type="sink")
    
    # Add edges from allocation
    for row in rows:
        s_node = f"S_{row['student']}"
        t_node = f"T_{row['assigned_topic']}"
        c_node = f"C_{row['assigned_coach']}"
        
        G_full.add_edge("SOURCE", s_node)
        G_full.add_edge(s_node, t_node)
        G_full.add_edge(t_node, c_node)
        G_full.add_edge(c_node, "SINK")
    
    # Find actual patterns in the allocation
    traces = []
    
    # Pattern 1: Topics with many students (Star patterns)
    topic_to_students = defaultdict(list)
    for row in rows:
        topic_to_students[row['assigned_topic']].append(row['student'])
    
    # Get all topics with multiple students
    top_topics = [(topic, students) for topic, students in sorted(topic_to_students.items(), key=lambda x: len(x[1]), reverse=True) if len(students) > 1]
    
    # Pattern 2: Coaches with many students (Hub patterns)
    coach_to_students = defaultdict(list)
    for row in rows:
        coach_to_students[row['assigned_coach']].append(row['student'])
    
    # Get all coaches with multiple students
    top_coaches = [(coach, students) for coach, students in sorted(coach_to_students.items(), key=lambda x: len(x[1]), reverse=True) if len(students) > 1]
    
    # Get some single-student patterns to show individual allocations (for reference, not currently visualized)
    # single_topic_allocations = [(topic, [s]) for topic, students in topic_to_students.items() if len(students) == 1]
    # single_coach_allocations = [(coach, [s]) for coach, students in coach_to_students.items() if len(students) == 1]
    
    # Visualize found patterns
    spacing = 6
    patterns_per_row = 3
    pattern_idx = 0
    
    # Visualize topic star patterns
    for topic, student_list in top_topics:
        if len(student_list) > 1:
            # Create a star subgraph centered on this topic
            subgraph_nodes = [f"T_{topic}"]
            for student in student_list:
                subgraph_nodes.append(f"S_{student}")
            
            # Create mini star graph for visualization
            pos = {}
            center_x = (pattern_idx % patterns_per_row) * spacing
            center_y = -(pattern_idx // patterns_per_row) * spacing
            
            # Position topic in center
            pos[f"T_{topic}"] = (center_x, center_y)
            
            # Position students around the topic
            num_students = len(student_list)
            for i, student in enumerate(student_list):
                angle = 2 * np.pi * i / num_students
                radius = 1.5
                pos[f"S_{student}"] = (
                    center_x + radius * np.cos(angle),
                    center_y + radius * np.sin(angle)
                )
            
            # Color
            color = f"rgba({100 + (pattern_idx * 40) % 155}, {150 + (pattern_idx * 60) % 105}, {200 + (pattern_idx * 80) % 55}, 0.7)"
            
            # Add edges
            edge_x, edge_y = [], []
            for student in student_list:
                edge_x += [pos[f"T_{topic}"][0], pos[f"S_{student}"][0], None]
                edge_y += [pos[f"T_{topic}"][1], pos[f"S_{student}"][1], None]
            
            traces.append(go.Scatter(
                x=edge_x, y=edge_y,
                mode='lines',
                line=dict(width=2, color=color),
                hoverinfo='skip',
                showlegend=False
            ))
            
            # Add nodes
            node_x = [pos[n][0] for n in subgraph_nodes]
            node_y = [pos[n][1] for n in subgraph_nodes]
            
            hover_texts = []
            for n in subgraph_nodes:
                if n.startswith('T_'):
                    hover_texts.append(
                        f"<b>📚 Topic: {topic}</b><br><br>"
                        f"<b>Pattern:</b> Popular Topic (Star)<br>"
                        f"This topic is assigned to {len(student_list)} student(s), making it a popular choice.<br><br>"
                        f"<b>Student(s):</b> {', '.join(student_list)}"
                    )
                else:
                    student = n.replace('S_', '')
                    hover_texts.append(
                        f"<b>👤 Student: {student}</b><br><br>"
                        f"<b>Topic:</b> {topic}<br>"
                        f"<b>Role:</b> Assigned student in popular topic star pattern"
                    )
            
            traces.append(go.Scatter(
                x=node_x, y=node_y,
                mode='markers+text',
                marker=dict(size=30, color=color, line=dict(width=2, color="white")),
                text=[f"📚\n{topic}" if n.startswith('T_') else f"👤\n{n.replace('S_', '')}" for n in subgraph_nodes],
                textfont=dict(size=9, color="white"),
                textposition="middle center",
                hovertext=hover_texts,
                hoverinfo='text',
                showlegend=False
            ))
            
            pattern_idx += 1
    
    # Visualize coach hub patterns  
    for coach, student_list in top_coaches:
        if len(student_list) > 1:
            # Create a star subgraph centered on this coach
            subgraph_nodes = [f"C_{coach}"]
            for student in student_list:
                subgraph_nodes.append(f"S_{student}")
            
            pos = {}
            center_x = (pattern_idx % patterns_per_row) * spacing
            center_y = -(pattern_idx // patterns_per_row) * spacing
            
            pos[f"C_{coach}"] = (center_x, center_y)
            
            num_students = len(student_list)
            for i, student in enumerate(student_list):
                angle = 2 * np.pi * i / num_students
                radius = 1.5
                pos[f"S_{student}"] = (
                    center_x + radius * np.cos(angle),
                    center_y + radius * np.sin(angle)
                )
            
            color = f"rgba({150 + (pattern_idx * 50) % 105}, {100 + (pattern_idx * 40) % 155}, {200 + (pattern_idx * 70) % 55}, 0.7)"
            
            edge_x, edge_y = [], []
            for student in student_list:
                edge_x += [pos[f"C_{coach}"][0], pos[f"S_{student}"][0], None]
                edge_y += [pos[f"C_{coach}"][1], pos[f"S_{student}"][1], None]
            
            traces.append(go.Scatter(
                x=edge_x, y=edge_y,
                mode='lines',
                line=dict(width=2, color=color),
                hoverinfo='skip',
                showlegend=False
            ))
            
            node_x = [pos[n][0] for n in subgraph_nodes]
            node_y = [pos[n][1] for n in subgraph_nodes]
            
            hover_texts = []
            for n in subgraph_nodes:
                if n.startswith('C_'):
                    hover_texts.append(
                        f"<b>🏛️ Coach: {coach}</b><br><br>"
                        f"<b>Pattern:</b> Coach Hub (Star)<br>"
                        f"This coach is supervising {len(student_list)} student(s), making them a hub in the allocation network.<br><br>"
                        f"<b>Student(s):</b> {', '.join(student_list)}"
                    )
                else:
                    student = n.replace('S_', '')
                    hover_texts.append(
                        f"<b>👤 Student: {student}</b><br><br>"
                        f"<b>Coach:</b> {coach}<br>"
                        f"<b>Role:</b> Supervised student in coach hub pattern"
                    )
            
            traces.append(go.Scatter(
                x=node_x, y=node_y,
                mode='markers+text',
                marker=dict(size=30, color=color, line=dict(width=2, color="white")),
                text=[f"🏛️\n{coach}" if n.startswith('C_') else f"👤\n{n.replace('S_', '')}" for n in subgraph_nodes],
                textfont=dict(size=9, color="white"),
                textposition="middle center",
                hovertext=hover_texts,
                hoverinfo='text',
                showlegend=False
            ))
            
            pattern_idx += 1
    
    # Create figure
    fig = go.Figure(data=traces)
    
    # Count total students shown
    total_students_shown = sum(len(students) for _, students in top_topics) + sum(len(students) for _, students in top_coaches)
    
    fig.update_layout(
        title={
            "text": f"Analytical Atlas - {pattern_idx} Patterns Detected ({total_students_shown} students shown)<br><sub>Real patterns from your allocation: Popular topics and coach hubs</sub>",
            "x": 0.5,
            "xanchor": "center",
            "font": {"size": 20}
        },
        showlegend=False,
        hovermode='closest',
        height=800,
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
    )
    
    html_content = fig.to_html(include_plotlyjs=True, full_html=True)
    
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html_content)
    
    return output_path


def create_pattern_analysis(rows, output_path="visualisations/patterns.html"):
    """Create a visualization showing different patterns in the actual allocation."""
    
    # Build simplified graphs from allocation
    G = nx.DiGraph()
    
    student_alloc = defaultdict(list)
    for row in rows:
        s = row["student"]
        t = row["assigned_topic"]
        c = row["assigned_coach"]
        student_alloc[s].append((t, c))
    
    # Pattern 1: Star patterns (topics with many students)
    topic_counts = defaultdict(int)
    for student, allocs in student_alloc.items():
        for topic, _ in allocs:
            topic_counts[topic] += 1
    
    # Find topics with multiple students (star patterns)
    popular_topics = [t for t, count in topic_counts.items() if count >= 2][:10]
    
    # Pattern 2: Coach load (who has many students)
    coach_counts = defaultdict(int)
    for student, allocs in student_alloc.items():
        for _, coach in allocs:
            coach_counts[coach] += 1
    
    # Create visualization showing distribution patterns
    from plotly.subplots import make_subplots
    
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=("Topic Distribution (Star Patterns)", "Coach Load Distribution"),
    )
    
    # Topic distribution
    topic_names = list(topic_counts.keys())[:20]
    topic_vals = [topic_counts[t] for t in topic_names]
    
    fig.add_trace(
        go.Bar(x=topic_names, y=topic_vals, name="Students per Topic"),
        row=1, col=1
    )
    
    # Coach distribution
    coach_names = list(coach_counts.keys())[:20]
    coach_vals = [coach_counts[c] for c in coach_names]
    
    fig.add_trace(
        go.Bar(x=coach_names, y=coach_vals, name="Students per Coach"),
        row=1, col=2
    )
    
    fig.update_layout(
        title_text="Allocation Pattern Analysis",
        showlegend=False,
        height=500
    )
    
    fig.update_xaxes(tickangle=-45)
    
    html_content = fig.to_html(include_plotlyjs=True, full_html=True)
    
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html_content)
    
    return output_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create graph atlas visualization")
    parser.add_argument("--allocation", required=True, help="Path to allocation.csv")
    parser.add_argument("--output", default="visualisations/atlas.html", help="Output HTML path")
    args = parser.parse_args()
    
    print(f"Loading allocation from {args.allocation}...")
    rows = load_allocation(args.allocation)
    
    print("Generating graph atlas visualization...")
    create_atlas_visualization(rows, args.output)
    print(f"✓ Graph atlas visualization created: {args.output}")
    
    print("Generating analytical atlas from your allocation...")
    analytical_output = args.output.replace('atlas.html', 'analytical_atlas.html')
    create_analytical_atlas(rows, analytical_output)
    print(f"✓ Analytical atlas created: {analytical_output}")
    
    print("Generating pattern analysis...")
    pattern_output = args.output.replace('atlas.html', 'patterns.html')
    create_pattern_analysis(rows, pattern_output)
    print(f"✓ Pattern analysis created: {pattern_output}")

