"""
Really Advanced Charts - Tier 1 and Tier 2 Visualizations
All visualizations for deep analysis of allocation results
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from scipy.stats import entropy


def create_preference_funnel(allocation_df):
    """
    Create Preference Satisfaction Funnel showing progression through satisfaction levels
    """
    # Count by preference category
    pref_counts = {
        '1st Choice': len(allocation_df[allocation_df['preference_rank'] == 10]),
        '2nd Choice': len(allocation_df[allocation_df['preference_rank'] == 11]),
        '3rd Choice': len(allocation_df[allocation_df['preference_rank'] == 12]),
        '4th Choice': len(allocation_df[allocation_df['preference_rank'] == 13]),
        '5th Choice': len(allocation_df[allocation_df['preference_rank'] == 14]),
        'Tier 1': len(allocation_df[allocation_df['preference_rank'] == 0]),
        'Tier 2': len(allocation_df[allocation_df['preference_rank'] == 1]),
        'Tier 3': len(allocation_df[allocation_df['preference_rank'] == 2]),
        'Unranked': len(allocation_df[allocation_df['preference_rank'] == 999]),
    }
    
    # Remove zeros
    pref_counts = {k: v for k, v in pref_counts.items() if v > 0}
    
    fig = go.Figure(go.Funnel(
        y=list(pref_counts.keys()),
        x=list(pref_counts.values()),
        marker=dict(
            color=['#2ca02c', '#1f77b4', '#9467bd', '#ff7f0e', '#d62728',
                   '#2ca02c', '#1f77b4', '#9467bd', '#808080'][:len(pref_counts)],
            line=dict(color='rgba(0,0,0,0.3)', width=2)
        ),
        textposition="inside",
        textinfo="value+percent initial"
    ))
    
    fig.update_layout(
        title="Preference Satisfaction Funnel",
        xaxis_title="Number of Students",
        yaxis_title="Satisfaction Level",
        height=500,
        hovermode='closest'
    )
    
    return fig


def create_cost_violin_plot(allocation_df):
    """
    Create Violin Plot showing cost distribution by preference type
    """
    # Create preference labels
    def get_pref_label(rank):
        if rank == 0: return 'Tier 1'
        elif rank == 1: return 'Tier 2'
        elif rank == 2: return 'Tier 3'
        elif rank == 10: return '1st Choice'
        elif rank == 11: return '2nd Choice'
        elif rank == 12: return '3rd Choice'
        elif rank == 13: return '4th Choice'
        elif rank == 14: return '5th Choice'
        elif rank == 999: return 'Unranked'
        elif rank == -1: return 'Forced'
        else: return 'Other'
    
    allocation_df_copy = allocation_df.copy()
    allocation_df_copy['pref_label'] = allocation_df_copy['preference_rank'].apply(get_pref_label)
    
    fig = px.violin(allocation_df_copy, x='pref_label', y='effective_cost',
                    title='Cost Distribution by Preference Type',
                    labels={'pref_label': 'Preference Type', 'effective_cost': 'Effective Cost'},
                    color='pref_label',
                    color_discrete_sequence=['#2ca02c', '#1f77b4', '#9467bd', '#ff7f0e', 
                                            '#d62728', '#ff7f0e', '#9467bd', '#1f77b4', 
                                            '#808080', '#000000'])
    
    fig.update_layout(
        height=500,
        showlegend=False,
        xaxis_tickangle=-45
    )
    
    return fig


def create_topic_demand_vs_capacity(allocation_df, repo):
    """
    Create horizontal bar chart showing Topic Demand vs Capacity Mismatch
    """
    topic_counts = allocation_df['assigned_topic'].value_counts().sort_index()
    
    topics_data = []
    for topic_id in sorted(repo.topics.keys()):
        topic = repo.topics[topic_id]
        assigned = topic_counts.get(topic_id, 0)
        capacity = topic.topic_cap
        mismatch = assigned - capacity
        
        topics_data.append({
            'Topic': topic_id,
            'Assigned': assigned,
            'Capacity': capacity,
            'Mismatch': mismatch,
            'Utilization': (assigned / capacity * 100) if capacity > 0 else 0,
            'Status': 'Over' if mismatch > 0 else ('Under' if mismatch < 0 else 'Balanced')
        })
    
    df_topics = pd.DataFrame(topics_data).sort_values('Mismatch', ascending=True)
    
    # Create stacked bar chart
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        y=df_topics['Topic'],
        x=df_topics['Capacity'],
        name='Capacity',
        marker_color='rgba(46, 204, 113, 0.7)',
        orientation='h'
    ))
    
    fig.add_trace(go.Bar(
        y=df_topics['Topic'],
        x=df_topics['Assigned'],
        name='Assigned',
        marker_color='rgba(52, 152, 219, 0.7)',
        orientation='h'
    ))
    
    # Add mismatch annotation
    colors = ['#d62728' if x > 0 else '#2ca02c' if x < 0 else '#808080' 
              for x in df_topics['Mismatch']]
    
    fig.add_trace(go.Bar(
        y=df_topics['Topic'],
        x=df_topics['Mismatch'],
        name='Mismatch',
        marker_color=colors,
        orientation='h'
    ))
    
    fig.update_layout(
        title='Topic Demand vs Capacity Mismatch',
        xaxis_title='Count',
        yaxis_title='Topic',
        barmode='stack',
        height=600,
        hovermode='closest'
    )
    
    return fig


def create_cost_breakdown_pie(allocation_df):
    """
    Create Pie Chart showing allocation cost breakdown by category
    """
    def get_cost_category(row):
        rank = row['preference_rank']
        if rank == 0: return 'Tier 1'
        elif rank == 1: return 'Tier 2'
        elif rank == 2: return 'Tier 3'
        elif 10 <= rank <= 14: return 'Ranked Choice'
        elif rank == 999: return 'Unranked'
        else: return 'Other'
    
    allocation_df_copy = allocation_df.copy()
    allocation_df_copy['cost_category'] = allocation_df_copy.apply(get_cost_category, axis=1)
    
    category_counts = allocation_df_copy['cost_category'].value_counts()
    
    colors = {
        'Tier 1': '#2ca02c',
        'Tier 2': '#1f77b4',
        'Tier 3': '#9467bd',
        'Ranked Choice': '#ff7f0e',
        'Unranked': '#d62728',
        'Other': '#808080'
    }
    
    fig = px.pie(
        values=category_counts.values,
        names=category_counts.index,
        title='Allocation Cost Breakdown by Category',
        color=category_counts.index,
        color_discrete_map=colors
    )
    
    fig.update_traces(textposition='inside', textinfo='percent+label')
    fig.update_layout(height=500)
    
    return fig


def create_fairness_radar(metrics):
    """
    Create Radar Chart for multi-dimensional fairness comparison
    """
    categories = [
        'Cost Fairness',
        'Topic Balance',
        'Coach Balance',
        'Dept Balance',
        'Preference Sat',
        'Overall Score'
    ]
    
    values = [
        (1 - metrics.get('gini_cost', 0)) * 100,
        metrics.get('topic_balance', 0) * 100,
        metrics.get('coach_balance', 0) * 100,
        metrics.get('dept_balance', 0) * 100,
        metrics.get('ranked_satisfaction', 0) * 100,
        75  # Target benchmark
    ]
    
    fig = go.Figure(data=go.Scatterpolar(
        r=values,
        theta=categories,
        fill='toself',
        name='Current Allocation',
        line_color='#1f77b4',
        marker=dict(size=8)
    ))
    
    # Add benchmark line
    benchmark = [75, 80, 80, 80, 70, 75]
    fig.add_trace(go.Scatterpolar(
        r=benchmark,
        theta=categories,
        fill='toself',
        name='Benchmark',
        line_color='#d62728',
        opacity=0.5
    ))
    
    fig.update_layout(
        title='Fairness Comparison Radar',
        polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
        height=600,
        hovermode='closest'
    )
    
    return fig


def create_student_satisfaction_scatter(allocation_df, repo, x_range=None, y_range=None):
    """
    Create Scatter Plot showing Cost vs Preference Rank with scattered bubbles
    """
    # Add jitter to make bubbles more scattered
    np.random.seed(42)  # For reproducible jitter
    jitter_x = np.random.normal(0, 0.1, len(allocation_df))
    jitter_y = np.random.normal(0, 0.1, len(allocation_df))
    
    # Create jittered coordinates
    jittered_x = allocation_df['preference_rank'] + jitter_x
    jittered_y = allocation_df['effective_cost'] + jitter_y
    
    fig = go.Figure()
    
    # Add scatter plot with jittered coordinates
    fig.add_trace(go.Scatter(
        x=jittered_x,
        y=jittered_y,
        mode='markers',
        marker=dict(
            size=8,
            opacity=0.7,
            line=dict(width=1, color='white')
        ),
        text=allocation_df['student'],
        hovertemplate='<b>%{text}</b><br>' +
                     'Preference Rank: %{x:.1f}<br>' +
                     'Effective Cost: %{y:.1f}<br>' +
                     'Department: ' + allocation_df['department_id'].astype(str) + '<br>' +
                     'Topic: ' + allocation_df['assigned_topic'].astype(str) + '<br>' +
                     'Coach: ' + allocation_df['assigned_coach'].astype(str) +
                     '<extra></extra>',
        name='Students'
    ))
    
    # Auto-detect data ranges if not provided
    if x_range is None:
        x_min = allocation_df['preference_rank'].min()
        x_max = allocation_df['preference_rank'].max()
        x_range = [x_min, x_max]
    
    if y_range is None:
        y_min = allocation_df['effective_cost'].min()
        y_max = allocation_df['effective_cost'].max()
        y_range = [y_min, y_max]
    
    fig.update_layout(
        title='Student Satisfaction: Cost vs Preference Rank',
        xaxis_title='Preference Rank (Lower = Better)',
        yaxis_title='Effective Cost (Lower = Better)',
        height=600,
        hovermode='closest'
    )
    
    # Set axis ranges
    fig.update_xaxes(range=x_range)
    fig.update_yaxes(range=y_range)
    
    # Add diagonal reference line
    fig.add_shape(
        type="line",
        x0=0, y0=0, x1=1000, y1=1000,
        line=dict(color="gray", width=1, dash="dash"),
        opacity=0.5
    )
    
    return fig


def create_coach_specialization_heatmap(allocation_df, repo):
    """
    Create Heatmap showing Coach × Topic specialization
    """
    # Create pivot table
    coach_topic_data = allocation_df.groupby(['assigned_coach', 'assigned_topic']).size().unstack(fill_value=0)
    
    # Sort both axes alphabetically
    coach_topic_data = coach_topic_data.sort_index()  # Sort coaches alphabetically
    coach_topic_data = coach_topic_data.sort_index(axis=1)  # Sort topics alphabetically
    
    fig = go.Figure(data=go.Heatmap(
        z=coach_topic_data.values,
        x=coach_topic_data.columns,
        y=coach_topic_data.index,
        colorscale='YlOrRd',
        colorbar=dict(title='Students'),
        hovertemplate='Coach: %{y}<br>Topic: %{x}<br>Count: %{z}<extra></extra>'
    ))
    
    fig.update_layout(
        title='Coach Specialization Heatmap (Coach × Topic)',
        xaxis_title='Topic',
        yaxis_title='Coach',
        height=600,
        xaxis_tickangle=-45
    )
    
    return fig


def create_department_diversity_analysis(allocation_df):
    """
    Create Department Diversity Analysis using Shannon Entropy
    """
    dept_diversity = []
    
    for dept_id in allocation_df['department_id'].unique():
        dept_data = allocation_df[allocation_df['department_id'] == dept_id]
        topic_dist = dept_data['assigned_topic'].value_counts() / len(dept_data)
        shannon_entropy = entropy(topic_dist)
        
        dept_diversity.append({
            'Department': dept_id,
            'Entropy': shannon_entropy,
            'Students': len(dept_data),
            'Topics': len(topic_dist)
        })
    
    df_div = pd.DataFrame(dept_diversity).sort_values('Department', ascending=True)  # Sort alphabetically
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=df_div['Department'],
        y=df_div['Entropy'],
        name='Diversity (Entropy)',
        marker_color='rgba(52, 152, 219, 0.7)',
        text=df_div['Topics'],
        texttemplate='%{text} topics',
        textposition='outside'
    ))
    
    fig.update_layout(
        title='Department Diversity Analysis (Shannon Entropy)',
        xaxis_title='Department',
        yaxis_title='Entropy (Higher = More Diverse)',
        height=500,
        hovermode='closest'
    )
    
    return fig


if __name__ == "__main__":
    print("Really Advanced Charts Module Loaded")
    print("Available visualizations:")
    print("  - create_preference_funnel()")
    print("  - create_cost_violin_plot()")
    print("  - create_topic_demand_vs_capacity()")
    print("  - create_cost_breakdown_pie()")
    print("  - create_fairness_radar()")
    print("  - create_student_satisfaction_scatter()")
    print("  - create_coach_specialization_heatmap()")
    print("  - create_department_diversity_analysis()")

