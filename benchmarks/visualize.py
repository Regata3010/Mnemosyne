"""
Visual Benchmark Dashboard for Mnemosyne.

Creates publication-quality charts comparing memory systems.
Exports to PNG for README and portfolio.
"""

import json
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px


@dataclass
class BenchmarkData:
    """Parsed benchmark results."""
    systems: List[str]
    recall_precision: List[float]
    storage_efficiency: List[float]
    memories_stored: List[int]
    memories_filtered: List[int]
    write_conflicts: List[int]
    task_success_rate: List[float]


def load_benchmark_results(filepath: str = None) -> BenchmarkData:
    """Load benchmark results from JSON."""
    if filepath is None:
        filepath = Path(__file__).parent / "results" / "benchmark_results.json"
    
    with open(filepath) as f:
        data = json.load(f)
    
    results = data["results"]
    
    return BenchmarkData(
        systems=[r["system"] for r in results],
        recall_precision=[r["metrics"]["recall_precision"] for r in results],
        storage_efficiency=[r["metrics"]["storage_efficiency"] for r in results],
        memories_stored=[r["metrics"]["memories_stored"] for r in results],
        memories_filtered=[r["metrics"]["memories_filtered"] for r in results],
        write_conflicts=[r["metrics"]["write_conflicts"] for r in results],
        task_success_rate=[r["metrics"]["task_success_rate"] for r in results],
    )


def create_storage_comparison_chart(data: BenchmarkData) -> go.Figure:
    """Create stacked bar chart showing storage efficiency."""
    
    fig = go.Figure()
    
    # Colors
    colors = {
        "stored": "#3498db",   # Blue
        "filtered": "#2ecc71", # Green
    }
    
    # Stored memories (bottom)
    fig.add_trace(go.Bar(
        name="Stored",
        x=data.systems,
        y=data.memories_stored,
        marker_color=colors["stored"],
        text=data.memories_stored,
        textposition="inside",
        textfont=dict(color="white", size=14, family="Arial Black"),
    ))
    
    # Filtered memories (top)
    fig.add_trace(go.Bar(
        name="Filtered",
        x=data.systems,
        y=data.memories_filtered,
        marker_color=colors["filtered"],
        text=data.memories_filtered,
        textposition="inside",
        textfont=dict(color="white", size=14, family="Arial Black"),
    ))
    
    fig.update_layout(
        title=dict(
            text="<b>Storage Efficiency Comparison</b><br><sup>Lower stored count = more efficient filtering</sup>",
            font=dict(size=20),
            x=0.5,
        ),
        barmode="stack",
        xaxis_title="Memory System",
        yaxis_title="Number of Memories",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        template="plotly_white",
        width=800,
        height=500,
        font=dict(family="Arial", size=14),
    )
    
    return fig


def create_metrics_radar_chart(data: BenchmarkData) -> go.Figure:
    """Create radar chart comparing all metrics."""
    
    categories = [
        "Recall Precision",
        "Storage Efficiency", 
        "Write Safety",
        "Task Performance",
    ]
    
    fig = go.Figure()
    
    colors = ["#e74c3c", "#3498db", "#2ecc71"]  # Red, Blue, Green
    
    for i, system in enumerate(data.systems):
        # Normalize metrics to 0-1 scale
        values = [
            data.recall_precision[i],
            data.storage_efficiency[i],
            1.0 if data.write_conflicts[i] == 0 else 0.0,  # Binary: safe or not
            data.task_success_rate[i],
        ]
        
        fig.add_trace(go.Scatterpolar(
            r=values + [values[0]],  # Close the polygon
            theta=categories + [categories[0]],
            fill="toself",
            name=system,
            line=dict(color=colors[i], width=2),
            fillcolor=f"rgba{tuple(list(px.colors.hex_to_rgb(colors[i])) + [0.2])}",
        ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1],
                tickformat=".0%",
            )
        ),
        title=dict(
            text="<b>Memory System Comparison</b><br><sup>Multi-dimensional performance analysis</sup>",
            font=dict(size=20),
            x=0.5,
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.2,
            xanchor="center",
            x=0.5
        ),
        template="plotly_white",
        width=700,
        height=600,
        font=dict(family="Arial", size=14),
    )
    
    return fig


def create_efficiency_bar_chart(data: BenchmarkData) -> go.Figure:
    """Create horizontal bar chart of storage efficiency."""
    
    # Sort by efficiency
    sorted_indices = sorted(range(len(data.systems)), 
                           key=lambda i: data.storage_efficiency[i], 
                           reverse=True)
    
    systems = [data.systems[i] for i in sorted_indices]
    efficiency = [data.storage_efficiency[i] * 100 for i in sorted_indices]
    
    # Color gradient based on efficiency
    colors = ["#2ecc71" if e > 50 else "#3498db" if e > 0 else "#95a5a6" 
              for e in efficiency]
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=efficiency,
        y=systems,
        orientation="h",
        marker_color=colors,
        text=[f"{e:.0f}%" for e in efficiency],
        textposition="outside",
        textfont=dict(size=16, family="Arial Black"),
    ))
    
    fig.update_layout(
        title=dict(
            text="<b>Memory Filtering Rate</b><br><sup>% of low-value memories filtered at ingestion</sup>",
            font=dict(size=20),
            x=0.5,
        ),
        xaxis_title="Filtering Rate (%)",
        xaxis=dict(range=[0, 105]),
        template="plotly_white",
        width=700,
        height=400,
        font=dict(family="Arial", size=14),
    )
    
    return fig


def create_summary_table(data: BenchmarkData) -> go.Figure:
    """Create a summary table as a figure."""
    
    headers = ["Metric", *data.systems]
    
    cells = [
        ["Memories Stored", "Memories Filtered", "Recall Precision", 
         "Storage Efficiency", "Write Conflicts", "Task Success"],
        *[
            [
                str(data.memories_stored[i]),
                str(data.memories_filtered[i]),
                f"{data.recall_precision[i]:.1%}",
                f"{data.storage_efficiency[i]:.1%}",
                str(data.write_conflicts[i]),
                f"{data.task_success_rate[i]:.1%}",
            ]
            for i in range(len(data.systems))
        ]
    ]
    
    # Highlight best values
    fill_colors = [["white"] * 6 for _ in range(len(data.systems) + 1)]
    
    fig = go.Figure(data=[go.Table(
        header=dict(
            values=[f"<b>{h}</b>" for h in headers],
            fill_color="#2c3e50",
            font=dict(color="white", size=14),
            align="center",
            height=35,
        ),
        cells=dict(
            values=cells,
            fill_color=[
                ["#ecf0f1"] * 6,  # First column (metrics)
                *[["white"] * 6 for _ in range(len(data.systems))]
            ],
            font=dict(size=13),
            align="center",
            height=30,
        ),
    )])
    
    fig.update_layout(
        title=dict(
            text="<b>Benchmark Results Summary</b>",
            font=dict(size=18),
            x=0.5,
        ),
        width=900,
        height=350,
        margin=dict(l=20, r=20, t=60, b=20),
    )
    
    return fig


def create_all_charts(data: BenchmarkData, output_dir: str = None) -> Dict[str, go.Figure]:
    """Create all benchmark charts."""
    
    if output_dir is None:
        output_dir = Path(__file__).parent / "charts"
    else:
        output_dir = Path(output_dir)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    charts = {
        "storage_comparison": create_storage_comparison_chart(data),
        "metrics_radar": create_metrics_radar_chart(data),
        "efficiency_bar": create_efficiency_bar_chart(data),
        "summary_table": create_summary_table(data),
    }
    
    # Save all charts
    for name, fig in charts.items():
        # Save as PNG
        png_path = output_dir / f"{name}.png"
        fig.write_image(str(png_path), scale=2)
        print(f"Saved: {png_path}")
        
        # Save as HTML (interactive)
        html_path = output_dir / f"{name}.html"
        fig.write_html(str(html_path))
        print(f"Saved: {html_path}")
    
    return charts


def create_combined_dashboard(data: BenchmarkData, output_path: str = None) -> go.Figure:
    """Create a combined dashboard with all charts."""
    
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            "<b>Storage Efficiency</b>",
            "<b>Performance Radar</b>",
            "<b>Filtering Rate</b>",
            "<b>Results Summary</b>",
        ),
        specs=[
            [{"type": "bar"}, {"type": "polar"}],
            [{"type": "bar"}, {"type": "table"}],
        ],
        vertical_spacing=0.15,
        horizontal_spacing=0.1,
    )
    
    colors = {"stored": "#3498db", "filtered": "#2ecc71"}
    system_colors = ["#e74c3c", "#3498db", "#2ecc71"]
    
    # Chart 1: Storage comparison
    fig.add_trace(
        go.Bar(name="Stored", x=data.systems, y=data.memories_stored,
               marker_color=colors["stored"], showlegend=True),
        row=1, col=1
    )
    fig.add_trace(
        go.Bar(name="Filtered", x=data.systems, y=data.memories_filtered,
               marker_color=colors["filtered"], showlegend=True),
        row=1, col=1
    )
    
    # Chart 2: Radar
    categories = ["Recall", "Efficiency", "Safety", "Task Perf."]
    for i, system in enumerate(data.systems):
        values = [
            data.recall_precision[i],
            data.storage_efficiency[i],
            1.0 if data.write_conflicts[i] == 0 else 0.0,
            data.task_success_rate[i],
        ]
        fig.add_trace(
            go.Scatterpolar(
                r=values + [values[0]],
                theta=categories + [categories[0]],
                name=system,
                line=dict(color=system_colors[i]),
                fill="toself",
                showlegend=False,
            ),
            row=1, col=2
        )
    
    # Chart 3: Efficiency bars
    fig.add_trace(
        go.Bar(
            x=[e * 100 for e in data.storage_efficiency],
            y=data.systems,
            orientation="h",
            marker_color=[system_colors[i] for i in range(len(data.systems))],
            showlegend=False,
        ),
        row=2, col=1
    )
    
    fig.update_layout(
        title=dict(
            text="<b>Mnemosyne Benchmark Dashboard</b><br><sup>Comparing memory system performance on 500 customer service conversations</sup>",
            font=dict(size=22),
            x=0.5,
        ),
        barmode="stack",
        template="plotly_white",
        width=1400,
        height=900,
        font=dict(family="Arial", size=12),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
    )
    
    # Save if output path provided
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.write_image(str(output_path), scale=2)
        print(f"Saved dashboard: {output_path}")
        
        html_path = output_path.with_suffix(".html")
        fig.write_html(str(html_path))
        print(f"Saved interactive: {html_path}")
    
    return fig


def main():
    """Generate all benchmark visualizations."""
    print("=" * 60)
    print("MNEMOSYNE BENCHMARK VISUALIZATION")
    print("=" * 60)
    
    # Load results
    print("\nLoading benchmark results...")
    try:
        data = load_benchmark_results()
        print(f"  Loaded results for {len(data.systems)} systems")
    except FileNotFoundError:
        print("  No benchmark results found. Run benchmarks first:")
        print("  python -m benchmarks.run_benchmark")
        return
    
    # Create individual charts
    print("\nGenerating charts...")
    charts = create_all_charts(data)
    print(f"  Created {len(charts)} charts")
    
    # Create combined dashboard
    print("\nGenerating combined dashboard...")
    dashboard_path = Path(__file__).parent / "charts" / "dashboard.png"
    create_combined_dashboard(data, str(dashboard_path))
    
    print("\n" + "=" * 60)
    print("Visualization complete!")
    print(f"Charts saved to: {Path(__file__).parent / 'charts'}")
    print("=" * 60)


if __name__ == "__main__":
    main()
