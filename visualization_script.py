#!/usr/bin/env python3
"""
NER Model Evaluation Results Visualization Script

This script creates comprehensive visualizations for NER model evaluation results,
comparing TITLE and NO-TITLE models across different datasets.

Usage:
    python visualization_script.py

Requirements:
    pip install pandas numpy matplotlib seaborn plotly scipy
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.figure_factory as ff
from pathlib import Path
import warnings
from datetime import datetime
import os

# Configure plotting
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")
warnings.filterwarnings('ignore')

# Set up paths
RESULTS_PATH = Path('/home/daniel-dorigo/Desktop/NER-proper-names/evaluation_results')
CHARTS_PATH = Path('/home/daniel-dorigo/Desktop/NER-proper-names/charts')
CHARTS_PATH.mkdir(exist_ok=True)

def load_latest_results():
    """Load the most recent evaluation results."""
    csv_files = list(RESULTS_PATH.glob("detailed_evaluation_results_*.csv"))
    
    if not csv_files:
        raise FileNotFoundError("No evaluation results found!")
    
    # Get the most recent file
    latest_file = max(csv_files, key=lambda x: x.stat().st_mtime)
    print(f" Loading results from: {latest_file.name}")
    
    df = pd.read_csv(latest_file)
    print(f" Loaded {len(df)} evaluation records")
    print(f" Models: {df['model_type'].unique()}")
    print(f" Datasets: {df['dataset'].unique()}")
    
    return df, latest_file.stem

def calculate_aggregated_stats(df):
    """Calculate aggregated statistics by model type."""
    agg_stats = []
    
    for model_type in df['model_type'].unique():
        model_data = df[df['model_type'] == model_type]
        
        stats = {
            'model_type': model_type,
            'num_datasets': len(model_data),
            'total_samples': model_data['num_samples'].sum(),
            'person_f1_mean': model_data['person_f1'].mean(),
            'person_f1_std': model_data['person_f1'].std(),
            'person_f1_min': model_data['person_f1'].min(),
            'person_f1_max': model_data['person_f1'].max(),
            'person_precision_mean': model_data['person_precision'].mean(),
            'person_precision_std': model_data['person_precision'].std(),
            'person_recall_mean': model_data['person_recall'].mean(),
            'person_recall_std': model_data['person_recall'].std(),
            'token_accuracy_mean': model_data['token_accuracy'].mean(),
            'token_accuracy_std': model_data['token_accuracy'].std(),
            'avg_inference_time_per_sample': model_data['inference_time_per_sample'].mean() * 1000,  # Convert to ms
            'model_size_mb': model_data['model_size_mb'].iloc[0]
        }
        agg_stats.append(stats)
    
    return pd.DataFrame(agg_stats)

def create_performance_comparison(df_results, results_timestamp):
    """Create comprehensive performance comparison charts."""
    
    # Create subplots
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('F1 Score by Dataset', 'Precision by Dataset', 
                       'Recall by Dataset', 'Token Accuracy by Dataset'),
        specs=[[{"secondary_y": False}, {"secondary_y": False}],
               [{"secondary_y": False}, {"secondary_y": False}]]
    )
    
    metrics = ['person_f1', 'person_precision', 'person_recall', 'token_accuracy']
    titles = ['F1 Score', 'Precision', 'Recall', 'Token Accuracy']
    positions = [(1, 1), (1, 2), (2, 1), (2, 2)]
    
    colors = {'TITLE': '#1f77b4', 'NO-TITLE': '#ff7f0e'}
    
    for metric, title, (row, col) in zip(metrics, titles, positions):
        for model_type in df_results['model_type'].unique():
            model_data = df_results[df_results['model_type'] == model_type]
            
            fig.add_trace(
                go.Bar(
                    x=model_data['dataset'],
                    y=model_data[metric],
                    name=f'{model_type} Model',
                    marker_color=colors[model_type],
                    showlegend=(row == 1 and col == 1)  # Only show legend once
                ),
                row=row, col=col
            )
    
    fig.update_layout(
        height=800,
        title_text="NER Model Performance Comparison Across Datasets",
        title_x=0.5,
        showlegend=True,
        template="plotly_white"
    )
    
    # Update y-axes
    for row in [1, 2]:
        for col in [1, 2]:
            fig.update_yaxes(range=[0, 1], row=row, col=col)
    
    # Save the chart
    fig.write_html(CHARTS_PATH / f"performance_comparison_{results_timestamp}.html")
    fig.write_image(CHARTS_PATH / f"performance_comparison_{results_timestamp}.png", width=1200, height=800)
    print(" Performance comparison chart saved!")
    
    return fig

def create_summary_dashboard(df_results, df_aggregated, results_timestamp):
    """Create a comprehensive summary dashboard."""
    
    # Create a 3x2 subplot layout
    fig = make_subplots(
        rows=3, cols=2,
        subplot_titles=('Model Performance Overview', 'Inference Speed Comparison',
                       'Dataset Performance Range', 'Model Size Comparison',
                       'Precision vs Recall', 'Performance Consistency'),
        specs=[[{"type": "bar"}, {"type": "bar"}],
               [{"type": "box"}, {"type": "bar"}],
               [{"type": "scatter"}, {"type": "bar"}]]
    )
    
    colors = {'TITLE': '#1f77b4', 'NO-TITLE': '#ff7f0e'}
    
    # 1. Model Performance Overview (F1 Score)
    for model_type in df_aggregated['model_type']:
        model_data = df_aggregated[df_aggregated['model_type'] == model_type].iloc[0]
        fig.add_trace(
            go.Bar(
                x=[model_type],
                y=[model_data['person_f1_mean']],
                name=f'{model_type} F1',
                marker_color=colors[model_type],
                showlegend=False,
                text=f"{model_data['person_f1_mean']:.3f}",
                textposition='outside'
            ),
            row=1, col=1
        )
    
    # 2. Inference Speed Comparison
    for model_type in df_aggregated['model_type']:
        model_data = df_aggregated[df_aggregated['model_type'] == model_type].iloc[0]
        fig.add_trace(
            go.Bar(
                x=[model_type],
                y=[model_data['avg_inference_time_per_sample']],
                name=f'{model_type} Speed',
                marker_color=colors[model_type],
                showlegend=False,
                text=f"{model_data['avg_inference_time_per_sample']:.1f}ms",
                textposition='outside'
            ),
            row=1, col=2
        )
    
    # 3. Dataset Performance Range (Box plot)
    for model_type in df_results['model_type'].unique():
        model_data = df_results[df_results['model_type'] == model_type]
        fig.add_trace(
            go.Box(
                y=model_data['person_f1'],
                name=model_type,
                marker_color=colors[model_type],
                showlegend=False
            ),
            row=2, col=1
        )
    
    # 4. Model Size Comparison
    for model_type in df_aggregated['model_type']:
        model_data = df_aggregated[df_aggregated['model_type'] == model_type].iloc[0]
        fig.add_trace(
            go.Bar(
                x=[model_type],
                y=[model_data['model_size_mb']],
                name=f'{model_type} Size',
                marker_color=colors[model_type],
                showlegend=False,
                text=f"{model_data['model_size_mb']:.1f}MB",
                textposition='outside'
            ),
            row=2, col=2
        )
    
    # 5. Precision vs Recall
    for model_type in df_results['model_type'].unique():
        model_data = df_results[df_results['model_type'] == model_type]
        fig.add_trace(
            go.Scatter(
                x=model_data['person_precision'],
                y=model_data['person_recall'],
                mode='markers',
                name=model_type,
                marker=dict(size=10, color=colors[model_type]),
                showlegend=False,
                text=model_data['dataset'],
                hovertemplate='%{text}<br>Precision: %{x:.3f}<br>Recall: %{y:.3f}<extra></extra>'
            ),
            row=3, col=1
        )
    
    # 6. Performance Consistency (Standard Deviation)
    for model_type in df_aggregated['model_type']:
        model_data = df_aggregated[df_aggregated['model_type'] == model_type].iloc[0]
        fig.add_trace(
            go.Bar(
                x=[model_type],
                y=[model_data['person_f1_std']],
                name=f'{model_type} Consistency',
                marker_color=colors[model_type],
                showlegend=False,
                text=f"{model_data['person_f1_std']:.3f}",
                textposition='outside'
            ),
            row=3, col=2
        )
    
    # Update layout
    fig.update_layout(
        height=1200,
        title_text="NER Model Evaluation Dashboard",
        title_x=0.5,
        template="plotly_white"
    )
    
    # Update specific axes
    fig.update_yaxes(title_text="F1 Score", row=1, col=1)
    fig.update_yaxes(title_text="Time (ms)", row=1, col=2)
    fig.update_yaxes(title_text="F1 Score", row=2, col=1)
    fig.update_yaxes(title_text="Size (MB)", row=2, col=2)
    fig.update_yaxes(title_text="Recall", row=3, col=1)
    fig.update_xaxes(title_text="Precision", row=3, col=1)
    fig.update_yaxes(title_text="Std Dev", row=3, col=2)
    
    # Save the dashboard
    fig.write_html(CHARTS_PATH / f"summary_dashboard_{results_timestamp}.html")
    fig.write_image(CHARTS_PATH / f"summary_dashboard_{results_timestamp}.png", width=1400, height=1200)
    print(" Summary dashboard saved!")
    
    return fig

def create_radar_chart(df_aggregated, results_timestamp):
    """Create radar chart for multi-dimensional model comparison."""
    
    # Prepare data for radar chart
    categories = ['F1 Score', 'Precision', 'Recall', 'Token Accuracy', 'Speed (1-normalized)', 'Efficiency (1/size)']
    
    fig = go.Figure()
    
    for model_type in df_aggregated['model_type']:
        model_data = df_aggregated[df_aggregated['model_type'] == model_type].iloc[0]
        
        # Normalize speed (lower is better, so invert)
        max_speed = df_aggregated['avg_inference_time_per_sample'].max()
        normalized_speed = 1 - (model_data['avg_inference_time_per_sample'] / max_speed)
        
        # Normalize efficiency (1/size, higher is better)
        max_size = df_aggregated['model_size_mb'].max()
        normalized_efficiency = 1 - (model_data['model_size_mb'] / max_size)
        
        values = [
            model_data['person_f1_mean'],
            model_data['person_precision_mean'],
            model_data['person_recall_mean'],
            model_data['token_accuracy_mean'],
            normalized_speed,
            normalized_efficiency
        ]
        
        fig.add_trace(go.Scatterpolar(
            r=values,
            theta=categories,
            fill='toself',
            name=f'{model_type} Model',
            opacity=0.6
        ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1]
            )
        ),
        title="Multi-dimensional Model Comparison (Radar Chart)",
        template="plotly_white",
        height=600
    )
    
    # Save the chart
    fig.write_html(CHARTS_PATH / f"radar_comparison_{results_timestamp}.html")
    fig.write_image(CHARTS_PATH / f"radar_comparison_{results_timestamp}.png", width=800, height=600)
    print(" Radar chart saved!")
    
    return fig

def generate_summary_report(df_results, df_aggregated, results_timestamp):
    """Generate a comprehensive text summary of the results."""
    
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    report = f"""
# NER Model Evaluation Summary Report
Generated: {timestamp}

## Overall Results

### Dataset Coverage:
- Total evaluations: {len(df_results)}
- Unique datasets: {len(df_results['dataset'].unique())}
- Model types evaluated: {', '.join(df_results['model_type'].unique())}

### Key Findings:

"""
    
    # Add model comparison
    for model_type in df_aggregated['model_type']:
        model_data = df_aggregated[df_aggregated['model_type'] == model_type].iloc[0]
        
        report += f"""
#### {model_type} Model Performance:
- **F1 Score**: {model_data['person_f1_mean']:.3f} ± {model_data['person_f1_std']:.3f} (range: {model_data['person_f1_min']:.3f} - {model_data['person_f1_max']:.3f})
- **Precision**: {model_data['person_precision_mean']:.3f} ± {model_data['person_precision_std']:.3f}
- **Recall**: {model_data['person_recall_mean']:.3f} ± {model_data['person_recall_std']:.3f}
- **Token Accuracy**: {model_data['token_accuracy_mean']:.3f} ± {model_data['token_accuracy_std']:.3f}
- **Model Size**: {model_data['model_size_mb']:.1f} MB
- **Avg Inference Time**: {model_data['avg_inference_time_per_sample']:.1f} ms per sample
- **Datasets Evaluated**: {model_data['num_datasets']}
- **Total Samples**: {model_data['total_samples']:,}
"""
    
    # Add best performing datasets
    report += "\n### Best Performing Datasets by Model:\n"
    
    for model_type in df_results['model_type'].unique():
        model_data = df_results[df_results['model_type'] == model_type]
        best_dataset = model_data.loc[model_data['person_f1'].idxmax()]
        
        report += f"""
**{model_type} Model - Best Dataset: {best_dataset['dataset']}**
- F1 Score: {best_dataset['person_f1']:.3f}
- Precision: {best_dataset['person_precision']:.3f}
- Recall: {best_dataset['person_recall']:.3f}
- Samples: {best_dataset['num_samples']:,}
"""
    
    # Add model comparison
    if len(df_aggregated) == 2:
        title_data = df_aggregated[df_aggregated['model_type'] == 'TITLE'].iloc[0]
        no_title_data = df_aggregated[df_aggregated['model_type'] == 'NO-TITLE'].iloc[0]
        
        f1_diff = title_data['person_f1_mean'] - no_title_data['person_f1_mean']
        speed_diff = title_data['avg_inference_time_per_sample'] - no_title_data['avg_inference_time_per_sample']
        
        better_f1 = "TITLE" if f1_diff > 0 else "NO-TITLE"
        faster = "TITLE" if speed_diff < 0 else "NO-TITLE"
        
        report += f"""
### Model Comparison Summary:
- **Better F1 Performance**: {better_f1} model ({abs(f1_diff):.3f} points difference)
- **Faster Inference**: {faster} model ({abs(speed_diff):.1f} ms difference)
- **Model Sizes**: Nearly identical (~{title_data['model_size_mb']:.1f} MB)

### Recommendations:
- For highest accuracy: Use {better_f1} model
- For fastest inference: Use {faster} model
- Both models show similar efficiency in terms of size/performance ratio
"""
    
    report += f"""
### Generated Visualizations:
The following charts have been saved to the charts directory:
1. Performance Comparison Chart
2. Multi-dimensional Radar Chart
3. Comprehensive Summary Dashboard

All charts are available in both HTML (interactive) and PNG (static) formats.
"""
    
    # Save the report
    report_path = CHARTS_PATH / f"evaluation_report_{results_timestamp}.txt"
    with open(report_path, 'w') as f:
        f.write(report)
    
    print(f" Summary report saved to: {report_path}")
    return report

def main():
    """Main function to run the visualization pipeline."""
    
    print(" Starting NER Model Evaluation Visualization Pipeline...")
    print("=" * 60)
    
    # Load data
    print("\n Loading evaluation results...")
    df_results, results_timestamp = load_latest_results()
    
    # Calculate aggregated statistics
    print("\n Calculating aggregated statistics...")
    df_aggregated = calculate_aggregated_stats(df_results)
    print(" Aggregated statistics calculated")
    
    # Create visualizations
    print("\n🎨 Creating visualizations...")
    
    # 1. Performance comparison
    print("   Creating performance comparison chart...")
    create_performance_comparison(df_results, results_timestamp)
    
    # 2. Radar chart
    print("   Creating radar chart...")
    create_radar_chart(df_aggregated, results_timestamp)
    
    # 3. Summary dashboard
    print("   Creating summary dashboard...")
    create_summary_dashboard(df_results, df_aggregated, results_timestamp)
    
    # 4. Generate report
    print("   Generating summary report...")
    summary_report = generate_summary_report(df_results, df_aggregated, results_timestamp)
    
    # Final summary
    print("\n Visualization Pipeline Complete!")
    print("=" * 50)
    
    print(f"\n Generated files in {CHARTS_PATH}:")
    chart_files = list(CHARTS_PATH.glob(f"*{results_timestamp}*"))
    for file_path in sorted(chart_files):
        file_size = file_path.stat().st_size / 1024  # Size in KB
        print(f"   {file_path.name} ({file_size:.1f} KB)")
    
    print(f"\n Total files generated: {len(chart_files)}")
    print(f" Interactive HTML files: {len([f for f in chart_files if f.suffix == '.html'])}")
    print(f"🖼️  Static PNG images: {len([f for f in chart_files if f.suffix == '.png'])}")
    print(f" Text reports: {len([f for f in chart_files if f.suffix == '.txt'])}")
    
    print(f"\n Next Steps:")
    print(f"  1. Open the HTML files in your browser for interactive exploration")
    print(f"  2. Use PNG files for presentations or reports")
    print(f"  3. Review the text summary report for key insights")
    print(f"  4. Share the summary dashboard for a comprehensive overview")
    
    print(f"\n Quick Access:")
    print(f"  Dashboard: {CHARTS_PATH / f'summary_dashboard_{results_timestamp}.html'}")
    print(f"  Report: {CHARTS_PATH / f'evaluation_report_{results_timestamp}.txt'}")

if __name__ == "__main__":
    main()
