#!/usr/bin/env python3

import argparse
import os
import sys

import matplotlib.pyplot as plt
import pandas as pd

# Add the parent directory to sys.path to import plot_style
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from draw.plot_style import apply_vldb_style


def main():
    parser = argparse.ArgumentParser(description="Generate ambiguity distribution plot")
    parser.add_argument("--pdf", action="store_true", help="Save PDF only")
    args = parser.parse_args()

    base_path = os.path.dirname(__file__)
    data_path = os.path.join(base_path, 'data', 'amb', 'kind_distribution.csv')
    output_dir = os.path.join(base_path, 'output')
    
    os.makedirs(output_dir, exist_ok=True)
    
    df = pd.read_csv(data_path)
    
    apply_vldb_style()
    
    # Group 7 ambiguity kinds into 3 categories
    group_map = {
        'Operation boundary': 'Operational',
        'Operation incomplete': 'Operational',
        'Operation inconsistent': 'Operational',
        'Group-level concept': 'Concept',
        'Row-level concept': 'Concept',
        'Single-table reference': 'Data',
        'Multi-table alignment': 'Data',
    }
    
    df['group'] = df['kind'].map(group_map)
    
    # Order: from 3 o'clock, clockwise Data → Concept → Operational
    group_order = ['Data', 'Concept', 'Operational']
    kind_order_in_group = {
        'Data': ['Single-table reference', 'Multi-table alignment'],
        'Concept': ['Group-level concept', 'Row-level concept'],
        'Operational': ['Operation incomplete', 'Operation inconsistent', 'Operation boundary'],
    }
    
    # Sort by group and kind order
    df['group'] = pd.Categorical(df['group'], categories=group_order, ordered=True)
    
    # Build flattened order list
    kind_order_flat = []
    for g in group_order:
        kind_order_flat.extend(kind_order_in_group[g])
    df['kind'] = pd.Categorical(df['kind'], categories=kind_order_flat, ordered=True)
    df = df.sort_values(['group', 'kind'])
    
    # Aggregate group counts
    group_data = df.groupby('group', observed=True)['count'].sum().reset_index()
    group_data['group'] = pd.Categorical(group_data['group'], categories=group_order, ordered=True)
    group_data = group_data.sort_values('group')

    fig, ax = plt.subplots(figsize=(10, 10))
    
    # Positional colors within each group (clockwise light -> dark)
    group_colors = ['#8B82E8', '#F5C065', '#2E8B7A']
    group_palettes = {
        'Data': ['#D4CFFF', '#AEA8F3'],
        'Concept': ['#FFE4A0', '#F9D79A'],
        'Operational': ['#9AE4D6', '#57C4AD', '#52AC9B'],
    }
    group_offsets = {g: 0 for g in group_order}
    kind_colors = []
    for _, row in df.iterrows():
        group = row['group']
        idx = group_offsets[group]
        palette = group_palettes[group]
        color = palette[min(idx, len(palette) - 1)]
        kind_colors.append(color)
        group_offsets[group] += 1

    radius_inner = 0.6
    width_inner = 0.3
    radius_outer = 0.9
    width_outer = 0.28
    
    ax.pie(
        group_data['count'], 
        radius=radius_inner, 
        colors=group_colors,
        labels=None, 
        wedgeprops=dict(width=width_inner, edgecolor='white', linewidth=3),
        startangle=0,
        counterclock=False
    )
    
    ax.pie(
        df['count'], 
        radius=radius_outer, 
        colors=kind_colors,
        labels=None, 
        wedgeprops=dict(width=width_outer, edgecolor='white', linewidth=3),
        startangle=0,
        counterclock=False
    )

    ax.set_xlim(-1.1, 1.1)
    ax.set_ylim(-1.1, 1.1)
    
    amb_output_dir = os.path.join(output_dir, 'amb')
    os.makedirs(amb_output_dir, exist_ok=True)
    
    output_png = os.path.join(amb_output_dir, 'plot_ambiguity_distribution.png')
    output_pdf = os.path.join(amb_output_dir, 'plot_ambiguity_distribution.pdf')
    
    if args.pdf:
        plt.savefig(output_pdf, bbox_inches='tight')
        print(f"Saved: {output_pdf}")
    else:
        plt.savefig(output_png, dpi=300, bbox_inches='tight')
        print(f"Saved: {output_png}")
    plt.close()


if __name__ == "__main__":
    main()
