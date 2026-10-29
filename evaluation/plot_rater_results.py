import pandas as pd
import os
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib import colors as col


EXPERIMENT_NAME = os.environ["EXPERIMENT_NAME"]
RUN_NAME = os.environ["RUN_NAME"]
RESULT_FOLDER = os.environ["RESULT_FOLDER"]
out_path = Path(RESULT_FOLDER, EXPERIMENT_NAME)


for scores_name in ["Scores_retro", "Scores_pro_BH", "Scores_pro_FB"]:
    # === Load Excel ===
    df = pd.read_excel(Path(RESULT_FOLDER, EXPERIMENT_NAME, f"Scores_v2.xlsx"), scores_name, index_col=0, header=[0, 1, 2])  
    # This reads a multi-level column structure: Rater → Category → Method

    # === Reshape to long form ===
    df_long = (
        df.melt(ignore_index=False, value_name='Score')
        .reset_index()
        .rename(columns={'index': 'Case'})
    )
    # Split the tuple column names (from melt) into separate columns
    df_long = df_long.rename(columns={
        'variable_0': 'Rater',
        'variable_1': 'Category',
        'variable_2': 'Method'
    })

    # === Define plotting parameters ===
    categories = ['ImageQuality', 'Artifacts', 'LesionConspicuity']
    category_labels = ['Image quality', 'Artifacts', 'Lesion conspicuity']
    if scores_name == "Scores_pro_BH" or scores_name == "Scores_pro_FB":
        methods = ['B', 'C' ]
        method_labels = ["MRI-bridge\n", "Clinical standard\n"]
        raters = ['Rater1', 'Rater2']
        rater_offset = 0.05 
    if scores_name == "Scores_retro":
        methods = ['A', 'B', 'C', 'D']
        method_labels = ["XD-GRASP\n(3 min)", "Soft-gating\n(20 s)", "DiscNet\n(20 s)",  "MRI-bridge\n(20 s)"]
        raters = ['Rater1', 'Rater2', 'Rater3']
        rater_offset = 0.1
    markers = ['o', 'o', 'o', 'D', '^', 'v']  # enough for several raters

    alpha_value = 1
    # === Boxplot width and offset parameters ===
    box_width = 0.05
    rater_colors = ['#1B264F', '#7789BB', '#233F80']

    # === Plot for each category ===
    for cat_i, category in enumerate(categories):
        plt.figure(figsize=(6, 5))
        for r_idx, rater in enumerate(raters):
            np.random.seed(23)
            data_sub = df_long[(df_long['Rater'] == rater) & (df_long['Category'] == category)]

            for m_idx, method in enumerate(methods):
                scores = data_sub[data_sub['Method'] == method]['Score'].dropna()

                scores[scores==0] = np.nan

                # x position with slight offset per rater
                if r_idx == 0:
                    x_pos = np.ones(len(scores)) * m_idx*0.5 - rater_offset - len(scores)/2*0.01 + 0.005 + np.repeat([range(len(scores))],  1, axis=0).flatten()*0.01 
                    box_center = m_idx*0.5 - rater_offset
                elif (len(raters) == 2 and r_idx == 1) | (len(raters) == 3 and r_idx == 2):
                        x_pos = np.ones(len(scores)) * m_idx*0.5 - len(scores)/2*0.01 + 0.005 + np.repeat([range(len(scores))], 1, axis=0).flatten()*0.01 + rater_offset
                        box_center = m_idx*0.5 + rater_offset
                else:
                    x_pos = np.ones(len(scores)) * m_idx*0.5 - len(scores)/2*0.01 + 0.005 + np.repeat([range(len(scores))], 1, axis=0).flatten()*0.01
                    box_center = m_idx*0.5
                # scatter points
                plt.scatter(x_pos, scores, 
                            alpha=alpha_value, marker=markers[r_idx % len(markers)], s=40, color=rater_colors[r_idx],
                            label=rater if m_idx == 0 else None, zorder=5)

                # boxplot for this rater/method
                scores_clean = scores.dropna().values
                if len(scores_clean) > 0:
                    bp = plt.boxplot([scores_clean], positions=[box_center], widths=box_width,
                                    patch_artist=True, manage_ticks=False,
                                    showfliers=False, zorder=3)
                    # Style the boxplot
                    for patch in bp['boxes']:
                        rgba = col.to_rgba(rater_colors[r_idx], alpha=0.3)
                        darkgray_rgba = col.to_rgba("darkgrey", alpha=1)
                        patch.set_facecolor((rgba))
                        patch.set_edgecolor((darkgray_rgba))
                        patch.set_linewidth(1.5)
                    for whisker in bp['whiskers']:
                        whisker.set_color((darkgray_rgba))
                        whisker.set_alpha(1)
                        whisker.set_linewidth(1.5)
                    for cap in bp['caps']:
                        cap.set_color((darkgray_rgba))
                        cap.set_alpha(1)
                        cap.set_linewidth(1.5)
                    for median in bp['medians']:
                        median.set_color('#D75F26')
                        median.set_linewidth(2)
                        median.set_zorder(5)
        if scores_name == 'Scores_retro':
            plt.xticks([0, 0.5, 1.0, 1.5], method_labels, fontsize=10, fontweight='bold')
            plt.xlim(-0.25, 1.75)
        else:
            plt.xticks([0, 0.5], method_labels, fontsize=10, fontweight='bold')
            plt.xlim(-0.25, 0.75)
        plt.yticks([1,2,3,4,5])
        plt.ylim(0.5, 5.5)

        plt.title(category_labels[cat_i], fontsize=15, fontweight='bold')
        plt.ylabel("Score", fontweight='bold', fontsize=15)
        plt.grid(True, alpha=0.3)

        if category == 'LesionConspicuity' and len(raters) == 2:
            custom_legend = [
                Line2D([0], [0], marker='o', color=rater_colors[0], label='Rater 1', alpha=alpha_value, markersize=10, linestyle='None',),
                Line2D([0], [0], marker='o', color=rater_colors[1], label='Rater 2', alpha=alpha_value, markersize=10, linestyle='None',),
            ]

            plt.legend(handles=custom_legend, fontsize=15, frameon=False, loc='best')
        elif category == 'LesionConspicuity' and len(raters) == 3:
            custom_legend = [
                Line2D([0], [0], marker='o', color=rater_colors[0], label='Rater 1', alpha=alpha_value, markersize=8, linestyle='None',),
                Line2D([0], [0], marker='o', color=rater_colors[1], label='Rater 2', alpha=alpha_value, markersize=8, linestyle='None',),
                Line2D([0], [0], marker='o', color=rater_colors[2], label='Rater 3', alpha=alpha_value, markersize=8, linestyle='None',),
            ]

            plt.legend(handles=custom_legend, fontsize=15, frameon=False, loc='best')



        plt.tight_layout()
        plt.savefig(f"{out_path}/{RUN_NAME}_{category.replace(' ', '_')}_{scores_name}.png", dpi=300, transparent=True)
        plt.close()

    print("✅ Plots saved for all categories!")
