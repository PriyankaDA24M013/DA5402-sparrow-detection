"""Prepare train/test data from raw inputs and optionally save EDA plots."""

import logging
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.model_selection import StratifiedGroupKFold


# Config logging
logging.basicConfig(level=logging.INFO, filename='pipe.log',
                    format="%(asctime)s -- [%(levelname)s]: %(message)s")

# Seed for reproducibility
SEED = 0

def draw_and_save_seaborn_plot(seaborn_plot_func=sns.lineplot, *, data=None, x=None, y=None,
                               figsize=(14, 6), x_label="", y_label="", title="",
                               xaxis_grid=False, yaxis_grid=False, x_ticks=None, y_ticks=None,
                               save_file_path=None, **kwargs):
    """Draw a plot using a given seaborn plot function (except 'seaborn.distplot')
    and save it if save_file_path is specified.
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Special handling for histplot to avoid 'mode.use_inf_as_null' issue
    if seaborn_plot_func == sns.histplot:
        # Direct plotting onto the axis without using seaborn's histplot
        if isinstance(data, pd.Series):
            data_values = data.values
        else:
            data_values = data
        ax.hist(data_values, **kwargs)
        ax.set(xlabel=x_label, ylabel=y_label, title=title)
    else:
        # For other plot types, use seaborn as normal
        ax = seaborn_plot_func(data=data, x=x, y=y, ax=ax, **kwargs)
        ax.set(xlabel=x_label, ylabel=y_label, title=title)
    
    if x_ticks:
        ax.set_xticks(x_ticks)
    if y_ticks:
        ax.set_yticks(y_ticks)
    ax.xaxis.grid(xaxis_grid)
    ax.yaxis.grid(yaxis_grid)
    ax.set_axisbelow(True)

    if save_file_path:
        save_file_path = Path(save_file_path)
        save_file_path.parent.mkdir(exist_ok=True, parents=True)
        fig.savefig(save_file_path)
        plt.close()

    return fig

def stratified_group_train_test_split(data, stratification_basis, groups, random_state=0):
    """Split data in a stratified way into training and test sets,
    taking into account groups, and return the corresponding indices.
    """
    split = StratifiedGroupKFold(n_splits=2, shuffle=True, random_state=random_state)
    train_ids, test_ids = next(split.split(X=data, y=stratification_basis, groups=groups))
    return train_ids, test_ids

def expand_img_df_with_avg_box_sizes(info_df, bbox_df, selected_images):
    """Add avg bbox width/height to selected images from bounding_boxes.csv."""
    avg_df = (bbox_df[bbox_df['image_name'].isin(selected_images)]
              .groupby('image_name')[['bbox_width', 'bbox_height']].mean()
              .reset_index()
              .rename(columns={
                  'image_name': 'Name',
                  'bbox_width': 'avg_bbox_width',
                  'bbox_height': 'avg_bbox_height'
              }))

    expanded_df = (info_df[info_df['Name'].isin(selected_images)]
                   .merge(avg_df, on='Name', how='left'))

    return expanded_df


def prepare_data(project_root: Path, save_eda_plots=False):
    """Load, split, and save train/test data; optionally generate EDA plots."""
    raw_data_dir = project_root / 'data' / 'raw'
    processed_data_dir = project_root / 'data' / 'processed'
    processed_data_dir.mkdir(parents=True, exist_ok=True)

    # Load raw data
    img_info_path = raw_data_dir / 'image_info.csv'
    bboxes_path = raw_data_dir / 'bboxes' / 'bounding_boxes.csv'

    img_info_df = pd.read_csv(img_info_path)
    bbox_df = pd.read_csv(bboxes_path)
    
    # Check column names to debug
    logging.info(f"Image info columns: {img_info_df.columns.tolist()}")

    # Train/test split
    train_ids, test_ids = stratified_group_train_test_split(
        data=img_info_df['Name'],
        stratification_basis=img_info_df['Number_HSparrows'],
        groups=img_info_df['Author'],
        random_state=SEED
    )

    if len(train_ids) < len(test_ids):
        train_ids, test_ids = test_ids, train_ids

    # Save CSVs
    split_info = [('train', train_ids), ('test', test_ids)]
    dfs = {}

    for split_name, ids in split_info:
        selected_names = img_info_df['Name'].iloc[ids]
        split_df = expand_img_df_with_avg_box_sizes(img_info_df, bbox_df, selected_names)
        output_csv = processed_data_dir / f"{split_name}.csv"
        split_df.to_csv(output_csv, index=False)
        dfs[split_name] = split_df
        logging.info(f"{split_name.capitalize()} CSV saved at: {output_csv}")

    # Optional EDA
    if save_eda_plots:
        plot_dir = project_root / 'outputs' / 'plots' / 'eda'
        plot_dir.mkdir(parents=True, exist_ok=True)

        train_df = dfs['train']
        eda_plots = []

        dist_plot_cfg = {
            'Number_HSparrows': {
                'x_label': "Number of house sparrows",
                'y_label': "Number of images",
                'title': f"House Sparrow Distribution (Train, {train_df.shape[0]} samples)"
            },
            'Author': {
                'x_label': "Number of images",
                'y_label': "Number of authors",
                'title': "Train Images per Author"
            }
        }

        for col, cfg in dist_plot_cfg.items():
            data = train_df[col]
            if col == 'Author':
                data = data.value_counts()
                x_ticks = range(1, len(data), 2)
            else:
                x_ticks = range(1, data.max() + 1)

            plot_path = plot_dir / f"train_{col.lower()}_distribution.jpg"
            plot = draw_and_save_seaborn_plot(
                sns.histplot,
                data=data,
                save_file_path=plot_path,
                edgecolor='#1B6FA6',
                yaxis_grid=True,
                linewidth=2,
                bins=int(data.max() - data.min()) if col != 'Author' else None,
                x_ticks=list(x_ticks),
                **cfg
            )
            eda_plots.append(plot)
            logging.info(f"{cfg['title']} saved at: {plot_path}")

        # Determine correct column names for width and height 
        # Check if expected column names exist, otherwise try alternatives
        width_col = 'image_width' if 'image_width' in train_df.columns else 'Width'
        height_col = 'image_height' if 'image_height' in train_df.columns else 'Height'
        
        logging.info(f"Using width column: {width_col}, height column: {height_col}")

        # Size plots with corrected column names
        for x, y, title, fname in [(width_col, height_col, "Train Image Sizes", 'img'),
                                  ('avg_bbox_width', 'avg_bbox_height', "Avg BBox Sizes", 'bbox')]:
            
            # Skip if columns don't exist
            if x not in train_df.columns or y not in train_df.columns:
                logging.warning(f"Skipping plot: columns {x} or {y} not found in DataFrame")
                continue
                
            plot_path = plot_dir / f"train_{fname}_sizes.jpg"
            plot = draw_and_save_seaborn_plot(
                sns.scatterplot,
                x=x, y=y,
                data=train_df,
                figsize=(6, 6),
                xaxis_grid=True,
                yaxis_grid=True,
                save_file_path=plot_path,
                x_label=x,
                y_label=y,
                title=title
            )
            eda_plots.append(plot)
            logging.info(f"{title} saved at: {plot_path}")

        return dfs, eda_plots

    return dfs

if __name__ == "__main__":
    prepare_data(Path.cwd(), save_eda_plots=True)