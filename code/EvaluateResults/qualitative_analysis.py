import json
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
import plotly.figure_factory as ff
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go

# import plotly.io as pio
# pio.renderers.default = 'svg'
#
# # display all columns and complete row of dataframe
# pd.set_option('display.max_columns', None)
# pd.set_option('display.max_rows', None)
#
# # display complete content of row in dataframe
# pd.set_option('display.expand_frame_repr', False)
#
all_detailed_files = [f for f in os.listdir('../../data/results/winoqueer/') if f.startswith('detailed')]

blue = '#004080'
red = '#ea3b06'
yellow = '#ffc900'
yellow = '#ffae00'
# Define colors and hatches
colors = [blue, red, yellow]




def create_overview_files():
    #
    # #
    categories = ['sexual', 'gender']
    modes = ['lora', 'pretrained', 'soft-prompt']

    result = {}

    qualitative_df = pd.DataFrame()
    header = ['orig_idx', 'model', 'mode', 'term_x', 'term_y', 'sentence_x','sentence_y' , 'diff', 'score_bias', 'score_cis']
    all_diff_Df = pd.DataFrame()
    header2 = ['model', 'mode', 'cat', 'diff']
    result = []
    result_all = []

    layout = None

    for mode in modes:
        mode_files = [f for f in all_detailed_files if mode.lower() in f]
        for file in mode_files:
            if mode == 'lora':
                mode = 'LoRA'
            if 'noeval' in file:
                continue

            if 'sexual' in file:
                target = pd.read_csv(f'../../data/datasets/winoqueer_sexual_identity.csv')
                cat = 'sexual'
            else:
                target = pd.read_csv(f'../../data/datasets/winoqueer_gender_identity.csv')
                cat = 'gender'
            model = file.split('_')[1].split(cat)[0]


            df = pd.read_csv(f'../../data/results/winoqueer/{file}')
            df = df.merge(target, on='Unnamed: 0')

            df['diff'] = abs(df['sent_more_score']) - abs(df['sent_less_score'])

            min_diff = df['diff'].min()
            max_diff = df['diff'].max()

            # normalize diff to [-1,1]
            # df['diff'] = ((df['diff'] - min_diff) / (max_diff - min_diff)) * 2 - 1

            df = df.drop(columns=['Unnamed: 0', 'sent_more', 'sent_less', 'score'])
    #         print(file)
            # sort values by diff
            df = df.sort_values(by='diff')
    #
    #         # print 5 smallest and 5 largest diffs
    #


            df_min = df[df['diff'] == df['diff'].min()]
            df_max = df[df['diff'] == df['diff'].max()]
    #
            result.append({'orig_idx': df_min.index, 'model': model, 'mode': mode, 'term_x': df_min['Gender_ID_x'].values[0],
                           'term_y': df_min['Gender_ID_y'].values[0],
                           'sentence_x': df_min['sent_x'].values[0],
                           'sentence_y':df_min['sent_y'].values[0],
                           'diff': round(df_min['diff'].values[0], 4),
                           'score_bias':  round(df_min['sent_more_score'].values[0], 4),
                           'score_cis':  round(df_min['sent_less_score'].values[0], 4)})
            result.append({'orig_idx': df_max.index,'model': model, 'mode': mode, 'term_x': df_max['Gender_ID_x'].values[0],
                            'term_y': df_max['Gender_ID_y'].values[0],
                           'sentence_x': df_max['sent_x'].values[0],
                           'sentence_y':df_max['sent_y'].values[0],
                           'diff': round(df_max['diff'].values[0], 4),
                           'score_bias': round(df_max['sent_more_score'].values[0], 4),
                            'score_cis': round(df_max['sent_less_score'].values[0], 4)})

    #             # print(df['diff'].describe())
            result_all.append({'model': model, 'mode': mode if mode != 'prompt' else 'soft-prompt', 'cat': cat, 'diff': df['diff'].to_list()})


    qualitative_df = pd.DataFrame(result, columns=header)
    qualitative_df.to_csv(f'../../data/results/winoqueer/qualitative_analysis.csv', index=False)
    #
    # sort data frame:
    # fist by diff, then by model, mode by ['pretrained', 'LoRA', 'soft-prompt']
    qualitative_df = qualitative_df.sort_values(by=['diff', 'model', 'mode', 'sentence_x'])

    # ensure mode is sorted like ['pretrained', 'LoRA', 'soft-prompt']
    # qualitative_df = qualitative_df.sort_values(by=['diff', 'model', 'mode'])

    for i, row in qualitative_df.iterrows():
        model = row['model'].replace("Meta", "").replace("-", " ").replace("b", "B").replace("v", "").strip().capitalize()
        mode = row['mode'].replace('q', '')
        if row['diff'] > 0:
            print(f"{model} & {mode} & {row['sentence_y']} & {row['term_x']} \\\\ ")
        else:
            print(f"{model} & {mode} & {row['sentence_x']} & {row['term_y']} \\\\ ")

    #
    all_diff_Df = pd.DataFrame(result_all, columns=header2)
    all_diff_Df.to_csv(f'../../data/results/winoqueer/qualitative_analysis_all_diff_2.csv', index=False)

# create_overview_files()
def parse_cell(x):
    return json.loads(x)

def add_fig_to_layout(df_orig, categories):
    models = sorted(df_orig['model'].unique())
    modes = ['pretrained', 'LoRA', 'soft-prompt']
    titles = [f'{model.replace("Meta-", "").capitalize().replace("-", " ").replace("b", "B")}' for model in models]
    # double each title entry
    titles = [val for val in titles for _ in (0, 1)]
    fig = make_subplots(rows=3, cols=2, subplot_titles=titles, vertical_spacing=0.1)
    df_orig['diff'] = df_orig['diff'].apply(lambda x: parse_cell(x))

    for k, cat in enumerate(categories):
        df_cat = df_orig[df_orig['cat'] == cat]
        for i, model in enumerate(models):
            df = df_cat[df_cat['model'] == model]
            #get min and max values from all diff per model
            min_diff = df[df['model'] == model]['diff'].min()
            max_diff = df[df['model'] == model]['diff'].max()


            for j, mode in enumerate(modes):
                print(model, mode)
                if mode == 'noeval':
                    continue

                df_subset = df[(df['model'] == model) & (df['mode'].replace('q', '') == mode)]
                data_points = df_subset['diff'].to_list()
                if len(data_points) == 1:
                    data_points = data_points[0]
                # plot = go.Histogram(x=data_points, marker_color=colors[j])

                # fig.add_trace(go.Scatter(x=[0, 0], y=[0, y_axis_range],
                #                       mode='lines', line=dict(color='black', width=1), showlegend=False), row=i+1, col=j+1)

                plot = go.Box(y=data_points,  line_color=colors[j], name=mode, showlegend=False,
                              line_width=1, boxpoints=False
                              )

                fig.add_trace(plot, row=i + 1, col=k + 1)
    fig.update_layout(
                       showlegend=False,
                      width=1000,
                      height=900,
                      )
    fig.write_image(f'../../data/results/winoqueer/boxplot_diff_comb.pdf')
    fig.show()


#
df = pd.read_csv(f'../../data/results/winoqueer/qualitative_analysis_all_diff_2.csv')
categories = ['gender', 'sexual']
add_fig_to_layout(df, categories)

# calculate std deviation for diff for each model and mode
models = df['model'].unique()
modes = df['mode'].unique()
# df['diff'] = df['diff'].apply(lambda x: json.loads(x))

for model in models:
    for mode in modes:
        for cat in categories:
            df_subset = df[(df['model'] == model) & (df['mode'] == mode) & (df['cat'] == cat)]
            data_points = df_subset['diff'].to_list()
            if len(data_points) == 1:
                data_points = data_points[0]
            std_dev = np.std(data_points)
            print(f'{model} {mode} {cat} {std_dev}')
