import json
import os
import pandas as pd
import numpy as np
import json

all_detailed_files = [f for f in os.listdir('../data/results/winoqueer/') if f.startswith('detailed')]

categories = ['sexual', 'gender']
modes = ['lora', 'pretrained', 'soft-prompt']

result = {}
for mode in modes:
        result[mode] = []
        for file in all_detailed_files:
                if mode in file:
                        df = pd.read_csv(f'../data/results/winoqueer/{file}')
                        print(file)
                        logmax_mean_bias = df['sent_more_score'].mean()
                        print(df['sent_more_score'].describe())
                        logmax_mean_cis = df['sent_less_score'].mean()
                        print(df['sent_less_score'].describe())
                        print()
                        result[mode].append({'logmax_mean_bias': logmax_mean_bias, 'logmax_mean_cis': logmax_mean_cis})
        overall_mode_score = np.mean([x['logmax_mean_bias'] for x in result[mode]])
        overall_mode_cis = np.mean([x['logmax_mean_cis'] for x in result[mode]])
        result[mode].append({'overall_mean_score': overall_mode_score, 'overall_mean_cis': overall_mode_cis})
print(json.dumps(result, indent=4))


