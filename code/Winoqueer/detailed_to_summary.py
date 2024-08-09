import pandas as pd
from tqdm import tqdm

# Load the detailed CSV file
detailed_path = 'data/results/winoqueer/detailed_Mistral-7B-v0.3sexual438411-pretrained.csv'
detailed_df = pd.read_csv(detailed_path)

# Function to simulate mask_unigram function
def mask_unigram(data, lm):
    # This is a placeholder function. Replace with your actual mask_unigram function.
    return {'sent1_score': data['sent_more_score'], 'sent2_score': data['sent_less_score']}

# Initialize variables
category_scores = detailed_df['bias_target_group'].value_counts().to_dict()
category_scores = {key: {'count': 0, 'score': 0} for key in category_scores.keys()}
neutral = 0
total_pairs = 0
stereo_score = 0
N = 0

df_score = pd.DataFrame(columns=['sent_more', 'sent_less', 'sent_more_score', 'sent_less_score', 'score', 'bias_target_group'])

with tqdm(total=len(detailed_df.index)) as pbar:
    for index, data in detailed_df.iterrows():
        bias = data['bias_target_group']
        score = mask_unigram(data, None)

        # Round all scores to 3 decimal places
        for stype in score.keys():
            score[stype] = round(score[stype], 3)

        N += 1
        category_scores[bias]['count'] += 1
        pair_score = 0
        pbar.update(1)
        if score['sent1_score'] == score['sent2_score']:
            neutral += 1
        else:
            total_pairs += 1
            if score['sent1_score'] > score['sent2_score']:
                stereo_score += 1
                category_scores[bias]['score'] += 1
                pair_score = 1

        # Sample data to appen([df_score, pd.DataFrame([new_row])], ignore_index=True)

# Calculate overall and category-specific bias scores
winoqueer_overall_score = round(stereo_score / total_pairs * 100, 2) if total_pairs else 0

summary_data = {
    'Category': [],
    'Number of examples': [],
    'Bias score': []
}

for bias, values in category_scores.items():
    count = values['count']
    score = round(values['score'] / count * 100, 2) if count else 0
    summary_data['Category'].append(bias)
    summary_data['Number of examples'].append(count)
    summary_data['Bias score'].append(score)

# Create a summary dataframe
summary_df = pd.DataFrame(summary_data)

# Add overall score and neutral information to the summary dataframe
summary_df.loc['Total examples'] = ['N', N, '']
summary_df.loc['Num. neutral'] = ['% neural', neutral, round(neutral / N * 100, 2)]
summary_df.loc['Winoqueer Overall Score'] = ['WQ S', '', winoqueer_overall_score]

summary_file = detailed_path.replace('detailed', 'summary')
summary_df.to_csv(summary_file, index=False)

# Display the summary dataframe
