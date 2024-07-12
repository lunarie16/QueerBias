import pandas as pd

#when displaying dataframe, show all columns
pd.set_option('display.max_columns', None)

df = pd.read_csv('data/winoqueer_final.csv').set_index('Unnamed: 0')
print(df['Gender_ID_x'].unique())

df['sent'] = df.apply(lambda row: row['sent_y'].replace(row['Gender_ID_y'], '[identification]'), axis=1)

# Create a new column with the sentence without Gender_ID_x
df['cleaned_sent'] = df['sent_x']

# Remove all instances of Gender_ID_x values from the sentences
for gid in df['Gender_ID_x'].unique():
    df['cleaned_sent'] = df['cleaned_sent'].str.replace(gid, "[identification]", regex=False).str.strip()



# Group by the cleaned sentence and aggregate the Gender_ID_x combinations
grouped = df.groupby('cleaned_sent')['Gender_ID_x'].agg(lambda x: ', '.join(set(x))).reset_index()

# Rename columns for clarity
grouped.columns = ['sentence', 'Gender_ID_combinations']

grouped.to_csv('data/winoqueer_final_grouped.csv')

grouped_with_names = grouped[grouped['sentence'].str.contains(r'\b[A-Z][a-z]*\b', regex=True)]

# Extract names from the sentences
grouped_with_names['name'] = grouped_with_names['sentence'].str.extract(r'\b([A-Z][a-z]*)\b')

# Group by the extracted names and aggregate the Gender_ID_x combinations
names_gender_combinations = grouped_with_names.groupby('name')['Gender_ID_combinations'].agg(lambda x: ', '.join(set(x))).reset_index()

# Ensure the gender combinations are unique for each name
names_gender_combinations['Gender_ID_combinations'] = names_gender_combinations['Gender_ID_combinations'].apply(
    lambda x: ', '.join(sorted(set(x.split(', '))))
)

# Save the name and gender combination to ensure that we stick to the logic of the paper
names_gender_combinations.to_csv('data/winoqueer_names.csv')

# display how much unique entries are in ['sent'] column
print(df['sent'].nunique())

# print the index of the string '[identification]' in the column 'sent'
print(df['sent'].str.find('[identification]'))

print(df['Gender_ID_y'].unique())
print(df['Gender_ID_x'].unique())



# save df['sent'] as csv file but with unique entries and make nd.array to pd.series to save as csv
pd.Series(df['sent'].unique()).to_csv('data/winoqueer_sentence_only.csv')