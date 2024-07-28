import re
import pandas as pd

#when displaying dataframe, show all columns
pd.set_option('display.max_columns', None)

df = pd.read_csv('./data/winoqueer_sentence_only.csv').set_index('Unnamed: 0')
df_names = pd.read_csv('./data/winoqueer_names.csv').set_index('name').drop(columns='Unnamed: 0')
df_grouped = pd.read_csv('./data/winoqueer_final_grouped.csv').set_index('Unnamed: 0')

options = {'lgbtq': ['queer'],
           'gay': ['gay', 'homo', 'homosexual'],
           'lesbian': ['lesbian', 'homo', 'homosexual'],
           'pansexual': ['pan', 'pansexual'],
           'nb': ['enby', 'non-binary', 'gender-fluid'],
           'transgender': ['trans*', 'transgender', 'inter*', 'intersexual'],
           'asexual': ['asexual', 'ace', 'demisexual', 'demi'],
           'bisexual': ['bi', 'bisexual'],
           'queer': ['queer']
           }



def create_new_dataset(sentences, original_comb, gender_x, gender_y):
    df_content = []
    for i, sent in enumerate(sentences):
        original_combination = original_comb[i].lower().split(', ')
        mapped_options = [options[x] for x in original_combination]
        # flatten list of original_combination
        options_left = set([x for xs in mapped_options for x in xs])
        gender_left = list(options_left.intersection(set(gender_y)))
        for g_x in gender_x:
            for g_y in original_combination:
                # name = re.match(r'\b([A-Z][a-z]*)\b', sent)
                # if name is not None:
                #     name = name.group(0)
                #     if name in df_names.index:
                #         gender_options = df_names.loc[name]['Gender_ID_combinations'].split(', ')
                #         if 'NB' not in gender_options and g_y in ['non-binary', 'enby']:
                #             skipped.add((name, g_y))
                #             continue
                #         if 'Lesbian' not in gender_options and g_y == 'lesbian':
                #             skipped.add((name, g_y))
                #             continue
                #         if 'Gay' not in gender_options and g_y == 'gay':
                #             skipped.add((name, g_y))
                #             continue

                df_content.append({'Gender_ID_y': g_x,
                                        'Gender_ID_x': g_y,
                                        'sent_y': sent.replace('[identification]', g_x),
                                        'sent_x': sent.replace('[identification]', g_y)})
    df = pd.DataFrame(columns=['Gender_ID_x', 'Gender_ID_y', 'sent_x', 'sent_y'], data=df_content)
    return df

gender_x = ['cis', 'cisgender']
gender_y = ['non-binary', 'enby', 'trans*', 'transgender',
            'inter*', 'intersex', 'gender-fluid' ]


skipped = set()
gender_df = create_new_dataset(df_grouped['sentence'],df_grouped['Gender_ID_combinations'], gender_x, gender_y)
print(skipped)


gender_df.to_csv('./data/winoqueer_gender_identity.csv')

gender_x1 = ['heterosexual', 'hetero', 'straight']
gender_y1 = ['bisexual', 'bi', 'homosexual', 'homo', 'gay', 'lesbian', 'queer', 'pansexual', 'pan',
             'asexual', 'ace', 'demisexual', 'demi']

skipped = set()
sexual_df = create_new_dataset(df_grouped['sentence'],df_grouped['Gender_ID_combinations'], gender_x1, gender_y1)
print(skipped)
sexual_df.to_csv('./data/winoqueer_sexual_identity.csv')