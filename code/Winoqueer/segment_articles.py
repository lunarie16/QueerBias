import pandas as pd
import spacy

from datasets import Dataset, DatasetDict
from tqdm import tqdm

# ARGUMENTS
# sys.argv[1] dataset path (pickled pandas df)
# sys.argv[2] save location
import re

# define some helper functions
# sentence segmentation for news
nlp = spacy.load("en_core_web_sm")
def spacy_seg(row):
    if isinstance(row['text'], float):
        return None
    text = row['text'].replace('\n', ' ').replace('\\', '')
    doc = nlp(text)
    return [sent.text for sent in doc.sents]

tqdm.pandas()

# load dataset
print("loading dataset...")
df = pd.read_csv('data/datasets/article_texts.csv')

# do sentence segmentation
print("Segmenting sentences...")
columns_to_drop = ['stories_id', 'authors', 'publish_date', 'media_outlet']
df = df.drop(columns=[col for col in columns_to_drop if col in df.columns])
# drop all columns that start with "unnamed"
df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
df['sentence'] = df.progress_apply(lambda row: spacy_seg(row), axis=1)
df = df.explode("sentence")

# deal with whitespace and empty strings
df['sentence'] = df['sentence'].str.strip()
df = df[df.sentence != '']
df = df[df['sentence'].notna()]
print(df)
# save
# filehandler = open("../data/datasets/queer_news.pkl","wb")
# pickle.dump(df,filehandler)
# filehandler.close()
df.to_csv('data/datasets/queer_news.csv', index=False)
print("done!")


# dataset_pd = Dataset.from_pandas(df)
# train_test = dataset_pd.train_test_split(test_size=.1)
# test_eval = train_test['test'].train_test_split(test_size=.5)
# dataset =  DatasetDict({
#                         "train": train_test["train"],
#                         "test": test_eval["test"],
#                         "validation": test_eval["train"]})
# for split, split_dataset in dataset.items():
#     split_dataset.to_json(f"queer_news-{split}.jsonl")
