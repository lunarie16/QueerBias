import pandas as pd
from newspaper import Article
from tqdm import tqdm
from logging import getLogger

logger = getLogger(__name__)


# articles = pd.read_csv('./data/checkpoints/article_texts-21000.csv')

# drop all columns with nan

error = 0
success = 0

checkpoint = 21000
for i, row in tqdm(articles.iterrows()):
    if i <= checkpoint:
        continue
    if error % 100 == 0:
        logger.warning(f"Error count: {error}")
    if success % 100 == 0:
        logger.warning(f"Success count: {success}")

    if i % 100 == 0:
        articles.to_csv('./data/article_texts.csv')
    if i % 1000 == 0:
        articles.to_csv(f'./data/checkpoints/article_texts-{i}.csv')
    url = row['url']
    if not isinstance(row['url'], str):
        articles.loc[i, 'text'] = None
        articles.loc[i, 'error'] = 'not a string'
        # logger.warning(f'Row {i} has a non-string URL: {url}')
        error += 1
        continue
    article = Article(url)
    try:
        article.download()
        article.parse()
        article_text = article.text
        articles.loc[i, 'text'] = article_text
        articles.loc[i, 'error'] = None
        success += 1
    except Exception as e:
        articles.loc[i, 'error'] = e
        articles.loc[i, 'text'] = None
        # logger.error(f'Error downloading article {url}: {e}')
        error += 1

articles.to_csv('./data/article_texts.csv')
articles.to_csv(f'./data/checkpoints/article_texts-{i}.csv')