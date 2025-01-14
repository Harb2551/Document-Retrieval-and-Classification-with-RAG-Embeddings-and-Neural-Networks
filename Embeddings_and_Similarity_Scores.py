import google.generativeai as genai
from kaggle_secrets import UserSecretsClient

GOOGLE_API_KEY = UserSecretsClient().get_secret("GOOGLE_API_KEY")
genai.configure(api_key=GOOGLE_API_KEY)

for model in genai.list_models():
  if 'embedContent' in model.supported_generation_methods:
    print(model.name)

texts = [
    'The quick brown fox jumps over the lazy dog.',
    'The quick rbown fox jumps over the lazy dog.',
    'teh fast fox jumps over the slow woofer.',
    'a quick brown fox jmps over lazy dog.',
    'brown fox jumping over dog',
    'fox > dog',
    'The five boxing wizards jump quickly.',
    'Lorem ipsum dolor sit amet, consectetur adipiscing elit. Vivamus et hendrerit massa. Sed pulvinar, nisi a lobortis sagittis, neque risus gravida dolor, in porta dui odio vel purus.',
]

response = genai.embed_content(model='models/text-embedding-004',
                               content=texts,
                               task_type='semantic_similarity')

def truncate(t: str, limit: int = 50) -> str:
  if len(t) > limit:
    return t[:limit-3] + '...'
  else:
    return t

truncated_texts = [truncate(t) for t in texts]

import pandas as pd
import seaborn as sns

df = pd.DataFrame(response['embedding'], index=truncated_texts)

sim = df @ df.T

sns.heatmap(sim, vmin=0, vmax=1);

sim['The quick brown fox jumps over the lazy dog.'].sort_values(ascending=False)
