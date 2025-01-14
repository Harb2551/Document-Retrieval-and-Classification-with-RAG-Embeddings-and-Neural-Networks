import google.generativeai as genai
from kaggle_secrets import UserSecretsClient

GOOGLE_API_KEY = UserSecretsClient().get_secret("GOOGLE_API_KEY")
genai.configure(api_key=GOOGLE_API_KEY)

from sklearn.datasets import fetch_20newsgroups

newsgroups_train = fetch_20newsgroups(subset="train")
newsgroups_test = fetch_20newsgroups(subset="test")

newsgroups_train.target_names

print(newsgroups_train.data[0])

import email
import re
import pandas as pd

def preprocess_newsgroup_row(data):
    msg = email.message_from_string(data)
    text = f"{msg['Subject']}\n\n{msg.get_payload()}"
    text = re.sub(r"[\w\.-]+@[\w\.-]+", "", text)
    text = text[:5000]
    return text

def preprocess_newsgroup_data(newsgroup_dataset):
    df = pd.DataFrame(
        {"Text": newsgroup_dataset.data, "Label": newsgroup_dataset.target}
    )
    df["Text"] = df["Text"].apply(preprocess_newsgroup_row)
    df["Class Name"] = df["Label"].map(lambda l: newsgroup_dataset.target_names[l])
    return df

df_train = preprocess_newsgroup_data(newsgroups_train)
df_test = preprocess_newsgroup_data(newsgroups_test)

def sample_data(df, num_samples, classes_to_keep):
    df = (
        df.groupby("Label")[df.columns]
        .apply(lambda x: x.sample(num_samples))
        .reset_index(drop=True)
    )
    df = df[df["Class Name"].str.contains(classes_to_keep)]
    df["Class Name"] = df["Class Name"].astype("category")
    df["Encoded Label"] = df["Class Name"].cat.codes
    return df

TRAIN_NUM_SAMPLES = 100
TEST_NUM_SAMPLES = 25
CLASSES_TO_KEEP = "sci"

df_train = sample_data(df_train, TRAIN_NUM_SAMPLES, CLASSES_TO_KEEP)
df_test = sample_data(df_test, TEST_NUM_SAMPLES, CLASSES_TO_KEEP)

df_train.value_counts("Class Name")

df_test.value_counts("Class Name")

from google.api_core import retry
from tqdm.rich import tqdm

tqdm.pandas()

@retry.Retry(timeout=300.0)
def embed_fn(text: str) -> list[float]:
    response = genai.embed_content(
        model="models/text-embedding-004", content=text, task_type="classification"
    )
    return response["embedding"]

def create_embeddings(df):
    df["Embeddings"] = df["Text"].progress_apply(embed_fn)
    return df

df_train = create_embeddings(df_train)
df_test = create_embeddings(df_test)

df_train.head()

import keras
from keras import layers

def build_classification_model(input_size: int, num_classes: int) -> keras.Model:
    return keras.Sequential(
        [
            layers.Input([input_size], name="embedding_inputs"),
            layers.Dense(input_size, activation="relu", name="hidden"),
            layers.Dense(num_classes, activation="softmax", name="output_probs"),
        ]
    )

embedding_size = len(df_train["Embeddings"].iloc[0])

classifier = build_classification_model(
    embedding_size, len(df_train["Class Name"].unique())
)
classifier.summary()

classifier.compile(
    loss=keras.losses.SparseCategoricalCrossentropy(),
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    metrics=["accuracy"],
)

import numpy as np

NUM_EPOCHS = 20
BATCH_SIZE = 32

y_train = df_train["Encoded Label"]
x_train = np.stack(df_train["Embeddings"])
y_val = df_test["Encoded Label"]
x_val = np.stack(df_test["Embeddings"])

early_stop = keras.callbacks.EarlyStopping(monitor="accuracy", patience=3)

history = classifier.fit(
    x=x_train,
    y=y_train,
    validation_data=(x_val, y_val),
    callbacks=[early_stop],
    batch_size=BATCH_SIZE,
    epochs=NUM_EPOCHS,
)

classifier.evaluate(x=x_val, y=y_val, return_dict=True)

new_text = """
First-timer looking to get out of here.

Hi, I'm writing about my interest in travelling to the outer limits!

What kind of craft can I buy? What is easiest to access from this 3rd rock?

Let me know how to do that please.
"""
embedded = embed_fn(new_text)

inp = np.array([embedded])
[result] = classifier.predict(inp)

for idx, category in enumerate(df_test["Class Name"].cat.categories):
    print(f"{category}: {result[idx] * 100:0.2f}%")
