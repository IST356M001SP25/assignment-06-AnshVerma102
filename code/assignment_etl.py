import sys
from concurrent.futures import ThreadPoolExecutor
import pandas as pd
import streamlit as st

# ensure our apicalls module is on the path
sys.path.append("code")
from apicalls import (
    get_google_place_details,
    get_azure_sentiment,
    get_azure_named_entity_recognition,
)

PLACE_IDS_SOURCE_FILE = "cache/place_ids.csv"
CACHE_REVIEWS_FILE = "cache/reviews.csv"
CACHE_SENTIMENT_FILE = "cache/reviews_sentiment_by_sentence.csv"
CACHE_ENTITIES_FILE = "cache/reviews_sentiment_by_sentence_with_entities.csv"


def _load_df(source):
    """Load a DataFrame from path or passthrough if already a DataFrame."""
    return pd.read_csv(source) if isinstance(source, str) else source


def reviews_step(place_ids: str | pd.DataFrame) -> pd.DataFrame:
    """
    1. place_ids --> reviews_step --> reviews: place_id, name, author_name, rating, text
    """
    df = _load_df(place_ids)
    place_ids_list = df["Google Place ID"].tolist()

    # Parallelize the Google Places API calls
    with ThreadPoolExecutor() as executor:
        results = list(executor.map(
            lambda pid: get_google_place_details(pid)["result"],
            place_ids_list
        ))

    # Normalize nested "reviews" arrays into a flat DataFrame
    reviews_df = pd.json_normalize(
        results,
        record_path="reviews",
        meta=["place_id", "name"],
    )

    reviews_df = reviews_df[[
        "place_id", "name", "author_name", "rating", "text"
    ]]

    reviews_df.to_csv(CACHE_REVIEWS_FILE, index=False)
    return reviews_df


def sentiment_step(reviews: str | pd.DataFrame) -> pd.DataFrame:
    """
    2. reviews --> sentiment_step --> review_sentiment_by_sentence
    """
    reviews_df = _load_df(reviews)
    texts = reviews_df["text"].tolist()
    metas = reviews_df[[
        "place_id", "name", "author_name", "rating"
    ]].to_dict("records")

    # Parallelize the Azure sentiment API calls
    with ThreadPoolExecutor() as executor:
        docs = list(executor.map(get_azure_sentiment, texts))

    # Attach metadata to each document result
    sentiments = []
    for meta, doc in zip(metas, docs):
        item = doc["results"]["documents"][0]
        item.update(meta)
        sentiments.append(item)

    sentiment_df = pd.json_normalize(
        sentiments,
        record_path="sentences",
        meta=["place_id", "name", "author_name", "rating"],
    )

    sentiment_df.rename(
        columns={
            "text": "sentence_text",
            "sentiment": "sentence_sentiment",
        },
        inplace=True
    )

    sentiment_df = sentiment_df[[
        "place_id", "name", "author_name", "rating",
        "sentence_text", "sentence_sentiment",
        "confidenceScores.positive",
        "confidenceScores.neutral",
        "confidenceScores.negative",
    ]]

    sentiment_df.to_csv(CACHE_SENTIMENT_FILE, index=False)
    return sentiment_df


def entity_extraction_step(sentiment: str | pd.DataFrame) -> pd.DataFrame:
    """
    3. review_sentiment_by_sentence --> entity_extraction_step --> review_sentiment_entities_by_sentence
    """
    sentiment_df = _load_df(sentiment)
    sentences = sentiment_df["sentence_text"].tolist()
    metas = sentiment_df.to_dict("records")

    # Parallelize the Azure NER API calls
    with ThreadPoolExecutor() as executor:
        docs = list(executor.map(get_azure_named_entity_recognition, sentences))

    # Attach metadata to each NER result
    entities = []
    for meta, doc in zip(metas, docs):
        item = doc["results"]["documents"][0]
        item.update(meta)
        entities.append(item)

    entities_df = pd.json_normalize(
        entities,
        record_path="entities",
        meta=list(sentiment_df.columns),
    )

    entities_df.rename(
        columns={
            "text": "entity_text",
            "category": "entity_category",
            "subcategory": "entity_subcategory",
            "confidenceScore": "confidenceScores.entity",
        },
        inplace=True
    )

    entities_df = entities_df[[
        "place_id", "name", "author_name", "rating",
        "sentence_text", "sentence_sentiment",
        "confidenceScores.positive", "confidenceScores.neutral",
        "confidenceScores.negative", "entity_text",
        "entity_category", "entity_subcategory",
        "confidenceScores.entity",
    ]]

    entities_df.to_csv(CACHE_ENTITIES_FILE, index=False)
    return entities_df


if __name__ == "__main__":
    reviews_df = reviews_step(PLACE_IDS_SOURCE_FILE)
    sentiment_df = sentiment_step(reviews_df)
    entities_df = entity_extraction_step(sentiment_df)
    st.write(entities_df)
