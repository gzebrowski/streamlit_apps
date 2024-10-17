from typing import Optional, TypedDict

import streamlit as st
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

from streamlit_env import Env

env = Env('.env')

QDRANT_URL = env["QDRANT_URL"]
QDRANT_KEY = env.get("QDRANT_KEY")
QDRANT_PATH = env.get("QDRANT_PATH")


class QdrantSearchResult(TypedDict):
    text: str
    score: Optional[float]


class QdrantEmbedingItem(TypedDict):
    name: str
    values: list[str]


class QdrantHelper:
    EMBEDDING_DIM = 1536  # 3072
    EMBEDDING_MODEL = 'text-embedding-3-small'  # text-embedding-3-large

    def __init__(self, openai_client):
        self._openai_client = openai_client

    @classmethod
    @st.cache_resource
    def get_qdrant_client(cls):
        qdrant_params = {}
        if QDRANT_URL:
            qdrant_params['url'] = QDRANT_URL
            if QDRANT_KEY:
                qdrant_params['api_key'] = QDRANT_KEY
        elif QDRANT_PATH:
            qdrant_params['path'] = QDRANT_PATH
        return QdrantClient(**qdrant_params)

    def assure_db_collection_exists(self, collection_name: str) -> bool:
        qdrant_client = self.get_qdrant_client()
        if not qdrant_client.collection_exists(collection_name):
            qdrant_client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(
                    size=self.EMBEDDING_DIM,
                    distance=Distance.COSINE,
                ),
            )
            return False
        return True

    def index_embedings(self, items: list[QdrantEmbedingItem]):
        for item in items:
            if not self.assure_db_collection_exists(item['name']):
                for nr, value in enumerate(item['values']):
                    self.embeding_to_db(item['name'], value, idx=nr + 1)

    def get_embedding(self, text: str):
        openai_client = self._openai_client
        result = openai_client.embeddings.create(
            input=[text],
            model=self.EMBEDDING_MODEL,
            dimensions=self.EMBEDDING_DIM,
        )
        return result.data[0].embedding

    def embeding_to_db(self, collection_name: str, text: str, idx: Optional[int] = None, wait=False):
        qdrant_client = self.get_qdrant_client()
        if idx is None:
            points_count = qdrant_client.count(
                collection_name=collection_name,
                exact=True,
            )
            idx = points_count.count + 1
        qdrant_client.upsert(
            collection_name=collection_name,
            wait=wait,
            points=[
                PointStruct(
                    id=idx,
                    vector=self.get_embedding(text=text),
                    payload={
                        "text": text,
                    },
                )
            ]
        )

    def search_values_from_db(
            self, collection_name: str, query: Optional[str] = None, limit=10) -> list[QdrantSearchResult]:
        qdrant_client = self.get_qdrant_client()
        if not query:
            values = qdrant_client.scroll(collection_name=collection_name, limit=limit)[0]
            result = []
            for value in values:
                result.append({
                    "text": value.payload["text"],
                    "score": None,
                })

            return result
        else:
            values = qdrant_client.search(
                collection_name=collection_name,
                query_vector=self.get_embedding(text=query),
                limit=limit,
            )
            result: list[QdrantSearchResult] = []
            for value in values:
                result.append({
                    "text": value.payload["text"],
                    "score": value.score,
                })

            return result
