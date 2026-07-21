"""
Setup (não-graded): popula a `bbc_collection` a partir de `data/bbc_data.joblib`.

Por que existe:
    A `bbc_collection` pré-construída do curso não está disponível localmente.
    Só temos os artigos brutos (`bbc_data.joblib`), então precisamos criar a
    coleção e inseri-la em modo *chunked* — o notebook e o `unittests.py`
    esperam as propriedades `chunk` e `chunk_index`.

Uso:
    Rodar UMA vez, depois do vectorizer estar no ar (`flask_app.start_vectorizer()`)
    e antes de executar os exercícios do notebook:

        python build_collection.py

    Ou importar `build_collection(client, bbc_data)` de uma célula de setup.

Observação: a receita de chunking original do curso é desconhecida; este chunker
por palavras é uma aproximação. Ajuste CHUNK_SIZE / CHUNK_OVERLAP se necessário.
"""

from datetime import timezone

import joblib
import weaviate
from weaviate.classes.config import Configure, DataType, Property

COLLECTION_NAME = "bbc_collection"
CHUNK_SIZE = 200      # palavras por chunk
CHUNK_OVERLAP = 20    # palavras de sobreposição entre chunks consecutivos


def chunk_text(text: str, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """Fatiar `text` em janelas de `size` palavras com `overlap` de sobreposição."""
    words = (text or "").split()
    if not words:
        return []
    step = max(1, size - overlap)
    return [" ".join(words[i : i + size]) for i in range(0, len(words), step)]


# Propriedades corrigidas para bater com o formato de bbc_data
# (title, pubDate, guid, link, description, article_content) + chunk/chunk_index gerados.
PROPERTIES = [
    Property(name="title", data_type=DataType.TEXT),
    Property(name="pubDate", data_type=DataType.DATE),   # Timestamp -> DATE
    Property(name="link", data_type=DataType.TEXT),
    Property(name="description", data_type=DataType.TEXT),
    Property(name="article_content", data_type=DataType.TEXT),
    Property(name="chunk", data_type=DataType.TEXT),         # gerado do article_content
    Property(name="chunk_index", data_type=DataType.INT),    # gerado
]


def create_collection(client: weaviate.WeaviateClient):
    """(Re)cria a `bbc_collection` do zero com vetor nomeado `main_vector` e reranker."""
    if client.collections.exists(COLLECTION_NAME):
        client.collections.delete(COLLECTION_NAME)

    return client.collections.create(
        name=COLLECTION_NAME,
        properties=PROPERTIES,
        vector_config=Configure.Vectors.text2vec_transformers(
            name="main_vector",
            source_properties=["chunk"],
        ),
        reranker_config=Configure.Reranker.transformers(),
    )


def insert_data(client: weaviate.WeaviateClient, bbc_data: list[dict]) -> int:
    """Fatiar cada artigo e inserir os chunks em batch. Retorna o total inserido."""
    collection = client.collections.get(COLLECTION_NAME)
    inserted = 0

    with collection.batch.dynamic() as batch:
        for art in bbc_data:
            pub = art["pubDate"].to_pydatetime()
            if pub.tzinfo is None:                      # DATE exige timezone (RFC3339)
                pub = pub.replace(tzinfo=timezone.utc)

            for idx, ch in enumerate(chunk_text(art["article_content"])):
                batch.add_object(
                    properties={
                        "title": art["title"],
                        "pubDate": pub,
                        "link": art["link"],
                        "description": art["description"],
                        "article_content": art["article_content"],
                        "chunk": ch,
                        "chunk_index": idx,
                    }
                )
                inserted += 1

    failed = collection.batch.failed_objects
    if failed:
        print(f"[aviso] {len(failed)} objetos falharam. Exemplo: {failed[0].message}")

    return inserted


def build_collection(client: weaviate.WeaviateClient, bbc_data: list[dict]):
    """Fluxo completo: cria a coleção e insere os dados chunked."""
    collection = create_collection(client)
    total = insert_data(client, bbc_data)
    print(f"Inseridos {total} chunks a partir de {len(bbc_data)} artigos.")
    print(f"len(collection) = {len(collection)}")
    return collection


if __name__ == "__main__":
    bbc_data = joblib.load("data/bbc_data.joblib")
    client = weaviate.connect_to_local(port=8079, grpc_port=50050)
    try:
        build_collection(client, bbc_data)
    finally:
        client.close()
