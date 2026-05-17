"""
Модуль для создания векторных индексов с использованием ChromaDB.
ChromaDB работает встроенно — никакого отдельного сервера не нужно.
"""

import os
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")
import time
import uuid
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from sentence_transformers import SentenceTransformer
from PIL import Image
import logging
import chromadb

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class VectorIndexer:
    """Класс для создания и управления векторными индексами через ChromaDB."""

    def __init__(self,
                 persist_dir: str = "./chroma_db",
                 text_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
                 image_model_name: str = "sentence-transformers/clip-ViT-B-32"):
        logger.info(f"Инициализация ChromaDB в директории: {persist_dir}")
        os.makedirs(persist_dir, exist_ok=True)
        self.client = chromadb.PersistentClient(path=persist_dir)
        logger.info("ChromaDB успешно инициализирован")

        logger.info(f"Загрузка текстовой модели: {text_model_name}")
        self.text_model = SentenceTransformer(text_model_name)

        logger.info(f"Загрузка модели изображений: {image_model_name}")
        try:
            self.image_model = SentenceTransformer(image_model_name)
        except Exception as e:
            logger.warning(f"Ошибка загрузки модели изображений: {e}. Используется текстовая модель.")
            self.image_model = self.text_model

        self.text_collection = self.client.get_or_create_collection(
            name="text_documents",
            metadata={"hnsw:space": "cosine"}
        )
        self.image_collection = self.client.get_or_create_collection(
            name="image_documents",
            metadata={"hnsw:space": "cosine"}
        )

        logger.info("VectorIndexer успешно инициализирован")

    def create_collections(self, text_collection_name: str = "text_documents",
                           image_collection_name: str = "image_documents"):
        """Совместимость — коллекции уже созданы в __init__."""
        pass

    def clear_all_documents(self):
        """Полная очистка всех данных из обеих коллекций."""
        try:
            self.client.delete_collection("text_documents")
            self.client.delete_collection("image_documents")
            self.text_collection = self.client.get_or_create_collection(
                name="text_documents", metadata={"hnsw:space": "cosine"}
            )
            self.image_collection = self.client.get_or_create_collection(
                name="image_documents", metadata={"hnsw:space": "cosine"}
            )
            logger.info("Все данные успешно очищены")
        except Exception as e:
            logger.error(f"Ошибка очистки данных: {e}")

    def delete_document(self, document_id: str):
        """Удаление конкретного документа по ID."""
        try:
            self.text_collection.delete(where={"document_id": document_id})
            self.image_collection.delete(where={"document_id": document_id})
            logger.info(f"Документ {document_id} удалён")
        except Exception as e:
            logger.error(f"Ошибка удаления документа {document_id}: {e}")

    def get_all_text_for_document(self, document_id: str) -> List[Dict]:
        """Получение всех текстовых фрагментов конкретного документа."""
        try:
            results = self.text_collection.get(
                where={"document_id": document_id},
                include=["documents", "metadatas"]
            )
            output = []
            if results['ids']:
                for i in range(len(results['ids'])):
                    output.append({
                        'score': 1.0,
                        'content': results['documents'][i],
                        'type': results['metadatas'][i].get('type', ''),
                        'page': results['metadatas'][i].get('page', 0),
                        'document_id': results['metadatas'][i].get('document_id', ''),
                        'metadata': {}
                    })
            return output
        except Exception as e:
            logger.error(f"Ошибка получения текста документа: {e}")
            return []

    def create_text_embeddings(self, texts: List[str]) -> np.ndarray:
        try:
            embeddings = self.text_model.encode(texts, convert_to_numpy=True)
            logger.info(f"Создано {len(embeddings)} текстовых эмбеддингов")
            return embeddings
        except Exception as e:
            logger.error(f"Ошибка создания текстовых эмбеддингов: {e}")
            return np.array([])

    def create_image_embeddings(self, image_paths: List[str]) -> Tuple[np.ndarray, List[str]]:
        try:
            images, valid_paths = [], []
            for path in image_paths:
                try:
                    images.append(Image.open(path))
                    valid_paths.append(path)
                except Exception as e:
                    logger.warning(f"Не удалось загрузить изображение {path}: {e}")

            if images:
                embeddings = self.image_model.encode(images, convert_to_numpy=True)
                logger.info(f"Создано {len(embeddings)} эмбеддингов изображений")
                return embeddings, valid_paths
            return np.array([]), []
        except Exception as e:
            logger.error(f"Ошибка создания эмбеддингов изображений: {e}")
            return np.array([]), []

    def index_documents(self,
                        extracted_content: Dict[str, Any],
                        document_id: str,
                        text_collection_name: str = "text_documents",
                        image_collection_name: str = "image_documents"):
        """Индексация документа в ChromaDB."""
        try:
            if extracted_content.get('text'):
                texts = [item['content'] for item in extracted_content['text']]
                if texts:
                    embeddings = self.create_text_embeddings(texts)
                    if len(embeddings) > 0:
                        ids = [str(uuid.uuid4()) for _ in texts]
                        metadatas = [{
                            'document_id': document_id,
                            'type': item.get('type', 'text'),
                            'page': item.get('page', 0),
                        } for item in extracted_content['text']]

                        self.text_collection.add(
                            ids=ids,
                            embeddings=embeddings.tolist(),
                            documents=texts,
                            metadatas=metadatas
                        )
                        logger.info(f"Проиндексировано {len(texts)} текстовых фрагментов")

            if extracted_content.get('images'):
                image_paths = [item['path'] for item in extracted_content['images']]
                if image_paths:
                    embeddings, valid_paths = self.create_image_embeddings(image_paths)
                    if len(embeddings) > 0:
                        path_to_item = {item['path']: item for item in extracted_content['images']}
                        ids, emb_list, docs, metadatas = [], [], [], []

                        for embedding, path in zip(embeddings, valid_paths):
                            if path in path_to_item:
                                item = path_to_item[path]
                                ids.append(str(uuid.uuid4()))
                                emb_list.append(embedding.tolist())
                                docs.append(item.get('ocr_text', '') or item['filename'])
                                metadatas.append({
                                    'document_id': document_id,
                                    'filename': item['filename'],
                                    'path': item['path'],
                                    'ocr_text': item.get('ocr_text', ''),
                                    'page': item.get('page', 0),
                                })

                        if ids:
                            self.image_collection.add(
                                ids=ids,
                                embeddings=emb_list,
                                documents=docs,
                                metadatas=metadatas
                            )
                            logger.info(f"Проиндексировано {len(ids)} изображений")

        except Exception as e:
            logger.error(f"Ошибка индексации документа {document_id}: {e}")

    def search_similar_text(self,
                            query: str,
                            collection_name: str = "text_documents",
                            limit: int = 5,
                            document_id: Optional[str] = None) -> List[Dict]:
        """Поиск похожих текстовых фрагментов с опциональной фильтрацией по документу."""
        try:
            where = {"document_id": document_id} if document_id else None
            count = self.text_collection.count()
            if count == 0:
                return []

            query_embedding = self.text_model.encode([query])[0]
            query_params = dict(
                query_embeddings=[query_embedding.tolist()],
                n_results=min(limit, count)
            )
            if where:
                query_params["where"] = where

            results = self.text_collection.query(**query_params)

            output = []
            if results['ids'] and results['ids'][0]:
                for i in range(len(results['ids'][0])):
                    score = 1.0 - results['distances'][0][i]
                    # Порог 0.1 — возвращаем всё хоть немного релевантное
                    if score >= 0.1:
                        output.append({
                            'score': score,
                            'content': results['documents'][0][i],
                            'type': results['metadatas'][0][i].get('type', ''),
                            'page': results['metadatas'][0][i].get('page', 0),
                            'document_id': results['metadatas'][0][i].get('document_id', ''),
                            'metadata': {}
                        })

            logger.info(f"Найдено {len(output)} похожих текстовых фрагментов")
            return output
        except Exception as e:
            logger.error(f"Ошибка поиска текста: {e}")
            return []

    def search_similar_images(self,
                              query: str,
                              collection_name: str = "image_documents",
                              limit: int = 3,
                              document_id: Optional[str] = None) -> List[Dict]:
        """Поиск похожих изображений по текстовому запросу (CLIP)."""
        try:
            where = {"document_id": document_id} if document_id else None
            count = self.image_collection.count()
            if count == 0:
                return []

            query_embedding = self.image_model.encode([query])[0]
            query_params = dict(
                query_embeddings=[query_embedding.tolist()],
                n_results=min(limit, count)
            )
            if where:
                query_params["where"] = where

            results = self.image_collection.query(**query_params)

            output = []
            if results['ids'] and results['ids'][0]:
                for i in range(len(results['ids'][0])):
                    score = 1.0 - results['distances'][0][i]
                    if score >= 0.1:
                        output.append({
                            'score': score,
                            'filename': results['metadatas'][0][i].get('filename', ''),
                            'path': results['metadatas'][0][i].get('path', ''),
                            'ocr_text': results['metadatas'][0][i].get('ocr_text', ''),
                            'page': results['metadatas'][0][i].get('page', 0),
                            'document_id': results['metadatas'][0][i].get('document_id', ''),
                            'metadata': {}
                        })

            logger.info(f"Найдено {len(output)} похожих изображений")
            return output
        except Exception as e:
            logger.error(f"Ошибка поиска изображений: {e}")
            return []


if __name__ == "__main__":
    indexer = VectorIndexer()
    print("Модуль vector_indexer готов к использованию")
