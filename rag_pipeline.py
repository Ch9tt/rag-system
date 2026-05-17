"""
Модуль для RAG pipeline с поддержкой мультимодальных LLM.
"""

import os
import base64
from typing import List, Dict, Any, Optional
from PIL import Image
import logging
from typing import Any, List, Optional
from pydantic import Field
from langchain_core.language_models.llms import LLM
from langchain_core.callbacks import CallbackManagerForLLMRun
import requests
import json

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MultimodalLLM(LLM):
    """
    Кастомная реализация мультимодальной LLM для LangChain.
    Поддерживает как локальные модели, так и API.
    """
    
    model_type: str = Field(default="openai", description="Тип модели")
    model_name: str = Field(default="gpt-4o-mini", description="Название модели")
    api_key: Optional[str] = Field(default=None, description="API ключ")
    base_url: Optional[str] = Field(default=None, description="Базовый URL")
    
    def __init__(self,
                 model_type: str = "groq",  # "groq", "openai", "ollama", "local"
                 model_name: str = "llama-3.3-70b-versatile",
                 api_key: Optional[str] = None,
                 base_url: Optional[str] = None,
                 **kwargs):
        """
        Args:
            model_type: Тип модели ("groq", "openai", "ollama", "local")
            model_name: Название модели
            api_key: API ключ
            base_url: Базовый URL для API
        """
        resolved_key = api_key or os.getenv("GROQ_API_KEY") or os.getenv("OPENAI_API_KEY")
        super().__init__(
            model_type=model_type,
            model_name=model_name,
            api_key=resolved_key,
            base_url=base_url,
            **kwargs
        )

        if model_type == "groq" and not resolved_key:
            logger.warning("API ключ Groq не найден. Получите бесплатно на https://console.groq.com/")
        if model_type == "openai" and not resolved_key:
            logger.warning("API ключ OpenAI не найден.")
    
    def _call(self,
              prompt: str,
              stop: Optional[List[str]] = None,
              run_manager: Optional[CallbackManagerForLLMRun] = None,
              **kwargs: Any) -> str:
        """Базовый вызов LLM для текстового промпта."""
        return self.generate_response(prompt, images=None)

    def generate_response(self,
                          prompt: str,
                          images: Optional[List[str]] = None) -> str:
        """
        Генерация ответа с поддержкой изображений.
        
        Args:
            prompt: Текстовый промпт
            images: Список путей к изображениям
            
        Returns:
            Сгенерированный ответ
        """
        if self.model_type == "groq":
            return self._call_groq_api(prompt, images)
        elif self.model_type == "openai":
            return self._call_openai_api(prompt, images)
        elif self.model_type == "ollama":
            return self._call_ollama_api(prompt, images)
        else:
            return self._call_local_model(prompt, images)
    
    def _call_groq_api(self, prompt: str, images: Optional[List[str]] = None) -> str:
        """Вызов Groq API — бесплатный, быстрый, работает из России."""
        try:
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            data = {
                "model": self.model_name,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 2048,
                "temperature": 0.7
            }
            response = requests.post(
                "https://api.groq.com/openai/v1/chat/completions",
                headers=headers,
                json=data,
                timeout=60
            )
            if response.status_code == 200:
                return response.json()["choices"][0]["message"]["content"]
            elif response.status_code == 401:
                return (
                    "⚠️ Неверный Groq API ключ!\n\n"
                    "💡 Получите бесплатный ключ на https://console.groq.com/\n"
                    "Затем вставьте его в поле 'Groq API Key' в боковой панели."
                )
            else:
                error = response.json().get("error", {}).get("message", response.text)
                logger.error(f"Ошибка Groq API: {response.status_code} - {error}")
                return f"Ошибка Groq API ({response.status_code}): {error}"
        except Exception as e:
            logger.error(f"Ошибка вызова Groq API: {e}")
            return f"Ошибка при подключении к Groq: {e}"

    def _call_openai_api(self, prompt: str, images: Optional[List[str]] = None) -> str:
        """Вызов OpenAI API."""
        try:
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            
            messages = []
            
            if images:
                # Мультимодальный запрос
                content = [{"type": "text", "text": prompt}]
                
                for image_path in images:
                    if os.path.exists(image_path):
                        ext = os.path.splitext(image_path)[1].lower()
                        mime_map = {'.jpg': 'image/jpeg', '.jpeg': 'image/jpeg',
                                    '.png': 'image/png', '.gif': 'image/gif',
                                    '.webp': 'image/webp'}
                        mime_type = mime_map.get(ext, 'image/jpeg')
                        with open(image_path, "rb") as image_file:
                            base64_image = base64.b64encode(image_file.read()).decode('utf-8')
                            content.append({
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:{mime_type};base64,{base64_image}"
                                }
                            })
                
                messages.append({"role": "user", "content": content})
            else:
                # Только текстовый запрос
                messages.append({"role": "user", "content": prompt})
            
            data = {
                "model": self.model_name,
                "messages": messages,
                "max_tokens": 1500,
                "temperature": 0.7
            }
            
            response = requests.post(
                "https://api.openai.com/v1/chat/completions",
                headers=headers,
                json=data,
                timeout=60
            )
            
            if response.status_code == 200:
                result = response.json()
                return result["choices"][0]["message"]["content"]
            elif response.status_code == 403:
                error_data = response.json() if response.text else {}
                error_code = error_data.get("error", {}).get("code", "")
                if "unsupported_country" in error_code or "country" in str(error_data).lower():
                    error_msg = (
                        "⚠️ OpenAI API недоступен в вашем регионе.\n\n"
                        "💡 Рекомендации:\n"
                        "1. Используйте Ollama (локальная модель) - выберите 'ollama' в настройках\n"
                        "2. Или используйте VPN для доступа к OpenAI API\n"
                        "3. Или используйте другого провайдера LLM\n\n"
                        "Для использования Ollama:\n"
                        "- Установите Ollama: https://ollama.ai/\n"
                        "- Загрузите модель: ollama pull llama2\n"
                        "- Выберите 'ollama' в настройках приложения"
                    )
                    logger.error(f"OpenAI API недоступен в регионе: {error_data}")
                    return error_msg
                else:
                    logger.error(f"Ошибка API OpenAI: {response.status_code} - {response.text}")
                    return f"Ошибка OpenAI API ({response.status_code}): {error_data.get('error', {}).get('message', 'Неизвестная ошибка')}"
            else:
                logger.error(f"Ошибка API OpenAI: {response.status_code} - {response.text}")
                error_msg = f"Ошибка API OpenAI ({response.status_code})"
                try:
                    error_data = response.json()
                    if "error" in error_data:
                        error_msg += f": {error_data['error'].get('message', 'Неизвестная ошибка')}"
                except:
                    pass
                return f"Извините, {error_msg}. Попробуйте использовать Ollama или другую модель."
                
        except Exception as e:
            logger.error(f"Ошибка вызова OpenAI API: {e}")
            return "Извините, произошла ошибка при генерации ответа."
    
    def _call_ollama_api(self, prompt: str, images: Optional[List[str]] = None) -> str:
        """Вызов локального Ollama API."""
        try:
            # Используем chat API для лучшей совместимости
            url = (self.base_url or "http://localhost:11434") + "/api/chat"
            
            messages = [{"role": "user", "content": prompt}]
            
            # Добавление изображений для мультимодальных моделей
            images_base64 = []
            if images:
                for img_path in images:
                    if img_path and os.path.exists(img_path):
                        with open(img_path, "rb") as image_file:
                            base64_image = base64.b64encode(image_file.read()).decode('utf-8')
                            images_base64.append(base64_image)
            
            data = {
                "model": self.model_name,
                "messages": messages,
                "stream": False
            }
            
            if images_base64:
                data["images"] = images_base64
            
            response = requests.post(url, json=data, timeout=120)
            
            if response.status_code == 200:
                result = response.json()
                # Ollama chat API возвращает ответ в message.content
                if "message" in result and "content" in result["message"]:
                    return result["message"]["content"]
                elif "response" in result:
                    return result["response"]
                else:
                    return "Нет ответа от модели"
            elif response.status_code == 404:
                error_msg = (
                    f"⚠️ Модель '{self.model_name}' не найдена!\n\n"
                    f"💡 Загрузите модель командой:\n"
                    f"   ollama pull {self.model_name}\n\n"
                    f"Или выберите другую модель в настройках."
                )
                logger.error(f"Модель Ollama не найдена: {self.model_name}")
                return error_msg
            else:
                error_text = response.text
                logger.error(f"Ошибка Ollama API: {response.status_code} - {error_text}")
                
                # Более понятное сообщение об ошибке
                if "model" in error_text.lower() and "not found" in error_text.lower():
                    return (
                        f"⚠️ Модель '{self.model_name}' не найдена!\n\n"
                        f"💡 Загрузите модель: ollama pull {self.model_name}"
                    )
                else:
                    return f"Извините, произошла ошибка при генерации ответа (код {response.status_code}). Проверьте, что Ollama запущен."
                
        except requests.exceptions.ConnectionError as e:
            error_msg = (
                "⚠️ Не удалось подключиться к Ollama!\n\n"
                "💡 Проверьте:\n"
                "1. Запущен ли Ollama? (должно быть открыто приложение Ollama)\n"
                "2. Работает ли Ollama на http://localhost:11434?\n"
                "3. Попробуйте перезапустить Ollama\n\n"
                "Для проверки выполните в командной строке:\n"
                "   ollama list"
            )
            logger.error(f"Ошибка подключения к Ollama: {e}")
            return error_msg
        except Exception as e:
            logger.error(f"Ошибка вызова Ollama API: {e}")
            return f"Извините, произошла ошибка при генерации ответа: {str(e)}"
    
    def _call_local_model(self, prompt: str, images: Optional[List[str]] = None) -> str:
        """Заглушка для локальных моделей."""
        return f"Ответ от локальной модели на запрос: {prompt[:100]}..."
    
    @property
    def _llm_type(self) -> str:
        return "multimodal_llm"
    
    @property
    def _identifying_params(self) -> dict:
        """Параметры для идентификации модели."""
        return {
            "model_type": self.model_type,
            "model_name": self.model_name
        }


class RAGPipeline:
    """Основной класс для RAG pipeline."""
    
    def __init__(self,
                 vector_indexer,
                 llm: MultimodalLLM,
                 max_context_length: int = 10000):
        """
        Инициализация RAG pipeline.
        
        Args:
            vector_indexer: Экземпляр VectorIndexer
            llm: Мультимодальная LLM
            max_context_length: Максимальная длина контекста
        """
        self.vector_indexer = vector_indexer
        self.llm = llm
        self.max_context_length = max_context_length
    
    def generate_answer(self,
                       question: str,
                       text_limit: int = 5,
                       image_limit: int = 3,
                       document_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Генерация ответа на вопрос с использованием RAG.
        
        Args:
            question: Вопрос пользователя
            text_limit: Максимальное количество текстовых фрагментов
            image_limit: Максимальное количество изображений
            
        Returns:
            Словарь с ответом и использованными источниками
        """
        try:
            # Определяем — запрос на пересказ/резюме или конкретный вопрос
            summary_keywords = [
                'пересказ', 'перескажи', 'резюме', 'summary', 'суммаризируй',
                'краткое', 'о чём', 'о чем', 'содержание', 'главное', 'основное',
                'что в документе', 'что в книге', 'о чём книга', 'что описывает',
                'расскажи про', 'опиши документ', 'what is this about'
            ]
            is_summary_query = any(kw in question.lower() for kw in summary_keywords)

            # Для пересказа берём больше фрагментов
            effective_limit = text_limit * 3 if is_summary_query else text_limit

            text_results = self.vector_indexer.search_similar_text(
                query=question,
                limit=effective_limit,
                document_id=document_id
            )

            # Если summary-запрос и результатов мало — добираем из документа напрямую
            if is_summary_query and len(text_results) < 5 and document_id:
                all_chunks = self.vector_indexer.get_all_text_for_document(document_id)
                existing_contents = {r['content'] for r in text_results}
                for chunk in all_chunks:
                    if chunk['content'] not in existing_contents:
                        text_results.append(chunk)
                    if len(text_results) >= effective_limit:
                        break

            # Поиск релевантных изображений
            image_results = self.vector_indexer.search_similar_images(
                query=question,
                limit=image_limit,
                document_id=document_id
            )
            
            # Формирование контекста
            context_text = self._build_text_context(text_results)
            image_paths = self._get_image_paths(image_results)
            
            # Логирование для отладки
            logger.info(f"Найдено текстовых фрагментов: {len(text_results)}")
            logger.info(f"Найдено изображений: {len(image_results)}")
            if text_results:
                logger.info(f"Первый найденный фрагмент (первые 200 символов): {text_results[0].get('content', '')[:200]}")
            
            # Создание промпта
            prompt = self._create_prompt(question, context_text, image_results)
            
            # Генерация ответа
            answer = self.llm.generate_response(prompt, image_paths)
            
            return {
                'answer': answer,
                'sources': {
                    'text_sources': text_results,
                    'image_sources': image_results
                },
                'context_used': {
                    'text_fragments': len(text_results),
                    'images': len(image_results)
                }
            }
            
        except Exception as e:
            logger.error(f"Ошибка генерации ответа: {e}")
            return {
                'answer': "Извините, произошла ошибка при обработке вашего вопроса.",
                'sources': {'text_sources': [], 'image_sources': []},
                'context_used': {'text_fragments': 0, 'images': 0}
            }
    
    def _build_text_context(self, text_results: List[Dict]) -> str:
        """Построение текстового контекста из найденных фрагментов."""
        if not text_results:
            return ""
        
        context_parts = []
        current_length = 0
        
        for i, result in enumerate(text_results):
            content = result.get('content', '').strip()
            if not content:
                continue
                
            page = result.get('page', 'неизвестно')
            source_info = f"[Источник {i+1}, страница {page}]"
            
            fragment = f"{source_info}\n{content}"
            
            if current_length + len(fragment) > self.max_context_length:
                # Добавляем хотя бы часть, если это первый фрагмент
                if i == 0:
                    available = self.max_context_length - current_length - len(source_info) - 50
                    if available > 100:
                        fragment = f"{source_info}\n{content[:available]}..."
                        context_parts.append(fragment)
                break
            
            context_parts.append(fragment)
            current_length += len(fragment)
        
        result_text = "\n\n".join(context_parts)
        logger.info(f"Построен контекст длиной {len(result_text)} символов из {len(context_parts)} фрагментов")
        return result_text
    
    def _get_image_paths(self, image_results: List[Dict]) -> List[str]:
        """Получение путей к изображениям."""
        paths = []
        for result in image_results:
            path = result.get('path', '')
            if path and os.path.exists(path):
                paths.append(path)
        return paths
    
    def _create_prompt(self, 
                      question: str, 
                      context_text: str, 
                      image_results: List[Dict]) -> str:
        """Создание промпта для LLM."""
        
        # Проверка наличия контекста
        if not context_text or context_text.strip() == "":
            context_text = "Контекст из документов не найден. Возможно, документы не были загружены или не содержат релевантной информации."
        
        system_prompt = f"""Ты — умный ассистент для анализа документов. Отвечай на вопросы на основе предоставленного контекста.

ПРАВИЛА:
1. Используй информацию из контекста документов как главный источник
2. Отвечай на том же языке, что и вопрос
3. Если вопрос — просьба пересказать или резюмировать, составь краткое и связное изложение всего контекста
4. Если спрашивают о конкретном факте, а его нет в контексте — честно скажи об этом
5. Давай развёрнутые ответы с конкретными деталями из документа
6. При возможности указывай номер страницы источника

КОНТЕКСТ ИЗ ДОКУМЕНТОВ:
{context_text}
"""
        
        if image_results:
            image_context = "\n\nИЗОБРАЖЕНИЯ В ДОКУМЕНТАХ:\n"
            for i, img in enumerate(image_results):
                ocr_text = img.get('ocr_text', '')
                if ocr_text:
                    image_context += f"Изображение {i+1} (страница {img.get('page', 'неизвестно')}): {ocr_text}\n"
            system_prompt += image_context
        
        user_prompt = f"\n\nВОПРОС ПОЛЬЗОВАТЕЛЯ: {question}\n\nОТВЕТ (на основе контекста из документов):"
        
        return system_prompt + user_prompt


# Пример использования
if __name__ == "__main__":
    # Пример инициализации
    # from vector_indexer import VectorIndexer
    # 
    # indexer = VectorIndexer()
    # llm = MultimodalLLM(model_type="openai", model_name="gpt-4o-mini")
    # pipeline = RAGPipeline(indexer, llm)
    # 
    # result = pipeline.generate_answer("Как решить квадратное уравнение?")
    # print(result['answer'])
    
    print("Модуль rag_pipeline готов к использованию")
