"""
Главное приложение Streamlit для RAG системы.
"""

import streamlit as st
import os
import tempfile
import uuid
from datetime import datetime
import logging

from document_processor import DocumentProcessor
from vector_indexer import VectorIndexer
from rag_pipeline import RAGPipeline, MultimodalLLM

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Загружаем API ключи из Streamlit Secrets (если запущено на Streamlit Cloud)
try:
    for key in ("GROQ_API_KEY", "OPENAI_API_KEY"):
        if key in st.secrets and not os.getenv(key):
            os.environ[key] = st.secrets[key]
except Exception:
    pass

st.set_page_config(
    page_title="RAG Система для Документов",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Цвета работают и в тёмной, и в светлой теме — используем полупрозрачный фон + inherit текст
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .upload-section {
        padding: 1.5rem;
        border-radius: 10px;
        margin-bottom: 2rem;
        border: 1px solid rgba(128,128,128,0.2);
    }
    .chat-message {
        padding: 1rem;
        border-radius: 10px;
        margin-bottom: 1rem;
        color: inherit;
    }
    .user-message {
        background-color: rgba(33, 150, 243, 0.12);
        border-left: 4px solid #2196f3;
        color: inherit;
    }
    .assistant-message {
        background-color: rgba(76, 175, 80, 0.1);
        border-left: 4px solid #4caf50;
        color: inherit;
    }
    .source-info {
        padding: 0.5rem;
        border-radius: 5px;
        margin-top: 0.5rem;
        font-size: 0.9rem;
        border: 1px solid rgba(128,128,128,0.2);
    }
</style>
""", unsafe_allow_html=True)

# Лучшая модель для RAG на русском: llama-3.3-70b-versatile
# - 70B параметров, 128k контекст, хорошо понимает русский язык
# - Бесплатна через Groq API

MODEL_OPTIONS = {
    "groq": [
        "llama-3.3-70b-versatile",
        "llama-3.1-8b-instant",
        "mixtral-8x7b-32768",
        "gemma2-9b-it",
    ],
    "openai": ["gpt-4o-mini", "gpt-4o", "gpt-4-turbo"],
    "ollama": ["llama3.2", "llama3.1", "mistral", "gemma2", "phi3", "llava"],
    "local":  ["local-model"],
}

# Описания моделей: что умеет, для чего подходит
MODEL_INFO = {
    "llama-3.3-70b-versatile": {
        "desc": "⭐ Лучший для текстов, литературы, отчётов. 128k контекст. Хорошо понимает русский язык.",
        "math_warning": True,
    },
    "llama-3.1-8b-instant": {
        "desc": "⚡ Самый быстрый, но слабее качеством. Подходит для простых вопросов и коротких документов.",
        "math_warning": True,
    },
    "mixtral-8x7b-32768": {
        "desc": "🔢 Лучший выбор для технических и математических документов. Контекст 32k токенов.",
        "math_warning": False,
    },
    "gemma2-9b-it": {
        "desc": "🤖 Модель Google. Хорош для структурированных документов и таблиц.",
        "math_warning": True,
    },
    "gpt-4o-mini": {
        "desc": "💰 Платный (OpenAI). Хорошее качество/цена. Поддерживает изображения напрямую.",
        "math_warning": False,
    },
    "gpt-4o": {
        "desc": "💰 Платный (OpenAI). Самый мощный. Лучшее понимание формул и изображений.",
        "math_warning": False,
    },
    "gpt-4-turbo": {
        "desc": "💰 Платный (OpenAI). Быстрый GPT-4, длинный контекст.",
        "math_warning": False,
    },
    "llama3.2": {
        "desc": "🏠 Локально. Лёгкий (3B), быстрый старт. Для коротких документов.",
        "math_warning": True,
    },
    "llama3.1": {
        "desc": "🏠 Локально. Сбалансированный (8B). Хорош для русскоязычных текстов.",
        "math_warning": True,
    },
    "mistral": {
        "desc": "🏠 Локально. Хорошо справляется с техническими и структурированными текстами.",
        "math_warning": False,
    },
    "gemma2": {
        "desc": "🏠 Локально. Модель Google, стабильная для аналитики.",
        "math_warning": True,
    },
    "phi3": {
        "desc": "🏠 Локально. Очень лёгкий (3.8B), быстрый. Для простых запросов.",
        "math_warning": True,
    },
    "llava": {
        "desc": "🏠 Локально + изображения. Единственная локальная модель, которая видит картинки напрямую.",
        "math_warning": False,
    },
    "local-model": {
        "desc": "🔧 Заглушка для тестирования без реальной модели.",
        "math_warning": False,
    },
}


def get_indexer():
    """Получает или создаёт VectorIndexer в session_state."""
    if 'indexer' not in st.session_state:
        st.session_state.indexer = VectorIndexer()
    return st.session_state.indexer


def initialize_pipeline(llm_type: str, model_name: str, api_key: str = None):
    """Инициализация pipeline (без пересоздания indexer)."""
    try:
        if 'processor' not in st.session_state:
            st.session_state.processor = DocumentProcessor(output_dir="temp_extracted")

        indexer = get_indexer()

        llm = MultimodalLLM(
            model_type=llm_type,
            model_name=model_name,
            api_key=api_key or None
        )
        pipeline = RAGPipeline(indexer, llm)
        return pipeline

    except Exception as e:
        logger.error(f"Ошибка инициализации pipeline: {e}")
        st.error(f"❌ Ошибка инициализации: {e}")
        return None


def main():
    st.markdown('<h1 class="main-header">📚 RAG Система для Документов</h1>',
                unsafe_allow_html=True)

    # Инициализация session_state
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    if 'documents' not in st.session_state:
        st.session_state.documents = {}   # doc_id -> {filename, pages, ...}
    if 'api_keys' not in st.session_state:
        st.session_state.api_keys = {}    # llm_type -> api_key
    if 'selected_doc_id' not in st.session_state:
        st.session_state.selected_doc_id = None

    # ─── Боковая панель ───────────────────────────────────────────────────────
    with st.sidebar:
        st.header("⚙️ Настройки модели")

        llm_type = st.selectbox(
            "Провайдер:",
            ["groq", "ollama", "openai", "local"],
            help="Groq — рекомендуется (бесплатно, быстро)"
        )

        model_name = st.selectbox("Модель:", MODEL_OPTIONS[llm_type])

        # Карточка с описанием выбранной модели
        info = MODEL_INFO.get(model_name, {})
        if info.get("desc"):
            st.caption(info["desc"])
        if info.get("math_warning"):
            st.warning(
                "⚠️ **Математические документы**\n\n"
                "Эта модель плохо работает с формулами и уравнениями.\n"
                "Для учебников по математике/физике выберите:\n"
                "- **Groq → mixtral-8x7b-32768**\n"
                "- **OpenAI → gpt-4o**\n"
                "- **Ollama → mistral**"
            )

        # API ключ — запоминается отдельно для каждого провайдера
        api_key = None
        if llm_type in ("groq", "openai"):
            label = "Groq API Key:" if llm_type == "groq" else "OpenAI API Key:"
            env_var = "GROQ_API_KEY" if llm_type == "groq" else "OPENAI_API_KEY"
            saved_key = st.session_state.api_keys.get(llm_type, os.getenv(env_var, ""))

            if llm_type == "groq":
                st.info("💡 Бесплатный ключ: console.groq.com")

            api_key = st.text_input(label, type="password", value=saved_key,
                                    help="Ключ сохраняется автоматически при переключении между провайдерами")
            if api_key:
                st.session_state.api_keys[llm_type] = api_key
                os.environ[env_var] = api_key

        elif llm_type == "ollama":
            st.info("💡 Убедитесь, что Ollama запущен:\n`ollama pull llama3.2`")

        # Слайдер количества фрагментов для поиска
        st.divider()
        st.caption("🔍 Параметры поиска")
        text_limit = st.slider(
            "Фрагментов для поиска:",
            min_value=3, max_value=20, value=7,
            help=(
                "Сколько отрывков документа передаётся модели.\n\n"
                "• 3–7 — быстро, для конкретных вопросов\n"
                "• 10–15 — для пересказа и анализа\n"
                "• 15–20 — для больших книг и романов"
            )
        )
        st.session_state.text_limit = text_limit

        # Проверяем нужна ли переинициализация pipeline
        need_reinit = (
            'pipeline' not in st.session_state
            or st.session_state.get('_llm_type') != llm_type
            or st.session_state.get('_model_name') != model_name
            or st.session_state.get('_api_key') != api_key
        )

        if need_reinit:
            pipeline = initialize_pipeline(llm_type, model_name, api_key)
            if pipeline:
                st.session_state.pipeline = pipeline
                st.session_state._llm_type = llm_type
                st.session_state._model_name = model_name
                st.session_state._api_key = api_key

        st.divider()

        # ── Список документов с возможностью выбора и удаления ───────────────
        st.header("📄 Документы")

        if st.session_state.documents:
            doc_names = {doc_id: info['filename']
                         for doc_id, info in st.session_state.documents.items()}

            options = ["🗂 Все документы"] + list(doc_names.values())
            selected_label = st.radio(
                "Поиск по:",
                options,
                index=0,
                help="Выберите конкретный документ, чтобы нейронка отвечала только по нему"
            )

            if selected_label == "🗂 Все документы":
                st.session_state.selected_doc_id = None
            else:
                for did, fname in doc_names.items():
                    if fname == selected_label:
                        st.session_state.selected_doc_id = did
                        break

            # Статистика выбранного документа
            sel_id = st.session_state.selected_doc_id
            if sel_id and sel_id in st.session_state.documents:
                info = st.session_state.documents[sel_id]
                st.caption(
                    f"📊 Страниц: {info.get('pages', '?')} | "
                    f"Чанков: {info.get('text_blocks', '?')} | "
                    f"Изображений: {info.get('images', '?')}"
                )

            st.divider()
            st.caption("Удалить документ:")
            for doc_id, info in list(st.session_state.documents.items()):
                col1, col2 = st.columns([3, 1])
                with col1:
                    st.caption(info['filename'])
                with col2:
                    if st.button("✕", key=f"del_{doc_id}",
                                 help="Удалить этот документ из базы"):
                        get_indexer().delete_document(doc_id)
                        del st.session_state.documents[doc_id]
                        if st.session_state.selected_doc_id == doc_id:
                            st.session_state.selected_doc_id = None
                        st.rerun()
        else:
            st.write("Документы не загружены")

        st.divider()
        col_a, col_b = st.columns(2)
        with col_a:
            if st.button(
                "💬 Очистить чат",
                help="Удаляет историю переписки. Документы остаются загруженными."
            ):
                st.session_state.messages = []
                st.rerun()
        with col_b:
            if st.button(
                "🗑️ Очистить всё",
                help="Удаляет все загруженные документы из базы и очищает чат. Потребуется загрузить документы заново."
            ):
                get_indexer().clear_all_documents()
                st.session_state.messages = []
                st.session_state.documents = {}
                st.session_state.selected_doc_id = None
                st.rerun()

    # ─── Загрузка документов ──────────────────────────────────────────────────
    st.markdown('<div class="upload-section">', unsafe_allow_html=True)
    st.header("📤 Загрузка документов")
    uploaded_files = st.file_uploader(
        "Выберите документы (PDF, DOCX):",
        type=['pdf', 'docx'],
        accept_multiple_files=True,
        help=(
            "Поддерживаемые форматы: PDF, DOCX. Максимум: 1 ГБ.\n\n"
            "После загрузки документ автоматически индексируется — "
            "нейронка сможет отвечать на вопросы по его содержимому."
        )
    )
    if uploaded_files:
        existing_names = {info['filename'] for info in st.session_state.documents.values()}
        for uploaded_file in uploaded_files:
            if uploaded_file.name not in existing_names:
                process_uploaded_file(uploaded_file)
    st.markdown('</div>', unsafe_allow_html=True)

    # Показываем по какому документу идёт поиск
    if st.session_state.selected_doc_id:
        doc_name = st.session_state.documents[st.session_state.selected_doc_id]['filename']
        st.info(f"🔍 Поиск по документу: **{doc_name}**")
    elif st.session_state.documents:
        st.info(f"🔍 Поиск по всем документам ({len(st.session_state.documents)} шт.)")

    # ─── Чат ──────────────────────────────────────────────────────────────────
    st.header("💬 Задайте вопрос по документам")

    for message in st.session_state.messages:
        display_message(message)

    if prompt := st.chat_input("Введите ваш вопрос..."):
        if not st.session_state.documents:
            st.warning("⚠️ Сначала загрузите документы!")
        elif 'pipeline' not in st.session_state:
            st.error("❌ Система не инициализирована. Проверьте настройки.")
        else:
            handle_user_question(prompt)


def process_uploaded_file(uploaded_file):
    """Обработка загруженного файла."""
    try:
        with st.spinner(f"Обработка {uploaded_file.name}..."):
            suffix = "." + uploaded_file.name.rsplit('.', 1)[-1]
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                tmp_path = tmp_file.name

            extracted = st.session_state.processor.process_document(tmp_path)

            if extracted['text'] or extracted['images']:
                doc_id = str(uuid.uuid4())
                get_indexer().index_documents(extracted, doc_id)

                st.session_state.documents[doc_id] = {
                    'filename': uploaded_file.name,
                    'upload_time': datetime.now(),
                    'pages': extracted['metadata'].get('total_pages', len(extracted['text'])),
                    'text_blocks': len(extracted['text']),
                    'images': len(extracted['images'])
                }

                st.success(f"✅ {uploaded_file.name} — загружен")
                st.info(f"📊 {len(extracted['text'])} текстовых блоков, {len(extracted['images'])} изображений")
            else:
                st.error(f"❌ Не удалось извлечь контент из {uploaded_file.name}")

            os.unlink(tmp_path)

    except Exception as e:
        st.error(f"❌ Ошибка: {e}")
        logger.error(f"Ошибка обработки файла: {e}")


def handle_user_question(question: str):
    """Обработка вопроса пользователя."""
    st.session_state.messages.append({
        "role": "user",
        "content": question,
        "timestamp": datetime.now()
    })
    display_message(st.session_state.messages[-1])

    with st.spinner("Думаю..."):
        try:
            result = st.session_state.pipeline.generate_answer(
                question,
                text_limit=st.session_state.get('text_limit', 7),
                document_id=st.session_state.selected_doc_id
            )
            msg = {
                "role": "assistant",
                "content": result['answer'],
                "sources": result['sources'],
                "context_used": result['context_used'],
                "timestamp": datetime.now()
            }
            st.session_state.messages.append(msg)
            display_message(msg)
        except Exception as e:
            msg = {
                "role": "assistant",
                "content": f"Ошибка при генерации ответа: {e}",
                "timestamp": datetime.now()
            }
            st.session_state.messages.append(msg)
            display_message(msg)
            logger.error(f"Ошибка генерации: {e}")


def display_message(message: dict):
    """Отображение сообщения в чате."""
    role = message["role"]
    content = message["content"]
    ts = message.get("timestamp", datetime.now()).strftime("%H:%M")

    if role == "user":
        st.markdown(f'''
        <div class="chat-message user-message">
            <strong>👤 Вы ({ts}):</strong><br>{content}
        </div>''', unsafe_allow_html=True)
    else:
        st.markdown(f'''
        <div class="chat-message assistant-message">
            <strong>🤖 Ассистент ({ts}):</strong><br>{content}
        </div>''', unsafe_allow_html=True)

        if 'sources' in message:
            sources = message['sources']
            ctx = message.get('context_used', {})
            if sources['text_sources'] or sources['image_sources']:
                label = (f"📚 Источники — {ctx.get('text_fragments', 0)} текст."
                         f" / {ctx.get('images', 0)} изобр.")
                with st.expander(label):
                    if sources['text_sources']:
                        st.subheader("📄 Текстовые источники:")
                        for i, src in enumerate(sources['text_sources'], 1):
                            st.markdown(f'''
                            <div class="source-info">
                                <strong>Источник {i}</strong>
                                (стр. {src.get('page', '?')}, релевантность: {src.get('score', 0):.2f})<br>
                                <em>{src.get('content', '')[:300]}…</em>
                            </div>''', unsafe_allow_html=True)

                    if sources['image_sources']:
                        st.subheader("🖼️ Изображения:")
                        for i, img in enumerate(sources['image_sources'], 1):
                            c1, c2 = st.columns([1, 3])
                            with c1:
                                if os.path.exists(img.get('path', '')):
                                    st.image(img['path'], width=150)
                            with c2:
                                st.markdown(f'''
                                <div class="source-info">
                                    <strong>Изображение {i}</strong>
                                    (стр. {img.get('page', '?')}, релевантность: {img.get('score', 0):.2f})<br>
                                    <strong>Файл:</strong> {img.get('filename', '?')}<br>
                                    <strong>OCR:</strong> {img.get('ocr_text', '—')[:150]}
                                </div>''', unsafe_allow_html=True)


if __name__ == "__main__":
    main()
