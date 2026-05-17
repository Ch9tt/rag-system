"""
Модуль для обработки документов (PDF и DOCX) с извлечением текста и изображений.
Текст разбивается на перекрывающиеся чанки для лучшего retrieval на больших документах.
"""

import os
import fitz  # PyMuPDF
import pytesseract
from PIL import Image
from docx import Document
from docx.oxml.table import CT_Tbl
from docx.oxml.text.paragraph import CT_P
from docx.table import Table
from docx.text.paragraph import Paragraph
import zipfile
from typing import List, Dict, Optional, Any
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Размер чанка в словах и перекрытие
CHUNK_SIZE = 600
CHUNK_OVERLAP = 80


class DocumentProcessor:
    """Класс для обработки документов различных форматов."""

    def __init__(self, output_dir: str = "extracted_content"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    # ─── Чанкинг ──────────────────────────────────────────────────────────────

    def _chunk_text(self, text: str, page: int = 0,
                    chunk_type: str = "direct_text") -> List[Dict]:
        """
        Разбивает текст на перекрывающиеся чанки по ~CHUNK_SIZE слов.
        Короткие тексты возвращаются одним блоком без разбивки.
        """
        words = text.split()
        if len(words) <= CHUNK_SIZE:
            return [{'page': page, 'content': text, 'type': chunk_type}]

        chunks = []
        start = 0
        while start < len(words):
            end = min(start + CHUNK_SIZE, len(words))
            chunk_text = ' '.join(words[start:end])
            chunks.append({'page': page, 'content': chunk_text, 'type': chunk_type})
            if end == len(words):
                break
            start += CHUNK_SIZE - CHUNK_OVERLAP

        return chunks

    # ─── PDF ──────────────────────────────────────────────────────────────────

    def process_pdf(self, pdf_path: str) -> Dict[str, Any]:
        """Обработка PDF: текст разбивается на чанки, изображения OCR."""
        try:
            doc = fitz.open(pdf_path)
            extracted_content = {
                'text': [],
                'images': [],
                'metadata': {
                    'total_pages': len(doc),
                    'title': doc.metadata.get('title', ''),
                    'author': doc.metadata.get('author', ''),
                    'subject': doc.metadata.get('subject', '')
                }
            }

            for page_num in range(len(doc)):
                page = doc[page_num]

                # Извлечение и чанкинг текста
                text = page.get_text()
                if text.strip():
                    chunks = self._chunk_text(text.strip(), page=page_num + 1)
                    extracted_content['text'].extend(chunks)

                # Извлечение изображений
                for img_index, img in enumerate(page.get_images()):
                    try:
                        xref = img[0]
                        pix = fitz.Pixmap(doc, xref)
                        if pix.n - pix.alpha < 4:  # GRAY или RGB
                            img_filename = f"page_{page_num + 1}_img_{img_index + 1}.png"
                            img_path = os.path.join(self.output_dir, img_filename)
                            pix.save(img_path)
                            ocr_text = self._extract_text_from_image(img_path)
                            extracted_content['images'].append({
                                'page': page_num + 1,
                                'filename': img_filename,
                                'path': img_path,
                                'ocr_text': ocr_text,
                                'type': 'image'
                            })
                        pix = None
                    except Exception as e:
                        logger.warning(f"Ошибка изображения на стр.{page_num + 1}: {e}")

            doc.close()
            logger.info(f"PDF обработан: {len(extracted_content['text'])} чанков, "
                        f"{len(extracted_content['images'])} изображений")
            return extracted_content

        except Exception as e:
            logger.error(f"Ошибка обработки PDF {pdf_path}: {e}")
            return {'text': [], 'images': [], 'metadata': {}}

    # ─── DOCX ─────────────────────────────────────────────────────────────────

    def process_docx(self, docx_path: str) -> Dict[str, Any]:
        """Обработка DOCX: параграфы накапливаются и чанкуются."""
        try:
            doc = Document(docx_path)
            extracted_content = {
                'text': [],
                'images': [],
                'metadata': {
                    'title': doc.core_properties.title or '',
                    'author': doc.core_properties.author or '',
                    'subject': doc.core_properties.subject or ''
                }
            }

            # Накапливаем весь текст DOCX, потом чанкуем
            full_text_parts = []
            for element in doc.element.body:
                if isinstance(element, CT_P):
                    paragraph = Paragraph(element, doc)
                    if paragraph.text.strip():
                        full_text_parts.append(paragraph.text.strip())
                elif isinstance(element, CT_Tbl):
                    table = Table(element, doc)
                    table_text = self._extract_table_text(table)
                    if table_text.strip():
                        # Таблицы — отдельный блок, не разбиваем
                        extracted_content['text'].append({
                            'content': table_text,
                            'type': 'table',
                            'page': 0
                        })

            # Чанкуем весь накопленный текст
            if full_text_parts:
                full_text = '\n'.join(full_text_parts)
                chunks = self._chunk_text(full_text, page=0, chunk_type='paragraph')
                extracted_content['text'].extend(chunks)

            # Изображения
            images = self._extract_images_from_docx(docx_path)
            for img_index, img_data in enumerate(images):
                img_filename = f"docx_img_{img_index + 1}.png"
                img_path = os.path.join(self.output_dir, img_filename)
                with open(img_path, 'wb') as f:
                    f.write(img_data)
                ocr_text = self._extract_text_from_image(img_path)
                extracted_content['images'].append({
                    'filename': img_filename,
                    'path': img_path,
                    'ocr_text': ocr_text,
                    'type': 'image',
                    'page': 0
                })

            logger.info(f"DOCX обработан: {len(extracted_content['text'])} чанков, "
                        f"{len(extracted_content['images'])} изображений")
            return extracted_content

        except Exception as e:
            logger.error(f"Ошибка обработки DOCX {docx_path}: {e}")
            return {'text': [], 'images': [], 'metadata': {}}

    # ─── Вспомогательные ──────────────────────────────────────────────────────

    def _extract_table_text(self, table: Table) -> str:
        rows = []
        for row in table.rows:
            rows.append(" | ".join(cell.text.strip() for cell in row.cells))
        return "\n".join(rows)

    def _extract_images_from_docx(self, docx_path: str) -> List[bytes]:
        images = []
        try:
            with zipfile.ZipFile(docx_path, 'r') as zf:
                for info in zf.filelist:
                    if info.filename.startswith('word/media/'):
                        if any(info.filename.lower().endswith(ext)
                               for ext in ['.png', '.jpg', '.jpeg', '.gif', '.bmp']):
                            images.append(zf.read(info.filename))
        except Exception as e:
            logger.warning(f"Ошибка извлечения изображений из DOCX: {e}")
        return images

    def _extract_text_from_image(self, image_path: str) -> str:
        try:
            image = Image.open(image_path)
            text = pytesseract.image_to_string(image, config='--oem 3 --psm 6', lang='rus+eng')
            return text.strip()
        except Exception as e:
            logger.warning(f"Ошибка OCR для {image_path}: {e}")
            return ""

    def process_document(self, file_path: str) -> Dict[str, Any]:
        """Универсальная обработка: определяет формат по расширению."""
        ext = os.path.splitext(file_path)[1].lower()
        if ext == '.pdf':
            return self.process_pdf(file_path)
        elif ext == '.docx':
            return self.process_docx(file_path)
        else:
            logger.error(f"Неподдерживаемый формат: {ext}")
            return {'text': [], 'images': [], 'metadata': {}}


if __name__ == "__main__":
    print("Модуль document_processor готов к использованию")
