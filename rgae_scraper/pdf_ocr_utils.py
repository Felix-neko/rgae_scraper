import logging
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

import fitz
import ocrmypdf
import pikepdf
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def get_optimal_jobs() -> int:
    """
    Вычисляет оптимальное количество параллельных задач для OCR.
    Не менее 1, но не более 75% от доступных ядер.

    Returns:
        Количество параллельных задач
    """
    cpu_count = os.cpu_count() or 1
    max_jobs = max(1, int(cpu_count * 0.75))
    return max_jobs


def _fix_mediabox(input_pdf: Path, fixed_pdf: Path) -> None:
    """
    Нормализует MediaBox = CropBox для каждой страницы.

    В некоторых PDF MediaBox больше CropBox (реальной видимой области),
    из-за чего ocrmypdf рендерит огромный белый холст, и Tesseract
    не может найти текст. Эта функция устанавливает MediaBox = CropBox
    и удаляет CropBox/TrimBox/ArtBox/BleedBox (они становятся не нужны).

    Args:
        input_pdf: Путь к исходному PDF
        fixed_pdf: Путь для сохранения нормализованного PDF
    """
    pdf = pikepdf.open(input_pdf)
    fixed_count = 0
    for page in pdf.pages:
        crop_box = page.get("/CropBox")
        media_box = page.get("/MediaBox")
        if crop_box is not None and media_box is not None:
            # Сравниваем значения MediaBox и CropBox
            mb = [float(x) for x in media_box]
            cb = [float(x) for x in crop_box]
            if mb != cb:
                page["/MediaBox"] = crop_box
                # Удаляем лишние box-ы — они теперь совпадают с MediaBox
                for box_key in ["/CropBox", "/TrimBox", "/ArtBox", "/BleedBox"]:
                    if box_key in page:
                        del page[box_key]
                fixed_count += 1
    pdf.save(fixed_pdf)
    pdf.close()
    if fixed_count > 0:
        logger.info(f"  Нормализован MediaBox на {fixed_count} страницах")


def process_pdf_with_ocr(
    input_pdf: Path,
    output_pdf: Path,
    language: str = "rus",
    skip_existing: bool = True,
    jobs: int | None = None,
    dpi: int = 300,
    pages: str | None = None,
    clean: bool = True,
    oversample: int = 600,
) -> bool:
    """
    Добавляет текстовый слой к PDF-документу с помощью ocrmypdf.

    Использует ocrmypdf, который сохраняет оригинальные изображения без перекодирования
    и добавляет невидимый текстовый слой на основе Tesseract OCR.

    Перед вызовом ocrmypdf нормализует MediaBox = CropBox, чтобы Tesseract
    корректно обрабатывал страницы (без пустого белого холста).

    Args:
        input_pdf: Путь к исходному PDF
        output_pdf: Путь для сохранения результата
        language: Язык распознавания (rus, eng и т.д.)
        skip_existing: Пропускать уже обработанные файлы
        jobs: Количество параллельных задач для Tesseract
        dpi: DPI для рендеринга страниц (по умолчанию 300)
        pages: Диапазон страниц для обработки (например "5-7"), None — все страницы
        clean: Удаление шума через unpaper перед OCR (не затрагивает финальное изображение)
        oversample: Повышение разрешения до указанного DPI перед OCR (0 — без повышения)

    Returns:
        True если обработка успешна, False иначе
    """
    if skip_existing and output_pdf.exists():
        logger.info(f"Пропускаю {input_pdf.name} — уже обработан")
        return True

    try:
        logger.info(f"Обрабатываю {input_pdf.name}...")

        if jobs is None:
            jobs = get_optimal_jobs()

        # Нормализуем MediaBox во временный файл
        with tempfile.TemporaryDirectory(prefix="ocr_fix_") as tmpdir:
            fixed_path = Path(tmpdir) / "fixed.pdf"
            _fix_mediabox(input_pdf, fixed_path)

            # language для ocrmypdf — список строк
            lang_list = [lang.strip() for lang in language.split("+")]

            # ocrmypdf добавляет текстовый слой, не перекодируя изображения
            # --output-type pdf — без принудительной конвертации в PDF/A (минимальные изменения)
            # --skip-text — пропускать страницы, где уже есть текст
            # --optimize 0 — без дополнительной оптимизации (сохраняем оригинал)
            # --clean — удаление шума через unpaper (только для OCR, не для финала)
            # --oversample — повышение разрешения для лучшего распознавания
            # --rotate-pages — автоопределение ориентации (для повёрнутых на 90° таблиц)
            kwargs: dict = dict(
                language=lang_list,
                jobs=jobs,
                output_type="pdf",
                skip_text=True,
                optimize=0,
                clean=clean,
                oversample=oversample,
                rotate_pages=True,
            )

            if pages is not None:
                kwargs["pages"] = pages

            ocrmypdf.ocr(fixed_path, output_pdf, **kwargs)

        logger.info(f"✓ Успешно обработан {input_pdf.name}")
        return True

    except ocrmypdf.exceptions.PriorOcrFoundError:
        logger.info(f"⊘ {input_pdf.name} — уже содержит текстовый слой, пропускаю")
        return True

    except Exception as e:
        logger.error(f"✗ Ошибка при обработке {input_pdf.name}: {e}")
        import traceback

        traceback.print_exc()
        return False


def verify_ocr_text(pdf_path: Path, page_num: int = 0) -> str:
    """
    Извлекает текст с указанной страницы PDF для проверки корректности OCR.

    Args:
        pdf_path: Путь к PDF-файлу
        page_num: Номер страницы (0-индексация)

    Returns:
        Извлечённый текст
    """
    doc = fitz.open(pdf_path)
    if page_num >= len(doc):
        doc.close()
        return ""
    page = doc[page_num]
    text = page.get_text()
    doc.close()
    return text


def process_directory(
    input_dir: Path,
    output_dir: Path,
    language: str = "rus",
    skip_existing: bool = False,
    jobs: int | None = None,
) -> tuple[int, int]:
    """
    Рекурсивно обрабатывает все PDF-файлы в директории и поддиректориях.

    Структура подпапок сохраняется: файл input_dir/sub/file.pdf
    будет сохранён как output_dir/sub/file.pdf.

    Args:
        input_dir: Корневая директория с исходными PDF
        output_dir: Корневая директория для сохранения результатов
        language: Язык распознавания
        skip_existing: Пропускать уже обработанные файлы
        jobs: Количество параллельных задач

    Returns:
        Кортеж (успешно обработано, всего файлов)
    """
    if not input_dir.exists():
        logger.error(f"Директория {input_dir} не существует")
        return 0, 0

    output_dir.mkdir(parents=True, exist_ok=True)

    # Рекурсивный поиск всех PDF во вложенных директориях
    pdf_files = sorted(input_dir.rglob("*.pdf"))
    if not pdf_files:
        logger.warning(f"В директории {input_dir} (рекурсивно) не найдено PDF-файлов")
        return 0, 0

    total = len(pdf_files)
    logger.info(f"Найдено {total} PDF-файлов (рекурсивно)")

    if jobs is None:
        jobs = get_optimal_jobs()

    logger.info(f"Используется {jobs} параллельных задач для OCR")

    success_count = 0
    pbar = tqdm(pdf_files, desc=f"Обработка PDF [0/{total}]", total=total)
    for pdf_file in pbar:
        # Сохраняем относительный путь от input_dir
        rel_path = pdf_file.relative_to(input_dir)
        output_file = output_dir / rel_path

        # Создаём промежуточные подпапки, если нужно
        output_file.parent.mkdir(parents=True, exist_ok=True)

        if process_pdf_with_ocr(pdf_file, output_file, language, skip_existing, jobs):
            success_count += 1

        # Обновляем описание прогресс-бара с текущим счётчиком
        pbar.set_description(f"Обработка PDF [{success_count}/{total}]")

    return success_count, total


def test_on_sample_pages():
    """
    Тестовый запуск: обрабатывает несколько страниц из середины первого PDF
    и проверяет, что русский текст корректно распознан.
    """
    script_dir = Path(__file__).parent
    input_dir = script_dir / "ocr_src"
    output_dir = script_dir / "ocr_dest"
    output_dir.mkdir(parents=True, exist_ok=True)

    pdf_files = sorted(input_dir.glob("*.pdf"))
    if not pdf_files:
        logger.error("Нет PDF-файлов для тестирования")
        return

    test_pdf = pdf_files[0]
    test_output = output_dir / f"_test_{test_pdf.name}"

    # Удаляем предыдущий тестовый файл, если есть
    if test_output.exists():
        test_output.unlink()

    logger.info(f"Тестовый запуск на {test_pdf.name}, страницы 5-7")

    # Обрабатываем только страницы 5-7 из середины
    success = process_pdf_with_ocr(
        input_pdf=test_pdf,
        output_pdf=test_output,
        language="rus",
        skip_existing=False,
        pages="5-7",
    )

    if not success:
        logger.error("Тестовый запуск провалился!")
        return

    # Проверяем размеры файлов
    original_size = test_pdf.stat().st_size
    result_size = test_output.stat().st_size
    logger.info(f"Размер оригинала: {original_size / 1024 / 1024:.1f} МБ")
    logger.info(f"Размер результата: {result_size / 1024 / 1024:.1f} МБ")
    logger.info(f"Соотношение: {result_size / original_size:.2f}x")

    # Извлекаем текст со страниц 5-7 (0-индексация: 4-6)
    for page_idx in [4, 5, 6]:
        text = verify_ocr_text(test_output, page_idx)
        preview = text[:300].strip()
        has_cyrillic = any("\u0400" <= ch <= "\u04ff" for ch in text)
        logger.info(f"--- Страница {page_idx + 1} (кириллица: {'✓' if has_cyrillic else '✗'}) ---")
        logger.info(f"{preview}")

    # Удаляем тестовый файл
    test_output.unlink()
    logger.info("Тестовый файл удалён")


def main():
    """Основная функция для запуска обработки."""
    # Определяем пути относительно текущего файла
    script_dir = Path(__file__).parent
    # input_dir = script_dir / "ocr_src"
    input_dir = Path("/mnt/dump3/DOWN/Плановое хозяйство (1931-1989)")
    output_dir = Path("/mnt/dump3/DOWN/Плановое хозяйство (1931-1989) [распознанное]")

    logger.info(f"Исходная директория: {input_dir}")
    logger.info(f"Целевая директория: {output_dir}")

    success, total = process_directory(
        input_dir=input_dir,
        output_dir=output_dir,
        language="rus",
        skip_existing=True,
    )

    logger.info(f"\n{'=' * 50}")
    logger.info(f"Обработка завершена: {success}/{total} файлов успешно обработано")
    logger.info(f"{'=' * 50}")


if __name__ == "__main__":
    main()
