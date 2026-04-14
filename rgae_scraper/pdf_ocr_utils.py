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


SPREAD_ASPECT_THRESHOLD: float = 288.0 / 250.0  # ≈ 1.152: разворот vs обычная страница


class _TqdmLoggingHandler(logging.StreamHandler):
    """Перенаправляет log-сообщения через tqdm.write, чтобы не ломать прогресс-бары."""

    def emit(self, record: logging.LogRecord) -> None:
        try:
            tqdm.write(self.format(record))
        except Exception:
            self.handleError(record)


class _TesseractLogFilter(logging.Filter):
    """Фильтрует избыточные сообщения от Tesseract."""

    def filter(self, record: logging.LogRecord) -> bool:
        # Подавляем сообщения о rotation confidence и diacritics
        msg = record.getMessage()
        if "page is facing" in msg or "diacritics" in msg:
            return False
        # Подавляем "Too few characters" и "Error during processing"
        if "Too few characters" in msg or "Error during processing" in msg:
            return False
        return True


_root = logging.getLogger()
_root.setLevel(logging.INFO)
for _h in _root.handlers[:]:
    _root.removeHandler(_h)
_tqdm_handler = _TqdmLoggingHandler()
_tqdm_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
_tqdm_handler.addFilter(_TesseractLogFilter())
_root.addHandler(_tqdm_handler)
logger = logging.getLogger(__name__)

# Настраиваем логирование ocrmypdf и tesseract
for logger_name in ["ocrmypdf", "ocrmypdf.exec.tesseract", "ocrmypdf._exec.tesseract"]:
    ocr_logger = logging.getLogger(logger_name)
    ocr_logger.setLevel(logging.WARNING)  # Показываем только WARNING и выше
    ocr_logger.handlers = []
    ocr_logger.addHandler(_tqdm_handler)
    ocr_logger.propagate = False


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


def _restore_original_images(source_pdf: Path, ocr_pdf: Path, output_pdf: Path) -> None:
    """
    Заменяет изображения в OCR-версии на оригиналы из source_pdf.

    Используется при oversample > 0: ocrmypdf рендерит страницы на высоком DPI для Tesseract
    и встраивает апсемплированные картинки в PDF. Эта функция берёт текстовые слои из
    ocr_pdf, но картинки подменяет оригинальными из source_pdf.

    Это работает корректно, потому что текстовый слой хранится в координатах PDF (pt),
    а не в пикселях — позиции слов не зависят от разрешения встроенного изображения.

    Args:
        source_pdf: PDF с оригинальными (нетронутыми) изображениями
        ocr_pdf: PDF с текстовым слоем после ocrmypdf (может содержать апсемплированные картинки)
        output_pdf: Путь для сохранения результата
    """
    src = pikepdf.open(source_pdf)
    ocr = pikepdf.open(ocr_pdf)
    replaced = 0

    for src_page, ocr_page in zip(src.pages, ocr.pages):
        # НЕ восстанавливаем MediaBox и CropBox — они уже нормализованы после fix_mediabox
        # Восстанавливаем только изображения, чтобы сохранить оригинальное качество
        
        # Восстанавливаем изображения
        src_res = src_page.get("/Resources") or pikepdf.Dictionary()
        src_xobj = src_res.get("/XObject") or {}
        ocr_res = ocr_page.get("/Resources") or pikepdf.Dictionary()
        ocr_xobj = ocr_res.get("/XObject") or {}

        # Собираем оригинальные Image XObjects из source по имени
        src_images = {k: v for k, v in src_xobj.items() if v.get("/Subtype") == "/Image"}
        if not src_images:
            continue

        for key in list(ocr_xobj.keys()):
            obj = ocr_xobj[key]
            if obj.get("/Subtype") != "/Image":
                continue
            # Берём оригинал с тем же именем, иначе — первый доступный
            src_img = src_images.get(key) or next(iter(src_images.values()))
            ocr_xobj[key] = ocr.copy_foreign(src_img)
            replaced += 1

    ocr.save(output_pdf)
    src.close()
    ocr.close()
    if replaced > 0:
        logger.info(f"  Восстановлены оригинальные картинки на {replaced} страницах")


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
                # Нормализуем CropBox к (0, 0) перед установкой в MediaBox
                x0, y0, x1, y1 = cb
                width = x1 - x0
                height = y1 - y0
                page["/MediaBox"] = pikepdf.Array([0, 0, width, height])
                # Удаляем лишние box-ы — они теперь совпадают с MediaBox
                for box_key in ["/CropBox", "/TrimBox", "/ArtBox", "/BleedBox"]:
                    if box_key in page:
                        del page[box_key]
                fixed_count += 1
    pdf.save(fixed_pdf)
    pdf.close()
    if fixed_count > 0:
        logger.info(f"  Нормализован MediaBox на {fixed_count} страницах")


def _split_landscape_pages(input_pdf: Path, output_pdf: Path) -> list[tuple[int, int, str]]:
    """
    Разбивает страницы-развороты на левую и правую половины без перекодирования изображений.

    Страница считается разворотом, если CropBox (или MediaBox) имеет width/height > SPREAD_ASPECT_THRESHOLD.
    Разбивка работает с оригинальными размерами: берёт CropBox как видимую область, делит его
    пополам по X, и для каждой половины устанавливает новый CropBox. MediaBox остаётся оригинальным
    (с полным изображением), что позволяет сохранить качество.

    Args:
        input_pdf: Входной PDF (может содержать MediaBox >> CropBox)
        output_pdf: Выходной PDF с разбитыми страницами

    Returns:
        page_map: список (out_idx, src_idx, side) где side — 'left', 'right' или 'full'
    """
    src = pikepdf.open(input_pdf)
    out = pikepdf.Pdf.new()
    page_map: list[tuple[int, int, str]] = []

    for src_idx, src_page in enumerate(src.pages):
        # Берём CropBox (видимая область) или MediaBox, если CropBox нет
        crop_box = src_page.get("/CropBox")
        media_box = src_page.get("/MediaBox")
        visible_box = crop_box if crop_box is not None else media_box
        
        if visible_box is None:
            # Страница без размеров — просто копируем
            out.pages.append(src_page)
            page_map.append((len(out.pages) - 1, src_idx, "full"))
            continue
        
        x0, y0, x1, y1 = [float(v) for v in visible_box]
        width = x1 - x0
        height = y1 - y0
        is_spread = width > height and (width / height) > SPREAD_ASPECT_THRESHOLD

        if is_spread:
            # Разворот — делим пополам по X
            mid = x0 + width / 2.0
            for half_x0, half_x1, side in [(x0, mid, "left"), (mid, x1, "right")]:
                out_idx = len(out.pages)
                out.pages.append(src_page)
                new_page = out.pages[out_idx]
                # Устанавливаем CropBox на половину, MediaBox оставляем оригинальным
                new_page["/CropBox"] = pikepdf.Array([half_x0, y0, half_x1, y1])
                # Удаляем лишние box-ы
                for box_key in ["/TrimBox", "/ArtBox", "/BleedBox"]:
                    if box_key in new_page:
                        del new_page[box_key]
                page_map.append((out_idx, src_idx, side))
        else:
            # Обычная страница — копируем как есть
            out_idx = len(out.pages)
            out.pages.append(src_page)
            page_map.append((out_idx, src_idx, "full"))

    out.save(output_pdf)
    src.close()

    spread_count = sum(1 for _, _, s in page_map if s == "left")
    if spread_count:
        logger.info(f"  Разворотов разбито: {spread_count}, страниц в результате: {len(page_map)}")
    return page_map


def process_pdf_with_ocr(
    input_pdf: Path,
    output_pdf: Path,
    language: str = "rus",
    skip_existing: bool = True,
    jobs: int | None = None,
    pages: str | None = None,
    page: int | None = None,
    start_with_one: bool = True,
    clean: bool = True,
    oversample: int = 600,
    deskew: bool = False,
    split_landscape: bool = False,
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
        pages: Диапазон страниц для обработки (например "5-7"), None — все страницы.
               Игнорируется при split_landscape=True (нумерация меняется после разбивки).
        page: Номер одной страницы для обработки. Перекрывает pages.
              Игнорируется при split_landscape=True.
        start_with_one: Если True (по умолчанию), page считается 1-based.
                        Если False — 0-based (конвертируется в 1-based внутри).
        clean: Удаление шума через unpaper перед OCR (не затрагивает финальное изображение)
        oversample: Повышение разрешения до указанного DPI перед OCR (0 — отключено).
                ВНИМАНИЕ: если значение > DPI исходника, ocrmypdf растеризует страницу
                заново и встраивает апсемплированную картинку в выходной PDF.
                Для сканов с оригинальным DPI=300 следует оставлять 0.
        deskew: Исправление наклона листа через unpaper перед OCR.
                ВНИМАНИЕ: в отличие от --clean, deskew ИЗМЕНЯЕТ финальное изображение
                (физически поворачивает и перекодирует JPEG). Если важно сохранить
                оригинальную картинку — держите False.
        split_landscape: Если True — перед OCR разбивает страницы-развороты на левую
                и правую половины (без перекодирования). Oversample и deskew применяются
                к каждой половине отдельно. Финальные изображения берутся из разбитого
                PDF (т.е. оригинальные сканы с MediaBox, обрезающим нужную половину).

    Returns:
        True если обработка успешна, False иначе
    """
    if skip_existing and output_pdf.exists():
        logger.info(f"Пропускаю {input_pdf.name} — уже обработан")
        return True

    try:
        # Определяем количество страниц для информации
        doc = fitz.open(input_pdf)
        total_pages = len(doc)
        doc.close()
        
        logger.info(f"Обрабатываю {input_pdf.name} ({total_pages} страниц)...")

        if jobs is None:
            jobs = get_optimal_jobs()

        # Определяем диапазон страниц (только для обычного режима, не split_landscape)
        effective_pages: str | None = None
        if page is not None:
            page_1based = page if start_with_one else page + 1
            effective_pages = str(page_1based)
        elif pages is not None:
            effective_pages = pages

        # Нормализуем MediaBox во временный файл
        with tempfile.TemporaryDirectory(prefix="ocr_fix_") as tmpdir:
            fixed_path = Path(tmpdir) / "fixed.pdf"
            _fix_mediabox(input_pdf, fixed_path)

            # language для ocrmypdf — список строк
            lang_list = [lang.strip() for lang in language.split("+")]

            # Базовые параметры ocrmypdf:
            # --output-type pdf — без принудительной конвертации в PDF/A
            # --skip-text — пропускать страницы, где уже есть текст
            # --optimize 0 — без дополнительной оптимизации (сохраняем оригинал)
            # --clean — удаление шума через unpaper (только для OCR, не для финала)
            # --deskew — исправление наклона через unpaper (МЕНЯЕТ финальную картинку)
            # --oversample — повышение разрешения для лучшего распознавания
            # --rotate-pages — автоопределение ориентации (для повёрнутых на 90° таблиц)
            kwargs: dict = dict(
                language=lang_list,
                jobs=jobs,
                output_type="pdf",
                skip_text=True,
                optimize=0,
                clean=clean,
                deskew=deskew,
                oversample=oversample,
                rotate_pages=True,
                progress_bar=False,  # Отключаем встроенный прогресс-бар
            )

            if split_landscape:
                # Пайплайн с разбивкой разворотов:
                # 1) Разбиваем развороты на левую+правую половины (работаем с оригиналом, устанавливаем CropBox)
                # 2) Нормализуем MediaBox=CropBox для разбитых страниц (для корректного рендеринга в OCR)
                # 3) OCR по разбитым страницам (oversample/deskew/rotate для каждой половины)
                # 4) Подменяем изображения в OCR-PDF на оригиналы из split_fixed (нормализованные координаты)
                
                # Шаг 1: Разбиваем развороты на оригинальном PDF (до fix_mediabox)
                split_path = Path(tmpdir) / "split.pdf"
                page_map = _split_landscape_pages(input_pdf, split_path)
                logger.info(f"  После разбивки: {len(page_map)} страниц")
                
                # Если указана конкретная страница — найдём её в page_map
                split_pages: str | None = None
                if effective_pages is not None:
                    # Парсим effective_pages (может быть "5" или "5-7")
                    if "-" in effective_pages:
                        start_str, end_str = effective_pages.split("-", 1)
                        src_pages = list(range(int(start_str) - 1, int(end_str)))
                    else:
                        src_pages = [int(effective_pages) - 1]
                    
                    # Находим соответствующие страницы в разбитом PDF
                    out_indices = [out_idx for out_idx, src_idx, _ in page_map if src_idx in src_pages]
                    if out_indices:
                        # Формируем диапазон для ocrmypdf (1-based)
                        out_indices_1based = [idx + 1 for idx in out_indices]
                        if len(out_indices_1based) == 1:
                            split_pages = str(out_indices_1based[0])
                        else:
                            # Формируем диапазон (может быть несколько несмежных страниц)
                            split_pages = ",".join(str(idx) for idx in out_indices_1based)
                        logger.info(
                            f"  Исходная страница {effective_pages} → разбитые страницы {split_pages}"
                        )
                
                # Шаг 2: Нормализуем MediaBox=CropBox для разбитых страниц
                split_fixed = Path(tmpdir) / "split_fixed.pdf"
                _fix_mediabox(split_path, split_fixed)
                
                # Шаг 3: OCR
                if split_pages is not None:
                    kwargs["pages"] = split_pages
                
                # При split_landscape НЕ восстанавливаем изображения,
                # потому что ocrmypdf уже правильно рендерит их по CropBox
                logger.info(f"  [{input_pdf.name}] Запуск OCR на {len(page_map)} страницах (oversample={oversample}, jobs={jobs})...")
                ocrmypdf.ocr(split_fixed, output_pdf, **kwargs)
                logger.info(f"  [{input_pdf.name}] OCR завершён")

            else:
                # Обычный пайплайн без разбивки
                if effective_pages is not None:
                    kwargs["pages"] = effective_pages

                if oversample > 0:
                    logger.info(f"  [{input_pdf.name}] Запуск OCR на {total_pages} страницах (oversample={oversample}, jobs={jobs})...")
                    ocr_hires = Path(tmpdir) / "ocr_hires.pdf"
                    ocrmypdf.ocr(fixed_path, ocr_hires, **kwargs)
                    logger.info(f"  [{input_pdf.name}] OCR завершён, восстановление оригинальных изображений...")
                    _restore_original_images(fixed_path, ocr_hires, output_pdf)
                else:
                    logger.info(f"  [{input_pdf.name}] Запуск OCR на {total_pages} страницах (jobs={jobs})...")
                    ocrmypdf.ocr(fixed_path, output_pdf, **kwargs)
                    logger.info(f"  [{input_pdf.name}] OCR завершён")

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
    clean: bool = True,
    oversample: int = 600,
    deskew: bool = False,
    split_landscape: bool = False,
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
        clean: Удаление шума через unpaper перед OCR
        oversample: Повышение разрешения до указанного DPI перед OCR (0 — отключено)
        deskew: Исправление наклона листа перед OCR (МЕНЯЕТ финальное изображение)
        split_landscape: Разбивать страницы-развороты на левую и правую половины

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

        if process_pdf_with_ocr(
            pdf_file,
            output_file,
            language,
            skip_existing,
            jobs,
            clean=clean,
            oversample=oversample,
            deskew=deskew,
            split_landscape=split_landscape,
        ):
            success_count += 1

        # Обновляем описание прогресс-бара с текущим счётчиком
        pbar.set_description(f"Обработка PDF [{success_count}/{total}]")

    return success_count, total


def _extract_pages_to_pdf(input_pdf: Path, output_pdf: Path, page_indices: list[int]) -> None:
    """
    Извлекает указанные страницы (0-based) в отдельный PDF без перекодирования.

    Args:
        input_pdf: Исходный PDF
        output_pdf: Путь для сохранения результата
        page_indices: Список 0-based индексов страниц
    """
    src = pikepdf.open(input_pdf)
    out = pikepdf.Pdf.new()
    for idx in page_indices:
        out.pages.append(src.pages[idx])
    out.save(output_pdf)
    src.close()


def test_on_sample_pages():
    """
    Тестовый запуск: обрабатывает несколько страниц из середины первого PDF
    и проверяет, что русский текст корректно распознан.
    """
    input_dir = Path(__file__).parent / "ocr_src"
    output_dir = Path(__file__).parent / "ocr_dest"
    output_dir.mkdir(parents=True, exist_ok=True)

    pdf_files = sorted(input_dir.glob("*.pdf"))
    if not pdf_files:
        logger.error("Нет PDF-файлов для тестирования")
        return

    test_pdf = pdf_files[0]
    test_output = output_dir / f"_test_{test_pdf.name}"

    if test_output.exists():
        test_output.unlink()

    logger.info(f"Тестовый запуск на {test_pdf.name}, страницы 5-7")

    success = process_pdf_with_ocr(
        input_pdf=test_pdf, output_pdf=test_output, language="rus", skip_existing=False, pages="5-7"
    )

    if not success:
        logger.error("Тестовый запуск провалился!")
        return

    original_size = test_pdf.stat().st_size
    result_size = test_output.stat().st_size
    logger.info(f"Размер оригинала: {original_size / 1024 / 1024:.1f} МБ")
    logger.info(f"Размер результата: {result_size / 1024 / 1024:.1f} МБ")
    logger.info(f"Соотношение: {result_size / original_size:.2f}x")

    for page_idx in [4, 5, 6]:
        text = verify_ocr_text(test_output, page_idx)
        preview = text[:300].strip()
        has_cyrillic = any("\u0400" <= ch <= "\u04ff" for ch in text)
        logger.info(f"--- Страница {page_idx + 1} (кириллица: {'✓' if has_cyrillic else '✗'}) ---")
        logger.info(f"{preview}")

    test_output.unlink()
    logger.info("Тестовый файл удалён")


def test_split_on_page(
    source_pdf: Path,
    page_idx: int = 122,
    output_dir: Path | None = None,
) -> None:
    """
    Тест разбивки разворота: обрабатывает одну страницу двумя способами
    (с разбивкой и без) и сравнивает качество OCR по количеству кириллических символов.

    Args:
        source_pdf: Исходный PDF
        page_idx: 0-based индекс страницы (по умолчанию 122)
        output_dir: Директория для сохранения результатов (по умолчанию рядом с source_pdf)
    """
    if not source_pdf.exists():
        logger.error(f"Файл не найден: {source_pdf}")
        return

    src_doc = fitz.open(source_pdf)
    total_pages = len(src_doc)
    src_doc.close()

    if page_idx >= total_pages:
        logger.error(f"Страница {page_idx} не существует в {source_pdf.name} ({total_pages} страниц)")
        return

    if output_dir is None:
        output_dir = source_pdf.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    stem = source_pdf.stem
    out_nosplit = output_dir / f"_test_{stem}_p{page_idx}_nosplit.pdf"
    out_split = output_dir / f"_test_{stem}_p{page_idx}_split.pdf"
    for p in [out_nosplit, out_split]:
        if p.exists():
            p.unlink()

    # Извлекаем только нужную страницу во временный PDF
    with tempfile.TemporaryDirectory(prefix="ocr_test_") as tmpdir:
        single_page_pdf = Path(tmpdir) / "single_page.pdf"
        _extract_pages_to_pdf(source_pdf, single_page_pdf, [page_idx])

        logger.info(f"=== Тест без разбивки: {source_pdf.name}, страница {page_idx} ===")
        process_pdf_with_ocr(
            input_pdf=single_page_pdf,
            output_pdf=out_nosplit,
            language="rus",
            skip_existing=False,
            split_landscape=False,
        )

        logger.info(f"=== Тест с разбивкой разворота: {source_pdf.name}, страница {page_idx} ===")
        process_pdf_with_ocr(
            input_pdf=single_page_pdf,
            output_pdf=out_split,
            language="rus",
            skip_existing=False,
            split_landscape=True,
        )

    # Сравниваем OCR-текст
    logger.info("=== Результаты ===")
    for label, pdf_path in [("без разбивки", out_nosplit), ("с разбивкой", out_split)]:
        if not pdf_path.exists():
            logger.warning(f"  {label}: файл не создан")
            continue
        size_mb = pdf_path.stat().st_size / 1024 / 1024
        doc = fitz.open(pdf_path)
        n_pages = len(doc)
        full_text = "\n".join(doc[i].get_text() for i in range(n_pages))
        doc.close()
        cyrillic_chars = sum(1 for ch in full_text if "\u0400" <= ch <= "\u04ff")
        logger.info(
            f"  {label}: {n_pages} стр., "
            f"{size_mb:.1f} МБ, кириллица: {cyrillic_chars} символов"
        )
        logger.info(f"  Текст ({label}): {full_text[:400].strip()}")
        logger.info(f"  Сохранён: {pdf_path}")


def main():
    """Основная функция для запуска обработки."""
    # Определяем пути относительно текущего файла
    script_dir = Path(__file__).parent
    # input_dir = script_dir / "ocr_src"
    input_dir = Path("/mnt/dump3/DOWN/Плановое хозяйство (1931-1989)") / "1939"
    output_dir = Path("/mnt/dump3/DOWN/Плановое хозяйство (1931-1989) [распознанное]")

    logger.info(f"Исходная директория: {input_dir}")
    logger.info(f"Целевая директория: {output_dir}")

    success, total = process_directory(
        input_dir=input_dir, output_dir=output_dir, language="rus", skip_existing=True, deskew=True, split_landscape=True
    )

    logger.info(f"\n{'=' * 50}")
    logger.info(f"Обработка завершена: {success}/{total} файлов успешно обработано")
    logger.info(f"{'=' * 50}")


if __name__ == "__main__":
    main()
