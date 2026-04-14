#!/usr/bin/env python3
"""
CLI для обработки PDF-файлов с помощью OCR.
"""
import argparse
from pathlib import Path

from rgae_scraper.pdf_ocr_utils import process_pdf_with_ocr, process_directory


def main():
    parser = argparse.ArgumentParser(
        description="Добавление текстового слоя к PDF-файлам с помощью Tesseract OCR"
    )
    parser.add_argument(
        "input",
        type=Path,
        help="Путь к PDF-файлу или директории с PDF-файлами",
    )
    parser.add_argument(
        "-o", "--output",
        type=Path,
        help="Путь для сохранения результата (файл или директория)",
    )
    parser.add_argument(
        "-l", "--language",
        default="rus",
        help="Язык распознавания (rus, eng и т.д.), по умолчанию: rus",
    )
    parser.add_argument(
        "-j", "--jobs",
        type=int,
        help="Количество параллельных задач для Tesseract (по умолчанию: 75%% от CPU)",
    )
    parser.add_argument(
        "--page",
        type=int,
        help="Номер одной страницы для обработки (по умолчанию: все страницы)",
    )
    parser.add_argument(
        "--pages",
        type=str,
        help='Диапазон страниц для обработки (например, "5-7")',
    )
    parser.add_argument(
        "--zero-based",
        action="store_true",
        help="Нумерация страниц начинается с 0 (по умолчанию: с 1)",
    )
    parser.add_argument(
        "--no-clean",
        action="store_true",
        help="Отключить удаление шума через unpaper",
    )
    parser.add_argument(
        "--deskew",
        action="store_true",
        help="Исправление наклона листа (ВНИМАНИЕ: изменяет финальное изображение)",
    )
    parser.add_argument(
        "--oversample",
        type=int,
        default=600,
        help="Повышение разрешения до указанного DPI перед OCR (0 — отключено), по умолчанию: 600",
    )
    parser.add_argument(
        "--split-landscape",
        action="store_true",
        help="Разбивать страницы-развороты на левую и правую половины перед OCR",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Перезаписывать существующие файлы",
    )

    args = parser.parse_args()

    # Проверяем входной путь
    if not args.input.exists():
        print(f"❌ Файл или директория не найдены: {args.input}")
        return 1

    # Определяем выходной путь
    if args.output is None:
        if args.input.is_file():
            args.output = args.input.parent / f"{args.input.stem}_ocr.pdf"
        else:
            args.output = args.input / "ocr_output"
    
    # Обработка
    if args.input.is_file():
        # Обработка одного файла
        success = process_pdf_with_ocr(
            input_pdf=args.input,
            output_pdf=args.output,
            language=args.language,
            skip_existing=not args.force,
            jobs=args.jobs,
            pages=args.pages,
            page=args.page,
            start_with_one=not args.zero_based,
            clean=not args.no_clean,
            oversample=args.oversample,
            deskew=args.deskew,
            split_landscape=args.split_landscape,
        )
        return 0 if success else 1
    else:
        # Обработка директории
        process_directory(
            input_dir=args.input,
            output_dir=args.output,
            language=args.language,
            skip_existing=not args.force,
            jobs=args.jobs,
            clean=not args.no_clean,
            oversample=args.oversample,
            deskew=args.deskew,
            split_landscape=args.split_landscape,
        )
        return 0


if __name__ == "__main__":
    exit(main())
