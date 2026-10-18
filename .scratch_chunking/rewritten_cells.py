import json
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Final, TypeAlias

import pandas  # type: ignore[import-untyped]  # pandas ships no type stubs; pandas-stubs is not a notebook prerequisite
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Each row is (product name, price in USD, price in EUR, rating).
ProductRowValues: TypeAlias = tuple[str, str, str, str]

# Each pair is (top-level header, sub-level header).
# An empty string marks a column that has no sub-header.
ColumnHeaderPair: TypeAlias = tuple[str, str]

PRODUCT_DATA_ROWS: Final[tuple[ProductRowValues, ...]] = (
    ("Model-A", "$299", "€257.25", "4.5 Stars"),
    ("Model-B", "$449", "€386.95", "4.2 Stars"),
    ("Model-C", "$199", "€171.50", "3.9 Stars"),
    ("Model-X", "$899", "€774.77", "4.8 Stars"),
    ("Model-Pro", "$1299", "€1122.15", "4.9 Stars"),
    ("Model-Lite", "$149", "€128.70", "3.5 Stars"),
    ("Model-Air", "$699", "€603.75", "4.4 Stars"),
    ("Model-Eco", "$99", "€85.52", "3.1 Stars"),
    ("Model-Max", "$1099", "€949.20", "4.7 Stars"),
    ("Model-Std", "$249", "€215.04", "4.0 Stars"),
    ("Model-Ultra", "$1599", "€1381.13", "4.8 Stars"),
    ("Model-Mini", "$349", "€301.44", "4.1 Stars"),
    ("Model-Go", "$179", "€154.60", "3.7 Stars"),
    ("Model-Titan", "$1999", "€1726.63", "4.9 Stars"),
)

COLUMN_HEADER_PAIRS: Final[tuple[ColumnHeaderPair, ...]] = (
    ("Product", ""),
    ("Price", "USD"),
    ("Price", "EUR"),
    ("Rating", ""),
)


class ProductTableTransformations:
    """Pure transformations that build the sample product table."""

    @staticmethod
    def build_product_dataframe(
        product_rows: Sequence[ProductRowValues],
        column_header_pairs: Sequence[ColumnHeaderPair],
    ) -> pandas.DataFrame:
        # pandas expects lists, so new lists are built from the read-only inputs.
        column_header_pair_list = list(column_header_pairs)
        product_row_list = list(product_rows)

        multi_level_columns = pandas.MultiIndex.from_tuples(tuples=column_header_pair_list)

        # pandas itself ships no type stubs. A type checker verifies the pandas annotations
        # in this notebook only when the pandas-stubs package is installed.
        product_dataframe = pandas.DataFrame(
            data=product_row_list,
            columns=multi_level_columns,
        )
        return product_dataframe

PREVIEW_ROW_COUNT: Final[int] = 4
JSON_INDENT_SPACES: Final[int] = 4


class TableFormatTransformations:
    """Pure transformations that select table rows and convert a table to text."""

    @staticmethod
    def select_preview_rows(
        table_dataframe: pandas.DataFrame,
        *,
        row_count: int,
    ) -> pandas.DataFrame:
        # head() returns a new DataFrame and does not change table_dataframe.
        preview_dataframe = table_dataframe.head(n=row_count)
        return preview_dataframe

    @staticmethod
    def format_dataframe_as_plain_text(table_dataframe: pandas.DataFrame) -> str:
        plain_table_text: str = table_dataframe.to_string()
        return plain_table_text

    @staticmethod
    def format_dataframe_as_json(
        table_dataframe: pandas.DataFrame,
        *,
        indent_spaces: int,
    ) -> str:
        # orient="columns" is the pandas default: {column_label: {row_label: cell_value}}.
        compact_json_text: str = table_dataframe.to_json(orient="columns")

        # json.loads returns Any. This annotation states the shape that orient="columns" produces
        # for a table of text cells; it is not checked at runtime.
        cells_by_column: dict[str, dict[str, str]] = json.loads(compact_json_text)

        # ensure_ascii=True is the json default: it writes "€" as the escape code "\u20ac".
        indented_json_text = json.dumps(
            cells_by_column,
            ensure_ascii=True,
            indent=indent_spaces,
        )
        return indented_json_text

    @staticmethod
    def format_dataframe_as_markdown(table_dataframe: pandas.DataFrame) -> str:
        # DataFrame.to_markdown() raises ImportError when the optional tabulate package is not installed.
        markdown_table_text: str = table_dataframe.to_markdown()
        return markdown_table_text

# The chunk size is intentionally small so the split breaks table rows apart.
CHUNK_SIZE_CHARACTERS: Final[int] = 100
CHUNK_OVERLAP_CHARACTERS: Final[int] = 20

# These are the library's default separators, written out so the split order is visible:
# paragraph break, then line break, then space, then individual characters.
CHUNK_SEPARATORS: Final[tuple[str, ...]] = ("\n\n", "\n", " ", "")


class ChunkTransformations:
    """Pure transformations that format text chunks for display."""

    @staticmethod
    def format_chunk_listing(
        table_chunks: Sequence[str],
        *,
        chunk_size_characters: int,
    ) -> str:
        chunk_count = len(table_chunks)
        listing_lines: list[str] = []

        header_line = f"\n--- SPLIT INTO {chunk_count} CHUNKS (Chunk Size: {chunk_size_characters}) ---"
        listing_lines.append(header_line)

        # start=1 numbers the chunks from 1 for display.
        for chunk_number, chunk_text in enumerate(table_chunks, start=1):
            # repr() shows the newline characters (\n) inside each chunk.
            chunk_representation = repr(chunk_text)
            chunk_heading_line = f"\n[CHUNK {chunk_number}]"

            listing_lines.append(chunk_heading_line)
            listing_lines.append(chunk_representation)

        chunk_listing_text = "\n".join(listing_lines)
        return chunk_listing_text

@dataclass(frozen=True, kw_only=True, slots=True)
class TableChunkingReport:
    """The text outputs of one table chunking demonstration."""

    plain_table_text: str
    preview_json_text: str
    markdown_table_text: str
    chunk_listing_text: str


class ConsoleReportWriter:
    """Prints a table chunking report to the notebook output."""

    def write_report(self, report: TableChunkingReport) -> None:
        self._write_section(section_title="Plain text table", section_text=report.plain_table_text)
        self._write_section(section_title="JSON preview", section_text=report.preview_json_text)
        self._write_section(section_title="Markdown table", section_text=report.markdown_table_text)
        self._write_section(section_title="Markdown chunks", section_text=report.chunk_listing_text)

    def _write_section(self, *, section_title: str, section_text: str) -> None:
        section_heading = f"=== {section_title} ==="

        # Printing to the notebook output is the purpose of this component. Logging would add
        # logger configuration and record prefixes to the output that the reader studies.
        print(section_heading)  # noqa: T201  # notebook output is the purpose of this component
        print(section_text)  # noqa: T201  # notebook output is the purpose of this component
        print()  # noqa: T201  # blank line between sections


class TableChunkingDemonstrationPipeline:
    """Builds the product table, converts it to text formats, and splits the Markdown text into chunks."""

    def __init__(
        self,
        *,
        text_splitter: RecursiveCharacterTextSplitter,
        preview_row_count: int,
        json_indent_spaces: int,
        chunk_size_characters: int,
    ) -> None:
        self._text_splitter = text_splitter
        self._preview_row_count = preview_row_count
        self._json_indent_spaces = json_indent_spaces

        # The splitter keeps its chunk size private, so the value for the display header is passed in.
        self._chunk_size_characters = chunk_size_characters

    def run(
        self,
        product_rows: Sequence[ProductRowValues],
        column_header_pairs: Sequence[ColumnHeaderPair],
    ) -> TableChunkingReport:
        product_dataframe = ProductTableTransformations.build_product_dataframe(
            product_rows,
            column_header_pairs,
        )

        plain_table_text = TableFormatTransformations.format_dataframe_as_plain_text(product_dataframe)

        preview_dataframe = TableFormatTransformations.select_preview_rows(
            product_dataframe,
            row_count=self._preview_row_count,
        )
        preview_json_text = TableFormatTransformations.format_dataframe_as_json(
            preview_dataframe,
            indent_spaces=self._json_indent_spaces,
        )

        markdown_table_text = TableFormatTransformations.format_dataframe_as_markdown(product_dataframe)

        table_chunks = self._text_splitter.split_text(markdown_table_text)

        chunk_listing_text = ChunkTransformations.format_chunk_listing(
            table_chunks,
            chunk_size_characters=self._chunk_size_characters,
        )

        table_chunking_report = TableChunkingReport(
            plain_table_text=plain_table_text,
            preview_json_text=preview_json_text,
            markdown_table_text=markdown_table_text,
            chunk_listing_text=chunk_listing_text,
        )
        return table_chunking_report


def main() -> None:
    # The splitter expects a list, so it receives a new list built from the constant tuple.
    chunk_separators = list(CHUNK_SEPARATORS)

    text_splitter = RecursiveCharacterTextSplitter(
        separators=chunk_separators,
        keep_separator=True,
        is_separator_regex=False,
        chunk_size=CHUNK_SIZE_CHARACTERS,
        chunk_overlap=CHUNK_OVERLAP_CHARACTERS,
        length_function=len,
        strip_whitespace=True,
    )

    table_chunking_pipeline = TableChunkingDemonstrationPipeline(
        text_splitter=text_splitter,
        preview_row_count=PREVIEW_ROW_COUNT,
        json_indent_spaces=JSON_INDENT_SPACES,
        chunk_size_characters=CHUNK_SIZE_CHARACTERS,
    )
    console_report_writer = ConsoleReportWriter()

    table_chunking_report = table_chunking_pipeline.run(PRODUCT_DATA_ROWS, COLUMN_HEADER_PAIRS)
    console_report_writer.write_report(table_chunking_report)


# Jupyter runs notebook cells with __name__ set to "__main__". main() returns no exit code,
# because raising SystemExit in a notebook prints an exception message instead of ending a program.
if __name__ == "__main__":
    main()
