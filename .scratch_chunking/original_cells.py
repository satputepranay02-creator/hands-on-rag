import json
from collections.abc import Sequence
from typing import Final

import pandas
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Each row is (product name, price in USD, price in EUR, rating).
PRODUCT_DATA_ROWS: Final[tuple[tuple[str, str, str, str], ...]] = (
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

# Each pair is (top-level header, sub-level header).
# An empty string marks a column that has no sub-header.
COLUMN_HEADER_PAIRS: Final[tuple[tuple[str, str], ...]] = (
    ("Product", ""),
    ("Price", "USD"),
    ("Price", "EUR"),
    ("Rating", ""),
)


def build_product_dataframe(
    product_rows: Sequence[tuple[str, str, str, str]],
    column_header_pairs: Sequence[tuple[str, str]],
) -> pandas.DataFrame:
    # pandas expects lists, so new lists are built from the read-only inputs.
    column_header_pair_list = list(column_header_pairs)
    product_row_list = list(product_rows)

    multi_level_columns = pandas.MultiIndex.from_tuples(tuples=column_header_pair_list)

    product_dataframe = pandas.DataFrame(
        data=product_row_list,
        columns=multi_level_columns,
    )
    return product_dataframe


# pandas itself ships no type stubs. A type checker verifies the pandas annotations
# in this notebook only when the pandas-stubs package is installed.
product_dataframe = build_product_dataframe(PRODUCT_DATA_ROWS, COLUMN_HEADER_PAIRS)

product_table_text: str = product_dataframe.to_string()

print(product_table_text)

PREVIEW_ROW_COUNT: Final[int] = 4
JSON_INDENT_SPACES: Final[int] = 4


def format_product_dataframe_as_json(
    product_dataframe: pandas.DataFrame,
    *,
    indent_spaces: int,
) -> str:
    # orient="columns" is the pandas default: {column_label: {row_label: cell_value}}.
    compact_json_text: str = product_dataframe.to_json(orient="columns")

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


preview_dataframe = product_dataframe.head(n=PREVIEW_ROW_COUNT)

preview_json_text = format_product_dataframe_as_json(
    preview_dataframe,
    indent_spaces=JSON_INDENT_SPACES,
)

print(preview_json_text)

# DataFrame.to_markdown() raises ImportError when the optional tabulate package is not installed.
markdown_table_text: str = product_dataframe.to_markdown()

print(markdown_table_text)

# The chunk size is intentionally small so the split breaks table rows apart.
CHUNK_SIZE_CHARACTERS: Final[int] = 100
CHUNK_OVERLAP_CHARACTERS: Final[int] = 20

# These are the library's default separators, written out so the split order is visible:
# paragraph break, then line break, then space, then individual characters.
CHUNK_SEPARATORS: Final[tuple[str, ...]] = ("\n\n", "\n", " ", "")

# The splitter expects a list, so we pass it a new list built from the constant tuple.
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

table_chunks = text_splitter.split_text(markdown_table_text)

chunk_count = len(table_chunks)

print(f"\n--- SPLIT INTO {chunk_count} CHUNKS (Chunk Size: {CHUNK_SIZE_CHARACTERS}) ---")

# start=1 numbers the chunks from 1 for display.
for chunk_number, chunk_text in enumerate(table_chunks, start=1):
    # repr() shows the newline characters (\n) inside each chunk.
    chunk_representation = repr(chunk_text)

    print(f"\n[CHUNK {chunk_number}]")
    print(chunk_representation)