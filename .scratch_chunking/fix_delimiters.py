import pathlib

builder_path = pathlib.Path("build_notebook.py")
builder_text = builder_path.read_text(encoding="utf-8")

triple_double = '"' * 3
escaped_triple_double = '\\"' * 3

for constant_name in ["IMPORTS_CODE", "CREATE_TABLE_CODE", "CONVERT_FORMATS_CODE", "CHUNKING_CODE", "RUN_CODE"]:
    opener = constant_name + " = r" + triple_double
    start = builder_text.index(opener)
    body_start = start + len(opener)
    search_position = body_start
    while True:
        end = builder_text.index(triple_double, search_position)
        if builder_text[end - 1] != "\\":
            break
        search_position = end + 3
    body = builder_text[body_start:end].replace(escaped_triple_double, triple_double)
    builder_text = builder_text[:start] + constant_name + " = r'''" + body + "'''" + builder_text[end + 3 :]

builder_text = builder_text.replace('as the escape code "€"', 'as the escape code "\\u20ac"')
builder_path.write_text(builder_text, encoding="utf-8")
