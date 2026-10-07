import sys
from typing import Final

import spacy
from spacy.language import Language
from spacy.tokens import Doc, Span

# Small pre-trained English pipeline. Sentence boundaries come from its
# dependency parser, so abbreviations like "Mr." and "A.I." do not split
# a sentence the way a naive split on "." would.
SPACY_MODEL_NAME: Final[str] = "en_core_web_sm"

SAMPLE_TEXT: Final[str] = (
    "Mr. Wang is a teacher. He teaches A.I. (?). "
    "Does he love his work? Of course!"
)

EXIT_CODE_SUCCESS: Final[int] = 0


class SpacyModelNotInstalledError(Exception):
    """Raised when the requested spaCy pipeline package is not installed."""


def load_language_model(model_name: str) -> Language:
    """Load an installed spaCy pipeline by name.

    Raises:
        SpacyModelNotInstalledError: If the named model package is not installed.
    """
    try:
        language_model: Language = spacy.load(model_name)
    except OSError as error:
        raise SpacyModelNotInstalledError(
            f"spaCy model {model_name!r} is not installed. "
            f"Install it with: python -m spacy download {model_name}"
        ) from error

    return language_model


def split_text_into_sentences(
    language_model: Language,
    text: str,
) -> list[str]:
    """Return the text of each sentence spaCy detects, in document order.

    The pipeline must set sentence boundaries (for example with a parser
    or a sentencizer); otherwise spaCy raises ValueError on ``Doc.sents``.
    """
    processed_document: Doc = language_model(text)

    sentence_texts: list[str] = []

    sentence_span: Span
    for sentence_span in processed_document.sents:
        sentence_text: str = sentence_span.text
        sentence_texts.append(sentence_text)

    return sentence_texts


def main() -> int:
    language_model: Language = load_language_model(SPACY_MODEL_NAME)

    sentence_texts: list[str] = split_text_into_sentences(
        language_model,
        SAMPLE_TEXT,
    )

    # Writing sentences to stdout is this script's output, one per line.
    sentence_text: str
    for sentence_text in sentence_texts:
        output_line: str = f"{sentence_text}\n"
        sys.stdout.write(output_line)

    return EXIT_CODE_SUCCESS


if __name__ == "__main__":
    raise SystemExit(main())
sent.text)