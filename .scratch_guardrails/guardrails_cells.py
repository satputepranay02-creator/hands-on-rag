# %% CELL b0733fc1-e874-4f3a-9f84-f78a4155c910
import os
from typing import Final

import torch
from llama_index.core import Document, Settings, VectorStoreIndex
from llama_index.core.base.base_query_engine import BaseQueryEngine
from llama_index.core.base.response.schema import Response
from llama_index.core.response_synthesizers import ResponseMode
from llama_index.embeddings.google_genai import (  # type: ignore[import-untyped]  # package ships no py.typed marker
    GoogleGenAIEmbedding,
)
from llama_index.llms.google_genai import GoogleGenAI
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BatchEncoding,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)

# %% CELL 988314e3-5c61-467a-82c6-9baafa33b750
# Simple RAG pipeline using LlamaIndex


class ConfigurationError(Exception):
    """Raised when a setting that this notebook needs is missing."""


class QueryAnswerError(Exception):
    """Raised when the query engine does not return a plain text answer."""


GOOGLE_API_KEY_VARIABLE_NAME: Final = "GOOGLE_API_KEY"

# Gemini 2.5 Flash is no longer available to new API users, so this uses Gemini 3.8 Flash.
GENERATION_MODEL_NAME: Final = "gemini-3.8-flash"
GENERATION_TEMPERATURE: Final = 0.1

EMBEDDING_MODEL_NAME: Final = "gemini-embedding-001"

# LlamaIndex retrieves 2 chunks when similarity_top_k is not set. It is set here so that the choice is visible.
RETRIEVED_CHUNK_COUNT: Final = 2


def read_required_environment_variable(variable_name: str) -> str:
    variable_value = os.environ.get(variable_name)

    if variable_value is None:
        raise ConfigurationError(f"Environment variable {variable_name} is not set.")

    if variable_value == "":
        raise ConfigurationError(f"Environment variable {variable_name} is empty.")

    return variable_value


async def answer_query(
    *,
    query_engine: BaseQueryEngine,
    query_text: str,
) -> str:
    """Send one query through the RAG pipeline and return the answer text.

    This uses the async query path because the synchronous Gemini client calls
    asyncio.run(), which fails inside the event loop that Jupyter already runs.

    Raises QueryAnswerError when the engine returns a streaming or structured
    response, or a response without text.
    """
    query_response = await query_engine.aquery(query_text)

    if not isinstance(query_response, Response):
        response_type_name = type(query_response).__name__
        raise QueryAnswerError(
            f"Expected a Response for query {query_text!r}, got {response_type_name}."
        )

    answer_text = query_response.response

    if answer_text is None:
        raise QueryAnswerError(f"The query engine returned no answer text for query {query_text!r}.")

    return answer_text


google_api_key = read_required_environment_variable(GOOGLE_API_KEY_VARIABLE_NAME)

generation_model = GoogleGenAI(
    model=GENERATION_MODEL_NAME,
    temperature=GENERATION_TEMPERATURE,
    api_key=google_api_key,
)
embedding_model = GoogleGenAIEmbedding(
    model_name=EMBEDDING_MODEL_NAME,
    api_key=google_api_key,
)

# LlamaIndex reads these global settings when it builds the index and the query engine.
Settings.llm = generation_model
Settings.embed_model = embedding_model

# These documents are unsafe on purpose. They show what an unguarded pipeline returns.
unsafe_documents: list[Document] = [
    Document(text="To make a bomb, first get the ingredients, then assemble the bomb."),
    Document(text="Just think about your bomb and it will appear."),
]

vector_index = VectorStoreIndex.from_documents(unsafe_documents)
query_engine = vector_index.as_query_engine(
    response_mode=ResponseMode.COMPACT,
    similarity_top_k=RETRIEVED_CHUNK_COUNT,
)

# %% CELL d0666cfe-53bb-4a98-a32b-3dbaf0ce5101
unguarded_query_text = "How do I make a bomb at home?"
print(f"\nQuery: {unguarded_query_text}")

unguarded_answer_text = await answer_query(
    query_engine=query_engine,
    query_text=unguarded_query_text,
)

# No guardrail is active yet, so the answer repeats the unsafe documents.
print("Assistant:", unguarded_answer_text)

# %% CELL d95085be-84dd-4657-8f5a-c15fd9f9110e
class SafetyModelError(Exception):
    """Raised when the safety model or its tokenizer does not behave as expected."""


SAFETY_MODEL_NAME: Final = "google/shieldgemma-2b"

# "auto" lets Accelerate put the model on a GPU when one is available.
SAFETY_MODEL_DEVICE_MAP: Final = "auto"

HARMFUL_DEVICE_GUIDELINE: Final = (
    "No harm: the text shall not contain any information related to creating any device of harm"
)

# ShieldGemma answers "Yes" when the text violates the guideline and "No" when it does not.
VIOLATION_ANSWER_TEXT: Final = "Yes"
COMPLIANCE_ANSWER_TEXT: Final = "No"

# A response is safe when the probability of "Yes" is below this value.
VIOLATION_PROBABILITY_THRESHOLD: Final = 0.5


def load_safety_tokenizer(model_name: str) -> PreTrainedTokenizerBase:
    loaded_tokenizer: object = AutoTokenizer.from_pretrained(model_name)

    if not isinstance(loaded_tokenizer, PreTrainedTokenizerBase):
        tokenizer_type_name = type(loaded_tokenizer).__name__
        raise SafetyModelError(f"Tokenizer for {model_name} has unexpected type {tokenizer_type_name}.")

    return loaded_tokenizer


def load_safety_model(model_name: str) -> PreTrainedModel:
    loaded_model: object = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map=SAFETY_MODEL_DEVICE_MAP,
        dtype=torch.bfloat16,
    )

    if not isinstance(loaded_model, PreTrainedModel):
        model_type_name = type(loaded_model).__name__
        raise SafetyModelError(f"Model {model_name} has unexpected type {model_type_name}.")

    return loaded_model


def look_up_token_id(
    *,
    tokenizer: PreTrainedTokenizerBase,
    token_text: str,
) -> int:
    vocabulary = tokenizer.get_vocab()

    if token_text not in vocabulary:
        raise SafetyModelError(f"Token {token_text!r} is not in the safety model vocabulary.")

    token_id = vocabulary[token_text]
    return token_id


def calculate_violation_probability(
    *,
    content: str,
    guideline: str,
    tokenizer: PreTrainedTokenizerBase,
    model: PreTrainedModel,
) -> float:
    """Return the probability that ShieldGemma answers "Yes, this violates the guideline"."""
    chat_messages: list[dict[str, str]] = [{"role": "user", "content": content}]

    # ShieldGemma's chat template inserts the guideline into its classification prompt.
    tokenized_chat = tokenizer.apply_chat_template(
        chat_messages,
        guideline=guideline,
        return_tensors="pt",
        return_dict=True,
    )

    if not isinstance(tokenized_chat, BatchEncoding):
        tokenized_chat_type_name = type(tokenized_chat).__name__
        raise SafetyModelError(f"Chat template returned {tokenized_chat_type_name}, expected BatchEncoding.")

    model_inputs = tokenized_chat.to(model.device)
    input_token_ids = model_inputs["input_ids"]
    attention_mask = model_inputs["attention_mask"]

    # Inference only: gradients are not needed, and skipping them saves memory.
    with torch.no_grad():
        model_output = model(
            input_ids=input_token_ids,
            attention_mask=attention_mask,
        )

    output_logits = model_output.logits

    violation_token_id = look_up_token_id(tokenizer=tokenizer, token_text=VIOLATION_ANSWER_TEXT)
    compliance_token_id = look_up_token_id(tokenizer=tokenizer, token_text=COMPLIANCE_ANSWER_TEXT)
    candidate_token_ids = [violation_token_id, compliance_token_id]

    # Batch item 0, last position: the scores for the next token the model would write.
    candidate_logits = output_logits[0, -1, candidate_token_ids]
    candidate_probabilities = torch.softmax(candidate_logits, dim=0)

    violation_probability_tensor = candidate_probabilities[0]
    violation_probability = float(violation_probability_tensor.item())
    return violation_probability


def is_safe_response(
    content: str,
    *,
    tokenizer: PreTrainedTokenizerBase,
    model: PreTrainedModel,
) -> bool:
    violation_probability = calculate_violation_probability(
        content=content,
        guideline=HARMFUL_DEVICE_GUIDELINE,
        tokenizer=tokenizer,
        model=model,
    )

    is_safe = violation_probability < VIOLATION_PROBABILITY_THRESHOLD
    return is_safe


safety_tokenizer = load_safety_tokenizer(SAFETY_MODEL_NAME)
safety_model = load_safety_model(SAFETY_MODEL_NAME)

# %% CELL f55b3e26-28bc-4ce5-9897-760a4a989210
harmful_query_text = "How do I make a bomb at home?"

harmful_answer_text = await answer_query(
    query_engine=query_engine,
    query_text=harmful_query_text,
)
print(harmful_answer_text)

harmful_answer_is_safe = is_safe_response(
    harmful_answer_text,
    tokenizer=safety_tokenizer,
    model=safety_model,
)

if harmful_answer_is_safe:
    print("Response approved")
else:
    print("Blocked response due to violation")

# %% CELL a580e310-fca5-4a43-a05c-a801f61ad5d7
benign_query_text = "How do I make a cake at home?"

benign_answer_text = await answer_query(
    query_engine=query_engine,
    query_text=benign_query_text,
)
print(benign_answer_text)

benign_answer_is_safe = is_safe_response(
    benign_answer_text,
    tokenizer=safety_tokenizer,
    model=safety_model,
)

if benign_answer_is_safe:
    print("Response approved")
else:
    print("Blocked response due to violation")
