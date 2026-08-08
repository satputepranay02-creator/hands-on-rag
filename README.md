# Hands-On RAG for Production

This repository contains the code for the O'Reilly book *Hands-On RAG for Production*, by Ofer Mendelevitch and Forrest Sheng Bao.

<a href="https://www.oreilly.com/library/view/hands-on-rag-for/9798341621701/"><img src="docs/hands-on-rag-front.jpg" width="250" alt="Hands-On RAG for Production book cover"></a>

Retrieval-augmented generation is easy to prototype and hard to ship. The book walks
through every phase of building a RAG application &mdash; parsing, chunking, embeddings,
and vector search, through to agentic RAG, multimodal RAG, and GraphRAG &mdash; and is
honest about the trade-offs at each step. This repository is the code: 35 notebooks you
can run locally or in Google Colab.

- **Book site** &mdash; https://ofermend.github.io/hands-on-rag/
- [O'Reilly](https://www.oreilly.com/library/view/hands-on-rag-for/9798341621701/) &mdash; read online
- [Amazon](https://www.amazon.com/Hands-RAG-Production-Production-Ready-Applications/dp/B0G48HGR81/) &mdash; print and Kindle
- [Bookshop.org](https://bookshop.org/p/books/hands-on-rag-for-production-design-develop-and-deploy-production-ready-rag-applications-ofer-mendelevitch/23535824) &mdash; supports independent bookstores
- ISBN 979-8-3416-2171-8

## Notebooks by chapter

Every notebook has an "Open in Colab" badge at the top, so you can run it without
installing anything. To run locally, install the chapter's `requirements.txt` first.

| Chapter | Notebooks | Folder |
| --- | --- | --- |
| **1. Introduction to Retrieval-Augmented Generation (RAG)**<br>A first end-to-end RAG pipeline | [sample-rag.ipynb](chapter1/sample-rag.ipynb) &middot; [Colab](https://colab.research.google.com/github/ofermend/hands-on-rag/blob/main/chapter1/sample-rag.ipynb) &mdash; End-to-end RAG with LangChain: load, split, embed, store, query | [chapter1/](chapter1) |
| **2. The Base RAG Stack**<br>Parsing, chunking, embeddings, vector search, generation | [parse-pdf.ipynb](chapter2/parse-pdf.ipynb) &middot; [Colab](https://colab.research.google.com/github/ofermend/hands-on-rag/blob/main/chapter2/parse-pdf.ipynb) &mdash; PDF text extraction with PyMuPDF<br>[parse-docx.ipynb](chapter2/parse-docx.ipynb) &middot; [Colab](https://colab.research.google.com/github/ofermend/hands-on-rag/blob/main/chapter2/parse-docx.ipynb) &mdash; DOCX parsing with python-docx<br>[parse-in-llm.ipynb](chapter2/parse-in-llm.ipynb) &middot; [Colab](https://colab.research.google.com/github/ofermend/hands-on-rag/blob/main/chapter2/parse-in-llm.ipynb) &mdash; Using an LLM to parse documents<br>[embedding.ipynb](chapter2/embedding.ipynb) &middot; [Colab](https://colab.research.google.com/github/ofermend/hands-on-rag/blob/main/chapter2/embedding.ipynb) &mdash; Sentence embeddings and similarity<br>[pgvector-simple.ipynb](chapter2/pgvector-simple.ipynb) &middot; [Colab](https://colab.research.google.com/github/ofermend/hands-on-rag/blob/main/chapter2/pgvector-simple.ipynb) &mdash; Vector search in PostgreSQL with pgvector<br>[sqlite-vec.ipynb](chapter2/sqlite-vec.ipynb) &middot; [Colab](https://colab.research.google.com/github/ofermend/hands-on-rag/blob/main/chapter2/sqlite-vec.ipynb) &mdash; Vector search with the SQLite-vec extension<br>[generative-llms.ipynb](chapter2/generative-llms.ipynb) &middot; [Colab](https://colab.research.google.com/github/ofermend/hands-on-rag/blob/main/chapter2/generative-llms.ipynb) &mdash; Generation with Claude | [chapter2/](chapter2) |
| **3. Scaling Your RAG Stack**<br>Parsing at scale, reranking, guardrails, hallucinations | [split-pdf.ipynb](chapter3/split-pdf.ipynb) &middot; [Colab](https://colab.research.google.com/github/ofermend/hands-on-rag/blob/main/chapter3/split-pdf.ipynb) &mdash; Splitting large PDFs by page count before parsing<br>[reranking.ipynb](chapter3/reranking.ipynb) &middot; [Colab](https://colab.research.google.com/github/ofermend/hands-on-rag/blob/main/chapter3/reranking.ipynb) &mdash; Reranking retrieved results<br>[guardrails.ipynb](chapter3/guardrails.ipynb) &middot; [Colab](https://colab.research.google.com/github/ofermend/hands-on-rag/blob/main/chapter3/guardrails.ipynb) &mdash; Input and output guardrails<br>[hallucinations.ipynb](chapter3/hallucinations.ipynb) &middot; [Colab](https://colab.research.google.com/github/ofermend/hands-on-rag/blob/main/chapter3/hallucinations.ipynb) &mdash; Detecting and handling hallucinations | [chapter3/](chapter3) |
| **4. Deploying RAG to Production**<br>Caching for latency and cost, redaction for sensitive data | [caching.ipynb](chapter4/caching.ipynb) &middot; [Colab](https://colab.research.google.com/github/ofermend/hands-on-rag/blob/main/chapter4/caching.ipynb) &mdash; Exact-match and semantic caching with Redis<br>[redaction.ipynb](chapter4/redaction.ipynb) &middot; [Colab](https://colab.research.google.com/github/ofermend/hands-on-rag/blob/main/chapter4/redaction.ipynb) &mdash; Redacting sensitive data | [chapter4/](chapter4) |
| **5. The RAG Platform**<br>Using a managed RAG platform instead of assembling one | [vectara-ingest.ipynb](chapter5/vectara-ingest.ipynb) &middot; [Colab](https://colab.research.google.com/github/ofermend/hands-on-rag/blob/main/chapter5/vectara-ingest.ipynb) &mdash; Ingesting documents into Vectara<br>[vectara-list-docs.ipynb](chapter5/vectara-list-docs.ipynb) &middot; [Colab](https://colab.research.google.com/github/ofermend/hands-on-rag/blob/main/chapter5/vectara-list-docs.ipynb) &mdash; Listing and managing indexed documents<br>[vectara-query.ipynb](chapter5/vectara-query.ipynb) &middot; [Colab](https://colab.research.google.com/github/ofermend/hands-on-rag/blob/main/chapter5/vectara-query.ipynb) &mdash; Querying Vectara | [chapter5/](chapter5) |
| **6. Evaluating Your RAG Application**<br>LLM-as-a-judge and retrieval metrics | [llm-as-a-judge.ipynb](chapter6/llm-as-a-judge.ipynb) &middot; [Colab](https://colab.research.google.com/github/ofermend/hands-on-rag/blob/main/chapter6/llm-as-a-judge.ipynb) &mdash; Scoring RAG answers with an LLM judge<br>[metrics.ipynb](chapter6/metrics.ipynb) &middot; [Colab](https://colab.research.google.com/github/ofermend/hands-on-rag/blob/main/chapter6/metrics.ipynb) &mdash; Retrieval and generation evaluation metrics<br>[umbrela.ipynb](chapter6/umbrela.ipynb) &middot; [Colab](https://colab.research.google.com/github/ofermend/hands-on-rag/blob/main/chapter6/umbrela.ipynb) &mdash; UMBRELA 0-3 passage relevance scoring | [chapter6/](chapter6) |
| **7. From RAG to AI Agents**<br>Tool calling, single agents, multi-agent systems | [tool-calling.ipynb](chapter7/tool-calling.ipynb) &middot; [Colab](https://colab.research.google.com/github/ofermend/hands-on-rag/blob/main/chapter7/tool-calling.ipynb) &mdash; LLM tool/function calling<br>[react-agent-langchain.ipynb](chapter7/react-agent-langchain.ipynb) &middot; [Colab](https://colab.research.google.com/github/ofermend/hands-on-rag/blob/main/chapter7/react-agent-langchain.ipynb) &mdash; ReAct agent with LangChain<br>[function-agent-llamaindex.ipynb](chapter7/function-agent-llamaindex.ipynb) &middot; [Colab](https://colab.research.google.com/github/ofermend/hands-on-rag/blob/main/chapter7/function-agent-llamaindex.ipynb) &mdash; Function-calling agent with LlamaIndex<br>[vectara-agent.ipynb](chapter7/vectara-agent.ipynb) &middot; [Colab](https://colab.research.google.com/github/ofermend/hands-on-rag/blob/main/chapter7/vectara-agent.ipynb) &mdash; RAG agent with Vectara<br>[crewai-multi-agent.ipynb](chapter7/crewai-multi-agent.ipynb) &middot; [Colab](https://colab.research.google.com/github/ofermend/hands-on-rag/blob/main/chapter7/crewai-multi-agent.ipynb) &mdash; Multi-agent system with CrewAI | [chapter7/](chapter7) |
| **8. Multimodal RAG**<br>Tables, images, and audio | [parse-tables.ipynb](chapter8/parse-tables.ipynb) &middot; [Colab](https://colab.research.google.com/github/ofermend/hands-on-rag/blob/main/chapter8/parse-tables.ipynb) &mdash; Extracting tables from PDFs with Docling<br>[multi-page-tables.ipynb](chapter8/multi-page-tables.ipynb) &middot; [Colab](https://colab.research.google.com/github/ofermend/hands-on-rag/blob/main/chapter8/multi-page-tables.ipynb) &mdash; Stitching tables that span page breaks<br>[simple-table-chunking.ipynb](chapter8/simple-table-chunking.ipynb) &middot; [Colab](https://colab.research.google.com/github/ofermend/hands-on-rag/blob/main/chapter8/simple-table-chunking.ipynb) &mdash; Table-to-text conversion and chunking pitfalls<br>[image-rag-langchain.ipynb](chapter8/image-rag-langchain.ipynb) &middot; [Colab](https://colab.research.google.com/github/ofermend/hands-on-rag/blob/main/chapter8/image-rag-langchain.ipynb) &mdash; Image RAG with LangChain and Docling<br>[image-retrieval-siglip.ipynb](chapter8/image-retrieval-siglip.ipynb) &middot; [Colab](https://colab.research.google.com/github/ofermend/hands-on-rag/blob/main/chapter8/image-retrieval-siglip.ipynb) &mdash; Image retrieval with SigLIP<br>[audio-transcribe.ipynb](chapter8/audio-transcribe.ipynb) &middot; [Colab](https://colab.research.google.com/github/ofermend/hands-on-rag/blob/main/chapter8/audio-transcribe.ipynb) &mdash; Audio transcription and indexing | [chapter8/](chapter8) |
| **9. Knowledge-Enhanced RAG**<br>Knowledge graphs and GraphRAG | [create-graph.ipynb](chapter9/create-graph.ipynb) &middot; [Colab](https://colab.research.google.com/github/ofermend/hands-on-rag/blob/main/chapter9/create-graph.ipynb) &mdash; Building a knowledge graph from movie data<br>[graph-query.ipynb](chapter9/graph-query.ipynb) &middot; [Colab](https://colab.research.google.com/github/ofermend/hands-on-rag/blob/main/chapter9/graph-query.ipynb) &mdash; Querying the knowledge graph with Cypher<br>[graphrag-create.ipynb](chapter9/graphrag-create.ipynb) &middot; [Colab](https://colab.research.google.com/github/ofermend/hands-on-rag/blob/main/chapter9/graphrag-create.ipynb) &mdash; Building an index with Microsoft GraphRAG<br>[graphrag-query.ipynb](chapter9/graphrag-query.ipynb) &middot; [Colab](https://colab.research.google.com/github/ofermend/hands-on-rag/blob/main/chapter9/graphrag-query.ipynb) &mdash; Querying with Microsoft GraphRAG | [chapter9/](chapter9) |
| **10. The Future of RAG** | No code | &mdash; |

## What each chapter needs

Set the keys as environment variables before starting Jupyter. Chapters that need a local
service won't run in Colab as-is &mdash; use those locally.

| Chapter | API keys | Local services |
| --- | --- | --- |
| 1 | `OPENAI_API_KEY` | &mdash; |
| 2 | `ANTHROPIC_API_KEY` | PostgreSQL with the pgvector extension, for `pgvector-simple.ipynb` |
| 3 | `OPENAI_API_KEY` | &mdash; |
| 4 | `OPENAI_API_KEY` | Docker &mdash; `caching.ipynb` starts and stops a Redis container itself |
| 5 | `VECTARA_API_KEY` | &mdash; |
| 6 | `OPENAI_API_KEY` | &mdash; |
| 7 | `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `VECTARA_API_KEY` | &mdash; |
| 8 | `OPENAI_API_KEY` | &mdash; |
| 9 | `OPENAI_API_KEY`, `NEO4J_URI`, `NEO4J_USER`, `NEO4J_PASSWORD` | Neo4j, local or in Docker |

The notebooks call paid APIs, so running them costs money. Chapter 9's GraphRAG indexing
makes the most LLM calls of any example here &mdash; check your usage before re-running it.

## Quick start

Python 3.10+.

```bash
git clone https://github.com/ofermend/hands-on-rag.git
cd hands-on-rag

python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
```

Each chapter has its own `requirements.txt` &mdash; install the one for the chapter you're
running rather than everything at once:

```bash
pip install -r chapter1/requirements.txt
export OPENAI_API_KEY="your-key-here"
jupyter lab
```

The `requirements.txt` at the repo root holds the dependencies shared across chapters.

## Citation

```bibtex
@book{mendelevitch2026handsonrag,
  author    = {Mendelevitch, Ofer and Bao, Forrest Sheng},
  title     = {Hands-On RAG for Production: Design, Develop, and Deploy
               Production-Ready RAG Applications},
  publisher = {O'Reilly Media},
  year      = {2026},
  isbn      = {979-8-3416-2171-8}
}
```

## Contributing

Issues and pull requests are welcome. If a notebook fails, open an issue with the chapter
and notebook name, the error message, and your Python version and OS.

## License

Apache License 2.0 &mdash; see [LICENSE](LICENSE). Free to use, modify, and distribute for
personal and commercial purposes.
