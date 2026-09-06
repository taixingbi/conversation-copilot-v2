# Copy this file to resume.md / projects.md and fill in your own STAR notes.
# example.md is not indexed unless PROFILE_INCLUDE_EXAMPLE=1

## RAG evaluation at Acme
Situation: Search quality dropped after we added a second corpus.
Task: I owned evaluation for the RAG pipeline.
Action: Built a 200-question golden set, measured recall@5 and answer faithfulness, then tuned chunk size and rerank.
Result: Recall@5 went from 0.61 to 0.78 and hallucination reports fell by about half.

## Python vs Java service
Situation: The team needed a new billing worker.
Task: I chose the runtime and shipped the first version.
Action: Used Python for the worker because the data team already had pandas jobs; kept the API in Java.
Result: We reused existing ETL code and shipped in two weeks instead of rewriting parsers.
