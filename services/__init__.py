# Service facades — the only layer server/ and main.py should import from.
#
# Internal packages (retrieval/, llm/, pipeline/, db/) are never imported
# directly by the server or CLI.  All calls go through:
#
#     services.rag        — retrieve, ingest, collections
#     services.llm        — answer, summarize, stream
#     services.extraction — (stub) large-doc branched extraction
