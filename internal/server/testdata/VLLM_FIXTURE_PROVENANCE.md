# SOL-04 vLLM Tier-A fixture provenance

Captured 2026-07-11 from vLLM commit
`01661cc57f48ce95c639efce7c88e6dd37349007` (Apache-2.0):

- `vllm/entrypoints/serve/tokenize/protocol.py`: `TokenizeResponse`
  (`count`, `max_model_len`, `tokens`) and `DetokenizeResponse` (`prompt`).
- `vllm/entrypoints/serve/tokenize/api_router.py`: `POST /tokenize` and
  `POST /detokenize` routes and error-envelope behavior.
- `vllm/entrypoints/openai/chat_completion/protocol.py`: nullable
  `max_tokens`, `max_completion_tokens` precedence, `stream_options`, and the
  rejection of stream options when `stream` is false.
- `vllm/entrypoints/openai/engine/protocol.py`: `StreamOptions.include_usage`
  and the nested `ErrorResponse.error` envelope.

The stream usage-only event shape additionally follows the OpenAI
`stream_options.include_usage` contract: one extra event immediately before
`[DONE]`, empty `choices`, terminal aggregate usage. No live vLLM GPU server was
available, so deterministic protocol-source fixtures are used instead of a
runtime capture.
