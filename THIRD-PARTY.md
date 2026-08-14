# Third-Party Notices

This sample is licensed under MIT-0 (see [LICENSE](LICENSE)). It also uses the
following third-party models at runtime, none of which are vendored in this
repository — each is fetched from its own upstream source when you run the
corresponding service.

| Model | Used by | License | Source |
|---|---|---|---|
| Qwen3-8B | orchestrator, Stage 3 PII detection (self-hosted vLLM) | Apache-2.0 | https://huggingface.co/Qwen/Qwen3-8B |
| PaddleOCR (PP-OCRv5) | ocr-service, `OCR_ENGINE=paddleocr` (default, CPU) | Apache-2.0 | https://github.com/PaddlePaddle/PaddleOCR |
| PaddleOCR-VL | ocr-service, `OCR_ENGINE=paddleocr-vl` (opt-in, GPU) | Apache-2.0 | https://huggingface.co/PaddlePaddle/PaddleOCR-VL |

All three are used as-is (no fine-tuning, no redistribution of weights in this
repository) via their respective Python packages / self-hosted inference
servers — see [`deploy/docker-compose.yml`](deploy/docker-compose.yml) and
[`deploy/ocr-gpu/Dockerfile`](deploy/ocr-gpu/Dockerfile).
