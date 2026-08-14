# PaddleOCR-VL, pre-baked and served with vLLM. This is what engine_paddleocr_vl.py
# (src/ocr-service/) talks to over PADDLE_VL_ENDPOINT — it is opt-in (OCR_ENGINE=
# paddleocr-vl) and needs a GPU host; the sample's default CPU engine (engine_
# paddleocr.py) does not need this image at all.
#
# base: PaddlePaddle's official vLLM-serving image, referenced by digest so the
# build is reproducible regardless of what "latest" resolves to later.
FROM ccr-2vdh3abv-pub.cnc.bj.baidubce.com/paddlepaddle/paddleocr-genai-vllm-server@sha256:5713fd30ab76094b7b6a20d95fd8e26fa9dc452bcc90ccb16f1fb056bd2a0f4d

# Model weights, pinned by exact HuggingFace revision for a reproducible build:
#   models/PaddleOCR-VL-1.5-0.9B  <- hf.co/PaddlePaddle/PaddleOCR-VL-1.5
#     @ revision 2a4195faa5e7914c12f2fc601d72c81caf8d2da5
#   paddlex-cache/PP-DocLayoutV2  <- hf.co/PaddlePaddle/PaddleOCR-VL (subfolder PP-DocLayoutV2)
#     @ revision 7fa00a8c55b735ba51ba49a9058f3f9c57a99a11
#
# Fetch them yourself before building, e.g.:
#   huggingface-cli download PaddlePaddle/PaddleOCR-VL-1.5 \
#     --revision 2a4195faa5e7914c12f2fc601d72c81caf8d2da5 \
#     --local-dir models/PaddleOCR-VL-1.5-0.9B
#   huggingface-cli download PaddlePaddle/PaddleOCR-VL \
#     --revision 7fa00a8c55b735ba51ba49a9058f3f9c57a99a11 \
#     --include "PP-DocLayoutV2/*" --local-dir paddlex-cache
#
# COPY-only build — no compilation, so the base image's platform (linux/amd64)
# is preserved regardless of the host architecture running `docker build`.
COPY models/PaddleOCR-VL-1.5-0.9B /models/PaddleOCR-VL-1.5-0.9B
COPY paddlex-cache/PP-DocLayoutV2 /root/.paddlex/official_models/PP-DocLayoutV2
