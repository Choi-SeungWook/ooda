# 실습장 PC(16 CPU, RTX 2080 x2)에서 학습/본실험을 돌리기 위한 이미지.
# RTX 2080(Turing, sm_75)은 CUDA 12.1 빌드의 PyTorch 에서 지원된다.
FROM pytorch/pytorch:2.3.1-cuda12.1-cudnn8-runtime

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    MPLBACKEND=Agg

WORKDIR /workspace

# 그림의 한글 표기를 위한 나눔 글꼴
RUN apt-get update && apt-get install -y --no-install-recommends fonts-nanum \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 소스 코드 복사 (results/ 는 볼륨으로 마운트하므로 이미지에 넣지 않음)
COPY ooda ./ooda
COPY scripts ./scripts

# 기본 명령: 설치 확인. 실제 실행은 docker compose run 으로 스크립트를 지정한다.
CMD ["python", "-c", "import torch, ooda; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), torch.cuda.device_count())"]
