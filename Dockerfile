# 실습장 PC(16 CPU, RTX 2080 x2)에서 학습/본실험을 돌리기 위한 이미지.
# RTX 2080(Turing, sm_75)은 CUDA 12.1 빌드의 PyTorch 에서 지원된다 (호스트 NVIDIA 드라이버 525 이상 필요).
FROM pytorch/pytorch:2.3.1-cuda12.1-cudnn8-runtime

# 스레드 1개: 프로세스를 여러 개 띄우므로 프로세스마다 스레드를 늘리면 CPU 경합만 생긴다
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    MPLBACKEND=Agg \
    MPLCONFIGDIR=/tmp/mpl \
    OMP_NUM_THREADS=1 \
    MKL_NUM_THREADS=1

WORKDIR /workspace

# 그림의 한글 표기를 위한 나눔 글꼴, 학습 로그용 git(커밋 해시 기록)
RUN apt-get update && apt-get install -y --no-install-recommends fonts-nanum git \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 소스 코드 복사. docker-compose 에서는 ./ooda, ./scripts 를 볼륨으로도 마운트하므로
# 코드만 바뀌었을 때는 다시 빌드하지 않아도 된다 (requirements.txt 가 바뀌면 다시 빌드).
COPY ooda ./ooda
COPY scripts ./scripts

# 기본 명령: 설치·GPU 확인
CMD ["python", "-c", "import torch, stable_baselines3, ooda; print('torch', torch.__version__, '| sb3', stable_baselines3.__version__, '| cuda', torch.cuda.is_available(), '| gpus', torch.cuda.device_count(), [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())])"]
