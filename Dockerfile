FROM pytorch/pytorch:2.6.0-cuda12.6-cudnn9-devel

# CUDA 라이브러리 경로 환경변수 설정 for onnxruntime-gpu
ENV LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda/lib64
ENV LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/opt/conda/lib/python3.11/site-packages/nvidia/cudnn/lib

# 기본 시스템 패키지 업데이트 및 필요한 도구 설치
RUN apt-get update && apt-get install -y \
    git \
    wget \
    curl \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Python 패키지 설치를 위한 requirements.txt를 복사 (필요한 경우)
COPY requirements.txt .
RUN pip install -r requirements.txt

# test 폴더의 파일들만 컨테이너로 복사
COPY test/ /test/

# 컨테이너가 실행될 때의 기본 명령
CMD ["bash"]
