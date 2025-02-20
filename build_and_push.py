import subprocess
import datetime
import re
from pathlib import Path

def get_base_tag():
    # Dockerfile에서 베이스 이미지 태그 추출
    dockerfile = Path('Dockerfile').read_text()
    match = re.search(r'FROM pytorch/pytorch:(.+)', dockerfile)
    if match:
        return match.group(1)
    raise ValueError("Could not find pytorch base image tag in Dockerfile")

def build_and_push():
    # 현재 시간을 YYYYMMDDHHMMSS 형식으로 가져오기
    timestamp = datetime.datetime.now().strftime('%Y%m%d%H%M%S')
    
    # 이미지 이름 설정
    base_tag = get_base_tag()
    image_name = "developer0hye/vision-ml-essential"
    image_tag = f"pytorch{base_tag}-{timestamp}"
    full_image_name = f"{image_name}:{image_tag}"
    
    try:
        # Docker 이미지 빌드
        print(f"Building Docker image: {full_image_name}")
        subprocess.run(["docker", "build", "-t", full_image_name, "."], check=True)
        
        # Docker Hub에 이미지 푸시
        print(f"Pushing Docker image to Docker Hub")
        subprocess.run(["docker", "push", full_image_name], check=True)
        
        print(f"Successfully built and pushed: {full_image_name}")
        
    except subprocess.CalledProcessError as e:
        print(f"Error occurred: {e}")
        raise
    
if __name__ == "__main__":
    build_and_push() 