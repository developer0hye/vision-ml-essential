import onnxruntime as ort
import json
import torch
import torch.nn as nn
import numpy as np
import os

class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.fc = nn.Linear(10, 2)
    
    def forward(self, x):
        return self.fc(x)

def test_onnx_gpu():
    # ONNX Runtime 버전 출력
    print(f"\n=== ONNX Runtime Version ===")
    print(f"Version: {ort.__version__}")
    
    # 임시 ONNX 파일 경로
    onnx_path = "test_model.onnx"
    
    # PyTorch 모델 생성 및 ONNX 변환
    model = SimpleModel()
    model.eval()
    
    # 더미 입력 생성
    dummy_input = torch.randn(1, 10)
    
    # ONNX 모델 내보내기
    torch.onnx.export(model, dummy_input, onnx_path,
                     input_names=['input'],
                     output_names=['output'],
                     dynamic_axes={'input': {0: 'batch_size'},
                                 'output': {0: 'batch_size'}})
    
    print("\n=== ONNX Model Created ===")
    
    # ONNX Runtime 프로바이더 정보 가져오기
    providers = ort.get_available_providers()
    print("\n=== Available ONNX Runtime Providers ===")
    print(json.dumps(providers, indent=2))

    # GPU 정보 가져오기
    gpu_info = ort.get_device()
    print("\n=== GPU Information ===")
    print(json.dumps(gpu_info, indent=2))

    # CUDA 프로바이더가 있는지 확인
    if 'CUDAExecutionProvider' in providers:
        print("\n✅ CUDA is properly configured for ONNX Runtime")
        
        # CUDA 세션 생성 테스트
        try:
            sess_options = ort.SessionOptions()
            session = ort.InferenceSession(onnx_path, sess_options, 
                                         providers=['CUDAExecutionProvider'])
            current_providers = session.get_providers()
            
            if 'CUDAExecutionProvider' not in current_providers:
                raise RuntimeError(f"CUDAExecutionProvider is not in the current providers. Current Providers: {current_providers}")
            
            print("✅ Successfully created CUDA session")
            
            # 실제 추론 테스트
            input_data = np.random.randn(1, 10).astype(np.float32)
            result = session.run(None, {'input': input_data})
            print("✅ Successfully ran inference")
            print(f"Input shape: {input_data.shape}")
            print(f"Output shape: {result[0].shape}")
            
        except Exception as e:
            print(f"❌ Failed to create CUDA session or run inference: {str(e)}")
    else:
        print("\n❌ CUDA is not available for ONNX Runtime")
    
    # 테스트 후 임시 파일 삭제
    if os.path.exists(onnx_path):
        os.remove(onnx_path)

if __name__ == "__main__":
    test_onnx_gpu() 