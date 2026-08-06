from src.pipeline import run_full_experiment
from src.config import OUTPUT_DIR  # 방금 수정한 config에서 경로를 직접 가져와 봅니다.

if __name__ == "__main__":
    # 실행하자마자 어떤 경로를 읽고 있는지 바로 출력해서 확인!
    print("★ 현재 파이썬이 읽고 있는 경로:", OUTPUT_DIR) 
    print("학습 파이프라인을 시작합니다...")
    run_full_experiment()