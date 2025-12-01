#!/usr/bin/env python3
import os
import shutil
import random
from pathlib import Path

def sample_caltech101_dataset(source_dir, target_dir, num_classes=100, sample_ratio=0.5):
    """
    Caltech-101 데이터셋에서 지정된 개수의 클래스를 선택하고,
    각 클래스의 이미지를 절반만 샘플링하여 새로운 폴더에 복사
    
    Args:
        source_dir (str): 원본 Caltech-101 데이터셋 경로
        target_dir (str): 새로운 데이터셋을 저장할 경로
        num_classes (int): 선택할 클래스 개수 (기본값: 100)
        sample_ratio (float): 각 클래스에서 샘플링할 비율 (기본값: 0.5)
    """
    
    # 경로 설정
    source_path = Path(source_dir)
    target_path = Path(target_dir)
    
    # 타겟 디렉토리 생성
    target_path.mkdir(parents=True, exist_ok=True)
    
    # 모든 클래스 디렉토리 찾기 (BACKGROUND_Google 제외)
    all_classes = []
    for item in source_path.iterdir():
        if item.is_dir() and item.name != "BACKGROUND_Google":
            all_classes.append(item.name)
    
    print(f"전체 클래스 개수 (BACKGROUND_Google 제외): {len(all_classes)}")
    
    # 재현성을 위한 시드 설정
    random.seed(42)
    
    # 100개 클래스 랜덤 선택
    if len(all_classes) >= num_classes:
        selected_classes = random.sample(all_classes, num_classes)
    else:
        selected_classes = all_classes
        print(f"경고: 전체 클래스가 {len(all_classes)}개뿐입니다. 모든 클래스를 사용합니다.")
    
    print(f"선택된 클래스 개수: {len(selected_classes)}")
    print(f"선택된 클래스들: {sorted(selected_classes)}")
    
    # 각 클래스별로 이미지 샘플링 및 복사
    total_copied = 0
    
    for class_name in selected_classes:
        source_class_dir = source_path / class_name
        target_class_dir = target_path / class_name
        
        # 타겟 클래스 디렉토리 생성
        target_class_dir.mkdir(exist_ok=True)
        
        # 이미지 파일 목록 가져오기
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff'}
        image_files = []
        
        for file_path in source_class_dir.iterdir():
            if file_path.is_file() and file_path.suffix.lower() in image_extensions:
                image_files.append(file_path)
        
        # 이미지 개수 출력
        print(f"{class_name}: {len(image_files)}개 이미지 발견")
        
        # 절반 샘플링
        num_to_sample = max(1, int(len(image_files) * sample_ratio))
        sampled_files = random.sample(image_files, num_to_sample)
        
        # 파일 복사
        copied_count = 0
        for img_file in sampled_files:
            try:
                target_file = target_class_dir / img_file.name
                shutil.copy2(img_file, target_file)
                copied_count += 1
            except Exception as e:
                print(f"파일 복사 실패: {img_file} -> {e}")
        
        print(f"  -> {copied_count}개 이미지 복사 완료")
        total_copied += copied_count
    
    print(f"\n총 {total_copied}개 이미지가 {len(selected_classes)}개 클래스에 대해 복사되었습니다.")
    print(f"새 데이터셋 경로: {target_path}")
    
    # 클래스 목록을 파일로 저장
    class_list_file = target_path / "class_list.txt"
    with open(class_list_file, 'w') as f:
        for class_name in sorted(selected_classes):
            f.write(f"{class_name}\n")
    
    print(f"클래스 목록이 {class_list_file}에 저장되었습니다.")


if __name__ == "__main__":
    # 설정
    SOURCE_DIR = "/data/caltech101/101_ObjectCategories"
    TARGET_DIR = "/data/caltech101/100_kd"
    NUM_CLASSES = 100
    SAMPLE_RATIO = 0.5  # 각 클래스의 50% 이미지만 사용
    
    # 데이터셋 샘플링 실행
    sample_caltech101_dataset(
        source_dir=SOURCE_DIR,
        target_dir=TARGET_DIR,
        num_classes=NUM_CLASSES,
        sample_ratio=SAMPLE_RATIO
    )
    
    # 결과 확인
    target_path = Path(TARGET_DIR)
    if target_path.exists():
        class_dirs = [d for d in target_path.iterdir() if d.is_dir()]
        print(f"\n최종 확인: {len(class_dirs)}개 클래스 디렉토리가 생성되었습니다.")
        
        # 각 클래스별 이미지 개수 확인
        total_images = 0
        for class_dir in sorted(class_dirs):
            image_count = len([f for f in class_dir.iterdir() if f.is_file()])
            print(f"  {class_dir.name}: {image_count}개 이미지")
            total_images += image_count
        
        print(f"\n전체 이미지 개수: {total_images}개")