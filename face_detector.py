#!/usr/bin/env python
# -*- coding: utf-8 -*-

import cv2
import mediapipe as mp

# =========================================
# 🧩 Mediapipe 초기화 (Face Detection으로 변경)
# =========================================
# Face Detection 솔루션 사용
mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils
mp_styles = mp.solutions.drawing_styles

face_detection = mp_face_detection.FaceDetection(
    model_selection=1,             # 0 (빠름) 또는 1 (정확, 원거리)
    min_detection_confidence=0.5   # 탐지 신뢰도
)

# =========================================
# 📸 카메라 연결
# =========================================
# cap = cv2.VideoCapture(0)            # 기본 카메라 사용시
cap = cv2.VideoCapture("face.mp4")   # 동영상 파일 사용 시 (실습을 위해 face.mp4로 가정) 

print("📷 MediaPipe Face Detection 시작 — ESC를 눌러 종료합니다.")

while cap.isOpened():
    success, image = cap.read()
    if not success:
        print("⚠️ 프레임을 읽지 못했습니다. 카메라 연결 또는 파일 경로를 확인하세요.")
        break

    # 좌우 반전 (셀카 뷰)
    image = cv2.flip(image, 1)

    # BGR → RGB 변환
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # 얼굴 검출 수행
    result = face_detection.process(image_rgb)

    # 🧑‍💻 얼굴 검출 결과 표시 (바운딩 박스)
    if result.detections:
        for detection in result.detections:
            # 얼굴 바운딩 박스와 6개의 키포인트 표시
            mp_drawing.draw_detection(
                image, 
                detection, 
                mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2),  # 바운딩 박스 스타일
                mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2)   # 키포인트 스타일
            )

    # 화면 표시
    cv2.imshow('🧑‍💻 MediaPipe Face Detection', image)

    # ESC 키로 종료
    if cv2.waitKey(5) & 0xFF == 27:
        print("👋 종료합니다.")
        break

# =========================================
# 🔚 종료 처리
# =========================================
cap.release()
cv2.destroyAllWindows()