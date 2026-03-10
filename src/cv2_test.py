import cv2

import numpy as np
from typing import Optional


def check_opencv_env() -> None:
    """
    驗證 OpenCV 環境配置與版本資訊。
    """
    # 取得版本資訊
    version: str = cv2.__version__
    print(f"OpenCV version: {version}")

    # 建立一個測試矩陣 (Black Image)
    # 使用 Type Hint 確保與 NumPy 的相容性
    test_img: np.ndarray = np.zeros((100, 100, 3), dtype=np.uint8)

    # 測試基本的繪圖功能
    cv2.circle(test_img, (50, 50), 20, (255, 255, 255), -1)

    print("OpenCV 環境驗證成功。")


if __name__ == "__main__":
    check_opencv_env()
