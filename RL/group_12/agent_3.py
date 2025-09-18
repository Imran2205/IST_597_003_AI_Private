import gymnasium
import miniwob
import numpy as np
import cv2
from image_analyzer_starter_code import ImageDetector
import time


class ShapeDetectionAgent:
    def __init__(self):
        self.detector = ImageDetector()

    def get_shape_color(self, image):
        """画像から主要な色を検出"""
        # BGR to HSV変換
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        # 色の範囲を定義
        color_ranges = {
            'green': ([40, 50, 50], [80, 255, 255]),
            'red': ([0, 50, 50], [10, 255, 255]),
            'blue': ([110, 50, 50], [130, 255, 255])
        }

        max_pixels = 0
        dominant_color = None

        for color, (lower, upper) in color_ranges.items():
            mask = cv2.inRange(hsv, np.array(lower), np.array(upper))
            pixel_count = np.sum(mask > 0)
            if pixel_count > max_pixels:
                max_pixels = pixel_count
                dominant_color = color

        return dominant_color if max_pixels > 100 else None

    def detect_shape(self, image):
        """画像から図形を検出"""
        # グレースケール変換
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # 二値化（閾値を調整）
        _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

        # 輪郭検出
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            return 'unknown'

        # 最大の輪郭を取得
        contour = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(contour)

        # 小さすぎる輪郭は無視（ノイズ対策）
        if area < 100:
            return 'unknown'

        perimeter = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.04 * perimeter, True)

        # 形状の特徴を分析
        hull = cv2.convexHull(contour)
        hull_area = cv2.contourArea(hull)
        solidity = float(area) / hull_area if hull_area > 0 else 0

        # デバッグ情報の出力
        print(f"Shape analysis - vertices: {len(approx)}, solidity: {solidity:.2f}")

        # 円の判定基準を厳格化
        if len(approx) > 6 and solidity > 0.90:
            return 'circle'
        # 三角形の判定
        elif len(approx) == 3:
            return 'triangle'
        # 四角形の判定（正方形と長方形の区別）
        elif len(approx) == 4:
            x, y, w, h = cv2.boundingRect(approx)
            aspect_ratio = float(w) / h
            if 0.95 <= aspect_ratio <= 1.05:
                return 'square'
            else:
                return 'rectangle'

        return 'unknown'

    def analyze_region(self, screenshot):
        """スクリーンショットから図形領域を分析"""
        # 図形が表示される領域を抽出（領域の調整が必要かもしれません）
        shape_region = screenshot[100:200, 100:200]

        # 色の検出を最初に行う
        color = self.get_shape_color(shape_region)

        if color:  # 色が検出された場合は図形である可能性が高い
            shape = self.detect_shape(shape_region)
            print(f"Color detected: {color}, Shape detected: {shape}")
            return shape

        # 色が検出されない場合のみOCRを試行
        ocr_result = self.detector.detect_content(shape_region)
        print(f"OCR result: {ocr_result}")

        if ocr_result['type'] in ['letter', 'number']:
            confidence = ocr_result.get('confidence', 0)
            if confidence > 0.8:  # 信頼度の閾値を設定
                return ocr_result['type']

        # 最後の手段として形状検出を試みる
        return self.detect_shape(shape_region)

    def find_button(self, dom_elements, target_type):
        """ボタン要素を検索"""
        print(f"Looking for button: {target_type}")  # デバッグ出力

        for element in dom_elements:
            # ボタンの属性をより詳細にチェック
            if ('tag' in element and
                    'text' in element and
                    element['tag'] == 'button'):
                print(f"Found button with text: {element['text']}")  # デバッグ出力

                if element['text'].strip().lower() == target_type.lower():
                    return element
        return None

    def get_action(self, button):
        """ボタンクリックのアクションを生成"""
        if button:
            # ボタンの座標を計算
            x = float(button['left'][0]) + float(button['width'][0]) / 2
            y = float(button['top'][0]) + float(button['height'][0]) / 2

            print(f"Clicking button with text: {button['text']} at ({x}, {y})")

            return {
                "action_type": 2,  # CLICK_ELEMENT
                "coords": np.array([x, y], dtype=np.float32),  # 座標も含める
                "ref": button["ref"]
            }

        print("No button found, returning NO_OP action")
        return {
            "action_type": 0,  # NO_OP
            "coords": np.array([0, 0], dtype=np.float32),
            "key": 0
        }

    def __call__(self, observation):
        """エージェントのメイン処理"""
        # 図形の種類を特定
        shape_type = self.analyze_region(observation['screenshot'])
        print(f"Detected shape: {shape_type}")

        # ボタンを検索
        button = self.find_button(observation['dom_elements'], shape_type)

        # アクションを返す
        return self.get_action(button)


def main():
    env = gymnasium.make('miniwob/identify-shape-v1', render_mode='human')
    agent = ShapeDetectionAgent()

    try:
        while True:
            observation, info = env.reset()
            time.sleep(0.5)

            while True:
                action = agent(observation)

                time.sleep(1.5)

                # アクションをそのまま環境に渡す
                observation, reward, terminated, truncated, info = env.step(action)

                if terminated or truncated:
                    print(f"Episode finished with reward: {reward}")
                    time.sleep(1)
                    break

    finally:
        env.close()


if __name__ == "__main__":
    main()