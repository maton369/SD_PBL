import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageDraw
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, datasets

class MNISTClassifierApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("MNIST CNN Classifier")

        # 開始ボタン
        self.start_button = tk.Button(self, text="開始", command=self.start_training)
        self.start_button.pack(pady=10)

        # 判別ボタン
        self.classify_button = tk.Button(self, text="Classify", command=self.classify_digit, state=tk.DISABLED)
        self.classify_button.pack(pady=10)

        # 判別結果表示用のラベル
        self.result_label = tk.Label(self, text="", font=("Helvetica", 16))
        self.result_label.pack(pady=20)

        # 学習中を示すラベル
        self.training_label = tk.Label(self, text="", font=("Helvetica", 16), fg="red")

        # MNIST CNNモデルの初期化
        self.model = None
        self.is_training = False

    def build_and_train_model(self):
        # MNISTデータセットを読み込む
        (train_images, train_labels), (test_images, test_labels) = datasets.mnist.load_data()

        # データの前処理
        train_images = train_images.reshape((60000, 28, 28, 1)) / 255.0
        test_images = test_images.reshape((10000, 28, 28, 1)) / 255.0

        # CNNモデルを構築
        model = models.Sequential([
            layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.Flatten(),
            layers.Dense(64, activation='relu'),
            layers.Dense(10, activation='softmax')
        ])

        # モデルをコンパイル
        model.compile(optimizer='adam',
                      loss='sparse_categorical_crossentropy',
                      metrics=['accuracy'])

        # モデルを学習
        model.fit(train_images, train_labels, epochs=1, batch_size=64)

        return model

    def preprocess_image(self, image):
        # 画像の前処理
        image = image.resize((28, 28))
        image = image.convert('L')  # グレースケールに変換
        image_array = np.array(image) / 255.0
        image_array = image_array.reshape((1, 28, 28, 1))
        return image_array

    def start_training(self):
        if not self.is_training:
            self.is_training = True
            self.start_button.config(state=tk.DISABLED)  # 開始ボタンを無効化
            self.training_label.config(text="学習中", fg="red")
            self.training_label.pack(pady=20)
            self.update_idletasks()  # UIを更新

            # モデルの初期化と学習
            self.model = self.build_and_train_model()

            # 学習完了メッセージとボタンの有効化
            self.start_button.config(state=tk.NORMAL)  # 開始ボタンを有効化
            self.classify_button.config(state=tk.NORMAL)  # 判別ボタンを有効化
            self.training_label.pack_forget()  # 学習中表示を非表示に
            messagebox.showinfo("Info", "学習が完了しました。")

    def classify_digit(self):
        # ファイルを選択して手書き数字画像を読み込む
        file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.png;*.jpg;*.jpeg")])
        if file_path:
            try:
                input_image = Image.open(file_path)

                # 画像を前処理
                input_image = self.preprocess_image(input_image)

                # モデルで数字を予測
                predictions = self.model.predict(input_image)
                predicted_digit = np.argmax(predictions)

                # 判別結果を表示
                self.result_label.config(text=f"Predicted Digit: {predicted_digit}")

            except Exception as e:
                messagebox.showerror("Error", f"Error occurred: {str(e)}")

if __name__ == "__main__":
    app = MNISTClassifierApp()
    app.mainloop()
