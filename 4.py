import tkinter as tk
from tkinter import messagebox
import tensorflow as tf
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

class IrisClassifierApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Iris Flower Classifier App")

        # テストデータ入力用のEntryウィジェット
        self.test_entry = tk.Entry(self)
        self.test_entry.pack(pady=20, padx=10, ipadx=50)

        # 分類結果表示用のラベル
        self.result_label = tk.Label(self, text="", font=("Helvetica", 16))
        self.result_label.pack(pady=20)

        # 学習ボタン
        self.train_button = tk.Button(self, text="Train Model", command=self.train_model)
        self.train_button.pack(pady=10)

        # テストボタン
        self.test_button = tk.Button(self, text="Test Model", command=self.predict_text)
        self.test_button.pack(pady=10)

        # モデルの初期化
        self.model = None

    def train_model(self):
        self.train_button.config(state=tk.DISABLED)  # 学習ボタンを無効化
        self.result_label.config(text="Training...", fg="orange")  # ラベルを学習中に変更
        self.update_idletasks()  # UIの更新を反映

        # アヤメのデータをロードしてモデルを学習
        X, y = load_iris(return_X_y=True)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # モデルの構築
        self.model = tf.keras.Sequential([
            tf.keras.layers.Dense(10, input_shape=(4,), activation='relu'),  # 入力層
            tf.keras.layers.Dense(8, activation='relu'),  # 隠れ層
            tf.keras.layers.Dense(3, activation='softmax')  # 出力層 (3クラス分類)
        ])

        # モデルのコンパイル
        self.model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

        # モデルの学習
        self.model.fit(X_train, y_train, epochs=50, batch_size=2, verbose=1)

        self.result_label.config(text="Training Completed", fg="green")  # 学習完了を表示
        self.train_button.config(state=tk.NORMAL)  # 学習ボタンを再度有効化

    def predict_text(self):
        try:
            # Entryからテストデータを取得
            test_input_str = self.test_entry.get().strip()
        
            # カンマで分割して4つの数値を取得
            values = test_input_str.split(',')
        
            if len(values) != 4:
                raise ValueError("Please enter four valid numeric values separated by commas.")

            # 数値に変換
            test_input = [float(val) for val in values]

            # 入力データをモデルに入力するために形状を整える
            test_input = np.array([test_input])

            # テストデータをモデルに入力して分類
            outputs = self.model.predict(test_input)
            predicted = np.argmax(outputs, axis=1)

            if predicted == 0:
                self.result_label.config(text="Predicted: Setosa", fg="green")
            elif predicted == 1:
                self.result_label.config(text="Predicted: Versicolor", fg="blue")
            elif predicted == 2:
                self.result_label.config(text="Predicted: Virginica", fg="red")

            # テストデータ入力欄をクリアする
            self.test_entry.delete(0, tk.END)

        except ValueError as ve:
            messagebox.showerror("Error", str(ve))
        except Exception as e:
            messagebox.showerror("Error", f"Error occurred during prediction: {str(e)}")

if __name__ == "__main__":
    app = IrisClassifierApp()
    app.mainloop()
