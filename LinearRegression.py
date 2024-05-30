import tkinter as tk
from tkinter import filedialog, messagebox
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

class LinearRegressionApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Linear Regression Model App")

        # ファイルを読み込むボタン
        self.load_button = tk.Button(self, text="Load TXT File", command=self.load_txt)
        self.load_button.pack(pady=20)

        # 学習ボタン
        self.train_button = tk.Button(self, text="Train Model", command=self.train_model, state=tk.DISABLED)
        self.train_button.pack(pady=10)

        # テストデータ入力用のエントリー
        self.test_entry = tk.Entry(self)
        self.test_entry.pack(pady=10, padx=20, ipadx=50)

        # テストボタン
        self.test_button = tk.Button(self, text="Test Model", command=self.test_model, state=tk.DISABLED)
        self.test_button.pack(pady=10)

        # モデルとデータ
        self.model = None
        self.X_test = None
        self.y_test = None

    def load_txt(self):
        file_path = filedialog.askopenfilename(filetypes=[("Text files", "*.txt")])
        if file_path:
            try:
                with open(file_path, 'r') as file:
                    lines = file.readlines()

                # 学習データを取得
                data = {'feature': [], 'target': []}
                for line in lines:
                    parts = line.strip().split(',')
                    if len(parts) == 2:
                        feature, target = float(parts[0]), float(parts[1])
                        data['feature'].append(feature)
                        data['target'].append(target)

                df = pd.DataFrame(data)

                # 特徴量とターゲットを分割
                X = df[['feature']]
                y = df['target']

                # 学習データとテストデータに分割
                X_train, self.X_test, y_train, self.y_test = train_test_split(X, y, test_size=0.2, random_state=42)

                # 学習ボタンを有効化
                self.train_button.config(state=tk.NORMAL)
                messagebox.showinfo("Success", "TXT file loaded successfully.")

            except Exception as e:
                messagebox.showerror("Error", f"Error occurred: {str(e)}")

    def train_model(self):
        if self.X_test is not None and self.y_test is not None:
            try:
                self.model = LinearRegression()
                self.model.fit(self.X_test, self.y_test)
                messagebox.showinfo("Success", "Model trained successfully.")
                self.test_button.config(state=tk.NORMAL)  # テストボタンを有効化
            except Exception as e:
                messagebox.showerror("Error", f"Error occurred during model training: {str(e)}")
        else:
            messagebox.showwarning("Warning", "No test data available. Please load a TXT file.")

    def test_model(self):
        if self.model is not None and self.X_test is not None and self.y_test is not None:
            try:
                # テストデータを入力から取得
                test_input_str = self.test_entry.get().strip()
                if not test_input_str:
                    raise ValueError("Please enter a valid numeric value for test input.")

                test_input = float(test_input_str)
                X_test_input = pd.DataFrame({'feature': [test_input]})

                # テストデータでモデルを評価
                y_pred = self.model.predict(X_test_input)

                messagebox.showinfo("Model Prediction", f"Predicted output: {y_pred[0]:.4f}")
            except ValueError as ve:
                messagebox.showerror("Error", str(ve))
            except Exception as e:
                messagebox.showerror("Error", f"Error occurred during model testing: {str(e)}")
        else:
            messagebox.showwarning("Warning", "No model or test data available. Please train the model first.")

if __name__ == "__main__":
    app = LinearRegressionApp()
    app.mainloop()
