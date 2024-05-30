import tkinter as tk
from tkinter import filedialog, messagebox
import pandas as pd
import matplotlib.pyplot as plt

class DataVisualizerApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Data Visualizer App")

        # ファイル選択ボタン
        self.select_button = tk.Button(self, text="Select File", command=self.load_file)
        self.select_button.pack(pady=20)

        # 表示用のフレーム
        self.display_frame = tk.Frame(self)
        self.display_frame.pack(pady=20)

    def load_file(self):
        file_path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv"), ("Text files", "*.txt")])
        if file_path:
            try:
                if file_path.endswith(".csv"):
                    self.visualize_csv(file_path)
                elif file_path.endswith(".txt"):
                    self.display_text(file_path)
                else:
                    messagebox.showerror("Error", "Unsupported file type")

            except Exception as e:
                messagebox.showerror("Error", f"Error occurred: {str(e)}")

    def visualize_csv(self, file_path):
        # CSVファイルを読み込んでデータフレームに変換
        df = pd.read_csv(file_path)

        # データをプロット
        plt.figure(figsize=(8, 6))
        for column in df.columns:
            plt.plot(df[column], label=column)
        plt.xlabel("Index")
        plt.ylabel("Value")
        plt.title("CSV File Data Visualization")
        plt.legend()
        plt.show()

    def display_text(self, file_path):
        # TXTファイルを読み込んでテキストデータを表示
        with open(file_path, 'r') as file:
            text = file.read()

        text_display = tk.Text(self.display_frame, width=60, height=20, wrap=tk.WORD)
        text_display.insert(tk.END, text)
        text_display.pack()

if __name__ == "__main__":
    app = DataVisualizerApp()
    app.mainloop()
