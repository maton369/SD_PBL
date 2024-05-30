import tkinter as tk
import numpy as np
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

class LinearRegressionApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Linear Regression with Tkinter")

        self.points = []  # データポイントを保持するリスト

        # Canvas設定
        self.canvas = tk.Canvas(self, width=400, height=400, bg="white")
        self.canvas.pack(padx=20, pady=20)

        # ボタン設定
        self.clear_button = tk.Button(self, text="Clear Points", command=self.clear_points)
        self.clear_button.pack(side=tk.LEFT, padx=10, pady=10)

        self.plot_button = tk.Button(self, text="Plot Regression Line", command=self.plot_regression)
        self.plot_button.pack(side=tk.LEFT, padx=10, pady=10)

        # プロット用のフィギュア
        self.fig = Figure(figsize=(5, 4), dpi=100)
        self.ax = self.fig.add_subplot(111)

        # グラフを描画するキャンバス
        self.graph_canvas = FigureCanvasTkAgg(self.fig, master=self)
        self.graph_canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)

        # Canvasにマウスイベントをバインド
        self.canvas.bind("<Button-1>", self.add_point)

    def add_point(self, event):
        # Canvas上でクリックされた座標を取得してデータポイントを追加
        self.points.append((event.x, event.y))
        self.canvas.create_oval(event.x - 3, event.y - 3, event.x + 3, event.y + 3, fill="black")

    def clear_points(self):
        # データポイントをクリアしてCanvasをクリア
        self.points = []
        self.canvas.delete("all")
        self.ax.clear()
        self.graph_canvas.draw()

    def plot_regression(self):
        if len(self.points) < 2:
            tk.messagebox.showerror("Error", "At least 2 points are required for linear regression.")
            return

        # データポイントをNumPy配列に変換
        x = np.array([p[0] for p in self.points]).reshape(-1, 1)
        y = np.array([p[1] for p in self.points])

        # 線形回帰モデルを構築してフィット
        model = LinearRegression()
        model.fit(x, y)

        # 回帰直線のプロット
        self.ax.clear()
        self.ax.scatter(x, y, color='blue', label='Data Points')
        self.ax.plot(x, model.predict(x), color='red', label='Regression Line')
        self.ax.set_xlabel('X')
        self.ax.set_ylabel('Y')
        self.ax.legend()

        # グラフを更新して表示
        self.graph_canvas.draw()

if __name__ == "__main__":
    app = LinearRegressionApp()
    app.mainloop()
