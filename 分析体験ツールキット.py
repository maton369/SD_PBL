import tkinter as tk
from PIL import Image, ImageTk
import subprocess

class FrameApp:

    def __init__(self, root):
        self.root = root
        self.root.geometry('800x600')
        self.root.title("分析体験ツールキット")

        frames_data = [
            {"row": 0, "column": 0, "image": "データ.jpg", "script": "1.py"},
            {"row": 0, "column": 1, "image": "gazo.jpg", "script": "2.py"},
            {"row": 0, "column": 2, "image": "無題 (6).png", "script": "3.py"},
            {"row": 1, "column": 0, "image": "ayame.jpg", "script": "4.py"},
            {"row": 1, "column": 1, "image": "画像認識プログラム.png", "script": "5.py"},
            {"row": 1, "column": 2, "image": "Positive_or_Negative.png", "script": "6.py"},
        ]

        for i in range(3):  
            self.root.grid_columnconfigure(i, weight=1)
        for i in range(2):  
            self.root.grid_rowconfigure(i, weight=1)

        for frame_data in frames_data:
            frame = tk.Frame(root, borderwidth=2, relief="groove")
            frame.grid(row=frame_data["row"], column=frame_data["column"], padx=5, pady=5, sticky="nsew")
            self.display_image(frame, frame_data["image"])
            self.bind_image_click(frame, frame_data["script"])

    def display_image(self, frame, image_path):
        img = Image.open(image_path)

        frame.original_image = img

        canvas = tk.Canvas(frame)
        canvas.pack(fill=tk.BOTH, expand=True)

        canvas.bind("<Configure>", lambda event, canvas=canvas, img=img: self.on_resize(event, canvas, img))

    def on_resize(self, event, canvas, img):
        canvas_width = event.width
        canvas_height = event.height

        img_resized = img.resize((canvas_width, canvas_height), Image.ANTIALIAS)
        img_tk = ImageTk.PhotoImage(img_resized)

        canvas.delete("all")
        canvas.create_image(0, 0, anchor=tk.NW, image=img_tk)

        canvas.image = img_tk

    def bind_image_click(self, frame, app_path):
        frame.children['!canvas'].bind("<Button-1>", lambda event: self.on_image_click(app_path))

    def on_image_click(self, app_path):
        subprocess.Popen(["python", app_path])

if __name__ == "__main__":
    root = tk.Tk()
    app = FrameApp(root)
    root.mainloop()
