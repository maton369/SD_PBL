import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model
import numpy as np
import tkinter as tk
from tkinter import filedialog, messagebox
import os

def build_rnn_model(input_dim, output_dim, input_length, lstm_units, dense_units, dropout_rate, activation):
    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(input_dim=input_dim, output_dim=output_dim, input_length=input_length),
        tf.keras.layers.LSTM(lstm_units, return_sequences=True),
        tf.keras.layers.Dropout(dropout_rate),
        tf.keras.layers.LSTM(lstm_units),
        tf.keras.layers.Dropout(dropout_rate),
        tf.keras.layers.Dense(dense_units, activation=activation),
        tf.keras.layers.Dense(2, activation='softmax')
    ])
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

def preprocess_text(text, tokenizer, max_len):
    sequences = tokenizer.texts_to_sequences([text])
    padded_sequences = pad_sequences(sequences, maxlen=max_len, padding='post')
    return padded_sequences

class TrainingCallback(tf.keras.callbacks.Callback):
    def __init__(self, app):
        super().__init__()
        self.app = app

    def on_epoch_end(self, epoch, logs=None):
        accuracy = logs.get('accuracy')
        self.app.update_training_status(epoch, accuracy)
        self.app.after(0, self.app.update_idletasks)

class SentimentAnalysisApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("感情分析レビュアー")

        self.description_label = tk.Label(self, text=(
            "このアプリケーションは、RNNモデルを使用して与えられたテキストの感情を分析します。\n"
            "まず、「学習開始」をクリックしてIMDbデータセットを使用してモデルを学習します。\n"
            "学習が完了したら、テキストを入力し、「判定」をクリックしてテキストがポジティブかネガティブかを確認します。"
        ), wraplength=700, justify="left")
        self.description_label.pack(pady=10)

        self.imdb_description_label = tk.Label(self, text=(
            "IMDbデータセットは、映画レビューのデータセットで、各レビューはポジティブ（好意的）またはネガティブ（否定的）に分類されています。\n"
            "このデータセットは自然言語処理（NLP）のタスクで広く使用されており、映画レビューのテキストを使用して感情分析を行うことができます。\n"
            "コンピュータは文章の意味は理解できなくても、文章の特徴を学習することはできるという一例です。"
        ), wraplength=700, justify="left")
        self.imdb_description_label.pack(pady=10)

        self.settings_frame = tk.Frame(self)
        self.settings_frame.pack(pady=10)

        tk.Label(self.settings_frame, text="エポック数:").grid(row=0, column=0)
        self.epochs_entry = tk.Entry(self.settings_frame, width=10)
        self.epochs_entry.grid(row=0, column=1)
        self.epochs_entry.insert(tk.END, '10') 

        tk.Label(self.settings_frame, text="LSTMユニット数:").grid(row=1, column=0)
        self.lstm_units_entry = tk.Entry(self.settings_frame, width=10)
        self.lstm_units_entry.grid(row=1, column=1)
        self.lstm_units_entry.insert(tk.END, '64') 

        tk.Label(self.settings_frame, text="Denseユニット数:").grid(row=2, column=0)
        self.dense_units_entry = tk.Entry(self.settings_frame, width=10)
        self.dense_units_entry.grid(row=2, column=1)
        self.dense_units_entry.insert(tk.END, '64')  

        tk.Label(self.settings_frame, text="ドロップアウト率:").grid(row=3, column=0)
        self.dropout_rate_entry = tk.Entry(self.settings_frame, width=10)
        self.dropout_rate_entry.grid(row=3, column=1)
        self.dropout_rate_entry.insert(tk.END, '0.2')  

        tk.Label(self.settings_frame, text="活性化関数:").grid(row=4, column=0)
        self.activation_var = tk.StringVar(value='relu')
        self.activation_menu = tk.OptionMenu(self.settings_frame, self.activation_var, 'sigmoid', 'relu', 'tanh')
        self.activation_menu.grid(row=4, column=1)

        self.text_entry = tk.Entry(self, width=50)
        self.text_entry.pack(pady=20)

        self.start_button = tk.Button(self, text="学習開始", command=self.start_training)
        self.start_button.pack(pady=10)

        self.analyze_button = tk.Button(self, text="判定", command=self.analyze_sentiment, state=tk.DISABLED)
        self.analyze_button.pack(pady=10)

        self.save_button = tk.Button(self, text="モデル保存", command=self.save_model, state=tk.DISABLED)
        self.save_button.pack(pady=10)

        self.load_button = tk.Button(self, text="モデル読み込み", command=self.load_new_model)
        self.load_button.pack(pady=10)

        self.result_label = tk.Label(self, text="", font=("Helvetica", 16))
        self.result_label.pack(pady=20)

        self.training_label = tk.Label(self, text="", font=("Helvetica", 16), fg="red")
        self.training_label.pack(pady=20)

        self.accuracy_label = tk.Label(self, text="", font=("Helvetica", 16), fg="blue")
        self.accuracy_label.pack(pady=20)

        try:
            (train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=10000)
        except Exception as e:
            messagebox.showerror("Error", f"データセットの読み込み中にエラーが発生しました: {e}")
            self.quit()

        word_index = imdb.get_word_index()
        reversed_word_index = dict([(value, key) for (key, value) in word_index.items()])

        self.train_texts = [self.decode_review(text, reversed_word_index) for text in train_data]
        self.train_labels = np.array(train_labels)
        self.test_texts = [self.decode_review(text, reversed_word_index) for text in test_data]
        self.test_labels = np.array(test_labels)

        all_texts = np.concatenate((self.train_texts, self.test_texts), axis=0)
        self.tokenizer = Tokenizer(num_words=10000)
        self.tokenizer.fit_on_texts(all_texts)

        self.model = None
        self.is_training = False

        if os.path.exists('sentiment_rnn_model.h5'):
            self.load_model()

    def decode_review(self, text, reversed_word_index):
        return ' '.join([reversed_word_index.get(i - 3, '?') for i in text])

    def start_training(self):
        if not self.is_training:
            self.is_training = True
            self.start_button.config(state=tk.DISABLED)
            self.training_label.config(text="学習中", fg="red")
            self.accuracy_label.config(text="")
            self.update_idletasks()

            input_dim = 10000
            output_dim = 16
            input_length = 100
            lstm_units = int(self.lstm_units_entry.get())
            dense_units = int(self.dense_units_entry.get())
            dropout_rate = float(self.dropout_rate_entry.get())
            activation = self.activation_var.get()
            epochs = int(self.epochs_entry.get())

            self.model = build_rnn_model(input_dim, output_dim, input_length, lstm_units, dense_units, dropout_rate, activation)

            train_sequences = self.tokenizer.texts_to_sequences(self.train_texts)
            train_padded_sequences = pad_sequences(train_sequences, maxlen=100, padding='post')

            self.model.fit(
                train_padded_sequences, 
                self.train_labels, 
                epochs=epochs, 
                verbose=1, 
                callbacks=[TrainingCallback(self)]
            )

            self.model.save('sentiment_rnn_model.h5')
            self.analyze_button.config(state=tk.NORMAL)
            self.save_button.config(state=tk.NORMAL)
            self.training_label.config(text="学習完了", fg="green")
            self.is_training = False

            self.start_button.config(state=tk.NORMAL)

    def update_training_status(self, epoch, accuracy):
        self.training_label.config(text=f"エポック: {epoch + 1}")
        self.accuracy_label.config(text=f"精度: {accuracy:.4f}")
        self.update_idletasks()

    def analyze_sentiment(self):
        self.result_label.config(text="")
        text = self.text_entry.get().strip()
        if not text:
            messagebox.showwarning("Warning", "テキストを入力してください。")
            return

        processed_text = preprocess_text(text, self.tokenizer, max_len=100)

        prediction = self.model.predict(processed_text)[0]

        if prediction[1] >= 0.5:
            result = "ポジティブ"
        else:
            result = "ネガティブ"

        self.result_label.config(text=f"テキストの感情は {result} です。")

        self.after(5000, self.clear_result)

    def clear_result(self):
        self.result_label.config(text="")

    def load_model(self):
        try:
            self.model = load_model('sentiment_rnn_model.h5')
            self.analyze_button.config(state=tk.NORMAL)
            self.save_button.config(state=tk.NORMAL)
            self.training_label.config(text="モデル読み込み完了", fg="green")
        except Exception as e:
            messagebox.showerror("Error", f"モデルの読み込み中にエラーが発生しました: {e}")

    def save_model(self):
        file_path = filedialog.asksaveasfilename(
            title="モデルファイルを保存",
            defaultextension=".h5",
            filetypes=(("H5 files", "*.h5"), ("All files", "*.*"))
        )
        if file_path:
            try:
                self.model.save(file_path)
                messagebox.showinfo("Info", "モデルを保存しました。")
            except Exception as e:
                messagebox.showerror("Error", f"モデルの保存中にエラーが発生しました: {e}")

    def load_new_model(self):
        file_path = filedialog.askopenfilename(
            title="モデルファイルを選択",
            filetypes=(("H5 files", "*.h5"), ("All files", "*.*"))
        )
        if file_path:
            try:
                self.model = load_model(file_path)
                self.analyze_button.config(state=tk.NORMAL)
                self.training_label.config(text="新しいモデル読み込み完了", fg="green")
            except Exception as e:
                messagebox.showerror("Error", f"モデルの読み込み中にエラーが発生しました: {e}")

if __name__ == "__main__":
    app = SentimentAnalysisApp()
    app.mainloop()
