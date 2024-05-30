import tkinter as tk
from tkinter import ttk
import pandas as pd
import re
from datetime import datetime

# 年号を西暦に変換する関数
def convert_to_gregorian(era_date):
    match = re.match(r'令和(\d+)年(\d+)月(\d+)日の週', era_date)
    if match:
        era_year, month, day = map(int, match.groups())
        year = 2018 + era_year  # 令和元年は2019年
        return datetime(year, month, day)
    return None

# Excelファイルを読み込む
df = pd.read_excel("vegetable_prices.xlsx")

# データを整形（列名に基づいて）
columns = ['year', 'きゃべつ', 'ねぎ', 'レタス', 'ばれいしょ', 'たまねぎ', 'きゅうり', 'トマト', 'にんじん', 'はくさい', 'だいこん']
df.columns = columns

# 年月を変換
df['year'] = df['year'].apply(convert_to_gregorian)

# Tkinterアプリの作成
root = tk.Tk()
root.title("Vegetable Price Checker")
root.geometry("800x600")

# 野菜の種類を選択するためのドロップダウンメニュー
vegetable_label = tk.Label(root, text="Select Vegetable:")
vegetable_label.pack(pady=10)

# 野菜の種類をリストにする
vegetables = columns[1:]  # 'year'を除く
vegetable_var = tk.StringVar(value=vegetables[0])
vegetable_dropdown = ttk.Combobox(root, textvariable=vegetable_var, values=vegetables)
vegetable_dropdown.pack(pady=10)

# 価格情報を表示するラベル
price_label = tk.Label(root, text="", font=("Helvetica", 12))
price_label.pack(pady=20)

# ボタンのクリックイベント
def check_price():
    selected_vegetable = vegetable_var.get()
    # 選択された野菜の価格データを抽出
    veg_data = df[['year', selected_vegetable]].dropna()
    veg_data['year_month'] = veg_data['year'].dt.to_period('M')
    
    # 最新の月を取得
    latest_month = veg_data['year_month'].max()
    latest_month_data = veg_data[veg_data['year_month'] == latest_month]
    latest_month_avg_price = latest_month_data[selected_vegetable].mean()
    
    # 最新の年度を除いた過去の同じ月の平均価格を計算
    latest_year = latest_month.year
    historical_data = veg_data[(veg_data['year_month'].dt.month == latest_month.month) & (veg_data['year'].dt.year < latest_year)]
    historical_avg_price = historical_data.groupby(historical_data['year'].dt.year)[selected_vegetable].mean()
    
    # 結果を表示（表形式）
    result_text = f"Average prices for {selected_vegetable} (Monthly):\n\n"
    result_text += f"{'Year':<10}{'Historical Price':<20}{'Comparison (%)':<20}\n"
    result_text += "-"*50 + "\n"
    
    for year, price in historical_avg_price.items():
        percentage_diff = (latest_month_avg_price - price) / price * 100
        if percentage_diff > 0:
            color = "red"
            sign = "+"
        else:
            color = "blue"
            sign = ""
        result_text += f"{year:<10}{price:<20.2f}\n"
        result_text += f"{'':<10}{'':<20}{sign}{percentage_diff:.2f}%\n"
    
    result_text += "-"*50 + "\n"
    result_text += f"{'Latest month ('+str(latest_month)+')':<10}{latest_month_avg_price:<20.2f}\n"
    
    price_label.config(text=result_text)

# 価格をチェックするボタン
check_button = tk.Button(root, text="Check Price", command=check_price)
check_button.pack(pady=20)

# メインループを開始
root.mainloop()




