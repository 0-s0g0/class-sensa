import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.lines as mlines # 凡例のために追加
import numpy as np # ダミーデータ作成のために追加

# グラフの色分けの範囲を定義
# black:[0-30, 90-120, 180-210, 270-300, 360-390, 450-480]
# blue:[30-90, 210-270, 390-450]
# red:[120-180, 300-360, 480-540]
color_ranges = {
    'black': [[0, 30], [90, 120], [180, 210], [270, 300], [360, 390], [450, 480]],
    'blue': [[30, 90], [210, 270], [390, 450]],
    'red': [[120, 180], [300, 360], [480, 540]]
}

# CSVファイルを読み込む（複数のエンコーディングを試す）
file_path = 'NIR2.csv'
encodings = ['utf-8', 'shift_jis', 'cp932', 'latin1', 'iso-8859-1']

df = None
for encoding in encodings:
    try:
        df = pd.read_csv(file_path, encoding=encoding)
        print(f"成功: {encoding} エンコーディングで読み込みました")
        break
    except Exception as e:
        print(f"失敗: {encoding} - {str(e)[:50]}")
        continue

if df is None:
    # ファイルが見つからなかった、または読み込めなかった場合はダミーデータを作成して続行する
    print("指定されたファイルが見つからなかったか、読み込めませんでした。ダミーデータを作成してプロットします。")
    # ダミーデータを作成
    x_dummy = np.linspace(0, 540, 1000)
    y_dummy = np.sin(x_dummy / 50) + np.cos(x_dummy / 20) * 0.5 + x_dummy / 540 + 2
    df = pd.DataFrame({'A列': x_dummy, 'B列': y_dummy})
    
# A列とB列のデータを取得（列名ではなく、列の位置で指定）
# x_data でソートされていない可能性があるため、一旦ソートし、インデックスをリセット
df_sorted = df.iloc[:, [0, 1]].sort_values(by=df.columns[0]).reset_index(drop=True)
x_data = df_sorted.iloc[:, 0] # A列（1列目）
y_data = df_sorted.iloc[:, 1] # B列（2列目）

# グラフを作成
plt.figure(figsize=(10, 6))

# 区間ごとにプロット
for color, ranges in color_ranges.items():
    for L, R in ranges:
        
        # L以上R以下のデータポイントのインデックスを取得
        segment_mask = (x_data >= L) & (x_data <= R)
        
        # 連続性を保つための追加処理:
        # Lの直前にある点（x_data < L のうち、x_data が最大のもの）
        prev_mask = x_data < L
        if prev_mask.any():
            prev_index = x_data[prev_mask].idxmax()
            if prev_index is not None:
                segment_mask = segment_mask | (x_data.index == prev_index)
        
        # Rの直後にある点（x_data > R のうち、x_data が最小のもの）
        next_mask = x_data > R
        if next_mask.any():
            next_index = x_data[next_mask].idxmin()
            if next_index is not None:
                segment_mask = segment_mask | (x_data.index == next_index)

        
        if segment_mask.any():
            # 抽出したインデックスで x と y データを取得
            segment_x = x_data.loc[segment_mask]
            segment_y = y_data.loc[segment_mask]
            
            # 各セグメントをプロット。
            plt.plot(segment_x, segment_y, color=color, linewidth=2, zorder=3)

# グラフの装飾
col_a = df_sorted.columns[0]
col_b = df_sorted.columns[1]

plt.xlabel('Time[s]')
plt.ylabel('Oxy Hb')
plt.title(f'{col_a} vs {col_b} ')
plt.grid(True)

# 凡例を手動で作成（表示の都合上、一部の範囲を省略）
def truncate_label(ranges_list):
    if len(ranges_list) > 3:
        return ', '.join(ranges_list[:3]) + ', ...'
    return ', '.join(ranges_list)

black_ranges_list = [f'[{l}-{r}]' for l, r in color_ranges['black']]
blue_ranges_list = [f'[{l}-{r}]' for l, r in color_ranges['blue']]
red_ranges_list = [f'[{l}-{r}]' for l, r in color_ranges['red']]

black_line = mlines.Line2D([], [], color='black', linewidth=2, label=f'黒: {truncate_label(black_ranges_list)} (s)')
blue_line = mlines.Line2D([], [], color='blue', linewidth=2, label=f'青: {truncate_label(blue_ranges_list)} (s)')
red_line = mlines.Line2D([], [], color='red', linewidth=2, label=f'赤: {truncate_label(red_ranges_list)} (s)')

plt.legend(handles=[black_line, blue_line, red_line], title=f"{col_a} の範囲", loc='best', fontsize='small')

# x軸の範囲を調整して、全ての区間（0から540）が見えるようにする
plt.xlim(0, 540)

# グラフをファイルに保存
plt.savefig('A_vs_B_colored.png')