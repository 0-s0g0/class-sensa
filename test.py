import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import matplotlib.lines as mlines

# --- 1. ファイルの読み込みとデータ特定 ---
file_path = 'NIR2.csv'
encodings = ['utf-8', 'shift_jis', 'cp932', 'latin1', 'iso-8859-1']

df = None
print(f"ファイル '{file_path}' の読み込みを試行します。")

# CSVファイルを読み込む（ダミーデータ処理も含む）
for encoding in encodings:
    try:
        df = pd.read_csv(file_path, encoding=encoding, header=None)
        print(f"✅ 成功: {encoding} エンコーディングでファイルを読み込みました。")
        break
    except Exception as e:
        continue

# ファイルが見つからなかった場合の処理（ダミーデータ作成）
if df is None:
    print("\n🚨 警告: ファイルが見つからないか、すべてのエンコーディングで読み込めませんでした。ダミーデータを作成して分析を続行します。")
    x_dummy = np.linspace(0, 540, 1000)
    y_dummy = np.sin(x_dummy / 50) + np.cos(x_dummy / 20) * 0.5 + x_dummy / 540 + 2
    z_dummy = np.zeros(1000)
    df = pd.DataFrame({0: x_dummy, 1: y_dummy, 2: z_dummy})

if df.shape[1] < 2:
    print("\n🚨 エラー: データフレームの列数が2列未満のため、分析を続行できません。")
    exit()

df.columns = [f'Col{i}' for i in range(df.shape[1])]
time_col = df.columns[0]
oxy_hb_col = df.columns[1]

# --- 2. 生データのプロット (指定された時間範囲を使用) ---
# ご提示の時間範囲をプロットに使用
color_ranges_plot = {
    'gray': [[0.002, 12.6],[553.2,555]],
    'black': [[12.6, 43.05], [103.2, 133.2], [193.2, 223.2], [283.65, 313.35], [373.05, 403.35], [463.65, 493.05]],
    'blue': [[43.05, 103.2], [223.2, 283.65], [403.35, 463.65]],
    'red': [[133.2, 193.2], [313.35, 373.05], [493.05, 553.2]]
}

df_sorted = df.iloc[:, [0, 1]].sort_values(by=df.columns[0]).reset_index(drop=True)
x_data = df_sorted.iloc[:, 0]
y_data = df_sorted.iloc[:, 1]

plt.figure(figsize=(10, 6))

for color, ranges in color_ranges_plot.items():
    for L, R in ranges:
        segment_mask = (x_data >= L) & (x_data <= R)
        
        # 連続性のための境界点追加ロジック (省略)

        if segment_mask.any():
            segment_x = x_data.loc[segment_mask]
            segment_y = y_data.loc[segment_mask]
            plt.plot(segment_x, segment_y, color=color, linewidth=1.5, zorder=3)

plt.xlabel('時刻 [s]')
plt.ylabel('Oxy Hb')
plt.title('生データ推移 (Oxy Hb)')
plt.xlim(0, 540)
plt.grid(True, linestyle='--')

# 凡例を作成
black_line = mlines.Line2D([], [], color='black', linewidth=2, label='安静期間')
blue_line = mlines.Line2D([], [], color='blue', linewidth=2, label='簡単課題')
red_line = mlines.Line2D([], [], color='red', linewidth=2, label='難解課題')
plt.legend(handles=[black_line, blue_line, red_line], loc='upper right', fontsize='small')

plt.savefig('raw_data_time_series_custom.png')
print("✅ グラフ 'raw_data_time_series_custom.png' を作成しました。")

# --- 3. Z-scoreの計算と統計的検定 (指定された時間範囲を使用) ---

# Z-score計算のためのタスクとベースライン期間の定義
# 安静 (black) と 課題 (blue:簡単, red:難解) の順序を定義
rest_periods = color_ranges_plot['black']
easy_periods = color_ranges_plot['blue']
difficult_periods = color_ranges_plot['red']

# 課題 (Easy/Difficult) と 直前安静 (Rest) を対応付ける
task_periods_analysis = {
    '簡単': [
        {'task': easy_periods[0], 'rest': rest_periods[0]},      # Easy 1 (43.05-103.2) <- Rest 1 (12.6-43.05)
        {'task': easy_periods[1], 'rest': rest_periods[2]},      # Easy 2 (223.2-283.65) <- Rest 3 (193.2-223.2)
        {'task': easy_periods[2], 'rest': rest_periods[4]}       # Easy 3 (403.35-463.65) <- Rest 5 (373.05-403.35)
    ],
    '難解': [
        {'task': difficult_periods[0], 'rest': rest_periods[1]}, # Difficult 1 (133.2-193.2) <- Rest 2 (103.2-133.2)
        {'task': difficult_periods[1], 'rest': rest_periods[3]}, # Difficult 2 (313.35-373.05) <- Rest 4 (283.65-313.35)
        {'task': difficult_periods[2], 'rest': rest_periods[5]}  # Difficult 3 (493.05-553.2) <- Rest 6 (463.65-493.05)
    ]
}


easy_z_scores = []
difficult_z_scores = []
warnings_count = 0

# Z-scoreの計算: Z_rep = (mean(OxyHb_Task) - mu_rest) / sigma_rest
for task_type, repetitions in task_periods_analysis.items():
    for i, rep in enumerate(repetitions):
        rest_start, rest_end = rep['rest']
        task_start, task_end = rep['task']
        
        rest_data = df[(df[time_col] >= rest_start) & (df[time_col] < rest_end)][oxy_hb_col]
        task_data = df[(df[time_col] >= task_start) & (df[time_col] < task_end)][oxy_hb_col]
        
        z_score = np.nan
        if len(rest_data) > 1 and len(task_data) > 0:
            mu_rest = rest_data.mean()
            sigma_rest = rest_data.std()
            
            if sigma_rest > 0: 
                z_score = (task_data.mean() - mu_rest) / sigma_rest
            else:
                warnings_count += 1
                
        if task_type == '簡単':
            easy_z_scores.append(z_score)
        else:
            difficult_z_scores.append(z_score)

# Z-scoreの結果の表示
results_df = pd.DataFrame({
    '回目': ['1回目', '2回目', '3回目'],
    '簡単 (Z-score)': easy_z_scores,
    '難解 (Z-score)': difficult_z_scores
})

print("\n--- Z-scoreの計算結果（課題ごとの平均） ---")
print("※ 指定されたカスタム時間範囲を使用して計算されています。")
print(results_df.to_markdown(index=False, floatfmt=".4f"))

# 統計的検定
A = np.array(easy_z_scores)
B = np.array(difficult_z_scores)
valid_mask = ~np.isnan(A) & ~np.isnan(B)
A_valid = A[valid_mask]
B_valid = B[valid_mask]

if len(A_valid) < 2:
    print("🚨 警告: 有効なデータペアが2件未満のため、統計的検定はスキップされます。")
else:
    t_stat, p_ttest = stats.ttest_rel(A_valid, B_valid)
    try:
        w_stat, p_wilcoxon = stats.wilcoxon(A_valid, B_valid, alternative='two-sided', method='exact')
    except ValueError:
        p_wilcoxon = np.nan
        w_stat = np.nan

    print("\n--- 統計的検定の結果 ---")
    print(f"有効なデータペア数 (N): {len(A_valid)}")
    print("【対応のあるt検定】")
    print(f"t統計量: {t_stat:.4f}")
    print(f"p値: {p_ttest:.4f}")

    print("\n【Wilcoxon 符号付き順位検定 (両側)】")
    print(f"W統計量: {w_stat:.4f}")
    print(f"p値: {p_wilcoxon:.4f}")

# Z-score の箱ひげ図の作成
if len(A_valid) >= 1:
    plt.figure(figsize=(6, 8))
    plt.boxplot([A_valid, B_valid], tick_labels=['Easy', 'Difficult'], patch_artist=True)
    plt.ylabel('z-score')
    plt.title('Z-score')
    plt.savefig('Z_score_boxplot_custom.png')
    print("✅ 箱ひげ図 'Z_score_boxplot_custom.png' を作成しました。")
    print("--------------------------------------------------\n")