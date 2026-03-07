import pandas as pd


def distribute_papers_from_csv(file_path, output_path):
    # 1. 讀取 CSV
    df = pd.read_csv(file_path)

    # 2. 預處理：將 'A1-A5' 轉為 list，並計算選擇數量
    df["possibilities"] = df["sessions_possibileties"].str.split("-")
    df["choice_count"] = df["possibilities"].apply(len)

    # 3. 排序：優先分配「沒得選」的論文 (choice_count 小的優先)
    # 這樣可以避免那些有選擇權的論文把唯一的坑位佔走
    df = df.sort_values(by="choice_count")

    # 4. 初始化 Session 計數器
    all_sessions = set([item for sublist in df["possibilities"] for item in sublist])
    session_counts = {s: 0 for s in all_sessions}
    assignments = {}

    # 5. 進行負載均衡分配
    for _, row in df.iterrows():
        pid = row["paper_id"]
        opts = row["possibilities"]

        # 挑選目前負載最輕的 session
        best_session = min(opts, key=lambda s: session_counts[s])

        assignments[pid] = best_session
        session_counts[best_session] += 1

    # 6. 將結果對回原始 DataFrame 並存檔
    df["assigned_session"] = df["paper_id"].map(assignments)

    # 移除輔助用的欄位並輸出
    result_df = df.drop(columns=["possibilities", "choice_count"])
    result_df.to_csv(output_path, index=False, encoding="utf-8-sig")

    print(f"分配完成！結果已儲存至: {output_path}")
    print("\n各 Session 分配統計：")
    for s, count in sorted(session_counts.items()):
        print(f"Session {s}: {count} 篇")


# 使用範例
distribute_papers_from_csv(
    "C:/Users/GAI/Downloads/papers3.csv", "C:/Users/GAI/Downloads/assigned_results3.csv"
)
