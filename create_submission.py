import pandas as pd
import os
import sys

# --- 配置 ---

# 1. 我们的答案来源文件 (已完成评估的文件)
# 假设列名: ['问题', '答案', '是否回答成功']
SOURCE_RESULTS_FILE = 'evaluation_results_final.csv'

# 2. 提交模板文件 (我们需要填充的文件)
# 假设列名: ['id', '问题'] (或者包含一个需要被覆盖的空 'answer' 列)
TEMPLATE_SUBMIT_FILE = 'sample_submit.csv'

# 3. 最终生成的提交文件
FINAL_OUTPUT_FILE = 'submit_result/final_submission_0910_17.csv'


# --- 脚本主逻辑 ---

def generate_submission_file():
    """
    读取评估结果，并将其答案填充到提交模板中。
    """
    print("--- 开始生成最终提交文件 ---")

    # --- 1. 验证文件是否存在 ---
    if not os.path.exists(SOURCE_RESULTS_FILE):
        print(f"错误：找不到答案来源文件: {SOURCE_RESULTS_FILE}")
        print("请确保已成功运行评估脚本（例如 evaluate.py）并生成了此文件。")
        sys.exit(1)

    if not os.path.exists(TEMPLATE_SUBMIT_FILE):
        print(f"错误：找不到提交模板文件: {TEMPLATE_SUBMIT_FILE}")
        sys.exit(1)

    try:
        # --- 2. 加载数据源并创建“答案查找字典” ---
        print(f"正在从 '{SOURCE_RESULTS_FILE}' 加载答案...")
        results_df = pd.read_csv(SOURCE_RESULTS_FILE)

        # 验证必需的列
        if '问题' not in results_df.columns or '答案' not in results_df.columns:
            print(f"错误: '{SOURCE_RESULTS_FILE}' 必须包含 '问题' 和 '答案' 列。")
            sys.exit(1)

        # 【关键】创建一个高效的 Python 字典 (哈希图) 作为我们的查找表
        # 这会自动处理重复的结果（只保留第一个遇到的答案）
        # 键 = 问题 (str), 值 = 答案 (str)
        print("正在创建答案查找映射...")
        # (确保问题列是字符串类型，以防万一)
        answer_map = pd.Series(
            results_df['答案'].values,
            index=results_df['问题'].astype(str)
        ).to_dict()

        # --- 3. 加载提交模板并映射答案 ---
        print(f"正在加载模板 '{TEMPLATE_SUBMIT_FILE}'...")
        submit_df = pd.read_csv(TEMPLATE_SUBMIT_FILE)

        if 'question' not in submit_df.columns:
            print(f"错误: 模板文件 '{TEMPLATE_SUBMIT_FILE}' 缺少 '问题' 列。")
            sys.exit(1)

        print("正在将答案映射到模板...")

        # 【核心逻辑】:
        # .map(answer_map) 会遍历 '问题' 列中的每一行：
        # 1. 如果“问题”是一个有效字符串 (例如 "问题A")，它会在字典中查找并返回 "答案A"。
        # 2. 【处理重复问题】：如果 "问题A" 出现 10 次，它会执行 10 次查找并正确返回 "答案A" 10 次。
        # 3. 【处理空问题】：如果“问题”是 NaN (空单元格)，.map() 会自动返回 NaN。
        submit_df['answer'] = submit_df['question'].map(answer_map)

        # --- 4. 清理与保存 ---

        # 【处理空问题 - 步骤 2】：
        # 将所有 NaN 值 (来自空问题 或 映射失败的问题) 替换为空字符串，以满足要求。
        submit_df['answer'] = submit_df['answer'].fillna('')
        print("空问题已设置为空回答。")

        # 保存最终文件
        # 我们使用 utf-8-sig 来确保 Excel 在打开中文 CSV 时不会出现乱码
        submit_df.to_csv(FINAL_OUTPUT_FILE, index=False, encoding='utf-8-sig')

        print("\n--- 成功！ ---")
        print(f"已成功生成最终提交文件: {FINAL_OUTPUT_FILE}")

    except Exception as e:
        print(f"\n--- 发生意外错误 ---")
        print(f"错误详情: {e}")
        sys.exit(1)


# --- 运行脚本 ---
if __name__ == "__main__":
    generate_submission_file()