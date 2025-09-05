import pandas as pd
import asyncio
from client import MCPClient
from utils import SYSTEM_PROMPT
from tqdm.asyncio import tqdm  # 确保导入异步版本的tqdm

# --- 配置 ---
QUESTION_FILE = 'question.csv'
SERVER_SCRIPT = 'mcp_server.py'
OUTPUT_FILE = 'evaluation_results_rate_limited.csv'

# 【新增】并发限制：设置同时运行的最大任务数量
# 这个值可以根据你的API提供商的速率限制来调整。5或10是一个比较安全的起始值。
CONCURRENCY_LIMIT = 1


async def run_evaluation():
    """
    执行完整的、带速率限制的批量评估流程。
    """
    print(f"1. 从 '{QUESTION_FILE}' 中读取问题...")
    try:
        questions_df = pd.read_csv(QUESTION_FILE)
        questions_to_run = questions_df['question'].dropna().tolist()
        print(f"   共找到 {len(questions_to_run)} 个有效问题。")
    except FileNotFoundError:
        print(f"错误：找不到问题文件 '{QUESTION_FILE}'。请确保文件存在。")
        return

    # 初始化客户端和结果列表
    client = MCPClient()
    results_list = []

    # 【新增】创建Semaphore对象
    semaphore = asyncio.Semaphore(CONCURRENCY_LIMIT)

    try:
        await client.connect_to_server(SERVER_SCRIPT)
        print(f"\n2. 开始批量处理问题 (并发限制为 {CONCURRENCY_LIMIT})...")

        # 【修改】创建一个“工人”函数，它会在执行前等待信号量
        async def worker(question: str):
            async with semaphore:
                # 进入这里时，我们保证已获得一张“通行证”
                return await process_single_question(client, question)

        # 【修改】为每个问题创建一个worker任务
        tasks = [worker(q) for q in questions_to_run]

        # 使用tqdm.gather来执行所有任务，它会自动处理并发和进度条
        results = await tqdm.gather(*tasks, desc="正在评估")
        results_list.extend(results)

        print("\n3. 所有问题处理完毕。")

    finally:
        print("   正在关闭客户端连接...")
        await client.exit_stack.aclose()
        print("   客户端已关闭。")

    if results_list:
        print(f"\n4. 正在将结果写入到 '{OUTPUT_FILE}'...")
        results_df = pd.DataFrame(results_list)
        results_df.to_csv(OUTPUT_FILE, index=False, encoding='utf-8-sig')
        print(f"   评估结果已成功保存到 '{OUTPUT_FILE}'！")


async def process_single_question(client: MCPClient, question: str) -> dict:
    """
    一个辅助函数，用于处理单个问题并捕获错误。(此函数保持不变)
    """
    try:
        result_dict = await client.process_query_with_history(question, SYSTEM_PROMPT)
        return {
            '问题': question,
            '决策content': result_dict.get('decision_content'),
            '答案': result_dict.get('final_answer')
        }
    except Exception as e:
        print(f"\n处理问题时发生严重错误: '{question}' -> {e}")
        return {
            '问题': question,
            '决策content': '处理失败',
            '答案': f'Error: {e}'
        }


if __name__ == "__main__":
    asyncio.run(run_evaluation())

