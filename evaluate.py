import pandas as pd
import asyncio
import json  # <-- 【新增】导入json库用于日志美化
from client import MCPClient
from utils import SYSTEM_PROMPT, setup_logger
from tqdm.asyncio import tqdm
from typing import Dict, Any
from datetime import date, datetime

# --- 配置 ---
QUESTION_FILE = 'question.csv'
SERVER_SCRIPT = 'mcp_server.py'
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
OUTPUT_FILE = f'evaluation_results_{timestamp}.csv'  # 使用新名称
CONCURRENCY_LIMIT = 1  # 评估时强烈建议保持并发为1，以确保日志清晰且状态绝对隔离
MAX_QUESTION_RETRIES = 2  # 【新需求】：每个问题最多允许的执行次数（1次运行 + 1次重试）

# 失败关键词列表
FAILURE_KEYWORDS = [
    "Error:",
    "我无法回答",
    "未得出最终结论",
    "模型未提供",
    "没有相应数据",
    "所有重试均失败"
]

# 配置专用于评估脚本的日志记录器
logger = setup_logger('Evaluation', log_file='evaluation.log')
logger.info("--- 启动新一轮批量评估 (V-Final 架构) ---")


async def run_evaluation():
    print(f"1. 从 '{QUESTION_FILE}' 中读取问题...")
    try:
        questions_df = pd.read_csv(QUESTION_FILE)
    except Exception as e:
        print(f"错误：无法读取问题文件 '{QUESTION_FILE}'. 错误: {e}")
        return

    questions_to_run = questions_df['question'].dropna().unique().tolist()
    print(f"   共找到 {len(questions_to_run)} 个有效问题。")

    client = MCPClient()
    results_list = []
    semaphore = asyncio.Semaphore(CONCURRENCY_LIMIT)

    try:
        await client.connect_to_server(SERVER_SCRIPT)
        print(f"\n2. 开始批量处理问题 (并发: {CONCURRENCY_LIMIT}, 问题重试: {MAX_QUESTION_RETRIES} 次)...")

        # 定义一个 worker 辅助函数，它将信号量和客户端传递给核心处理器
        async def worker(question: str):
            async with semaphore:
                # 调用我们新的、带重试和高级日志记录的核心处理函数
                return await process_single_question_with_retry(client, question)

        tasks = [worker(q) for q in questions_to_run]
        results = await tqdm.gather(*tasks, desc="正在评估")
        results_list.extend(results)

    finally:
        print("\n3. 所有问题处理完毕。正在关闭客户端连接...")
        await client.exit_stack.aclose()
        print("   客户端已关闭。")

    if results_list:
        print(f"\n4. 处理评估结果...")

        # --- 【新需求 4：将决策历史写入日志】 ---
        logger.info("\n--- 开始转储所有问题的决策历史记录 (Decision History Dumps) ---")
        for res_dict in results_list:
            log_header = f"DECISION_HISTORY (Q: {res_dict.get('问题')}) (Success: {res_dict.get('是否回答成功')})"
            try:
                # 美化JSON输出到日志文件
                history_json = json.dumps(res_dict.get('decision_history_log', {}), indent=2, ensure_ascii=False, default=str)
                logger.info(f"{log_header}\n{history_json}\n{'-' * 50}")
            except Exception as e:
                logger.error(f"{log_header} - (日志记录失败: {e}) - 原始数据: {res_dict.get('decision_history_log')}")
        logger.info("--- 决策历史记录转储完毕 ---")

        # --- 【新需求 5：仅保存三列到CSV】 ---
        results_df = pd.DataFrame(results_list)
        final_columns = ['问题', '答案', '是否回答成功']

        # 确保我们只保存这三列，即使字典中包含其他辅助键（如 decision_history_log）
        if all(col in results_df.columns for col in final_columns):
            final_df_to_save = results_df[final_columns]
            print(f"5. 正在将最终结果 (3列) 写入到 '{OUTPUT_FILE}'...")
            final_df_to_save.to_csv(OUTPUT_FILE, index=False, encoding='utf-8-sig')
            print(f"   评估结果已成功保存！")
        else:
            print(f"错误：处理结果时未能生成所需的列。可用列: {results_df.columns}")


async def process_single_question_with_retry(client: MCPClient, question: str) -> Dict[str, Any]:
    """
    【V-Final 核心处理器】: 结合了重试、隔离调用和结构化输出。
    """
    last_exception = None

    # --- 【新需求 1：问题级重试循环】 ---
    for attempt in range(MAX_QUESTION_RETRIES):
        try:
            # 1. 确保状态绝对隔离 (必须在每次重试开始时都运行)
            await client.session.call_tool("clear_cache", {"flush_all": True})

            # --- 【新需求 2：调用隔离的评估函数 (process_query_with_history)】 ---
            # result_dict 将包含: {'final_answer': ..., 'decision_history': [...]}
            result_dict = await client.process_query_with_history(question, SYSTEM_PROMPT)

            final_answer = result_dict.get('final_answer')
            history_log = result_dict.get('decision_history')

            # --- 【新需求 3：定义成功标准】 ---
            # 检查答案是否有效，且不包含任何已知的失败关键词
            is_success = (
                    final_answer is not None and
                    final_answer.strip() != "" and
                    not any(kw in final_answer for kw in FAILURE_KEYWORDS)
            )

            # 只要函数成功返回（即使答案是“失败”），我们就完成了，无需重试
            return {
                '问题': question,
                '答案': final_answer,
                '是否回答成功': is_success,
                'decision_history_log': history_log  # 传递给日志记录器
            }

        except Exception as e:
            # 这捕获了灾难性的失败（例如连接中断，或函数本身崩溃）
            logger.error(f"问题 '{question}' 在第 {attempt + 1}/{MAX_QUESTION_RETRIES} 次尝试中发生灾难性故障: {e}",
                         exc_info=True)
            last_exception = e

            # 如果这不是最后一次尝试，则等待一段时间后重试
            if attempt < MAX_QUESTION_RETRIES - 1:
                await asyncio.sleep(5)  # 等待5秒，防止快速连续失败

    # --- 如果所有重试都失败了 (即循环正常结束) ---
    failure_message = f"所有 {MAX_QUESTION_RETRIES} 次尝试均失败。最后错误: {last_exception}"
    return {
        '问题': question,
        '答案': failure_message,
        '是否回答成功': False,
        'decision_history_log': [{"role": "system", "content": f"Catastrophic Failure: {failure_message}"}]
    }


if __name__ == "__main__":
    asyncio.run(run_evaluation())