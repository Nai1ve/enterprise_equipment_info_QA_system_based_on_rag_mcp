import pandas as pd
import asyncio
import json
import sys
from client import MCPClient
from utils import SYSTEM_PROMPT, setup_logger
from tqdm.asyncio import tqdm
from typing import Dict, Any

# --- 配置 ---
# 【关键】：我们的输入和输出文件现在是同一个文件
EVALUATION_FILE = 'evaluation_results_final.csv'
SERVER_SCRIPT = 'mcp_server.py'
CONCURRENCY_LIMIT = 1
# 【重要】：我们复用了 evaluate.py 中的相同配置
MAX_QUESTION_RETRIES = 2
FAILURE_KEYWORDS = [
    "Error:",
    "我无法回答",
    "未得出最终结论",
    "模型未提供",
    "没有相应数据",
    "所有重试均失败"
]

# 配置日志记录器
logger = setup_logger('Evaluation_Retry', log_file='evaluation_retry.log')
logger.info("--- 启动评估重试脚本 (V-Final 架构) ---")


# --- 步骤 1: 我们必须从 evaluate.py 完整复制核心处理函数 ---
# (这个函数是独立的，我们直接复用它)

async def process_single_question_with_retry(client: MCPClient, question: str) -> Dict[str, Any]:
    """
    【V-Final 核心处理器】: (从 evaluate.py 完整复制而来)
    结合了重试、隔离调用和结构化输出。
    """
    last_exception = None

    for attempt in range(MAX_QUESTION_RETRIES):
        try:
            await client.session.call_tool("clear_cache", {"flush_all": True})

            result_dict = await client.process_query_with_history(question, SYSTEM_PROMPT)

            final_answer = result_dict.get('final_answer')
            history_log = result_dict.get('decision_history')

            is_success = (
                    final_answer is not None and
                    final_answer.strip() != "" and
                    not any(kw in final_answer for kw in FAILURE_KEYWORDS)
            )

            return {
                '问题': question,
                '答案': final_answer,
                '是否回答成功': is_success,
                'decision_history_log': history_log
            }

        except Exception as e:
            logger.error(f"问题 '{question}' 在第 {attempt + 1}/{MAX_QUESTION_RETRIES} 次尝试中发生灾难性故障: {e}",
                         exc_info=True)
            last_exception = e
            if attempt < MAX_QUESTION_RETRIES - 1:
                await asyncio.sleep(5)

    failure_message = f"所有 {MAX_QUESTION_RETRIES} 次尝试均失败。最后错误: {last_exception}"
    return {
        '问题': question,
        '答案': failure_message,
        '是否回答成功': False,
        'decision_history_log': [{"role": "system", "content": f"Catastrophic Failure: {failure_message}"}]
    }


# --- 步骤 2: 专用于重试的全新主函数 ---

async def run_retry_evaluation():
    """
    新的主业务逻辑：读取结果CSV，仅重试失败项，然后更新CSV。
    """
    print(f"1. 正在读取现有的评估文件: '{EVALUATION_FILE}'...")
    try:
        main_df = pd.read_csv(EVALUATION_FILE)
    except FileNotFoundError:
        print(f"错误: 找不到文件 '{EVALUATION_FILE}'。请先运行 'evaluate.py' 生成初始报告。")
        logger.error(f"文件未找到: {EVALUATION_FILE}。脚本退出。")
        return
    except Exception as e:
        print(f"读取文件时出错: {e}")
        return

    # 2. 查找所有失败的行 (确保我们正确处理布尔值)
    # Pandas 读取 'False' (字符串) 或 False (布尔值) 取决于文件。
    # 我们将该列转换为字符串并检查 'False' 的小写，这是最健壮的方法。
    if '是否回答成功' not in main_df.columns:
        print("错误: CSV文件中缺少 '是否回答成功' 列。")
        return

    # 强制将列转换为布尔值进行筛选 (False, 或被解释为 NaN/None 的也算作 False)
    failed_mask = (main_df['是否回答成功'].fillna(False).astype(bool) == False)
    failed_df = main_df[failed_mask]

    questions_to_retry = failed_df['问题'].dropna().unique().tolist()

    if not questions_to_retry:
        print("\n恭喜！在 '{EVALUATION_FILE}' 中未发现任何失败 (False) 的问题。无需重试。")
        logger.info("未发现失败问题。脚本执行完毕。")
        return

    print(f"   共找到 {len(questions_to_retry)} 个失败的问题。开始重试...")
    logger.info(f"开始重试 {len(questions_to_retry)} 个失败的问题。")

    # 3. 启动客户端并运行任务 (与 evaluate.py 相同)
    client = MCPClient()
    retry_results_list = []
    semaphore = asyncio.Semaphore(CONCURRENCY_LIMIT)

    try:
        await client.connect_to_server(SERVER_SCRIPT)
        print(f"\n4. 正在批量重试失败的问题 (并发: {CONCURRENCY_LIMIT})...")

        async def worker(question: str):
            async with semaphore:
                return await process_single_question_with_retry(client, question)

        tasks = [worker(q) for q in questions_to_retry]
        retry_results_list = await tqdm.gather(*tasks, desc="正在重试")

    finally:
        print("\n5. 重试处理完毕。正在关闭客户端...")
        await client.exit_stack.aclose()
        print("   客户端已关闭。")

    if not retry_results_list:
        print("重试期间未产生任何结果。文件未更新。")
        return

    # 6. 【核心更新逻辑】: 使用新结果更新旧的 DataFrame
    print(f"\n6. 正在用 {len(retry_results_list)} 个新结果更新主数据...")

    # a. 将我们的新结果列表转换为一个以“问题”为索引的 DataFrame
    new_results_df = pd.DataFrame(retry_results_list)

    # 过滤掉空的或无效的结果
    new_results_df = new_results_df.dropna(subset=['问题'])
    if new_results_df.empty:
        print("重试结果为空，跳过更新。")
        return

    new_results_df = new_results_df.set_index('问题')

    # b. 将我们的原始主 DataFrame 也设置为以“问题”为索引
    main_df = main_df.set_index('问题')

    # c. 【Pandas 魔法】: update()
    # 这将查找 main_df 索引中与 new_results_df 索引匹配的行，
    # 并仅用 new_results_df 中的值【覆盖】它们。
    # 所有原始的 True 行将保持不变。
    main_df.update(new_results_df)

    # d. 将“问题”从索引恢复为列
    main_df.reset_index(inplace=True)

    # 7. (可选): 将新的决策历史转储到我们的重试日志中
    logger.info("\n--- 开始转储【本次重试运行】的决策历史记录 ---")
    for res_dict in retry_results_list:
        log_header = f"RETRY_DECISION_HISTORY (Q: {res_dict.get('问题')}) (New Success: {res_dict.get('是否回答成功')})"
        try:
            history_json = json.dumps(res_dict.get('decision_history_log', {}), indent=2, ensure_ascii=False, default=str)
            logger.info(f"{log_header}\n{history_json}\n{'-' * 50}")
        except Exception as e:
            logger.error(f"{log_header} - (日志记录失败: {e})")

    # 8. 写回【同一文件】
    print(f"7. 更新完毕。正在将完整的（已修复的）结果写回到 '{EVALUATION_FILE}'...")
    try:
        # 确保我们只保存原始的三列 + 索引问题列
        final_columns = ['问题', '答案', '是否回答成功']
        final_df_to_save = main_df[final_columns]
        final_df_to_save.to_csv(EVALUATION_FILE, index=False, encoding='utf-8-sig')
        print("   文件已成功覆盖！")
    except Exception as e:
        print(f"   错误：写回文件失败: {e}")
        logger.error(f"写回 {EVALUATION_FILE} 失败: {e}")


# --- 步骤 3: 脚本入口点 ---
if __name__ == "__main__":
    asyncio.run(run_retry_evaluation())