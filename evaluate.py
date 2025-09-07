import pandas as pd
import asyncio
from client import MCPClient  # 假设 client.py 及其依赖 (utils.py) 已存在
from utils import SYSTEM_PROMPT, setup_logger  # 假设 SYSTEM_PROMPT 在 utils 中
from tqdm.asyncio import tqdm

# (此处的 logger 只是为了记录评估脚本本身的清理动作, 可选)
logger = setup_logger('Evaluation', log_file='evaluation.log')

# --- 配置 ---
QUESTION_FILE = 'question.csv'
SERVER_SCRIPT = 'mcp_server.py'  # 确保这里指向你刚保存的新 server.py
OUTPUT_FILE = 'evaluation_results_streaming_v2.csv'
CONCURRENCY_LIMIT = 1


async def run_evaluation():
    print(f"1. 从 '{QUESTION_FILE}' 中读取问题...")
    questions_df = pd.read_csv(QUESTION_FILE)
    questions_to_run = questions_df['question'].dropna().tolist()
    print(f"   共找到 {len(questions_to_run)} 个有效问题。")

    client = MCPClient()
    results_list = []
    semaphore = asyncio.Semaphore(CONCURRENCY_LIMIT)

    try:
        await client.connect_to_server(SERVER_SCRIPT)
        print(f"\n2. 开始批量处理问题 (架构: 统一句柄, 并发: {CONCURRENCY_LIMIT})...")

        async def worker(question: str):
            async with semaphore:
                return await process_single_question(client, question)

        tasks = [worker(q) for q in questions_to_run]
        results = await tqdm.gather(*tasks, desc="正在评估")
        results_list.extend(results)

    finally:
        print("\n3. 所有问题处理完毕。正在关闭客户端连接...")
        await client.exit_stack.aclose()
        print("   客户端已关闭。")

    if results_list:
        print(f"\n4. 正在将结果写入到 '{OUTPUT_FILE}'...")
        results_df = pd.DataFrame(results_list)
        results_df.to_csv(OUTPUT_FILE, index=False, encoding='utf-8-sig')
        print(f"   评估结果已成功保存到 '{OUTPUT_FILE}'！")


async def process_single_question(client: MCPClient, question: str) -> dict:
    """
    【已更新】: 使用多轮决策方法处理单个问题，并在 finally 块中确保清理服务器缓存。
    """
    try:
        # 1. (可选但推荐) 在处理新问题之前，先清理一次缓存，确保环境干净
        await client.session.call_tool("clear_cache", {"flush_all": True})

        # 2. 正常执行多轮决策 (这会调用 client.py 中的 process_query_with_history)
        result_dict = await client.process_query_with_history(question, SYSTEM_PROMPT)

        return {
            '问题': question,
            '决策过程': result_dict.get('decision_history'),
            '最终答案': result_dict.get('final_answer')
        }
    except Exception as e:
        print(f"\n处理问题时发生严重错误: '{question}' -> {e}")
        return {
            '问题': question,
            '决策过程': f'处理失败: Error: {e}',
            '最终答案': f'Error: {e}'
        }
    finally:
        # 3. 【关键】: 无论成功还是失败，都在问题结束后再次清理服务器缓存
        try:
            await client.session.call_tool("clear_cache", {"flush_all": True})
            logger.info(f"已清理问题 '{question[:20]}...' 的服务器缓存。")
        except Exception as ce:
            # 即便清理失败也不应影响主流程，但需要记录
            logger.error(f"清理缓存时失败: {ce}")


if __name__ == "__main__":
    asyncio.run(run_evaluation())