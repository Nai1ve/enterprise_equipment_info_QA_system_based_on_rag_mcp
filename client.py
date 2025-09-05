import asyncio
import json
import sys
from typing import Optional, Dict, Any, List
from contextlib import AsyncExitStack
# MCP 相关的导入
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
# LLM 相关的导入 (我们使用openai库)
from openai import AsyncOpenAI
# 用于加载环境变量
from dotenv import load_dotenv

import utils
from utils import setup_logger,robust_json_loads,retry_on_rate_limit

logger = setup_logger('MCP_Client',log_file='client.log')

load_dotenv()

LOCAL_LLM_MODEL_NAME = "Qwen/Qwen2.5-72B-Instruct"


class MCPClient:
    def __init__(self):
        self.session : Optional[ClientSession] = None
        self.exit_stack = AsyncExitStack()
        # 初始化一个指向本地模型的OpenAI客户端
        self.llm_client = AsyncOpenAI(
            base_url='https://api.siliconflow.cn/v1',
            api_key='sk-qwwyvlebeeusfqhywsqhtfvfpjyughkwokfomcgnrvkiwogn'
        )

    async def connect_to_server(self,server_script_path:str):
        """
        根据脚本路径链接到MCP服务器
        :param server_script_path:
        :return:
        """
        logger.info(f"正在尝试连接到服务器: {server_script_path}...")
        command = 'python'
        server_params = StdioServerParameters(
            command = command,
            args=[server_script_path]
        )

        stdio_transport = await self.exit_stack.enter_async_context(stdio_client(server_params))
        self.stdio, self.write = stdio_transport
        self.session = await self.exit_stack.enter_async_context(ClientSession(self.stdio, self.write))

        await self.session.initialize()

        # 从服务器获取可用工具列表并保存它们的元数据
        response = await self.session.list_tools()
        self.tools_metadata = [{
            "type": "function",
            "function": {
                "name": tool.name,
                "description": tool.description,
                "parameters": tool.inputSchema,
            }
        } for tool in response.tools]

        if not self.tools_metadata:
            logger.warning("\n警告：已连接到服务器，但未发现任何可用工具。")
        else:
            logger.info("\n成功连接到服务器，可用工具如下:")
            for tool in self.tools_metadata:
                logger.info(f"- {tool['function']['name']}")

    @retry_on_rate_limit(logger, max_retries=3, delay=60)
    async def _create_chat_completion_with_retry(self, **kwargs):
        """
        这个函数是对原始API调用的一个简单封装，
        我们的重试装饰器就应用在这里。
        """
        return await self.llm_client.chat.completions.create(**kwargs)

    async def process_query(self, query: str, chat_history: List[Dict[str, Any]], prompt: str) -> str:

        messages = [
            {"role": "system", "content": prompt},
            *chat_history,  # 附加之前的对话历史
            {"role": "user", "content": query}
        ]

        max_turns = 5
        for i in range(max_turns):
            logger.info(f"\n--- [ReAct 循环: 第 {i + 1} 轮] ---")

            # 1. 思考与行动决策
            logger.info("  -> 1. 请求LLM进行思考和决策...")
            response = await self._create_chat_completion_with_retry(
                model=LOCAL_LLM_MODEL_NAME,
                messages=messages,
                tools=self.tools_metadata,
                tool_choice="auto",
                max_tokens=4096
            )
            response_message = response.choices[0].message

            # 2. 将模型的思考过程(文本内容)和行动决策(tool_calls)都加入历史
            messages.append(response_message)

            # 3. 检查是否有行动（工具调用）
            if not response_message.tool_calls:
                logger.info("  -> 2. LLM未调用工具，认为当前内容已是最终答案。")
                return response_message.content if response_message.content else "任务已完成，但模型未提供最终文本。"

            logger.info(f"  -> 2. LLM 思考: {response_message.content}")
            logger.info(f"  -> 3. LLM 决定行动: 调用 {len(response_message.tool_calls)} 个工具。")

            # 4. 观察 (执行所有工具调用)
            for tool_call in response_message.tool_calls:
                function_name = tool_call.function.name

                # 依然使用 robust_json_loads 来确保参数健壮性
                function_args = await robust_json_loads(self.llm_client, tool_call.function.arguments, logger)

                if "error" in function_args:
                    observation = f"工具参数解析失败: {function_args}"
                else:
                    logger.info(f"     - 正在执行工具 '{function_name}'，参数: {function_args}")
                    try:
                        tool_result = await self.session.call_tool(function_name, function_args)
                        observation = str(tool_result.content)
                    except Exception as e:
                        observation = f"工具 '{function_name}' 执行失败: {e}"

                logger.info(f"  -> 4. 观察结果 (对于工具 {function_name}): {observation}")

                # 将观察结果加入历史，喂给下一轮的LLM
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "name": function_name,
                    "content": observation,
                })

        logger.warning("任务已达到最大循环次数，但仍未找到最终答案。")
        return "抱歉，处理您的问题步骤过长，我无法得出最终结论。"

    async def process_query_with_history(self, query: str, prompt: str) -> Dict[str, Any]:
        """
        使用多轮 ReAct 循环处理一个问题，并返回最终答案以及详细的决策历史记录。
        专为批量评估设计，以捕捉完整的思考链。
        """
        messages = [
            {"role": "system", "content": prompt},
            {"role": "user", "content": query}
        ]
        decision_history = []
        max_turns = 5  # 设置最大循环次数以防止无限循环

        for i in range(max_turns):
            # 1. 请求LLM进行思考和决策
            response = await self._create_chat_completion_with_retry(
                model=LOCAL_LLM_MODEL_NAME,
                messages=messages,
                tools=self.tools_metadata,
                tool_choice="auto",
                max_tokens=4096
            )
            response_message = response.choices[0].message
            messages.append(response_message)  # 将模型的回复加入对话历史

            # 准备记录当前这一轮的决策
            turn_log = {
                "turn": i + 1,
                "llm_thought": response_message.content if response_message.content else "",
                "tool_calls": []
            }

            # 2. 检查模型是否决定结束并给出最终答案
            if not response_message.tool_calls:
                final_answer = response_message.content if response_message.content else "任务已完成，但模型未提供最终文本。"
                decision_history.append(turn_log)
                return {
                    "decision_history": json.dumps(decision_history, ensure_ascii=False, indent=2),
                    "final_answer": final_answer
                }

            # 3. 如果模型决定调用工具，则执行并记录
            for tool_call in response_message.tool_calls:
                function_name = tool_call.function.name
                function_args = await robust_json_loads(self.llm_client, tool_call.function.arguments, logger)

                if "error" in function_args:
                    observation = f"工具参数解析失败: {function_args}"
                else:
                    try:
                        tool_result = await self.session.call_tool(function_name, function_args)
                        observation = str(tool_result.content)
                    except Exception as e:
                        observation = f"工具 '{function_name}' 执行失败: {e}"

                # 记录本次工具调用的所有信息
                turn_log["tool_calls"].append({
                    "tool_name": function_name,
                    "tool_args": function_args,
                    "observation": observation
                })

                # 将工具的返回结果加入对话历史，供下一轮思考
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "name": function_name,
                    "content": observation,
                })

            decision_history.append(turn_log)

        # 如果达到最大循环次数仍未结束
        final_answer = "抱歉，处理您的问题步骤过长，我无法得出最终结论。"
        # 尝试使用模型的最后一条回复作为答案
        if messages and messages[-1]["role"] == "assistant" and messages[-1]["content"]:
            final_answer = messages[-1]["content"]

        return {
            "decision_history": json.dumps(decision_history, ensure_ascii=False, indent=2),
            "final_answer": final_answer
        }


# (将这两个方法和主程序入口添加到 MCPClient class 的外部)
async def chat_loop(client: MCPClient):
    """运行交互式聊天循环"""
    print("\n--- MCP 客户端已启动！---")
    print("请输入你的问题，或输入 'quit' 退出。")

    chat_history = []  # 初始化聊天历史
    while True:
        try:
            query = input("\n你: ").strip()
            if query.lower() == 'quit':
                break

            response = await client.process_query(query, chat_history,utils.SYSTEM_PROMPT)
            print(f"\n助手: {response}")

        except (KeyboardInterrupt, EOFError):
            break
        except Exception as e:
            print(f"\n发生错误: {e}")


async def main():
    if len(sys.argv) < 2:
        logger.info("用法: python client.py <path_to_server_script>")
        sys.exit(1)

    server_path = sys.argv[1]
    client = MCPClient()
    try:
        await client.connect_to_server(server_path)
        await chat_loop(client)
    finally:
        logger.info("\n正在关闭客户端...")
        await client.exit_stack.aclose()  # 清理资源
        logger.info("客户端已关闭。")


if __name__ == "__main__":
    asyncio.run(main())