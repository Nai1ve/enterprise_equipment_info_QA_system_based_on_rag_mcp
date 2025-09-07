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
# 【新增】定义一个不应暴露给LLM（推理层）的系统/维护工具列表
# 这些工具仍然会被注册到服务器，供客户端（编排层）直接调用。
MAINTENANCE_TOOLS = {"clear_cache"}
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
        # 【新增】: 为交互式聊天会话添加持久的客户端句柄状态
        self.current_data_handle: Optional[str] = None

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
        } for tool in response.tools
            if tool.name not in MAINTENANCE_TOOLS
        ]

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

    # --- 在 client.py -> class MCPClient 内部，替换此函数 ---

    # --- 在 client.py -> class MCPClient 内部，替换此函数 ---

    async def process_query(self, query: str, chat_history: List[Dict[str, Any]], prompt: str) -> str:
        """
        【V4-最终架构版】:
        1. [FIX-A]: 实现了正确的观察结果解析 (content[0].text) 来更新状态。
        2. [FIX-B]: 实现了由您提议的“权威状态注入”（始终覆盖 input_handle）。
        """

        STREAM_AWARE_TOOLS = {
            "execute_query", "get_grouped_aggregation", "get_scalar_aggregation",
            "get_most_common_attributes", "consume_data_to_text"
        }

        messages = [
            {"role": "system", "content": prompt},
            *chat_history,
            {"role": "user", "content": query}
        ]

        max_turns = 5
        for i in range(max_turns):
            logger.info(f"\n--- [交互式 ReAct 循环: 第 {i + 1} 轮] ---")

            response = await self._create_chat_completion_with_retry(
                model=LOCAL_LLM_MODEL_NAME, messages=messages, tools=self.tools_metadata,
                tool_choice="auto", max_tokens=4096
            )
            response_message = response.choices[0].message
            messages.append(response_message)

            if not response_message.tool_calls:
                logger.info("  -> LLM未调用工具，返回最终答案。")
                return response_message.content if response_message.content else "任务完成。"

            logger.info(f"  -> LLM 决定行动: 调用 {len(response_message.tool_calls)} 个工具。")

            for tool_call in response_message.tool_calls:
                function_name = tool_call.function.name
                function_args_STRING = tool_call.function.arguments
                parsed_args = await robust_json_loads(self.llm_client, function_args_STRING, logger)

                observation_text = ""

                if not isinstance(parsed_args, dict):
                    observation_text = f"JSON解析彻底失败。LLM返回的原始字符串: {parsed_args}"
                    logger.error(observation_text)

                elif "error" in parsed_args:
                    observation_text = f"工具参数JSON解析失败: {parsed_args}"
                    logger.warning(observation_text)

                else:
                    function_args_DICT = parsed_args

                    # --- 【修复 B：权威状态注入 (您的提议)】 ---
                    # 我们不再检查LLM是否遗漏了句柄。我们假定它总是遗漏（根据提示词），并始终强制注入我们的本地状态。
                    if function_name in STREAM_AWARE_TOOLS:
                        function_args_DICT['input_handle'] = self.current_data_handle
                        logger.info(
                            f"[StateInject-AUTH]: 权威注入持久句柄: {self.current_data_handle} 到 {function_name}")
                    # ----------------------------------------------------

                    logger.info(f"     - 正在执行工具 '{function_name}'，(权威注入后)参数: {function_args_DICT}")
                    try:
                        tool_result = await self.session.call_tool(function_name, function_args_DICT)

                        # --- 【修复 A：状态更新BUG】 ---
                        # 1. 正确从 [TextContent] 列表中提取纯文本
                        if isinstance(tool_result.content, list) and tool_result.content and hasattr(
                                tool_result.content[0], 'text'):
                            observation_text = tool_result.content[0].text
                        else:
                            observation_text = str(tool_result.content)  # 回退

                        # 2. 在【正确的文本】上尝试状态更新
                        try:
                            obs_dict = json.loads(observation_text)  # <-- 这现在可以成功解析了
                            # 检查返回的是否是一个包含句柄的【流工具】响应
                            if obs_dict.get('status') == 'success' and 'data_handle' in obs_dict:
                                new_handle = obs_dict['data_handle']
                                if new_handle != self.current_data_handle:
                                    logger.info(
                                        f"[StateUpdate-Chat]: 持久句柄已更新! 旧: {self.current_data_handle}, 新: {new_handle}")
                                    self.current_data_handle = new_handle  # 更新【类实例】的状态
                        except Exception:
                            pass  # 观察结果不是JSON或不含句柄 (例如，它是一个终端工具的响应)，忽略状态更新
                        # --- 【修复 A 结束】 ---

                    except Exception as e:
                        logger.error(f"工具 '{function_name}' 执行失败!", exc_info=True)
                        observation_text = f"工具 '{function_name}' 执行失败: {e}"

                # 6. 添加观察结果
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "name": function_name,
                    "content": observation_text,  # <-- 确保使用纯文本
                })

            if not messages[-1].get("tool_calls") and messages[-1]["role"] == "assistant":
                return messages[-1]["content"] if messages[-1]["content"] else "操作已执行完毕。"

        logger.warning("任务已达到最大循环次数...")
        return "抱歉，处理您的问题步骤过长，我无法得出最终结论。"

    async def clear_client_and_server_state(self):
        """【新增】: 重置客户端和服务器的状态，用于新的聊天会话。"""
        logger.info("重置客户端持久句柄并清理服务器缓存...")
        self.current_data_handle = None  # 1. 重置客户端本地句柄
        if self.session:
            try:
                # 2. 调用服务器工具清理服务器缓存
                await self.session.call_tool("clear_cache", {"flush_all": True})
                logger.info("服务器缓存已成功清理。")
            except Exception as e:
                logger.error(f"在重置期间未能清理服务器缓存: {e}")
        else:
            logger.warning("客户端未连接，仅重置本地句柄。")

    # --- 在 client.py 中替换此函数 ---

    async def process_query_with_history(self, query: str, prompt: str) -> Dict[str, Any]:
        """
        【V4-最终架构版】: (评估函数) 同样应用 [FIX-A] (状态更新) 和 [FIX-B] (权威注入)。
        """
        messages = [
            {"role": "system", "content": prompt},
            {"role": "user", "content": query}
        ]
        decision_history = []
        max_turns = 5

        # 【评估函数】使用临时的本地句柄状态
        current_data_handle: Optional[str] = None

        STREAM_AWARE_TOOLS = {
            "execute_query", "get_grouped_aggregation", "get_scalar_aggregation",
            "get_most_common_attributes", "consume_data_to_text"
        }

        for i in range(max_turns):
            response = await self._create_chat_completion_with_retry(
                model=LOCAL_LLM_MODEL_NAME, messages=messages, tools=self.tools_metadata,
                tool_choice="auto", max_tokens=4096
            )
            response_message = response.choices[0].message
            messages.append(response_message)

            turn_log = {
                "turn": i + 1,
                "llm_thought": response_message.content if response_message.content else "",
                "tool_calls": []
            }

            if not response_message.tool_calls:
                final_answer = response_message.content if response_message.content else "任务完成。"
                decision_history.append(turn_log)
                return {
                    "decision_history": json.dumps(decision_history, ensure_ascii=False, indent=2),
                    "final_answer": final_answer
                }

            for tool_call in response_message.tool_calls:
                function_name = tool_call.function.name
                function_args_STRING = tool_call.function.arguments

                parsed_args = await robust_json_loads(self.llm_client, function_args_STRING, logger)
                observation_text = ""

                if not isinstance(parsed_args, dict):
                    observation_text = f"[EVAL] JSON解析彻底失败。原始字符串: {parsed_args}"
                    logger.error(observation_text)

                elif "error" in parsed_args:
                    observation_text = f"[EVAL] 工具参数JSON解析失败: {parsed_args}"
                    logger.warning(observation_text)

                else:
                    function_args_DICT = parsed_args

                    # --- 【修复 B：权威状态注入】 ---
                    if function_name in STREAM_AWARE_TOOLS:
                        function_args_DICT['input_handle'] = current_data_handle  # 注入【本地临时句柄】
                        logger.info(
                            f"[StateInject-AUTH-EVAL]: 权威注入临时句柄: {current_data_handle} 到 {function_name}")
                    # ------------------------------------

                    logger.info(f"     - [EVAL] 正在执行工具 '{function_name}'，(权威注入后)参数: {function_args_DICT}")
                    try:
                        tool_result = await self.session.call_tool(function_name, function_args_DICT)

                        # --- 【修复 A：状态更新BUG】 ---
                        # 1. 正确提取观察文本
                        if isinstance(tool_result.content, list) and tool_result.content and hasattr(
                                tool_result.content[0], 'text'):
                            observation_text = tool_result.content[0].text
                        else:
                            observation_text = str(tool_result.content)

                        # 2. 在【正确的文本】上尝试状态更新
                        try:
                            obs_dict = json.loads(observation_text)
                            if obs_dict.get('status') == 'success' and 'data_handle' in obs_dict:
                                new_handle = obs_dict['data_handle']
                                if new_handle != current_data_handle:
                                    logger.info(
                                        f"[StateUpdate-EVAL]: 临时句柄已更新! 旧: {current_data_handle}, 新: {new_handle}")
                                    current_data_handle = new_handle  # 更新【本地临时】状态
                        except Exception:
                            pass  # 非JSON或非句柄响应，忽略
                        # --- 【修复 A 结束】 ---

                    except Exception as e:
                        logger.error(f"[EVAL] 工具 '{function_name}' 执行失败!", exc_info=True)
                        observation_text = f"[EVAL] 工具 '{function_name}' 执行失败: {e}"

                # 记录日志并添加到历史
                turn_log["tool_calls"].append({
                    "tool_name": function_name,
                    "tool_args": parsed_args if isinstance(parsed_args, dict) else {"raw_str_error": parsed_args},
                    "observation": observation_text
                })
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "name": function_name,
                    "content": observation_text,
                })

            decision_history.append(turn_log)

        final_answer = "任务已达到最大循环次数，未能得出结论。"
        return {
            "decision_history": json.dumps(decision_history, ensure_ascii=False, indent=2),
            "final_answer": final_answer
        }


# (将这两个方法和主程序入口添加到 MCPClient class 的外部)
async def chat_loop(client: MCPClient):
    """运行交互式聊天循环（已升级，支持 'reset' 命令）"""
    print("\n--- MCP 客户端已启动！(V2: 状态管理模式) ---")
    print("请输入你的问题。输入 'reset' 来清除会话状态和服务器缓存，或输入 'quit' 退出。")

    chat_history = []
    while True:
        try:
            query = input("\n你: ").strip()
            if query.lower() == 'quit':
                break

            if query.lower() == 'reset':
                chat_history = []
                await client.clear_client_and_server_state()
                print("\n--- 客户端状态和服务器缓存已重置。请输入新问题。 ---")
                continue

            response_text = await client.process_query(query, chat_history, utils.SYSTEM_PROMPT)
            print(f"\n助手: {response_text}")

            # 添加历史记录
            chat_history.append({"role": "user", "content": query})
            chat_history.append({"role": "assistant", "content": response_text})  # 只保存最终的文本回复

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