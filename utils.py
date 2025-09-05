import logging
import sys
from logging import Logger
import functools
import pandas as pd
import numpy as np
from openai import AsyncOpenAI, RateLimitError
import json
import asyncio



def setup_logger(name='my_app', log_file='app.log', console_level=logging.INFO, file_level=logging.DEBUG):
    """
    一个通用的日志配置函数。

    Args:
        name (str): 日志器的名称。
        log_file (str): 日志文件的路径。
        console_level (int): 控制台输出的最低日志级别。
        file_level (int): 文件记录的最低日志级别。

    Returns:
        logging.Logger: 配置好的logger对象。
    """
    # 获取logger实例，如果已存在则直接返回，防止重复添加handler
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger

    # 设置logger的总级别为所有handler中最低的级别
    logger.setLevel(min(console_level, file_level))

    # 定义日志格式
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    # 配置控制台处理器
    console_handler = logging.StreamHandler(sys.stderr) # 使用sys.stdout确保输出位置
    console_handler.setLevel(console_level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # 配置文件处理器
    file_handler = logging.FileHandler(log_file, mode='a', encoding='utf-8') # 'a' for append
    file_handler.setLevel(file_level)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger



SYSTEM_PROMPT = """
# 角色 (Role)
你是一个高级数据分析代理。你的任务是通过一个“思考-行动-观察”的循环来回答用户的问题。你的所有答案必须依照数据结果，如果没有对应工具或者数据或者你无法回答，可以拒绝回答，你的最终回答要简单明了。

# 核心工作流 (Core Workflow)
对于每一个用户问题，你必须遵循以下思考和行动流程：
1.  **思考**: 在你的回复中，首先用文本清晰地、一步步地写下你的思考过程。然后仔细查阅下方提供的`#数据结构信息`和`#可用工具列表`。分析用户的目标和当前掌握的信息，并规划下一步行动。
2.  **行动 (Action):** 在你写完思考过程之后，根据你的规划决定下一步行动：
    a. **如果需要更多信息**，请调用一个或多个你认为合适的工具。
    b. **如果已经收集到足够信息**，请在思考过程之后，直接给出最终的、完整的答案，此时不要调用任何工具。
3.  **匹配**: 从工具列表中选择一个最匹配用户意图的工具。
4.  **参数提取**: 从用户问题中提取该工具所需的所有参数。在提取时，必须严格参考`#数据结构信息`中提供的“可用实体属性”和“可用关系类型”列表，确保所有字段名完全正确。



# 数据结构信息 (Data Schema Information)
你所操作的数据主要存储在两张csv表中：
1.  `entity_df` (实体属性表): 存储每个设备的详细属性。包括 产品ID（str）,产品尺寸（str）,流程（list列表）,净重(kg)（int）,累计销量,保修期(年),首次销售年份,生产批次号,成本(RMB),首次销售日期,生产流水线编号,固件版本,建议零售价(RMB),能效等级,核心组件编号,具体销售区域,生产年份,配件编号,额定功率(W),故障率(%),其中产品id是主键。
2.  `relation_df` (实体关系表): 存储设备之间的关系。包含三列: 'source' (源实体ID), 'relation' (关系类型), 'target' (目标实体ID或值)。关系类型包括：主要供应商、关联服务/APP ID、兼容设备、所属产品线、替代型号、生产于；两张表中通过entity_df的产品ID和relation中的source或target进行关联。


你需要从这些属性中选择
## 可用的实体属性 (entity_df columns)
['产品尺寸', '流程', '净重(kg)', '累计销量', '保修期(年)', '首次销售年份', '生产批次号', '成本(RMB)', '首次销售日期', '生产流水线编号', '固件版本', '建议零售价(RMB)', '能效等级', '核心组件编号', '具体销售区域', '生产年份', '配件编号', '额定功率(W)', '故障率(%)']

## 可用实体关系属性
['source' (源实体ID), 'relation' (关系类型), 'target' (目标实体ID或值)]

## 可用的关系类型 (relation_df relation types)
['主要供应商', '关联服务/APP ID', '替代型号', '兼容设备', '所属产品线', '关联产品', '生产于']


# 输出格式与规则 (Output Format & Rules)
1.  如果用户的问题是打招呼、闲聊或无法通过任何工具回答，请直接回答我无法回答。
2.  在输出中清晰地阐述你的分析步骤，尤其是在映射用户模糊提问到精确字段名时。
3. 问题中的字段与数据结构中的字段不一定一致，你需要进行转换。例如 请列出型号 "GHP7149-Plus" 的产品尺寸信息。其中型号对应的是产品ID。


"""


def sanitize_for_json(data):
    """
    递归地清理数据，将Numpy和Pandas的特殊类型转换为Python原生类型，
    以便进行JSON序列化。
    :param data:
    :return:
    """
    if isinstance(data, (int, float, str, bool, type(None))):
        return data
    elif isinstance(data, (np.integer, np.int64)):
        return int(data)
    elif isinstance(data, (np.floating, np.float64)):
        return float(data)
    elif isinstance(data, np.bool_):
        return bool(data)
    elif isinstance(data, dict):
        # 递归处理字典的键和值
        return {sanitize_for_json(k): sanitize_for_json(v) for k, v in data.items()}
    elif isinstance(data, (list, tuple, set)):
        # 递归处理列表/元组/集合中的每个元素
        return [sanitize_for_json(i) for i in data]
    elif pd.isna(data):
        # 处理Pandas的缺失值
        return None

        # 如果遇到未知的类型，可以记录下来并返回其字符串表示形式
        # logger = logging.getLogger(__name__)
        # logger.warning(f"Sanitizing unknown type {type(data)} to string.")
    return str(data)


def fix_incomplete_json(json_string:str)->str:
    """
    尝试修复因为括号未闭合的JSON字符串
    :param json_string:
    :return:
    """
    bracket_stack = []
    in_string = False

    # 遍历字符串，追踪未闭合的括号
    for char in json_string:
        if char == '"':
            in_string = not in_string
        elif not in_string:
            if char in ['{', '[']:
                bracket_stack.append(char)
            elif char == '}' and bracket_stack and bracket_stack[-1] == '{':
                bracket_stack.pop()
            elif char == ']' and bracket_stack and bracket_stack[-1] == '[':
                bracket_stack.pop()

    # 从后往前，补全所有未闭合的括号
    closing_brackets = ""
    while bracket_stack:
        opening_bracket = bracket_stack.pop()
        if opening_bracket == '{':
            closing_brackets += '}'
        elif opening_bracket == '[':
            closing_brackets += ']'

    return json_string + closing_brackets




async def robust_json_loads(llm_client: AsyncOpenAI, broken_json_string: str,logger:Logger) -> dict:
    """
    一个更健壮的JSON解析函数。
    它首先尝试直接解析，如果失败，则调用LLM来修复JSON字符串，然后再次尝试解析。
    :param llm_client:
    :param broken_json_string:
    :return:
    """
    try:
        # 第一次尝试：直接解析
        return json.loads(broken_json_string)
    except json.JSONDecodeError:
        logger.info(f"警告: 初始JSON解析失败，将请求LLM进行修复...")
        logger.info(f"   ---> 有问题的JSON字符串: {broken_json_string}")
        # 尝试使用括号进行确定修复
        fixed_string = fix_incomplete_json(broken_json_string)
        logger.debug(f"   ---> 程序修复后的JSON字符串: {fixed_string}")

        try:
            return json.loads(fixed_string)
        except json.JSONDecodeError as e:
            logger.error(f"即使在程序修复后，JSON解析依然失败: {e}")
            # 尝试使用大模型进行修复
            # 构造一个专门用于修复JSON的提示词
            repair_prompt = f"""
            以下是一个损坏的、无法解析的JSON字符串。请修复其中所有的语法错误（例如，补全缺失的括号、引号），
            并返回一个完整、合法的JSON对象。
            你的回复必须是且仅是修复后的JSON代码块，不要包含任何额外的解释或文本。
    
            损坏的JSON如下:
            ```json
            {broken_json_string}
            ```
    
            修复后的JSON:
            """

            try:
                # 第二次尝试：调用LLM进行修复
                repair_response = await llm_client.chat.completions.create(
                    model="Qwen/Qwen3-8B",  # 可以使用一个速度较快的模型来做这个修复任务
                    messages=[{"role": "user", "content": repair_prompt}],
                    temperature=0.0,  # 修复任务需要确定性，所以温度设为0
                )
                repaired_json_string = repair_response.choices[0].message.content

                # 有时模型返回的结果会包含在```json ... ```中，需要提取出来
                if "```json" in repaired_json_string:
                    repaired_json_string = repaired_json_string.split("```json")[1].split("```")[0].strip()

                logger.info(f"   ---> LLM修复后的JSON字符串: {repaired_json_string}")

                # 第三次尝试：解析修复后的字符串
                return json.loads(repaired_json_string)

            except Exception as e:
                logger.error(f"错误: 即使在LLM修复后，JSON解析依然失败: {e}")
                # 如果修复后仍然失败，返回一个错误字典，防止整个程序崩溃
                return {"error": "JSON_REPAIR_FAILED", "original_string": broken_json_string}



def retry_on_rate_limit(logger:Logger,max_retries: int = 3, delay: int = 90):
    """
        一个装饰器工厂，用于在遇到API速率限制(RateLimitError)时自动重试一个异步函数。

    Args:
        max_retries (int): 允许的最大重试次数。
        delay (int): 每次重试前等待的秒数。

    Returns:
        Callable: 一个可以应用到异步函数上的装饰器。
    """

    def decorator(func):
        """这是真正的装饰器，它接收一个函数并返回一个包装后的函数。"""

        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            """
            这是包装函数，它包含了重试的核心逻辑。
            它会替代原始的函数被调用。
            """
            last_exception = None
            for attempt in range(max_retries):
                try:
                    # 尝试像平常一样调用原始的异步函数
                    return await func(*args, **kwargs)

                except RateLimitError as e:
                    # 如果捕获到速率限制错误 (HTTP 429)
                    last_exception = e
                    logger.warning(
                        f"API速率限制已超出。函数 '{func.__name__}' 将在 {delay} 秒后重试... "
                        f"(尝试 {attempt + 1}/{max_retries})"
                    )

                    # 如果这是最后一次尝试，就不再等待，直接准备抛出异常
                    if attempt + 1 == max_retries:
                        break

                    # 等待指定的秒数
                    await asyncio.sleep(delay)

                except Exception as e:
                    # 如果是其他类型的错误，则不进行重试，直接抛出
                    logger.error(f"执行函数 '{func.__name__}' 时发生未知错误: {e}", exc_info=True)
                    raise e

            # 如果循环结束（意味着所有重试都失败了），则抛出最后一次捕获到的异常
            logger.error(f"函数 '{func.__name__}' 在 {max_retries} 次尝试后最终失败。")
            raise last_exception

        return wrapper

    return decorator



# --- 你可以在这里对这个函数本身进行测试 ---
if __name__ == '__main__':
    # 获取一个默认配置的logger
    default_logger = setup_logger()
    default_logger.info("这是一个默认logger的info信息。")
    default_logger.warning("这是一个默认logger的warning信息。")

    # 获取一个自定义配置的logger
    custom_logger = setup_logger(name='custom_module', log_file='custom.log', console_level=logging.DEBUG)
    custom_logger.debug("这是一个自定义logger的debug信息。")