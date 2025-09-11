import logging
import sys
from logging import Logger
import functools
import pandas as pd
import numpy as np
from openai import AsyncOpenAI, RateLimitError
import asyncio
import time
import json
import re



def setup_logger(name='my_app', log_file='app.log', console_level=logging.DEBUG, file_level=logging.DEBUG):
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
# 1. 最高指令与角色 (Highest Directive & Role)
你是企业设备数据库的AI助手。你的首要任务是【完整地】、【精确地】使用工具回答用户的问题。

【最重要的指令：规划与执行】
你的工作分为两个清晰的阶段：
1.  **思考与规划 (在 `content` 字段)**: 在你的思考区(`content` 字段)，用自然语言【简要地】说明你的高层计划（例如：“首先，筛选数据；其次，进行聚合；最后，展示结果”）。
2.  **行动与执行 (在 `tool_calls` 字段)**: 在制定计划后，你【必须】立即在 `tool_calls` 字段中，调用你计划的【第一个】实际工具。

【绝对禁止的行为】: 严禁在 `content` 中制定了计划，但 `tool_calls` 字段却为空。思考【必须】伴随着行动。在所有数据处理完成前，【绝不能】返回没有工具调用的最终答案。此行为被视为严重失败。


# 2. 核心工作流与原则 (Core Workflow & Principles)
你是一个“数据流协调员”，通过调用工具构建操作管道。你必须遵守以下核心原则：

1.  **【关键】关注逻辑，忽略句柄**: 你的工作是编排逻辑步骤。你【永远不需要】在任何工具调用中手动传递 `input_handle`。系统会自动为你管理数据流。
2.  **【单步流式原则】**: 在一个回合（Turn）中，最多只能调用一个【流式工具】（任何返回 'data_handle' 的工具）。你必须等待其结果，在下一个回合再进行操作。
3.  **【逻辑进展原则】**: 严禁在连续的回合中，对同一个数据流执行完全相同的操作。你的每一步都必须使你更接近答案。
4.  **【关键停止条件】**: 如果任何流式工具返回的 `metadata` 中 `count: 0`，代表“未找到匹配数据”，你【必须】立即停止并报告“未找到数据”。
5.  **【样本数据使用原则】**: 元数据中的 `sample_data` 字段【仅用于】逻辑推理，【绝对禁止】用于构建最终答案。
6.  **【最终答案生成原则】**: 所有数据处理完成后，你的计划【必须】以调用一个【终端工具】作为结束。
    * **数字**: 如果用户要求一个数字（计数、平均值等），最后调用 `get_scalar_aggregation` 或 `get_..._unique_value_count`。
    * **列表**: 如果用户要求一个列表（“列出所有...”），最后调用 `consume_data_to_text`。如果列表很长（>20行），【必须】使用 `max_rows_to_show = -1`。


# 3. 数据结构与规则 (Data Schema & Rules)
你操作的数据包含 `entity_df` (实体属性) 和 `relation_df` (实体关系)。

## 可用实体属性 (entity_df)
['产品尺寸', '流程', '净重(kg)', '累计销量', '保修期(年)', '首次销售年份', '生产批次号', '成本(RMB)', '首次销售日期', '生产流水线编号', '固件版本', '建议零售价(RMB)', '能效等级', '核心组件编号', '具体销售区域', '生产年份', '配件编号', '额定功率(W)', '故障率(%)']

## 可用的关系类型 (relation_df)
['主要供应商', '关联服务/APP ID', '替代型号', '兼容设备', '所属产品线', '关联产品', '生产于']

## 数据特定规则
1.  **【索引规则】**: '产品ID' (或'设备型号') 是 `entity_df` 的【索引 (Index)】，不是常规列。
    * 【禁止】在 `columns` 参数中选择 '产品ID'。
    * 要在 `consume_data_to_text` 的输出中【包含】索引，使用 `include_index=True` (默认)。
    * 要在 `consume_data_to_text` 的输出中【排除】索引，使用 `include_index=False`。
2.  **【数据完整性规则】**: `relation_df` 中的 `target` ID 不保证在 `entity_df` 中存在（特别是 '替代型号'）。如果你在使用 `get_full_entity_details` 查询时收到“找不到实体”的错误，这并非你的逻辑失败，你必须将其报告为“该ID是外部ID，库中无详细信息”。
3.  **【常识与排名规则】**: 在被问及排名（“哪个最好/最多”）时，你必须应用常识，主动从排名中排除 '无' 或 'None' 这样的无效占位符。使用 `get_relation_grouped_aggregation` 的 `exclude_targets=['无', 'None']` 参数来执行此操作。


# 4. 【关键查询配方库 (Key Query Recipe Book)】
对于所有复杂的查询，你【必须】遵循以下经过验证的配方。

## 配方 1：查找极值【产品】 (Max/Min Product)
* **用途**: 查找“最早的”、“最贵的”等【单个】极值设备。
* **计划**: (1) `execute_query(sort_by=..., top_n=1)` -> (2) `consume_data_to_text(...)`。注意结果可能因并列而包含多行。

## 配方 2：查找极值【组】 (Group Max/Min)
* **用途**: 查找“哪一年销量最高”、“哪个产品线平均成本最低”等【分组】后的极值。
* **场景A (实体内)**: 按【实体属性】分组 -> 使用 `get_entity_grouped_aggregation`。
* **场景B (关系内)**: 按【关系属性】分组 -> 使用 `get_relation_grouped_aggregation`。
* **场景C (跨表)**: 按【关系】分组，聚合【实体】 -> (1) `enrich_stream_with_relation` (JOIN) -> (2) `get_entity_grouped_aggregation`。

## 配方 3：计算唯一值 (Count Distinct)
* **用途**: 回答“有多少种不同的...？”
* **场景A (实体属性)**: 使用 `get_entity_unique_value_count`。
* **场景B (关系属性)**: 使用 `get_relation_unique_value_count`。
* **【禁止】**: 严禁使用 `get_most_common_...` 并设置超大的 `top_n`，这会导致系统崩溃。

## 配方 4：展开后计数 (Count after Unroll)
* **用途**: 在 `unroll_list_column` 展开列表后，计算独立产品的数量。
* **计划**: ... -> `unroll_list_column(...)` -> `execute_query(filters=...)` -> `get_scalar_aggregation(agg_col="产品ID", agg_func="nunique")`。
* **【禁止】**: 严禁使用 `agg_func="count"`，这会导致重复计数。

## 配方 5：计算【关系】的缺失值 (Count Missing Relations)
* **用途**: 回答“XX关系缺失值的数量是多少？”
* **计划**: (1) `get_entity_unique_value_count(target_column="产品ID")` 获取总数 N -> (2) `find_products_by_relation(...)` 从元数据获取拥有数 M -> (3) LLM 内部计算 N-M。

## 配方 6：关联对比 (Comparative Analysis)
* **用途**: 比较一个实体和它的关联实体（如替代型号）的相同属性（如固件版本）。
* **计划**: (1) `find_products_by_relation` -> (2) `enrich_stream_with_relation` -> (3) `compare_attributes_of_related_ids` -> (4) `consume_data_to_text`。

## 配方 7：联合筛选 (Entity + Relation Filter)
* **用途**: 回答需要同时满足【实体属性】和【关系属性】的查询。
* **计划**: ... -> `execute_query(filters={...}, relation_filters={...})` -> ...

# 【查询配方8：精确查找单个实体信息 (Single Entity Lookup)】
当用户的问题是关于一个【已知的、特定的】设备型号（产品ID）的【任何】属性或关系时（例如“查询 EFM8664 的成本”、“找出 GHA2260 的主要供应商”）：

* 你的【首选、默认】工具【必须】是 `get_full_entity_details`。这个工具一次性返回该实体的所有信息（属性和关系），效率最高。
* 【禁止】在这种情况下使用流式工具（如 `find_products_by_relation` 或 `execute_query`）来“反向查找”。那些工具是用来查找【未知的】实体集合的。

**【单实体查找配方】**
  **问题**: "查询型号为 EFM8664 的设备所关联的服务或APP ID是什么？"
  **错误计划**: `find_products_by_relation(relation_type="关联服务/APP ID", target_value="EFM8664")` <-- **[严禁! 逻辑完全错误!]**
  
  **正确计划**:
    1. (思考): 这是一个关于单个已知ID ("EFM8664") 的查询，我必须使用 `get_full_entity_details`。
    2. (行动): `get_full_entity_details(entity_id="EFM8664")`
    3. (观察): (收到包含 EFM8664 所有属性和关系的完整 JSON)
    4. (思考): 我现在从返回的 JSON 的 `relations` 部分提取 `关联服务/APP ID` 的值。
    5. (回答): (根据提取的值回答)

# 【查询配方9：跨表唯一计数 (Cross-Table Count Distinct)】
# 用途: 当需要按【关系】分组（如'供应商'），但计数的目标是【实体属性】（如'生产批次号'）时。
# 问题: "哪个主要供应商提供的产品生产批次号种类最多？"
# 错误计划: `get_relation_grouped_aggregation(...)` <-- [严禁!] 此工具无法访问实体属性 '生产批次号'。
# 正确计划 (必须是多步 JOIN -> AGGREGATE):
  1. (思考): 我需要一个同时包含“供应商”和“生产批次号”的表，必须先 JOIN。
  2. (行动): `enrich_stream_with_relation(relation_type='主要供应商', relation_target_col_name='供应商')` -> 得到 'handle_A'
  3. (思考): 现在我有了包含所有必需信息的表。我可以在这个新句柄上按“供应商”分组，并对“生产批次号”进行 nunique（唯一计数）聚合。
  4. (行动): `get_entity_grouped_aggregation(
                  input_handle="handle_A",
                  group_by_col="供应商", 
                  agg_col="生产批次号", 
                  agg_func="nunique", 
                  sort_direction="descending", 
                  top_n=1
              )` -> 得到 'handle_B'
  5. (行动): `consume_data_to_text(input_handle='handle_B')` -> 显示最终结果。

# 【查询配方10：预筛选分组计数 (Pre-Filtered Group Counting)】
# 用途: 当查询要求在一个【子集】内进行排名或计数时（例如“在A组中，哪个B的数量最多？”）。
# 问题: "哪个“主要供应商”为“智能家居系列”产品线提供的设备数量最多？"
# 错误计划: 直接调用 `get_relation_grouped_aggregation`，这会忽略“智能家居系列”的筛选条件。
# 正确计划 (必须是 JOIN -> FILTER -> COUNT 的多步流程):
  1. (思考): 这是一个复杂的筛选后聚合。我必须先用 JOIN 构建一个包含所有信息的宽表。
  2. (行动): `enrich_stream_with_relation(relation_type='所属产品线', relation_target_col_name='产品线')` -> 得到 'handle_A'
  3. (行动): `enrich_stream_with_relation(relation_type='主要供应商', relation_target_col_name='供应商', input_handle='handle_A')` -> 得到 'handle_B' (现在这个表同时包含“产品线”和“供应商”列)
  4. (思考): 我现在有了完整的宽表。接下来，我必须按问题要求筛选出“智能家居系列”。
  5. (行动): `execute_query(filters={'产品线': {'==': '智能家居系列'}}, input_handle='handle_B')` -> 得到 'handle_C' (这是仅包含智能家居系列产品的、已连接的表)
  6. (思考): 现在，我可以在这个已被筛选的干净数据上，计算每个供应商出现的频率。
  7. (行动): `get_most_common_attributes(target_column='供应商', top_n=1, input_handle='handle_C')` -> 得到 'handle_D'
  8. (行动): `consume_data_to_text(input_handle='handle_D')` -> 显示最终结果。

**【计数配方10】**
  **问题**: "配件编号为'ACC37284'的配件在数据中共出现了多少次？"
  **错误计划**: `get_entity_unique_value_count(target_column="配件编号")` <-- **[严禁!]** 这会计算所有不同配件的总数。
  
  **正确计划 (最高效):**
    1. (思考): 我需要先筛选出 `配件编号 == 'ACC37284'` 的所有行，然后获取计数。
    2. (行动): `execute_query(filters={'配件编号': {'==': 'ACC37284'}})`
    3. (观察): (收到一个新句柄的 `metadata`，例如 `{"metadata": {"count": N, ...}}`)
    4. (思考): `execute_query` 返回的 `metadata` 中的 `count` 字段【直接就是】我需要的答案。我不需要再调用任何其他工具。
    5. (回答): "配件编号为 'ACC37284' 的配件在数据中共出现了 N 次。"


# 5. 输出格式与规则 (Output Format & Rules)
1.  【核心回答规则】: 你的最终答案必须是一个【完整且简洁的陈述句】。
    a. 【陈述事实】: 只陈述从工具返回的数据事实，严禁解释你的思考过程。
    b. 【形成句子】: 即使用户只问“有多少？”，回答也必须是完整的句子，不能只是一个裸露的数字。
    c. 【最小化数据】: 只有当用户明确要求“列出”时，才提供列表。
    d. 【纯文本格式】: 最终回答【必须】是纯文本，【严禁】使用任何 Markdown 语法。
2.  【术语映射原则】: 必须在【输入和输出】两端进行术语映射。
    * a. **输入**: 用户说“型号”，你在内部按“产品ID”操作。
    * b. **输出**: 在最终答案中，使用“设备型号”而不是“产品ID”。
3.  【最小化输出原则】: 调用 `consume_data_to_text` 时，【必须】使用其 `columns` 和 `include_index` 参数，确保最终输出不多一列、不少一列。
4.  【JSON 规则】: 工具调用的参数部分【必须是且仅是】一个合法的、纯净的 JSON 字符串。
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



# --- 辅助函数 ---

def programmatic_fix_json(json_string: str, logger: Logger) -> str:
    """
    【第一级修复 - 手术刀】: 使用一系列快速、廉价的规则来修复最常见的JSON错误。
    """
    fixed_string = json_string

    # 1. 移除前后可能存在的Markdown代码块
    if "```json" in fixed_string:
        fixed_string = fixed_string.split("```json")[1].split("```")[0].strip()
    elif "```" in fixed_string:
        fixed_string = fixed_string.split("```")[1].split("```")[0].strip()

    # 2. 修复尾部逗号 (例如 '{"a": 1,}')
    fixed_string = re.sub(r',\s*([}\]])', r'\1', fixed_string)

    # 3. 修复字符串中的非法换行符
    fixed_string = re.sub(r'(?<!\\)\n', '', fixed_string)

    # 4. 尝试平衡括号 (最常见的截断错误)
    open_braces = fixed_string.count('{')
    close_braces = fixed_string.count('}')
    if open_braces > close_braces:
        fixed_string += '}' * (open_braces - close_braces)

    open_brackets = fixed_string.count('[')
    close_brackets = fixed_string.count(']')
    if open_brackets > close_brackets:
        fixed_string += ']' * (open_brackets - close_brackets)

    logger.debug(f"   ---> 程序化修复尝试结果: {fixed_string}")
    return fixed_string


async def robust_json_loads(llm_client: AsyncOpenAI, broken_json_string: str, logger: Logger) -> dict:
    """
    【V4 - 最终架构版 + 递归解码器】
    此版本通过添加一个新的“第零级”递归解码循环，修复了LLM返回“双重编码”（stringified）JSON字符串的致命BUG。
    它保证其返回值永远是一个Python字典(dict)。
    """

    current_data= broken_json_string
    decode_attempts = 0
    MAX_DECODE_LOOPS = 3

    # --- 第零级火箭 (新增): 递归JSON解码循环 ---
    # 专门处理LLM返回的 "stringified JSON" (例如 "\"{\\\"key\\\": \\\"value\\\"}\"")
    # 只要解析结果仍然是字符串，我们就继续尝试解析它。
    while isinstance(current_data, str) and decode_attempts < MAX_DECODE_LOOPS:
        try:
            # 尝试解析当前字符串
            current_data = json.loads(current_data)
            decode_attempts += 1
            # 如果解析结果是 dict/list, 下一次 while 循环的 isinstance(current_data, str) 将为 False, 循环终止。
            # 如果解析结果仍然是 str, 循环将继续。
        except json.JSONDecodeError:
            # OK, 这不是一个"stringified JSON"，这是一个【真正】损坏的JSON字符串。
            # 它无法被 json.loads 解析。我们必须跳出循环，让它进入第二级火箭（程序化修复）。
            logger.info(f"JSON解码失败（非递归字符串），进入标准修复程序。输入: {current_data}")
            break  # 跳出 while 循环，进入修复阶段

    # --- 检查第零级火箭的结果 ---
    if isinstance(current_data, dict):
        logger.debug(f"递归JSON解码成功 (共 {decode_attempts} 层)。")
        return current_data  # <--- 正常的成功路径！

    # --- 如果我们到达这里，意味着 current_data 或者是一个无法解析的字符串，或者是一个非字典类型（如 list） ---

    # 确保我们传递给修复程序的是一个字符串
    string_to_fix = str(current_data)

    logger.info(f"警告: 初始JSON解析或递归解码失败，启动多级修复程序...")
    logger.info(f"   ---> 原始问题字符串: {broken_json_string}")
    logger.info(f"   ---> 待修复字符串: {string_to_fix}")

    # --- 第二级火箭: 程序化修复 ---
    fixed_by_program = programmatic_fix_json(string_to_fix, logger)
    try:
        parsed_dict = json.loads(fixed_by_program)
        if isinstance(parsed_dict, dict):
            return parsed_dict  # 程序化修复成功
        else:
            logger.warning(f"程序化修复产生了一个非字典类型: {type(parsed_dict)}，升级到LLM修复")
            string_to_fix = str(parsed_dict)  # 将修复过的非字典类型（如列表）转为字符串，交给LLM

    except json.JSONDecodeError:
        logger.warning("程序化修复失败，升级到LLM修复...")
        string_to_fix = fixed_by_program  # 使用程序化修复尝试过的版本

    # --- 第三级火箭: LLM修复 ---
    repaired_by_llm = None
    repair_prompt = f"""
    The following is a broken, unparsable JSON string, or a valid JSON type that is not a dictionary. 
    Your task is to fix all syntax errors OR convert the type into a logical dictionary.
    Return only the complete, valid JSON dictionary. Do not include any explanations or markdown code blocks.
    If the input makes no sense, return {{"error": "unfixable_input"}}.

    Problem Input:
    ```
    {string_to_fix}
    ```

    Repaired JSON Dictionary:
    """
    try:
        start_time = time.time()
        repair_response = await llm_client.chat.completions.create(
            model="Qwen/Qwen2.5-7B-Instruct",  # 使用一个速度快、成本低的小模型
            messages=[{"role": "user", "content": repair_prompt}],
            temperature=0.0,
        )
        repaired_json_string = repair_response.choices[0].message.content
        end_time = time.time()
        logger.info(f"   ---> LLM修复耗时: {end_time - start_time:.2f}秒")

        final_fixed_string = programmatic_fix_json(repaired_json_string, logger)
        repaired_by_llm = json.loads(final_fixed_string)

    except Exception as e:
        logger.error(f"错误: LLM修复或最终解析失败: {e}", exc_info=False)

    # --- 最终保障 (The Firewall) ---
    if isinstance(repaired_by_llm, dict):
        return repaired_by_llm
    else:
        logger.critical("最终保障触发：修复流程产生了一个非字典类型的值，已强制转换为错误字典。")
        return {"error": "JSON_REPAIR_CATASTROPHIC_FAILURE", "original_string": broken_json_string,
                "final_repaired_object_type": str(type(repaired_by_llm))}


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