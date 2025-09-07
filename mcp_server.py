import pprint
import uuid
from typing import Dict, Any, List, Optional, Union
from mcp.server.fastmcp import FastMCP
import data
from utils import setup_logger, sanitize_for_json
import pandas as pd
import numpy as np

logger = setup_logger('MCP_Server_V2', log_file='server_v2.log')

# --- 1. 全局状态与初始化 ---
mcp = FastMCP(
    name='device_database_server_streaming_pandas'
)

# 【核心】缓存现在存储Pandas对象（DataFrames 或 Series）
DATA_CACHE: Dict[str, Union[pd.DataFrame, pd.Series]] = {}

# 加载数据
entity_df, relation_df = data.load_data(logger)
logger.info("数据加载完毕。服务器（统一句柄架构）准备就绪。")


# --- 2. 核心辅助函数 ---

def _create_metadata(data: Union[pd.DataFrame, pd.Series]) -> Dict[str, Any]:
    """【内部辅助】: 为LLM创建丰富的元信息包（数据结构和示例数据）。"""
    if data.empty:
        return {
            "count": 0,
            "data_type": str(type(data)),
            "columns": list(data.index.names) if isinstance(data, pd.Series) else list(data.columns),
            "sample_data": "(No data)"
        }

    sample_csv = data.head(3).to_csv(index=True)
    count = len(data)
    columns = []
    if isinstance(data, pd.DataFrame):
        columns = list(data.columns)
    elif isinstance(data, pd.Series):
        columns = [data.name if data.name else 'value']

    return {
        "count": count,
        "data_type": str(type(data)),
        "columns": columns,
        "sample_data": sample_csv
    }


def _get_base_data(input_handle: Optional[str]) -> Union[pd.DataFrame, pd.Series]:
    """【内部辅助】: 根据句柄获取数据，或回退到主实体DF"""
    if input_handle:
        if input_handle not in DATA_CACHE:
            raise ValueError(f"数据句柄 '{input_handle}' 无效或已过期。")
        return DATA_CACHE[input_handle]
    else:
        return entity_df  # 默认从主数据库开始


# --- 3. 流式工具 (Stream Tools) ---
# 这些工具总是返回 {'status': 'success', 'data_handle': str, 'metadata': {...}}

@mcp.tool()
def execute_query(
        filters: Optional[Dict[str, Any]] = None,
        columns: Optional[List[str]] = None,
        sort_by: Optional[str] = None,
        ascending: bool = True,
        top_n: Optional[int] = None,
        input_handle: Optional[str] = None
) -> Dict[str, Any]:
    """
    【核心流工具】: 对数据执行查询、筛选、排序、切片和列选择。
    这是最主要的中间过程工具。它始终缓存结果DF，并返回新句柄和元信息。

    Args:
        filters (Optional[Dict[str, Any]]): 筛选条件的字典。
            【结构】: `{ "列名": { "操作符": "值" } }`
            【支持的操作符】: '==', '>', '<', 'between' (值为 [min, max] 列表)
            【示例】: `{"filters": {"生产年份": {"==": 2020}, "成本(RMB)": {">": 5000}}}`

        columns (Optional[List[str]]): 【可选】要选择的列名列表。
            【示例】: `{"columns": ["成本(RMB)", "固件版本"]}`

        sort_by (Optional[str]): 【可选】用于排序的单个列名。
            【示例】: `{"sort_by": "累计销量"}`

        ascending (bool): 【可选】是否按升序排序。仅在提供了 'sort_by' 时生效。默认 True。
            【示例】: `{"ascending": false}`

        top_n (Optional[int]): 【可选】仅返回结果中的前N条记录。
            【示例】: `{"top_n": 5}`

        input_handle (Optional[str]): 【可选】流式输入句柄。如果提供，则在已缓存的数据上执行此操作，否则从主数据库开始。
            【示例】: `{"input_handle": "df_handle_123"}`

    Returns:
        一个包含新句柄和元信息的标准字典: {'status': 'success', 'data_handle': 'df_handle_456', 'metadata': {...}}
    """
    try:
        base_data = _get_base_data(input_handle)
        if not isinstance(base_data, pd.DataFrame):
            return {"status": "error", "message": "此操作需要DataFrame输入，但句柄指向了Series。"}

        query_df = base_data
        if filters:
            for column, condition in filters.items():
                if column not in query_df.columns: continue
                op, val = list(condition.items())[0]
                if op == '==':
                    query_df = query_df[query_df[column] == val]
                elif op == '>':
                    query_df = query_df[query_df[column] > val]
                elif op == '<':
                    query_df = query_df[query_df[column] < val]
                elif op == 'between':
                    query_df = query_df[query_df[column].between(val[0], val[1])]
        if sort_by and sort_by in query_df.columns:
            query_df = query_df.sort_values(by=sort_by, ascending=ascending)
        if top_n is not None and top_n > 0:
            query_df = query_df.head(top_n)
        if columns:
            query_df = query_df[[col for col in columns if col in query_df.columns]]

        new_handle_id = f"df_{uuid.uuid4()}"
        DATA_CACHE[new_handle_id] = query_df
        metadata = _create_metadata(query_df)
        return {"status": "success", "data_handle": new_handle_id, "metadata": metadata}
    except AttributeError as ae:
        logger.error(f"参数结构错误 (很可能是filters格式问题): {ae}", exc_info=True)
        return {"status": "error",
                "message": f"参数结构错误: {ae}。请检查 'filters' 参数是否遵循 '{{\"列名\": {{\"操作符\": \"值\"}}}}' 的嵌套格式。"}
    except Exception as e:
        logger.error(f"执行数据流操作时出错: {e}", exc_info=True)
        return {"status": "error", "message": f"执行数据流操作时出错: {e}"}


@mcp.tool()
def find_products_by_relation(relation_type: str, target_value: str) -> Dict[str, Any]:
    """
    【流工具-生产者】: 根据“关系”查找实体，创建一个新的数据流(DF)，并返回其句柄和元信息。
    这是一个“纯生产者”，它始终从全局关系表创建全新的数据流，不接受 input_handle。

    Args:
        relation_type (str): 要查询的关系名称。必须是数据菜单中的合法关系。
        target_value (str): 关系的目标值。

    【示例】: 查找由 'A电子元件厂' 供应的所有产品:
    `{"relation_type": "主要供应商", "target_value": "A电子元件厂"}`

    Returns:
        一个包含新句柄和元信息的标准字典: {'status': 'success', 'data_handle': 'df_handle_456', 'metadata': {...}}
    """
    try:
        matching_relations = relation_df[
            (relation_df['relation'] == relation_type) & (relation_df['target'] == target_value)]
        if matching_relations.empty:
            result_df = pd.DataFrame(columns=entity_df.columns)
        else:
            product_ids = matching_relations['source'].unique().tolist()
            result_df = entity_df.loc[product_ids]

        new_handle_id = f"df_{uuid.uuid4()}"
        DATA_CACHE[new_handle_id] = result_df
        metadata = _create_metadata(result_df)
        return {"status": "success", "data_handle": new_handle_id, "metadata": metadata}
    except Exception as e:
        return {"status": "error", "message": f"关系查找时出错: {e}"}


@mcp.tool()
def get_grouped_aggregation(
        group_by_col: str,
        agg_col: str,
        agg_func: str,
        input_handle: Optional[str] = None
) -> Dict[str, Any]:
    """
    【流工具-转换器/生产者】: 对数据流（或主数据）执行分组聚合操作 (GROUP BY)。
    返回一个指向新聚合结果(Series)的【新句柄】和元信息。

    Args:
        group_by_col (str): 用于分组的列名 (例如 '生产流水线编号')。
        agg_col (str): 需要进行聚合计算的数值列名 (例如 '成本(RMB)')。
        agg_func (str): 聚合函数名称 ('sum', 'mean', 'median', 'count' 等)。
        input_handle (Optional[str]): 【可选】流式输入句柄。如果为None，则对整个数据库执行操作。

    【示例 1 (全局)】: 计算“每个”流水线的“平均”成本:
    `{"group_by_col": "生产流水线编号", "agg_col": "成本(RMB)", "agg_func": "mean"}`

    【示例 2 (流式)】: 对句柄 "handle_A" 中的数据执行相同操作:
    `{"group_by_col": "...", "agg_col": "...", "agg_func": "...", "input_handle": "handle_A"}`

    Returns:
        一个包含新句柄和元信息的标准字典: {'status': 'success', 'data_handle': 'series_handle_789', 'metadata': {...}}
        注意：返回的数据类型将是 Series。
    """
    try:
        base_data = _get_base_data(input_handle)
        if not isinstance(base_data, pd.DataFrame):
            return {"status": "error", "message": "分组聚合必须在DataFrame上执行。"}

        result_series = base_data.groupby(group_by_col)[agg_col].agg(agg_func)

        new_handle_id = f"series_{uuid.uuid4()}"
        DATA_CACHE[new_handle_id] = result_series
        metadata = _create_metadata(result_series)
        return {"status": "success", "data_handle": new_handle_id, "metadata": metadata}
    except Exception as e:
        return {"status": "error", "message": f"分组聚合时发生错误: {e}"}


# --- 4. 终端工具 (Terminal Tools) ---
# 这些工具返回最终的JSON、文本或值，供LLM回答。它们不返回句柄。

@mcp.tool()
def consume_data_to_text(
        max_rows_to_show: int = 20,
        input_handle: Optional[str] = None  # <-- 【关键修改】: 变为可选的 input_handle
) -> Dict[str, Any]:
    """
    【终端工具-消费方法-已升级】: 将一个数据流（来自句柄或整个数据库）转换为人类可读的文本(CSV)。
    如果提供了 input_handle，则消费该流。
    如果未提供，则默认消费【整个数据库】（这可能非常大，应谨慎使用）。

    Args:
        max_rows_to_show (int): 【可选】最多将会话多少行数据转为文本。默认为 20。
        input_handle (Optional[str]): 【可选】要消费的数据句柄。
    """
    try:
        # 【关键逻辑】: 它现在也使用标准的基础数据获取器
        target_data = _get_base_data(input_handle)

        total_rows = len(target_data)
        if total_rows == 0:
            return {"status": "success", "text_data": "(无数据)", "total_rows": 0}

        final_text_data = target_data.head(max_rows_to_show).to_csv(index=True)
        if total_rows > max_rows_to_show:
            final_text_data += f"\n\n... (数据已被截断，仅显示前 {max_rows_to_show} 行，总行数为 {total_rows})"

        return {"status": "success", "text_data": final_text_data, "total_rows": total_rows}
    except Exception as e:
        return {"status": "error", "message": f"消费数据句柄时出错: {e}"}


@mcp.tool()
def get_scalar_aggregation(
        agg_col: str,
        agg_func: str,
        input_handle: Optional[str] = None
) -> Dict[str, Any]:
    """
    【终端工具-计算】: 对一个数据流(DF)或整个数据库执行聚合，并直接返回【单个标量值】（数字）。
    用于直接回答“总和是多少？”或“平均值是多少？”这类问题。

    Args:
        agg_col (str): 需要聚合的列名 (例如 '累计销量')。
        agg_func (str): 聚合函数 (如 'sum', 'mean')。
        input_handle (Optional[str]): 【可选】数据句柄。如果为None，则对【整个数据库】聚合。

    【示例 1 (全局)】: 计算所有产品的累计销量总和:
    `{"agg_col": "累计销量", "agg_func": "sum"}`

    【示例 2 (流式)】: 计算句柄 "handle_A" 中产品的平均成本:
    `{"agg_col": "成本(RMB)", "agg_func": "mean", "input_handle": "handle_A"}`

    Returns:
        一个包含最终计算值的字典: {'status': 'success', 'data': {'aggregation_result': 12345.6}}
    """
    try:
        base_data = _get_base_data(input_handle)
        if not isinstance(base_data, pd.DataFrame):
            return {"status": "error", "message": "此操作需要DataFrame输入。"}
        if base_data.empty:
            return {"status": "success", "data": {"aggregation_result": None}, "message": "数据为空。"}

        result = base_data[agg_col].agg(agg_func)
        result = float(result) if pd.notna(result) else None
        return {"status": "success", "data": {"aggregation_result": sanitize_for_json(result)}}
    except KeyError:
        return {"status": "error", "message": f"列 '{agg_col}' 不存在。"}
    except Exception as e:
        return {"status": "error", "message": f"标量聚合时出错: {e}"}


# --- 5. 独立终端工具 与 维护工具 ---
# 这些工具不参与流式处理，它们执行独立的、即时的查找或维护。

@mcp.tool()
def get_most_common_attributes(
        target_column: str,
        top_n: int = 1,
        input_handle: Optional[str] = None
) -> Dict[str, Any]:
    """
    【终端工具-计算】: 查找一个“属性”列中最常见的前N个值。
    如果提供了 input_handle，则在【该数据流】中查找。
    如果未提供 input_handle，则在【整个实体数据库】中查找。

    Args:
        target_column (str): 要统计的实体属性列名 (例如 '生产年份')。
        top_n (int): 返回排名前几的结果。默认为 1。
        input_handle (Optional[str], optional): 可选的数据句柄。

    【示例 1 (全局)】: 整个数据库中最常见的生产年份:
    `{"target_column": "生产年份", "top_n": 1}`

    【示例 2 (流式)】: 句柄 "handle_B" 中最常见的固件版本 (前3名):
    `{"target_column": "固件版本", "top_n": 3, "input_handle": "handle_B"}`

    Returns:
        一个包含最终计数字典的字典: {'status': 'success', 'data': {'项目1': 数量1, ...}}
    """
    try:
        base_data = _get_base_data(input_handle)
        if not isinstance(base_data, pd.DataFrame):
            return {"status": "error", "message": "此操作必须在DataFrame上执行。"}
        if target_column not in base_data.columns:
            return {"status": "error", "message": f"列 '{target_column}' 不存在。"}

        counts = base_data[target_column].value_counts()
        result_dict = counts.head(top_n).to_dict()
        return {"status": "success", "data": sanitize_for_json(result_dict)}
    except Exception as e:
        return {"status": "error", "message": f"计算最常见属性时出错: {e}"}


@mcp.tool()
def get_most_common_relations(relation_type: str, top_n: int = 1) -> Dict[str, Any]:
    """
    【独立终端工具】: 查找“关系”数据中最常见的目标。
    (注意: 此工具始终查询【全局关系表 (relation_df)】，它不接受 input_handle)。

    Args:
        relation_type (str): 要统计的关系名称 (例如 '主要供应商')。
        top_n (int): 返回排名前几的结果。默认为 1。

    【示例】: 查找排名第一的主要供应商:
    `{"relation_type": "主要供应商", "top_n": 1}`

    Returns:
        一个包含最终计数字典的字典: {'status': 'success', 'data': {'目标A': 数量X, ...}}
    """
    try:
        counts = relation_df[relation_df['relation'] == relation_type]['target'].value_counts()
        if counts.empty:
            return {"status": "success", "data": {}, "message": f"找不到关于 '{relation_type}' 的关系记录。"}
        result_dict = counts.head(top_n).to_dict()
        return {"status": "success", "data": sanitize_for_json(result_dict)}
    except Exception as e:
        return {"status": "error", "message": f"计算最常见关系时出错: {e}"}


@mcp.tool()
def get_full_entity_details(entity_id: str) -> Dict[str, Any]:
    """
    【独立终端工具】: 根据单一产品ID查询其全部详细信息（属性和关系）。
    这是一个即时查找工具，不参与流式处理。

    Args:
        entity_id (str): 【必需】需要查询的产品的唯一ID。

    【示例】:
    `{"entity_id": "EFM8664"}`

    Returns:
        一个包含该实体完整画像的字典: {'status': 'success', 'data': {'entity_id': ..., 'attributes': {...}, 'relations': {...}}}
    """
    if entity_id not in entity_df.index:
        return {"status": "error", "message": f"知识库中找不到实体 '{entity_id}'。"}
    try:
        attributes = entity_df.loc[entity_id].to_dict()
        attributes_cleaned = {key: value for key, value in attributes.items() if pd.notna(value)}
        relations = relation_df[relation_df['source'] == entity_id]
        relations_grouped = {rel_type: group['target'].tolist() for rel_type, group in relations.groupby('relation')}
        full_details = {"entity_id": entity_id, "attributes": attributes_cleaned, "relations": relations_grouped}
        return {"status": "success", "data": full_details}
    except Exception as e:
        return {"status": "error", "message": f"查询实体 '{entity_id}' 时发生内部错误: {e}"}


@mcp.tool()
def compare_entity_attributes(entity_id_1: str, entity_id_2: str, attribute_name: str) -> Dict[str, Any]:
    """
    【独立终端工具】: 比较两个指定实体的某个特定属性的值。不参与流式处理。

    Args:
        entity_id_1 (str): 第一个需要比较的产品ID。
        entity_id_2 (str): 第二个需要比较的产品ID。
        attribute_name (str): 需要比较的属性名称 (例如 '成本(RMB)')。

    【示例】:
    `{"entity_id_1": "EFM8664", "entity_id_2": "EFM8665", "attribute_name": "成本(RMB)"}`

    Returns:
        一个包含比较结果的字典: {'status': 'success', 'data': {'ID1': val1, 'ID2': val2, 'difference': diff}}
    """
    try:
        val1 = entity_df.loc[entity_id_1, attribute_name]
        val2 = entity_df.loc[entity_id_2, attribute_name]
        diff = None
        if pd.notna(val1) and pd.notna(val2) and isinstance(val1, (int, float, np.number)) and isinstance(val2,
                                                                                                          (int, float,
                                                                                                           np.number)):
            diff = float(val1 - val2)
        return {"status": "success",
                "data": {entity_id_1: sanitize_for_json(val1), entity_id_2: sanitize_for_json(val2),
                         "difference": sanitize_for_json(diff)}}
    except Exception as e:
        return {"status": "error", "message": f"比较属性时发生错误: {e}"}


@mcp.tool()
def clear_cache(flush_all: bool = True) -> Dict[str, Any]:
    """
    【维护工具】: 清理服务器端的所有临时数据缓存。
    (注意: 此工具由客户端自动调用，LLM不应（也无法）看到或调用此工具)。
    """
    global DATA_CACHE
    count = len(DATA_CACHE)
    DATA_CACHE.clear()
    return {"status": "success", "message": f"服务器缓存已清空，释放了 {count} 个缓存项。"}


# --- 6. 启动服务器 ---
if __name__ == "__main__":
    logger.info("正在启动MCP服务器 (统一句柄架构 V2)...")
    mcp.run(transport='stdio')