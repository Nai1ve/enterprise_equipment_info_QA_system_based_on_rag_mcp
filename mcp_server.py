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

# 【代码修复】：请用此 V5 最终版替换 server.py 中现有的 execute_query 函数

@mcp.tool()
def execute_query(
        filters: Optional[Dict[str, Any]] = None,
        columns: Optional[List[str]] = None,
        sort_by: Optional[str] = None,
        ascending: bool = True,
        top_n: Optional[int] = None,
        relation_filters: Optional[Dict[str, str]] = None,
        input_handle: Optional[str] = None
) -> Dict[str, Any]:
    """
    【核心流工具-V5最终版】: 对数据执行查询、筛选、排序、切片和列选择。

    Args:
        ... (filters, columns 参数不变) ...
        sort_by (Optional[str]): 【可选】用于排序的单个列名。
        ascending (bool): 【可选】是否按升序排序。仅在提供了 'sort_by' 时生效。默认 True。
        top_n (Optional[int]): 【可选】仅在提供了 'sort_by' 时生效。返回排序后的前 N 个结果，并【自动包含所有共享第N个位置的并列项】。
        ... (relation_filters, input_handle 参数不变) ...
    """
    if filters and "产品ID" in filters:
        return { "status": "error", "message": "逻辑错误：不能使用 'execute_query' 按 '产品ID' 筛选。请改用 'get_full_entity_details'。" }
    try:
        base_data = _get_base_data(input_handle)
        if not isinstance(base_data, pd.DataFrame):
            return {"status": "error", "message": "此操作需要DataFrame输入。"}
        query_df = base_data
        if filters:
            for column, condition in filters.items():
                if column not in query_df.columns:
                    logger.error(f"execute_query 收到对不存在的列 '{column}' 的筛选请求。")
                    raise KeyError(f"筛选列 '{column}' 在当前数据流中不存在。请检查列名，或使用 'relation_filters' 参数来按关系筛选。")
                op, val = list(condition.items())[0]
                if op == '==': query_df = query_df[query_df[column] == val]
                elif op == '>': query_df = query_df[query_df[column] > val]
                elif op == '<': query_df = query_df[query_df[column] < val]
                elif op == '>=': query_df = query_df[query_df[column] >= val]
                elif op == '<=': query_df = query_df[query_df[column] <= val]
                elif op == 'between': query_df = query_df[query_df[column].between(val[0], val[1])]
                else:
                    SUPPORTED_OPS = "==, >, <, >=, <=, between"
                    raise ValueError(f"不支持的操作符: '{op}'。支持的操作符包括: {SUPPORTED_OPS}。")
        if relation_filters:
            for rel_type, rel_target in relation_filters.items():
                product_ids_with_relation = relation_df[(relation_df['relation'] == rel_type) & (relation_df['target'] == rel_target)]['source'].unique()
                query_df = query_df[query_df.index.isin(product_ids_with_relation)]
        if sort_by:
            if sort_by not in query_df.columns:
                 raise KeyError(f"用于排序的列 '{sort_by}' 不存在。")
            if top_n is not None and top_n > 0:
                if ascending:
                    query_df = query_df.nsmallest(top_n, columns=sort_by, keep='all')
                else:
                    query_df = query_df.nlargest(top_n, columns=sort_by, keep='all')
            else:
                query_df = query_df.sort_values(by=sort_by, ascending=ascending)
        if columns:
            query_df = query_df[[col for col in columns if col in query_df.columns]]
        new_handle_id = f"df_{uuid.uuid4()}"
        DATA_CACHE[new_handle_id] = query_df
        metadata = _create_metadata(query_df)
        return {"status": "success", "data_handle": new_handle_id, "metadata": metadata}
    except (ValueError, TypeError, KeyError) as ve:
        return {"status": "error", "message": f"执行 execute_query 操作时出错: {ve}"}
    except Exception as e:
        return {"status": "error", "message": f"执行 execute_query 操作时出错: {e}"}


# 【代码修复】：请用此 V5 最终版替换 server.py 中现有的 consume_data_to_text 函数

@mcp.tool()
def consume_data_to_text(
        columns: Optional[List[str]] = None,  # <-- 【【革命性新功能】】
        max_rows_to_show: int = 20,
        include_index: bool = True,
        input_handle: Optional[str] = None
) -> Dict[str, Any]:
    """
    【终端工具-消费方法-已升级V5】: 将数据流转换为人类可读的文本(CSV)，并可选择最终输出的列。

    Args:
        columns (Optional[List[str]]): 【可选】指定最终输出中应包含哪些列。如果省略，则输出所有列。
        max_rows_to_show (int): 【可选】最多显示多少行。传入 -1 代表返回所有数据。默认为 20。
        include_index (bool): 【可选】是否在最终的 CSV 输出中包含索引（即'产品ID'）。默认为 True。
        input_handle (Optional[str]): 【可选】要消费的数据句柄。
    """
    try:
        target_data = _get_base_data(input_handle)

        # --- 【【新功能：执行最终的列选择】】 ---
        if columns:
            # 确保我们不会因为请求不存在的列而崩溃
            valid_columns = [col for col in columns if col in target_data.columns]
            if not valid_columns:
                # 如果用户请求的所有列都不存在，这是一个错误
                return {"status": "error", "message": f"请求的列 {columns} 在数据流中均不存在。"}
            target_data = target_data[valid_columns]
        # --- 新功能结束 ---

        total_rows = len(target_data)

        if total_rows == 0:
            return {"status": "success", "text_data": "(无数据)", "total_rows": 0, "is_truncated": False}

        text_generator = lambda data: data.to_csv(index=include_index)

        if max_rows_to_show == -1 or total_rows <= max_rows_to_show:
            final_text_data = text_generator(target_data)
            is_truncated = False
        else:
            final_text_data = text_generator(target_data.head(max_rows_to_show))
            final_text_data += f"\n\n... (数据已被截断，仅显示前 {max_rows_to_show} 行，总行数为 {total_rows})"
            is_truncated = True

        return {"status": "success", "text_data": final_text_data, "total_rows": total_rows,
                "is_truncated": is_truncated}
    except Exception as e:
        return {"status": "error", "message": f"消费数据句柄时出错: {e}"}

@mcp.tool()
def find_products_by_relation(
        relation_type: str,
        target_value: Optional[str] = None
) -> Dict[str, Any]:
    """
    【流工具-生产者-已升级V2】: 根据“关系”查找实体，创建一个新的数据流(DF)，并返回其句柄和元信息。
    这是一个“纯生产者”，它始终从全局关系表创建全新的数据流，不接受 input_handle。

    Args:
        relation_type (str): 【必需】要查询的关系名称。必须是数据菜单中的合法关系。
        target_value (Optional[str]): 【可选】关系的目标值。

    【用例 1：指定目标值 (和以前一样)】
    查找由 'A电子元件厂' 供应的所有产品:
    `{"relation_type": "主要供应商", "target_value": "A电子元件厂"}`

    【用例 2：查询关系存在性 (新功能)】
    查找所有【具有】“替代型号”（无论具体型号是什么）的产品:
    `{"relation_type": "替代型号"}` (只需省略 target_value)

    Returns:
        一个包含新句柄和元信息的标准字典: {'status': 'success', 'data_handle': 'df_handle_456', 'metadata': {...}}
    """
    try:
        # 1. 基础筛选：首先按关系类型筛选
        base_filter = (relation_df['relation'] == relation_type)

        # 2. 可选增强：如果提供了目标值，则进一步筛选
        if target_value:
            base_filter = base_filter & (relation_df['target'] == target_value)

        # 3. 应用筛选
        matching_relations = relation_df[base_filter]

        if matching_relations.empty:
            result_df = pd.DataFrame(columns=entity_df.columns)
        else:
            # 4. 获取所有匹配的源ID，并去重
            product_ids = matching_relations['source'].unique().tolist()
            # 5. 从主实体表 (entity_df) 中提取这些产品的完整信息
            result_df = entity_df.loc[product_ids]

        # 6. 缓存结果并返回句柄
        new_handle_id = f"df_{uuid.uuid4()}"
        DATA_CACHE[new_handle_id] = result_df
        metadata = _create_metadata(result_df)
        return {"status": "success", "data_handle": new_handle_id, "metadata": metadata}
    except KeyError:
        return {"status": "error", "message": f"关系类型 '{relation_type}' 不在数据库中。"}
    except Exception as e:
        return {"status": "error", "message": f"关系查找时出错: {e}"}


@mcp.tool()
def get_entity_unique_value_count(
        target_column: str,
        drop_na: bool = True,
        input_handle: Optional[str] = None
) -> Dict[str, Any]:
    """
    【终端工具-计算】【专用于实体表】: 计算一个指定【实体属性】列中的不同（唯一）值的数量 (COUNT DISTINCT)。
    这用于回答 “有多少种不同的...” 这类关于实体属性的问题。

    Args:
        target_column (str): 需要计算唯一值的列名 (例如 '生产年份')。
        drop_na (bool): 是否忽略空值 (NaN)。默认为 True。
        input_handle (Optional[str]): 【可选】数据句柄。如果为None，则对【整个 entity_df】计算。

    【示例 1 (全局)】: 计算整个数据库中有多少个不同的“生产流水线编号”：
    `{"target_column": "生产流水线编号"}`

    Returns:
        一个包含最终计数值的字典: {'status': 'success', 'data': {'target_column': '...', 'unique_count': 15}}
    """
    try:
        # 此函数通过 _get_base_data 访问 entity_df 或其派生流
        base_data = _get_base_data(input_handle)
        if not isinstance(base_data, pd.DataFrame):
            return {"status": "error", "message": "此操作需要DataFrame输入。"}

        if target_column not in base_data.columns:
            if target_column == base_data.index.name or target_column == "产品ID":
                count = base_data.index.nunique()
                return {"status": "success", "data": {"target_column": target_column, "unique_count": int(count)}}
            else:
                return {"status": "error", "message": f"列 '{target_column}' 不存在。"}

        count = base_data[target_column].nunique(dropna=drop_na)

        return {"status": "success", "data": {"target_column": target_column, "unique_count": int(count)}}
    except Exception as e:
        return {"status": "error", "message": f"计算实体唯一值数量时出错: {e}"}


@mcp.tool()
def get_relation_unique_value_count(
        target_column: str
) -> Dict[str, Any]:
    """
    【终端工具-计算】【专用于关系表】: 计算【关系表】指定列中不同（唯一）值的数量。
    用于回答 “有多少种不同的关系类型？” 或 “有多少个产品充当了关系的源头？”。
    它【不接受】input_handle，始终在完整的 relation_df 上操作。

    Args:
        target_column (str): 需要计算唯一值的列名 (必须是 'source', 'relation', 或 'target' 之一)。

    【示例】: 计算数据库中有多少种不同的“关系类型”：
    `{"target_column": "relation"}`

    Returns:
        一个包含最终计数值的字典: {'status': 'success', 'data': {'target_column': '...', 'unique_count': 7}}
    """
    try:
        # 硬编码到 relation_df
        base_data = relation_df

        RELATION_COLS = ['source', 'relation', 'target']
        if target_column not in RELATION_COLS:
            return {"status": "error",
                    "message": f"目标列 '{target_column}' 不是有效的关系列 ('source', 'relation', 'target')。"}

        count = base_data[target_column].nunique(dropna=True)

        return {"status": "success", "data": {"target_column": target_column, "unique_count": int(count)}}
    except Exception as e:
        return {"status": "error", "message": f"计算关系唯一值数量时出错: {e}"}

@mcp.tool()
def find_relations_for_stream(
        relation_type: str,
        input_handle: str
) -> Dict[str, Any]:
    """
    【流工具-转换器/连接】: 获取一个现有的数据流句柄（代表一组产品），并查找这【整组产品】的指定类型的【新关系】。

    这用于解决复杂的多跳(multi-hop)查询。例如：先找到A组产品，然后查找A组产品的所有B类关系。
    它返回一个【新的数据流】（一个DataFrame句柄），内容是匹配的关系表（列: source, relation, target）。

    Args:
        relation_type (str): 【必需】要为这个流中的所有产品查找的关系名称 (例如 '关联服务/APP ID')。
        input_handle (str): 【必需】必须提供一个指向数据流（DataFrame）的输入句柄。

    【工作流示例】:
    问题: "找出所有具有替代型号的设备，及其对应的 关联服务/APP ID。"
    计划:
    1. (思考): 首先，找到所有“具有替代型号”的设备。
    2. (行动): 调用 `find_products_by_relation(relation_type='替代型号')`
    3. (观察): (收到新句柄 'handle_A')
    4. (思考): 现在我有一个包含这些设备的流(handle_A)。我要查找这个流中【所有设备】的 '关联服务/APP ID' 关系。
    5. (行动): 调用 `find_relations_for_stream(relation_type='关联服务/APP ID', input_handle='handle_A')`
    6. (观察): (收到新句柄 'handle_B'，这是一个包含 [source_id, '关联服务/APP ID', app_id] 的关系表)
    7. (思考): 将这个最终的关系列表展示给用户。
    8. (行动): 调用 `consume_data_to_text(input_handle='handle_B')`

    Returns:
        一个包含新句柄和元信息的标准字典: {'status': 'success', 'data_handle': 'df_relations_456', 'metadata': {...}}
    """
    try:
        if not input_handle or input_handle not in DATA_CACHE:
            return {"status": "error", "message": "错误：此工具必须提供一个有效的 input_handle。"}

        base_data = _get_base_data(input_handle)
        if not isinstance(base_data, pd.DataFrame):
            return {"status": "error", "message": "输入句柄必须指向一个DataFrame。"}

        # 1. 从输入的数据流（这是一个来自 entity_df 的切片）中获取所有产品 ID
        product_ids_in_stream = base_data.index.unique().tolist()
        if not product_ids_in_stream:
            # 输入流为空，返回一个空的关系DF
            result_df = pd.DataFrame(columns=['source', 'relation', 'target'])
        else:
            # 2. 在【全局关系表 (relation_df)】中，筛选出(A)ID在流中 且 (B)关系类型匹配 的所有条目
            result_df = relation_df[
                (relation_df['source'].isin(product_ids_in_stream)) &
                (relation_df['relation'] == relation_type)
                ]

        # 3. 缓存这个新的关系DF并返回新句柄
        new_handle_id = f"df_relations_{uuid.uuid4()}"
        DATA_CACHE[new_handle_id] = result_df
        metadata = _create_metadata(result_df)
        return {"status": "success", "data_handle": new_handle_id, "metadata": metadata}
    except Exception as e:
        return {"status": "error", "message": f"为数据流查找关系时发生错误: {e}"}


# 【代码修复】：请用此修复了 Join 索引 BUG 的版本，替换 server.py 中现有的 enrich_stream_with_relation 函数

@mcp.tool()
def enrich_stream_with_relation(
        relation_type: str,
        relation_target_col_name: str,
        join_type: str = 'left',
        columns_to_keep: Optional[List[str]] = None,  # <-- 【【新功能】】
        input_handle: Optional[str] = None
) -> Dict[str, Any]:
    """
    【流工具-V2最终版】【连接(JOIN)并筛选】: 使用关系表数据来“丰富”当前数据流，并可选择性地只保留最终需要的列。

    Args:
        relation_type (str): 【必需】要用于连接(Join)的关系名称 (例如 '主要供应商')。
        relation_target_col_name (str): 【必需】为新添加的关系目标列指定名称 (例如 '供应商名称')。
        join_type (str): 【可选】连接类型, 'left' 或 'inner'。默认为 'left'。
        columns_to_keep (Optional[List[str]]): 【可选】在 JOIN 后，只保留这个列表中的列以及新添加的关系列。这用于生成干净的最终输出。
        input_handle (Optional[str]): 【可选】要丰富的句柄。如果为None，则丰富【整个数据库】。

    【工作流示例 - JOIN 后直接清理】:
    问题: "列出所有具有替代型号的设备型号及其核心组件编号和替代型号。"
    计划:
    1. `find_products_by_relation(relation_type='替代型号')` -> (得到 handle_A)
    2. `enrich_stream_with_relation(
            relation_type='替代型号',
            relation_target_col_name='替代型号',
            columns_to_keep=['核心组件编号'], # 只保留“核心组件编号”，新的“替代型号”列会自动保留
            input_handle='handle_A'
       )` -> (得到 handle_B，一个只含 2 列 + 索引的干净数据框)
    3. `consume_data_to_text(input_handle='handle_B', max_rows_to_show=-1)`
    """
    try:
        base_data = _get_base_data(input_handle).copy()
        original_index_name = base_data.index.name

        relation_data_to_join = relation_df[relation_df['relation'] == relation_type][['source', 'target']]
        relation_data_to_join = relation_data_to_join.rename(columns={'target': relation_target_col_name})

        enriched_df = pd.merge(
            base_data,
            relation_data_to_join,
            left_index=True,
            right_on='source',
            how=join_type
        )

        if 'source' in enriched_df.columns:
            enriched_df = enriched_df.set_index('source')
            enriched_df.index.name = original_index_name
        else:
            raise KeyError("Join 操作后未能找到 'source' 列，无法恢复索引。")

        # --- 【【新功能：执行最终的列选择】】 ---
        if columns_to_keep:
            # 最终要保留的列 = 用户指定的列 + 我们刚刚新加入的那一列
            final_columns_to_keep = columns_to_keep + [relation_target_col_name]

            # 过滤不存在的列以避免崩溃
            valid_columns = [col for col in final_columns_to_keep if col in enriched_df.columns]
            enriched_df = enriched_df[valid_columns]
        # --- 新功能结束 ---

        new_handle_id = f"df_enriched_{uuid.uuid4()}"
        DATA_CACHE[new_handle_id] = enriched_df
        metadata = _create_metadata(enriched_df)
        return {"status": "success", "data_handle": new_handle_id, "metadata": metadata}

    except Exception as e:
        logger.error(f"Enrich (Join) 操作失败: {e}", exc_info=True)
        return {"status": "error", "message": f"Enrich (Join) 操作失败: {e}"}


# 【代码修复】：请用此 V3 版本替换 server.py 中现有的 get_relation_grouped_aggregation

# 【代码修复】：请用此 V4 版本替换 server.py 中现有的 get_relation_grouped_aggregation

@mcp.tool()
def get_relation_grouped_aggregation(
        group_by_col: str,
        agg_col: str,
        agg_func: str,
        relation_type: Optional[str] = None,
        exclude_targets: Optional[List[str]] = None,  # <-- 【【新功能】】: 排除列表
        sort_direction: Optional[str] = None,
        top_n: Optional[int] = None,
) -> Dict[str, Any]:
    """
    【流工具-生产者-V4版】【专用于关系表】...

    Args:
        ... (参数不变) ...
        exclude_targets (Optional[List[str]]): 【可选】在聚合前，从 'target' 列中排除一个或多个指定的值 (例如 ['无', 'None'])。
        ... (其余参数不变) ...

    【工作流示例 - 排除无效值】:
    问题: "哪个【有效的】主要供应商提供的产品种类最多？"
    计划:
    1. (思考): 我需要查找主要供应商，但我应该排除 '无' 这个无效占位符。
    2. (行动): `get_relation_grouped_aggregation(
                    relation_type='主要供应商',
                    group_by_col='target',
                    agg_col='source',
                    agg_func='nunique',
                    sort_direction='descending',
                    top_n=1,
                    exclude_targets=['无', 'None']  # <-- 使用新功能
                )`
    ...
    """
    try:
        base_data = relation_df
        if relation_type:
            if relation_type not in base_data['relation'].unique():
                return {"status": "error", "message": f"关系类型 '{relation_type}' 不存在。"}
            base_data = base_data[base_data['relation'] == relation_type]

        # --- 【【新功能：执行排除】】 ---
        if exclude_targets:
            # 在执行任何分组前，先从 base_data 中过滤掉不想要的 target
            base_data = base_data[~base_data['target'].isin(exclude_targets)]
        # --- 新功能结束 ---

        # (验证和聚合逻辑保持不变)
        RELATION_COLS = ['source', 'relation', 'target']
        if group_by_col not in RELATION_COLS:
            return {"status": "error", "message": f"用于分组的列 '{group_by_col}' 不是有效的关系列。"}
        if agg_col not in RELATION_COLS:
            return {"status": "error", "message": f"用于聚合的列 '{agg_col}' 不是有效的关系列。"}

        grouped = base_data.groupby(group_by_col)[agg_col]

        SUPPORTED_FUNCS = "'count', 'nunique', 'max', 'min'"
        if agg_func == 'count':
            result_series = grouped.count()
        elif agg_func == 'nunique':
            result_series = grouped.nunique()
        elif agg_func == 'max':
            result_series = grouped.max()
        elif agg_func == 'min':
            result_series = grouped.min()
        else:
            raise ValueError(f"不支持的聚合函数: '{agg_func}'. 此工具仅支持: {SUPPORTED_FUNCS}。")

        # (排序和 Top-N 逻辑保持不变)
        if sort_direction:
            if sort_direction == 'ascending':
                result_series = result_series.sort_values(ascending=True)
                if top_n: result_series = result_series.nsmallest(top_n, keep='all')
            elif sort_direction == 'descending':
                result_series = result_series.sort_values(ascending=False)
                if top_n: result_series = result_series.nlargest(top_n, keep='all')
            else:
                raise ValueError(f"sort_direction 必须是 'ascending' 或 'descending'。")

        # (缓存和返回逻辑保持不变)
        new_handle_id = f"series_{uuid.uuid4()}"
        DATA_CACHE[new_handle_id] = result_series
        metadata = _create_metadata(result_series)
        return {"status": "success", "data_handle": new_handle_id, "metadata": metadata}
    except (ValueError, KeyError) as ve:
        logger.error(f"关系分组聚合失败: {ve}", exc_info=True)
        return {"status": "error", "message": f"关系分组聚合失败: {ve}"}
    except Exception as e:
        return {"status": "error", "message": f"关系分组聚合时发生未知错误: {e}"}


@mcp.tool()
def get_unique_relation_target_count(
        relation_type: str
) -> Dict[str, Any]:
    """
    【终端工具-计算】: 专用于关系表。计算指定“关系类型”的【不同（唯一）目标(target)的数量】。
    例如，用于回答 “有多少种不同的兼容设备？” 或 “有多少个不同的主要供应商？”。
    它自动忽略空值。

    Args:
        relation_type (str): 需要计算唯一目标的关系类型 (例如 '兼容设备', '主要供应商')。

    【示例】: 计算数据库中有多少种不同的“兼容设备”：
    `{"relation_type": "兼容设备"}`

    Returns:
        一个包含最终计数值的字典: {'status': 'success', 'data': {'relation_type': '...', 'unique_target_count': 141}}
    """
    try:
        if relation_type not in relation_df['relation'].unique():
            return {"status": "error", "message": f"关系类型 '{relation_type}' 不存在。"}

        # 【核心逻辑】: 1. 筛选关系类型  2. 在 'target' 列上执行 nunique()
        filtered_df = relation_df[relation_df['relation'] == relation_type]
        count = filtered_df['target'].nunique(dropna=True)

        return {"status": "success", "data": {"relation_type": relation_type, "unique_target_count": int(count)}}
    except Exception as e:
        return {"status": "error", "message": f"计算唯一关系目标时出错: {e}"}


# 【代码修复】：请用此 V5 版本替换 server.py 中现有的 get_entity_grouped_aggregation

@mcp.tool()
def get_entity_grouped_aggregation(
        group_by_col: str,
        agg_col: str,
        agg_func: str,
        sort_direction: Optional[str] = None,
        top_n: Optional[int] = None,
        input_handle: Optional[str] = None
) -> Dict[str, Any]:
    """
    【流工具-V5最终版】【专用于实体表】: ...

    Args:
        ... (参数不变) ...
        agg_func (str): 聚合函数。支持: 'sum', 'mean', 'median', 'count', 'nunique',
                        【新】'max', 'min'。
        ... (其余参数不变) ...
    """
    try:
        base_data = _get_base_data(input_handle)
        if not isinstance(base_data, pd.DataFrame):
            return {"status": "error", "message": "分组聚合必须在DataFrame上执行。"}

        if group_by_col not in base_data.columns:
            return {"status": "error", "message": f"用于分组的列 '{group_by_col}' 不存在。"}

        if agg_col not in base_data.columns:
            if agg_col == base_data.index.name or agg_col == "产品ID":
                logger.info(f"聚合目标 '{agg_col}' 是索引，正在将其重置为列以便操作。")
                base_data = base_data.reset_index()
            else:
                return {"status": "error", "message": f"用于聚合的列 '{agg_col}' 不存在。"}

        # 1. 执行核心聚合
        grouped = base_data.groupby(group_by_col)[agg_col]

        # --- 【【功能增强：添加 max 和 min】】 ---
        if agg_func == 'sum':
            result_series = grouped.sum()
        elif agg_func == 'mean':
            result_series = grouped.mean()
        elif agg_func == 'median':
            result_series = grouped.median()
        elif agg_func == 'count':
            result_series = grouped.count()
        elif agg_func == 'nunique':
            result_series = grouped.nunique()
        elif agg_func == 'max':  # <-- 新增
            result_series = grouped.max()
        elif agg_func == 'min':  # <-- 新增
            result_series = grouped.min()
        else:
            SUPPORTED_FUNCS = "'sum', 'mean', 'median', 'count', 'nunique', 'max', 'min'"
            raise ValueError(f"不支持的聚合函数: '{agg_func}'. 支持: {SUPPORTED_FUNCS}。")
        # --- 增强结束 ---

        # 2. 排序和并列感知Top-N逻辑 (不变)
        if sort_direction:
            # ... (这部分代码与您已有的版本相同, 此处省略以保持简洁)
            if sort_direction == 'ascending':
                result_series = result_series.sort_values(ascending=True)
                if top_n:
                    result_series = result_series.nsmallest(top_n, keep='all')
            elif sort_direction == 'descending':
                result_series = result_series.sort_values(ascending=False)
                if top_n:
                    result_series = result_series.nlargest(top_n, keep='all')
            else:
                raise ValueError(f"sort_direction 必须是 'ascending' 或 'descending'。")

        # 3. 缓存并返回 (不变)
        new_handle_id = f"series_{uuid.uuid4()}"
        DATA_CACHE[new_handle_id] = result_series
        metadata = _create_metadata(result_series)
        return {"status": "success", "data_handle": new_handle_id, "metadata": metadata}
    except (ValueError, KeyError) as ve:
        logger.error(f"实体分组聚合失败: {ve}", exc_info=True)
        return {"status": "error", "message": f"实体分组聚合失败: {ve}"}
    except Exception as e:
        return {"status": "error", "message": f"实体分组聚合时发生未知错误: {e}"}


# --- 4. 终端工具 (Terminal Tools) ---
# 这些工具返回最终的JSON、文本或值，供LLM回答。它们不返回句柄。

# 【代码修复】：请用此 V4 最终版替换 server.py 中现有的 consume_data_to_text 函数

@mcp.tool()
def consume_data_to_text(
        columns: Optional[List[str]] = None,
        max_rows_to_show: int = 20,
        include_index: bool = True,
        input_handle: Optional[str] = None
) -> Dict[str, Any]:
    """
    【终端工具-消费方法-已升级V6】: 将数据流（DataFrame 或 Series）转换为文本(CSV)。
    (其余 Docstring 不变...)
    """
    try:
        target_data = _get_base_data(input_handle)

        # --- 【【关键 BUG 修复：兼容 Series 类型】】 ---
        if isinstance(target_data, pd.Series):
            # 如果输入是 Series，它没有 .columns 属性。
            # 我们忽略 'columns' 参数，因为 Series 本身就是要展示的数据。
            logger.info("输入数据为 Series，将直接转换为 CSV。")
        elif isinstance(target_data, pd.DataFrame) and columns:
            # 仅当输入是 DataFrame 时，才执行列选择
            valid_columns = [col for col in columns if col in target_data.columns]
            if not valid_columns and columns:  # 如果请求了列但一列都不存在
                return {"status": "error", "message": f"请求的列 {columns} 在数据流中均不存在。"}
            if valid_columns:
                target_data = target_data[valid_columns]
        # --- 修复结束 ---

        total_rows = len(target_data)

        if total_rows == 0:
            return {"status": "success", "text_data": "(无数据)", "total_rows": 0, "is_truncated": False}

        text_generator = lambda data: data.to_csv(index=include_index, header=True)  # 确保 Series 也输出 header

        if max_rows_to_show == -1 or total_rows <= max_rows_to_show:
            final_text_data = text_generator(target_data)
            is_truncated = False
        else:
            final_text_data = text_generator(target_data.head(max_rows_to_show))
            final_text_data += f"\n\n... (数据已被截断，仅显示前 {max_rows_to_show} 行，总行数为 {total_rows})"
            is_truncated = True

        return {"status": "success", "text_data": final_text_data, "total_rows": total_rows,
                "is_truncated": is_truncated}
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
    (其余 Docstring 不变...)
    """
    try:
        base_data = _get_base_data(input_handle)
        if not isinstance(base_data, pd.DataFrame):
            return {"status": "error", "message": "此操作需要DataFrame输入。"}
        if base_data.empty:
            return {"status": "success", "data": {"aggregation_result": None}, "message": "数据为空。"}

        # --- 【【关键 BUG 修复：处理索引聚合】】 ---
        # 检查要聚合的列是否在列中
        if agg_col not in base_data.columns:
            # 如果不在列中，检查它是否是索引
            if agg_col == base_data.index.name or agg_col == "产品ID":
                # 为了能对其进行 .agg() 操作，我们必须先将索引重置为一个常规列
                logger.info(f"聚合目标 '{agg_col}' 是索引，正在将其重置为列以便操作。")
                base_data = base_data.reset_index()
            else:
                # 如果既不在列中，也不是索引，那么它确实不存在
                return {"status": "error", "message": f"用于聚合的列 '{agg_col}' 不存在。"}
        # --- 修复结束 ---

        # 此时，agg_col 必然是 base_data 的一个常规列
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
        ascending: bool = False,
        input_handle: Optional[str] = None
) -> Dict[str, Any]:
    """
    【终端工具-计算-已升级】: 按频率统计一个“属性”列的值，并返回最高（最常见）或最低（最不常见）的前N个值。

    如果提供了 input_handle，则在【该数据流】中统计。
    如果未提供 input_handle，则在【整个实体数据库】中统计。

    Args:
        target_column (str): 要统计的实体属性列名 (例如 '生产年份', '固件版本')。
        top_n (int): 返回排名前几的结果。默认为 5。
        ascending (bool): 【可选】排序方式。默认为 False。
            - False (默认): 按频率【降序】排列 (返回【最常见】的 N 个值)。
            - True: 按频率【升序】排列 (返回【最不常见】的 N 个值)。
        input_handle (Optional[str], optional): 可选的数据句柄。

    【示例 1 (全局, 默认降序)】: 查找整个数据库中【最常见】的 3 个生产年份:
    `{"target_column": "生产年份", "top_n": 3}` (ascending 默认为 False)

    【示例 2 (流式, 升序)】: 查找句柄 "handle_B" 中【最不常见】的 5 个固件版本:
    `{"target_column": "固件版本", "top_n": 5, "ascending": True, "input_handle": "handle_B"}`

    Returns:
        一个包含最终计数字典的字典: {'status': 'success', 'data': {'项目1': 数量1, ...}}
    """
    try:
        base_data = _get_base_data(input_handle)
        if not isinstance(base_data, pd.DataFrame):
            return {"status": "error", "message": "此操作必须在DataFrame上执行。"}
        if target_column not in base_data.columns:
            return {"status": "error", "message": f"列 '{target_column}' 不存在。"}

        counts = base_data[target_column].value_counts(ascending=ascending)
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


# 【代码修复】：请将这个 *新* 函数添加到 server.py

@mcp.tool()
def compare_attributes_of_related_ids(
        id_column_name: str,
        attribute_to_compare: str,
        input_handle: str
) -> Dict[str, Any]:
    """
    【流工具-转换器】【比较工具】: 这是一个高级工具。它获取一个数据流，该流中必须包含一个含有【其他实体ID】的列。
    然后，它会为这些ID查找指定的属性，将该属性作为新列添加，并添加一个布尔列来比较这两个属性是否相同。

    Args:
        id_column_name (str): 【必需】当前数据流中，包含需要进行二次查找的“实体ID”的列名 (例如 '替代型号')。
        attribute_to_compare (str): 【必需】需要为源实体和关联实体进行比较的属性名称 (例如 '固件版本')。
        input_handle (str): 【必需】必须提供一个指向源数据流的句柄。

    【工作流示例】:
    问题: "比较具有替代型号的设备与其替代型号的固件版本是否相同。"
    计划:
    1. `find_products_by_relation(relation_type='替代型号')` -> handle_A (获取源设备)
    2. `enrich_stream_with_relation(relation_type='替代型号', relation_target_col_name='替代型号', columns_to_keep=['固件版本'], input_handle='handle_A')` -> handle_B (获得包含[源固件版本, 替代型号ID]的表)
    3. `compare_attributes_of_related_ids(id_column_name='替代型号', attribute_to_compare='固件版本', input_handle='handle_B')` -> handle_C (获得最终的、包含所有比较列的表)
    4. `consume_data_to_text(input_handle='handle_C')` -> 显示结果

    Returns:
        一个包含新句柄和元信息的标准字典。新的数据框将包含原始列、新查找的属性列，以及一个比较结果列。
    """
    try:
        if not input_handle or input_handle not in DATA_CACHE:
            return {"status": "error", "message": "错误：此工具必须提供一个有效的 input_handle。"}

        base_data = _get_base_data(input_handle).copy()
        if not isinstance(base_data, pd.DataFrame):
            return {"status": "error", "message": "输入句柄必须指向一个DataFrame。"}
        if id_column_name not in base_data.columns or attribute_to_compare not in base_data.columns:
            return {"status": "error",
                    "message": f"指定的列 '{id_column_name}' 或 '{attribute_to_compare}' 不在输入数据流中。"}

        # 1. 从数据流的指定ID列中，获取所有需要二次查找的ID
        ids_to_lookup = base_data[id_column_name].dropna().unique().tolist()

        # 2. 从全局 entity_df 中，为这些ID查找它们对应的目标属性，创建一个映射字典
        #    注意：我们只查找那些真实存在的ID
        valid_ids_to_lookup = [id_val for id_val in ids_to_lookup if id_val in entity_df.index]
        attribute_map = entity_df.loc[valid_ids_to_lookup, attribute_to_compare].to_dict()

        # 3. 将查找到的属性值映射回原始数据框，创建一个新列
        new_col_name = f"{id_column_name}_{attribute_to_compare}"
        base_data[new_col_name] = base_data[id_column_name].map(attribute_map)

        # 4. 创建最终的比较列
        comparison_col_name = f"is_{attribute_to_compare}_same"
        base_data[comparison_col_name] = (base_data[attribute_to_compare] == base_data[new_col_name])

        # 5. 缓存并返回
        new_handle_id = f"df_compared_{uuid.uuid4()}"
        DATA_CACHE[new_handle_id] = base_data
        metadata = _create_metadata(base_data)
        return {"status": "success", "data_handle": new_handle_id, "metadata": metadata}

    except Exception as e:
        logger.error(f"比较关联ID属性时失败: {e}", exc_info=True)
        return {"status": "error", "message": f"比较关联ID属性时失败: {e}"}


@mcp.tool()
def unroll_list_column(
        column_to_unroll: str,
        input_handle: Optional[str] = None
) -> Dict[str, Any]:
    """
    【流工具-转换器】: 将包含【列表】的列“展开”(Unroll/Explode)为多行，以便对其内容进行筛选或聚合。

    这是处理 '流程' 字段 (该字段是一个列表) 的【关键前置步骤】。
    例如：一行 流程=["A", "B"] 的数据，将被转换为两行数据（一行 流程="A"，另一行 流程="B"）。
    此操作会返回一个指向【新数据框】的新句柄。

    Args:
        column_to_unroll (str): 需要展开的、包含列表的列名。必须是 "流程"。
        input_handle (Optional[str]): 【可选】流式输入句柄。如果为None，则对整个数据库执行操作。

    【工作流示例】:
    问题: "流程为'包装'的设备有多少?"
    计划:
    1. (思考): '流程' 是一个列表，我必须先展开它。
    2. (行动): 调用 `unroll_list_column(column_to_unroll='流程')`
    3. (观察): (收到新句柄 'df_exploded_handle')
    4. (思考): 现在 '流程' 是一个标量列了，我可以在这个新句柄上进行标准筛选。
    5. (行动): 调用 `execute_query(filters={'流程': {'==': '包装'}}, input_handle='df_exploded_handle')`
    6. (观察): (收到最终筛选的句柄 'df_final_handle')
    7. (思考): 计算总数。
    8. (行动): 调用 `get_scalar_aggregation(agg_col='产品ID', agg_func='count', input_handle='df_final_handle')`

    Returns:
        一个包含新句柄和元信息的标准字典: {'status': 'success', 'data_handle': 'df_exploded_123', 'metadata': {...}}
    """
    if column_to_unroll != "流程":
        return {"status": "error", "message": f"此工具目前仅配置为展开 '流程' 列。"}

    try:
        base_data = _get_base_data(input_handle)
        if not isinstance(base_data, pd.DataFrame):
            return {"status": "error", "message": "展开操作必须在DataFrame上执行。"}

        if column_to_unroll not in base_data.columns:
            return {"status": "error", "message": f"列 '{column_to_unroll}' 不存在。"}

        # 【核心逻辑】: 执行 Pandas Explode 操作
        # .dropna() 确保我们只保留那些爆炸后确实有流程的行
        exploded_df = base_data.explode(column_to_unroll)

        new_handle_id = f"df_exploded_{uuid.uuid4()}"
        DATA_CACHE[new_handle_id] = exploded_df
        metadata = _create_metadata(exploded_df)
        return {"status": "success", "data_handle": new_handle_id, "metadata": metadata}
    except Exception as e:
        return {"status": "error", "message": f"展开列表列时发生错误: {e}"}


if __name__ == "__main__":
    logger.info("正在启动MCP服务器 (统一句柄架构 V2)...")
    filter ={'流程': {'==': '包装'}}
    result = get_entity_grouped_aggregation(group_by_col='生产流水线编号',agg_col= '产品ID',agg_func='nunique',sort_direction='descending',top_n = 1)
    print(result)
    mcp.run(transport='stdio')