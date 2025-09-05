import pprint
from typing import Dict,Any,List,Optional
from mcp.server.fastmcp import FastMCP
import data
from utils import setup_logger,sanitize_for_json
import pandas as pd
import numpy as np

logger = setup_logger('MCP_Server',log_file='server.log')



# 1.初始化FastMcp
mcp = FastMCP(
    name = 'device_database_server_base_pandas'
)

# 2.加载数据
entity_df,relation_df = data.load_data(logger)

# --- 3. 定义工具函数 (MCP接口) ---

@mcp.tool()
def get_full_entity_details(entity_id: str) -> Dict[str, Any]:
    """根据产品ID查询其全部详细信息，包括所有属性和关联关系。

    这是工具箱中的核心查询工具，用于获取一个实体的完整画像。

    Args:
        entity_id (str): 需要查询的产品的唯一ID, 例如 'EFM8664'。

    Returns:
        Dict[str, Any]: 包含查询状态和结果的字典。
            成功时: {'entity_id': 'EFM8664', 'attributes': {'产品尺寸': '650x600x1850mm', '流程': "['原材料采购']", '累计销量': 325394, '首次销售年份': 2021, '生产批次号': 'BATCH20120913-454', '成本(RMB)': 756.71, '首次销售日期': Timestamp('2021-02-01 00:00:00'), '生产流水线编号': 'PL001', '固件版本': 'v1.8.3', '建议零售价(RMB)': 1329.0, '能效等级': '一级', '核心组件编号': 'COMPRESSOR-XYZ', '具体销售区域': '安徽省-合肥市', '生产年份': 2012, '配件编号': 'MOD36336', '额定功率(W)': 150.0, '故障率(%)': 4.67}, 'relations': {'主要供应商': ['A电子元件厂'], '关联服务/APP ID': ['EcoLifeApp-100'], '兼容设备': ['KLT6358-Plus, OPT6875-Pro', 'KLT6358-Plus', 'OPT6875-Pro'], '所属产品线': ['智能家居系列'], '生产于': ['PL001']}}}
            失败时: {'status': 'error', 'message': '错误信息'}
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
def find_products_by_relation(relation_type: str, target_value: str) -> Dict[str, Any]:
    """反向查询：根据一个关系的目标值，查找所有相关的源产品ID列表。

    例如，可以根据供应商名称查找其供应的所有产品，或根据产品线名称查找该产品线下的所有产品。

    Args:
        relation_type (str): 要查询的关系名称, 例如 '主要供应商', '所属产品线'。
        target_value (str): 关系的目标值, 例如 'A电子元件厂' 或 '节能系列'。

    Returns:
        Dict[str, Any]: 包含状态和产品ID列表的字典。
            成功时: {'status': 'success', 'data': ['产品ID1', '产品ID2', ...]}
            失败时: {'status': 'error', 'message': '错误信息'}
    """
    try:
        matching_relations = relation_df[(relation_df['relation'] == relation_type) & (relation_df['target'] == target_value)]
        if matching_relations.empty:
            return {"status": "success", "data": [], "message": f"没有找到 '{relation_type}' 为 '{target_value}' 的任何产品。"}
        product_ids = matching_relations['source'].unique().tolist()
        return {"status": "success", "data": product_ids}
    except Exception as e:
        return {"status": "error", "message": f"反向关联查询时发生错误: {e}"}

@mcp.tool()
def find_most_common(target_column: str, is_relation: bool, top_n: int = 1) -> Dict[str, Any]:
    """查找某个属性或关系中出现次数最多的前N个项。

    用于回答“哪个...最多/最常见？”或“排名前N的...”这类频率统计问题。

    Args:
        target_column (str): 要统计的属性名或关系名, 例如 '生产年份' 或 '主要供应商'。
        is_relation (bool): 标记target_column是否是一个关系类型 (True代表关系, False代表属性)。
        top_n (int, optional): 返回排名前几的结果。默认为 1。

    Returns:
        Dict[str, Any]: 包含状态和结果字典的字典。
            成功时: {'status': 'success', 'data': {'项目1': 数量1, '项目2': 数量2, ...}}
            失败时: {'status': 'error', 'message': '错误信息'}
    """
    try:
        if is_relation:
            counts = relation_df[relation_df['relation'] == target_column]['target'].value_counts()
        else:
            counts = entity_df[target_column].value_counts()
        if counts.empty:
            return {"status": "error", "message": f"找不到关于 '{target_column}' 的记录。"}
        return {"status": "success", "data": sanitize_for_json(counts.head(top_n).to_dict())}
    except Exception as e:
        return {"status": "error", "message": f"计算时发生错误: {e}"}

@mcp.tool()
def get_filtered_count(filters: Dict[str, Any]) -> Dict[str, Any]:
    """根据一组筛选条件，计算符合条件的设备总数。

    支持大于(>), 小于(<), 等于(==), 介于(between)操作。
    `filters`参数是一个字典，键为列名，值为另一个字典，其中包含操作符和操作数值。
    例如: `{"保修期(年)": {"==": 3}, "成本(RMB)": {">": 10000}}`

    Args:
        filters (Dict[str, Any]): 筛选条件的字典。

    Returns:
        Dict[str, Any]: 包含状态和计数值的字典。
            成功时: {'status': 'success', 'count': 123}
            失败时: {'status': 'error', 'message': '错误信息'}
    """
    try:
        query_df = entity_df
        for column, condition in filters.items():
            op, val = list(condition.items())[0]
            if op == '==': query_df = query_df[query_df[column] == val]
            elif op == '>': query_df = query_df[query_df[column] > val]
            elif op == '<': query_df = query_df[query_df[column] < val]
            elif op == 'between': query_df = query_df[query_df[column].between(val[0], val[1])]
        return {"status": "success", "count": sanitize_for_json(len(query_df))}
    except Exception as e:
        return {"status": "error", "message": f"筛选计数时发生错误: {e}"}

@mcp.tool()
def get_grouped_aggregation(
    group_by_col: str,
    agg_col: str,
    agg_func: str,
    top_n: Optional[int] = None
) -> Dict[str, Any]:
    """对数据按指定列分组后，对另一列进行聚合计算。可以选择只返回排名最高的N个结果。

    用于回答“每个...的平均...是多少？”或“列出所有...的总和...”这类问题。
    如果提供了 top_n 参数，则可用于回答“哪个...的...最高/最多？”的问题。

    Args:
        group_by_col (str): 用于分组的列名, 例如 '生产流水线编号'。
        agg_col (str): 需要进行聚合计算的数值列名, 例如 '成本(RMB)'。
        agg_func (str): 聚合函数名称, 支持 'sum', 'mean', 'median', 'count' 等Pandas原生聚合函数。
        top_n (Optional[int], optional): 如果提供此参数，则只返回聚合结果中数值最高的N个分组。默认为 None (返回所有结果)。

    Returns:
        Dict[str, Any]: 包含状态和结果字典的字典。
            成功时: {'status': 'success', 'data': {'分组1': 结果1, '分组2': 结果2, ...}}
            失败时: {'status': 'error', 'message': '错误信息'}
    """
    try:
        # 1. 执行分组聚合
        grouped_result = entity_df.groupby(group_by_col)[agg_col].agg(agg_func)

        # 2. 【新增】如果提供了 top_n，则筛选出最大的 N 个结果
        if top_n is not None and top_n > 0:
            if grouped_result.empty:
                result_dict = {} # 如果聚合结果为空，则直接返回空字典
            else:
                result_dict = grouped_result.nlargest(top_n).to_dict()
        else:
            result_dict = grouped_result.to_dict()

        return {"status": "success", "data": sanitize_for_json(result_dict)}
    except Exception as e:
        return {"status": "error", "message": f"分组聚合时发生错误: {e}"}

@mcp.tool()
def get_top_item_by_grouped_agg(group_by_col: str, agg_col: str, agg_func: str = 'sum', top_n: int = 1) -> Dict[str, Any]:
    """分组聚合后，返回排名最高的N个项。

    用于回答“哪个...的...最高/最多？”这类需要先计算再排序的问题。

    Args:
        group_by_col (str): 用于分组的列名, 例如 '具体销售区域'。
        agg_col (str): 用于计算的列名, 例如 '累计销量'。
        agg_func (str, optional): 聚合函数名称, 如 'sum', 'mean'等Pandas原生聚合函数。默认为 'sum'。
        top_n (int, optional): 返回排名前几的结果。默认为 1。

    Returns:
        Dict[str, Any]: 包含状态和结果字典的字典。
            成功时: {'status': 'success', 'data': {'排名第一的项': 数值1, ...}}
            失败时: {'status': 'error', 'message': '错误信息'}
    """
    try:
        grouped = entity_df.groupby(group_by_col)[agg_col].agg(agg_func)
        top_items = grouped.nlargest(top_n).to_dict()
        return {"status": "success", "data": sanitize_for_json(top_items)}
    except Exception as e:
        return {"status": "error", "message": f"分组聚合排名时发生错误: {e}"}

@mcp.tool()
def get_aggregation_by_relation(relation_type: str, target_value: str, agg_col: str, agg_func: str = 'sum') -> Dict[str, Any]:
    """反向聚合：先根据关系找到一组产品，再对这组产品的属性进行聚合。

    例如，计算某个供应商供应的所有产品的累计销量总和。

    Args:
        relation_type (str): 用于筛选的关系名称, 例如 '所属产品线'。
        target_value (str): 关系的目标值, 例如 '节能系列'。
        agg_col (str): 需要进行聚合计算的属性列名, 例如 '累计销量'。
        agg_func (str, optional): 聚合函数名称, 如 'sum', 'mean'等Pandas原生聚合函数。默认为 'sum'。

    Returns:
        Dict[str, Any]: 包含状态和单一聚合结果的字典。
            成功时: {'status': 'success', 'data': {'aggregation_result': 12345.6}}
            失败时: {'status': 'error', 'message': '错误信息'}
    """
    try:
        product_ids_result = find_products_by_relation(relation_type, target_value)
        if product_ids_result['status'] == 'error' or not product_ids_result['data']:
            return {"status": "success", "data": None, "message": "未找到相关产品。"}
        product_ids = product_ids_result['data']
        related_products_df = entity_df.loc[product_ids]
        result = related_products_df[agg_col].agg(agg_func)
        result = float(result) if pd.notna(result) else None
        return {"status": "success", "data": {"aggregation_result": sanitize_for_json(result)}}
    except Exception as e:
        return {"status": "error", "message": f"反向聚合查询时发生错误: {e}"}

@mcp.tool()
def compare_entity_attributes(entity_id_1: str, entity_id_2: str, attribute_name: str) -> Dict[str, Any]:
    """比较两个指定实体的某个特定属性的值，并计算数值差异（如果可计算）。

    Args:
        entity_id_1 (str): 第一个需要比较的产品ID。
        entity_id_2 (str): 第二个需要比较的产品ID。
        attribute_name (str): 需要比较的属性名称, 例如 '成本(RMB)'。

    Returns:
        Dict[str, Any]: 包含状态和比较结果的字典。
            成功时: {'status': 'success', 'data': {'ID1': val1, 'ID2': val2, 'difference': diff}}
            失败时: {'status': 'error', 'message': '错误信息'}
    """
    try:
        val1 = entity_df.loc[entity_id_1, attribute_name]
        val2 = entity_df.loc[entity_id_2, attribute_name]
        diff = None
        if pd.notna(val1) and pd.notna(val2) and isinstance(val1, (int, float, np.number)) and isinstance(val2, (int, float, np.number)):
            diff = float(val1 - val2)
        return {"status": "success", "data": {entity_id_1: sanitize_for_json(val1), entity_id_2: sanitize_for_json(val2), "difference": sanitize_for_json(diff)}}
    except Exception as e:
         return {"status": "error", "message": f"比较属性时发生错误: {e}"}

@mcp.tool()
def compare_all_substitutes_attribute(attribute_name: str) -> Dict[str, Any]:
    """找出所有存在替代型号的设备对，并比较它们的某个指定属性。

    Args:
        attribute_name (str): 需要进行比较的属性名称, 例如 '固件版本'。

    Returns:
        Dict[str, Any]: 包含状态和比较结果列表的字典。
            成功时: {'status': 'success', 'data': [{'source': ..., 'target': ..., 'is_same': ...}, ...]}
            失败时: {'status': 'error', 'message': '错误信息'}
    """
    try:
        substitutes_rel = relation_df[relation_df['relation'] == '替代型号']
        merged_df = pd.merge(substitutes_rel, entity_df[[attribute_name]], left_on='source', right_index=True, how='left')
        merged_df = merged_df.rename(columns={attribute_name: f'source_{attribute_name}'})
        merged_df = pd.merge(merged_df, entity_df[[attribute_name]], left_on='target', right_index=True, how='left')
        merged_df = merged_df.rename(columns={attribute_name: f'target_{attribute_name}'})
        merged_df['is_same'] = (merged_df[f'source_{attribute_name}'] == merged_df[f'target_{attribute_name}'])
        result_dict = merged_df[['source', 'target', f'source_{attribute_name}', f'target_{attribute_name}', 'is_same']].to_dict('records')
        return {"status": "success", "data":  sanitize_for_json(result_dict)}
    except Exception as e:
        return {"status": "error", "message": f"比较替代型号属性时发生错误: {e}"}

@mcp.tool()
@mcp.tool()
def execute_query(
    filters: Dict[str, Any],
    columns: List[str] = None,
    sort_by: Optional[str] = None,
    ascending: bool = True,
    top_n: Optional[int] = None
) -> Dict[str, Any]:
    """通用数据查询接口，可根据条件筛选、排序并限制返回数量。返回结果为CSV格式字符串。

    这是最灵活的工具，用于处理其他专用工具无法覆盖的复杂列表查询。
    例如，可以查询“成本高于5000的所有设备，并按‘累计销量’降序排列，只返回前5条记录”。

    Args:
        filters (Dict[str, Any]): 筛选条件的字典。例如: `{"保修期(年)": {"==": 3}}`
        columns (List[str], optional): 需要返回的列名列表。如果为None，则返回所有列。
        sort_by (Optional[str], optional): 用于排序的列名。如果为None则不排序。默认为 None。
        ascending (bool, optional): 是否按升序排序。默认为 True (升序)。
        top_n (Optional[int], optional): 返回结果的最大行数。如果为None则返回所有结果。默认为 None。

    Returns:
        Dict[str, Any]: 包含状态和CSV格式数据字符串的字典。
            成功时: {'status': 'success', 'data': 'CSV string...'}
            失败时: {'status': 'error', 'message': '错误信息'}
    """
    try:
        query_df = entity_df

        # 1. 应用筛选条件
        for column, condition in filters.items():
            if column not in query_df.columns:
                continue
            op, val = list(condition.items())[0]
            if op == '==': query_df = query_df[query_df[column] == val]
            elif op == '>': query_df = query_df[query_df[column] > val]
            elif op == '<': query_df = query_df[query_df[column] < val]
            elif op == 'between': query_df = query_df[query_df[column].between(val[0], val[1])]

        # 2. 应用排序
        if sort_by and sort_by in query_df.columns:
            query_df = query_df.sort_values(by=sort_by, ascending=ascending)

        # 3. 应用Top-N限制
        if top_n is not None and top_n > 0:
            query_df = query_df.head(top_n)

        # 4. 选择列
        if columns:
            query_df = query_df[[col for col in columns if col in query_df.columns]]

        # 5. 【已修改】序列化为CSV格式
        serialized_data = query_df.to_csv(index=True)

        return {"status": "success", "data": serialized_data}
    except Exception as e:
        return {"status": "error", "message": f"执行查询时发生错误: {e}"}

# --- 4. 启动服务器的入口 ---
if __name__ == "__main__":
    logger.info("正在启动MCP服务器...")
    # 测试
    result = find_most_common('生产年份',False,1)
    print(result)
    print(get_full_entity_details('EFM8664'))
    logger.info("已注册的“专家”工具:")
    mcp.run(transport='stdio')