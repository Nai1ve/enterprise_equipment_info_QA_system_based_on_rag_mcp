import ast

import pandas as pd
from pandas import DataFrame
from typing import Tuple
import logging
import json

def read_json_file(file_path):
    try:
        with open(file_path,'r') as file:
            data = json.load(file)
        return data
    except FileNotFoundError:
        print(f"错误：文件 '{file_path}' 未找到")
        return None
    except json.JSONDecodeError as e:
        print(f"错误：JSON解析失败 - {e}")
        return None
    except Exception as e:
        print(f"错误：读取文件时发生意外错误 - {e}")
        return None

def convert_to_list(value):
    """将字符串安全转换为列表"""
    try:
        # 处理空值
        if pd.isna(value) or value.strip() in ['', 'nan', 'NaN', 'null', 'None']:
            return []
        # 安全转换
        return ast.literal_eval(value)
    except (ValueError, SyntaxError, TypeError):
        # 转换失败时返回空列表
        print('转换错误')
        return []


def load_data(logger:logging) -> Tuple[DataFrame,DataFrame]:
    """加载项目文件夹下的entity和relation数据"""
    try:
        entity_df = pd.read_csv('data/entity.csv',
                                index_col='产品ID',
                                parse_dates=['首次销售日期'],
                                converters={'流程': convert_to_list})
        entity_df = csv_entity_data_type_convert(entity_df)
        relation_df = pd.read_csv('data/relation.csv')
        logger.info("MCP Server: 数据加载成功。")
    except FileNotFoundError:
        logger.error("error:MCP Server :服务器无法找到CSV文件。已创建空文件",FileNotFoundError)
        entity_df = pd.DataFrame()
        relation_df = pd.DataFrame()
    return entity_df,relation_df


def csv_entity_data_type_convert(entity_df :DataFrame) -> DataFrame:
    """
    entity的数据类型转换
    :param entity_df:
    :return:
    """
    # 转换 '首次销售日期' 为 datetime 类型
    entity_df['首次销售日期'] = pd.to_datetime(entity_df['首次销售日期'], errors='coerce')

    # 转换年份和保修期字段为可以处理缺失值的整数类型 (Int64)
    entity_df['首次销售年份'] = entity_df['首次销售年份'].astype('Int64')
    entity_df['生产年份'] = entity_df['生产年份'].astype('Int64')
    entity_df['保修期(年)'] = entity_df['保修期(年)'].astype('Int64')
    entity_df['累计销量'] = entity_df['累计销量'].astype('Int64')

    entity_df.info()
    return entity_df

