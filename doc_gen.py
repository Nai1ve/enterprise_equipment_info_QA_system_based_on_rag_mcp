import data



entity_df,relation_df = data.load_data()

print(relation_df.head(5))

answer_parts = []
#成本(RMB)最高的设备的成本是多少？
answer_parts.append(f"成本(RMB)最高的设备的成本是{entity_df['成本(RMB)'].max()}")
#哪一年是产品首次销售的最早年份？
answer_parts.append(f"{entity_df['首次销售年份'].min()}年是产品首次销售的最早年份")
#累计销量最高的产品销量是多少？
answer_parts.append(f"累计销量最高的产品销量是：{entity_df['累计销量'].max()}")
#故障率(%)最高的设备的故障率是多少？
answer_parts.append(f"故障率(%)最高的设备的故障率是{entity_df['故障率(%)'].max()}")
#哪一年是设备首次销售的最近年份？
answer_parts.append(f"{entity_df['首次销售年份'].max()}年是设备首次销售的最近年份")
#保修期为3年的设备共有多少种型号/台？
answer_parts.append(f"保修期为3年的设备共有{entity_df[entity_df['保修期(年)'] == 3].shape[0]}种型号/台")
#有多少种不同的兼容设备类型？
answer_parts.append(f"有{relation_df[relation_df['relation'] == '兼容设备']['target'].nunique()}种不同的兼容设备类型")
#净重(kg)大于50kg的设备有多少台？
answer_parts.append(f"净重(kg)大于50kg的设备有{entity_df[entity_df['净重(kg)'] > 50].shape[0]}台")
#能效等级为“一级”的设备数量是多少？
answer_parts.append(f"能效等级为“一级”的设备数量是{entity_df[entity_df['能效等级'] == '一级'].shape[0]}")
#建议零售价(RMB)在1000到5000之间的产品有多少个？
answer_parts.append(f"建议零售价(RMB)在1000到5000之间的产品有{entity_df[entity_df['建议零售价(RMB)'].between(1000, 5000)].shape[0]}个")
#"关联服务/APP ID为'IoTControl-120'的设备数量是多少？"
answer_parts.append(f"关联服务/APP ID为'IoTControl-120'的设备数量是{relation_df[(relation_df['relation'] == '关联服务/APP ID') & (relation_df['target'] == 'IoTControl-120')].shape[0]}")
#有多少种不同的替代型号？
answer_parts.append(f"有{relation_df[relation_df['relation'] == '替代型号']['target'].nunique()}种不同的替代型号")
#哪一年生产的设备数量最多？
answer_parts.append(f"{entity_df['生产年份'].value_counts().index[0]}年生产的设备数量最多")
#产品尺寸出现次数最多的前5种尺寸是哪些？
answer_parts.append(f"产品尺寸出现次数最多的前5种尺寸是:"+"\n".join([ f"产品尺寸为:{a}的产品共有:{b}个。"for a,b in entity_df['产品尺寸'].value_counts().head(5).items()]))
#额定功率(W)的平均值是多少？
answer_parts.append(f"额定功率(W)的平均值是{entity_df['额定功率(W)'].mean()}")
"核心组件编号为 'DISPLAY-ABC' 的设备累计销量的平均值/总和/最大值是多少？"
answer_parts.append(f"核心组件编号为 'DISPLAY-ABC' 的设备累计销量的平均值/总和/最大值分别是{entity_df[entity_df['核心组件编号'] == 'DISPLAY-ABC']['累计销量'].mean()}/"
                    f"{entity_df[entity_df['核心组件编号'] == 'DISPLAY-ABC']['累计销量'].sum()}/{entity_df[entity_df['核心组件编号'] == 'DISPLAY-ABC']['累计销量'].max()}")
#"生产流水线编号为'PL003'的产品累计销量是多少？"
answer_parts.append(f"生产流水线编号为'PL003'的产品累计销量是{entity_df[entity_df['生产流水线编号'] == 'PL003']['累计销量'].sum()}")
#建议零售价(RMB)的平均值是多少？
answer_parts.append(f"建议零售价(RMB)的平均值是{ entity_df['建议零售价(RMB)'].mean()}")
#哪个具体销售区域的设备累计销量最高？
answer_parts.append(f"{entity_df.groupby('具体销售区域')['累计销量'].sum().idxmax()}是所有销售区域中设备累计销量最高")
#哪个产品线的建议零售价中位数最高？
answer_parts.append(f"{entity_df.groupby('生产流水线编号')['建议零售价(RMB)'].median().idxmax()}产品线的建议零售价中位数最高")
#生产年份在2015年及之后的产品数量是多少？
answer_parts.append(f"生产年份在2015年及之后的产品数量是：{entity_df[entity_df['生产年份'] >= 2015].shape[0]}")
#各个能效等级的设备数量分别是多少？
answer_parts.append(f"各个能效等级的设备数量分别是\n"+"\n".join([f"能效等级为{level} 的设备共有{count}台。" for level,count in entity_df['能效等级'].value_counts().items()]))
print('-----------------生成文档内容-----------------------')
print(';\n'.join(answer_parts))


# 多问的将前置问题提取出来。的进行优化