import pandas as pd
import data



json_data = data.read_json_file('../rag_system/graph_data.json')
entities_dic = json_data['entities']
attributes_dic = json_data['attributes']
relations_list = json_data['relations']

# print(entities_dic['EFM8664'])


entity_list =[]

relation_set = set()

for key,value in entities_dic.items():
    att = attributes_dic.get(key,{})

    if not att:
        print(f'key:{key}所对应的属性为空')

    entity = {
        '产品ID': key,
        '产品尺寸':att.get('产品尺寸',None),
        '流程':att.get('流程',None),
        '净重(kg)':att.get('净重(kg)',None),
        '累计销量':att.get('累计销量',None),
        '保修期(年)':att.get('保修期(年)',None),
        '首次销售年份':att.get('首次销售年份',None),
        '生产批次号': att.get('生产批次号',None),
        '成本(RMB)': att.get('成本(RMB)',None),
        '首次销售日期': att.get('首次销售日期',None),
        '生产流水线编号': att.get('生产流水线编号',None),
        '固件版本': att.get('固件版本',None),
        '建议零售价(RMB)': att.get('建议零售价(RMB)',None),
        '能效等级': att.get('能效等级',None),
        '核心组件编号': att.get('核心组件编号',None),
        '具体销售区域': att.get('具体销售区域',None),
        '生产年份': att.get('生产年份',None),
        #'关联服务/APP ID': att.get('关联服务/APP ID',''),
        #'替代型号': att.get('替代型号',''),
        #'兼容设备': att.get('兼容设备',''),
        '配件编号': att.get('配件编号',None),
        #'所属产品线': att.get('所属产品线',''),
        '额定功率(W)': att.get('额定功率(W)',None),
        '故障率(%)': att.get('故障率(%)',None),
        #'关联产品': att.get('关联产品','')
    }

    # 主要供应商
    supply = att.get('主要供应商','')
    if supply:
        relation_set.add((key,'主要供应商',supply))

    #关联服务/APP
    relation_service = att.get('关联服务/APP ID','')
    if relation_service:
        relation_set.add((key,'关联服务/APP ID',relation_service))

    #替代型号
    substitute = att.get('替代型号','')
    if substitute:
        relation_set.add((key,'替代型号',substitute))


    #兼容设备
    compatible_devices = att.get('兼容设备','')
    if compatible_devices:
        relation_set.add((key,'兼容设备',compatible_devices))

    #所属产品线
    product_line = att.get('所属产品线','')
    if product_line:
        relation_set.add((key,'所属产品线',product_line))

    #关联产品
    relation_product = att.get('关联产品','')
    if relation_product:
        relation_set.add((key,'关联产品',relation_product))

    if value.get('关联服务/APP ID','') != att.get('关联服务/APP ID',''):
        print(f'key:{key} 的关联服务不相同')

    if value.get('所属产品线','') != att.get('所属产品线',''):
        print(f'key:{key}的所属产品线不相同')

    entity_list.append(entity)

for relation in relations_list:
    relation_set.add((relation['source'],relation['type'],relation['target']))





print("--- 准备创建DataFrame ---")
entity_df = pd.DataFrame(entity_list)

entity_df = data.csv_entity_data_type_convert(entity_df)
entity_df.to_csv('entity.csv',index=False)


relation_df = pd.DataFrame(list(relation_set),columns=['source','relation','target'])
relation_df.columns = ['source','relation','target']
relation_df.to_csv('relation.csv',index=False)