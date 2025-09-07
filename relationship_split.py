import pandas as pd
from pandas import DataFrame


def split(data_frame:DataFrame)->set:
    data_set = set()

    for i, row in data_frame.iterrows():
        target_list = str(row['target']).split(',')
        for target in target_list:
            data_set.add((row['source'],target))

    return data_set

def to_disk(path:str,columns:list,data:list)->None:
    df = pd.DataFrame(data,columns=columns)
    df.to_csv(path,index=False)

relations_df = pd.read_csv('data/relation.csv')
# 将数据按照关系划分为主要供应商、关联服务/APP ID、兼容设备、所属产品线、替代型号、生产于
#procut_id,primary_supplier
print('--------------开始拆分主要供应商----------------------')
primary_supplier_df= relations_df[relations_df['relation'] == '主要供应商']
primary_supplier_data_set = split(primary_supplier_df)
to_disk('data/relation_primary_supplier.csv',['product_id','primary_supplier'],list(primary_supplier_data_set))
print('--------------开始拆分关联服务/APPID----------------------')
#product_id,associated_service_app_id
associated_service_APP_ID_df= relations_df[relations_df['relation'] == '关联服务/APP ID']
associated_service_set = split(associated_service_APP_ID_df)
to_disk('data/relation_associated_service.csv',['product_id','associated_service_app_id'],list(associated_service_set))
print('--------------开始拆分兼容设备----------------------')
#product_id,compatible_device
compatible_devices_df= relations_df[relations_df['relation'] == '兼容设备']
compatible_devices_set = split(compatible_devices_df)
to_disk('data/relation_compatible_devices.csv',['product_id','compatible_device'],list(compatible_devices_set))
print('--------------开始拆分所属产品线----------------------')
#product_id,product_line
product_line_df= relations_df[relations_df['relation'] == '所属产品线']
product_line_set = split(primary_supplier_df)
to_disk('data/relation_product_line.csv',['product_id','product_line'],list(product_line_set))
#product_id,alternative_model
print('--------------开始拆分替代型号----------------------')
alternative_model_df= relations_df[relations_df['relation'] == '替代型号']
alternative_model_set = split(alternative_model_df)
to_disk('data/relation_alternative_model.csv',['product_id','alternative_model'],list(alternative_model_set))
#product_id,production_id
print('--------------开始拆分生产于----------------------')
production_date_df= relations_df[relations_df['relation'] == '生产于']
production_date_set = split(production_date_df)
to_disk('data/relation_production_date.csv',['product_id','production_id'],list(production_date_set))
print('完成')


