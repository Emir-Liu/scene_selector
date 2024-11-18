# read scene selector data from xlsx

import json

import pandas as pd

from config import data_path



def get_dataset():
    # 读取Excel文件
    df = pd.read_excel(data_path)

    # 将数据框转换为JSON格式
    json_data = json.loads(df.to_json(orient='records', force_ascii=False))

    # 获取标签列表
    label_set = set()

    for tmp_json_data in json_data:
        print(f'tmp_json_data:{tmp_json_data}')
        label_set.add(tmp_json_data['scene'])

    return json_data, label_set
    print(f'label set:{label_set}')

    # 打印JSON数据
    print(f'data:{json_data}')