# read scene selector data from xlsx

import pandas as pd

from config import data_path

# 读取Excel文件
df = pd.read_excel(data_path)

# 将数据框转换为JSON格式
json_data = df.to_json(orient='records', force_ascii=False)

# 打印JSON数据
print(json_data)