from typing import List

# from langchain_openai import ChatOpenAI
from langchain import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

from llm_operator import LLMOperator


template = """从下面的几种渠道中获取和问题相关的数据，仅选择其中一个。

提供的渠道有:
{scenes}

问题:
{question}

输出格式:

```json
{{
    'scene': string, \\\\ 选择的渠道
}}
```
"""


def scene_selector(scene_list: List[str] = [], question: str='') -> str:
    """select scene by llm

    Args:
        scene_list (List[str]): scene list
        question (str): input content

    Returns:
        scene (str): scene choosen
    """
    if not scene_list:
        scene_list = ["报销单", "加工厂产值"]

    # question = '加工厂产值'
    # question = '奖励'
    # question = '刘一鸣的报销单'
    # question = '罗青逢的报销单'
    # question = '高温笔'
    # question = '2023年，不同客户的加工厂产值'
    if not question:
        question = "2023年，设计部的软件费用"


    llm = LLMOperator().get_llm_model()

    scenes_info_str = ""

    for scene in scene_list:
        desc = f"获取{scene}相关数据"
        scenes_info_str += f"> {scene} : {desc}\n"

    prompt_template = PromptTemplate(
        input_variables=["scenes", "question"], template=template
    )

    chain = prompt_template | llm | StrOutputParser()
    input_params = {"scenes": scenes_info_str, "question": question}
    print(f"input:{input_params}")
    ans = chain.invoke(input_params)
    print(f"ans:{ans}")

    scene_res = ""
    for tmp_scene in scene_list:
        if tmp_scene in ans:
            scene_res = tmp_scene
            break
    print(f"scene res:{scene_res}")

    return scene_res
