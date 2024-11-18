# from modelscope.models import Model
# from modelscope.pipelines import pipeline
# from modelscope.utils.constant import Tasks
import numpy as np

from config import embedding_model

def l2_distance_numpy(vector1, vector2):
    """cal L2 distance

    Args:
        vector1 (_type_): _description_
        vector2 (_type_): _description_

    Returns:
        _type_: _description_
    """
    return np.linalg.norm(np.array(vector1) - np.array(vector2))


from langchain_community.embeddings.sentence_transformer import (
    SentenceTransformerEmbeddings,
)

model_path = embedding_model
# 使用句子模型
embedding_function = SentenceTransformerEmbeddings(model_name=model_path)


def find_sim_content(documents_list: dict):

    source_embeddings_list = np.array(
        embedding_function.embed_documents(documents_list["source_sentence"])
    )
    # print(f"source_embeddings_list:{source_embeddings_list.shape}")

    embeddings_list = np.array(
        embedding_function.embed_documents(documents_list["sentence_to_compare"])
    )
    # print(f"embeddings_list:{embeddings_list.shape}")

    l2_dis_list = []
    for tmp_source_embeddings in source_embeddings_list:
        tmp_l2_dis_list = []
        for tmp_embeddings in embeddings_list:
            tmp_l2_distance = l2_distance_numpy(tmp_source_embeddings, tmp_embeddings)
            tmp_l2_dis_list.append(tmp_l2_distance)
        l2_dis_list.append(tmp_l2_dis_list)
    # print(f"l2_dis_list:{l2_dis_list}")
    # return l2_dis_list

    scores = l2_dis_list
    sentence_to_compare = documents_list["sentence_to_compare"]
    source_sentence = documents_list["source_sentence"]

    sorted_res = {}
    for tmp_source_sentence_id, tmp_source_sentence in enumerate(source_sentence):
        # 将分数和对应的句子对存储在列表中
        score_pairs = list(zip(scores[tmp_source_sentence_id], sentence_to_compare))
        # print(f"score_pairs:{score_pairs}")
        # 根据分数对列表进行排序
        sorted_score_pairs = sorted(score_pairs, key=lambda x: x[0])
        # print(f"sorted_score_pairs:{sorted_score_pairs}")

        # 创建一个结果列表来存储排序后的句子对和分数
        sorted_results = []

        # 遍历排序后的列表，提取信息并添加到结果列表中
        for score_pair, sentence in sorted_score_pairs:

            sorted_results.append(
                {
                    "sentence_to_compare": sentence,
                    "distance": score_pair,
                }
            )
        sorted_res[tmp_source_sentence] = sorted_results

    # print(f"sorted_res:{sorted_res}")
    return sorted_res


if __name__ == "__main__":
    documents_list = {
        # "source_sentence": ["2023年不同客户的产值"],
        "source_sentence": ["2023年，不同月份的客户产值"],
        "sentence_to_compare": ["报销单", "加工厂产值"],
    }

    sim_res = find_sim_content(documents_list=documents_list)
    print(f"sim_res:{sim_res}")
