
from embedding_model_operator import find_sim_content
from read_data import get_dataset

import pandas as pd

if __name__ == '__main__':
    dataset_json ,label_set = get_dataset()
    label_list = list(label_set)
    print(f'dataset json:{dataset_json}')
    print(f'label_list:{label_list}')

    num_total_question = 0
    num_right_question = 0

    res_json = []
    for tmp_data in dataset_json:


        print(f'tmp data:{tmp_data}')
        tmp_question = tmp_data['question']
        tmp_label = tmp_data['scene']
        documents_list = {
            "source_sentence": [tmp_question],
            "sentence_to_compare": label_list,
        }
        res = find_sim_content(documents_list=documents_list)
        ans = res[tmp_question][0]['sentence_to_compare']
        print(f'res:{res}')
        print(f'ans:{ans}')
        tmp_res_json = {
            'question': tmp_question,
            'label': tmp_label,
            'pred': ans
        }

        if tmp_label == ans:
            num_right_question += 1
        num_total_question += 1

        res_json.append(tmp_res_json)

    res = pd.DataFrame(res_json)

    res.to_excel(f'embedding_{num_right_question}_{num_total_question}.xlsx')


        # if ans == tmp_data['scene']