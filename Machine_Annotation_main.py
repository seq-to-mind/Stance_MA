import json
import re
import time
from tqdm import tqdm
import random
import requests

global_num_to_label_mapping = {"0": "AGAINST", "1": "FAVOR", "2": "NONE"}
global_label_to_num_mapping = {"AGAINST": "0", "FAVOR": "1", "NONE": "2"}

config_multi_label_two_hop_inference = True
config_multi_target_adversarial_sampling = True
config_3_class_labeling = True

llm_predict_api = "http://127.0.0.1:2121/predict"
parse_model_api = "http://127.0.0.1:2121/parse"


def extract_stance_label(one_model_output):
    one_model_label = "NONE"
    for tmp_i in ["against", "disagree", "critic", "negative", "oppose"]:
        if tmp_i in one_model_output.lower():
            one_model_label = "AGAINST"
            break
    for tmp_i in ["favor", "favour", "agree", "support", "positive", "inclined", "\"for\"", "advot"]:
        if tmp_i in one_model_output.lower():
            one_model_label = "FAVOR"
            break
    return one_model_label


def stance_label_prediction(one_target, one_text, one_original_label, two_hop_inference):
    if one_target.lower() == "atheism":
        one_target = "No belief in god"

    """ adding the two-hop inference """
    if two_hop_inference is True:
        two_hop_step_1_input = 'USER: Describe the tweet\'s explicit or implicit relation to the target \"%s\".\nTweet: \"%s\"\nASSISTANT:' % (one_target, one_text)

        two_hop_step_1_response = requests.post(llm_predict_api, data={"llm_input": two_hop_step_1_input, "LLM_gen_len": 80})
        two_hop_step_1_response = json.loads(two_hop_step_1_response.text)["llm_output"].strip()
        one_mentioned = two_hop_step_1_response.replace("<Output>", "").replace("</s>", "").strip()
        if len(one_mentioned) < 3:
            one_mentioned = "No relation to the target."
        two_hop_step_2_input = 'USER: Classify the tweet\'s stance label on \"%s\" into "Favor", "Against", or "None".\nTweet: \"%s\" (%s)\nASSISTANT:' % (one_target, one_text, one_mentioned)
        one_llm_prediction_input = two_hop_step_2_input
    else:
        one_llm_prediction_input = 'USER: Classify the tweet\'s stance label on \"%s\" into "Favor", "Against", or "None".\nTweet: \"%s\"\nASSISTANT:' % (one_target, one_text)

    response = requests.post(llm_predict_api, data={"llm_input": one_llm_prediction_input, "LLM_gen_len": 50})
    one_llm_prediction_output = json.loads(response.text)["llm_output"].strip()
    one_model_label = extract_stance_label(one_llm_prediction_output)

    one_res_dict = {"Target": one_target, "Text": one_text, "Label": one_model_label,
                    "Prompt_Prediction": one_llm_prediction_input.strip(),
                    "Model_Output": one_llm_prediction_output, "Gold_Label": one_original_label}

    return one_res_dict


if __name__ == '__main__':

    one_corpus_file = "./Stance_Data_JSON/TweetTask_A_Train.json"
    original_sample_list = json.load(open(one_corpus_file, encoding="utf-8"))
    machine_annotated_sample_list = []
    for one_item in tqdm(original_sample_list[:]):
        stance_predict_A = stance_label_prediction(one_target=one_item["Target"], one_text=one_item["Text"], one_original_label=one_item["Label"],
                                                   two_hop_inference=config_multi_label_two_hop_inference)
        if stance_predict_A["Label"] != "NONE" or config_3_class_labeling:
            if config_multi_target_adversarial_sampling is True:
                one_phrase_list = requests.post(parse_model_api, data={"parse_input": one_item["Text"]})
                one_phrase_list = json.loads(one_phrase_list.text)["parse_output"]
                for tmp_target in one_phrase_list:
                    stance_predict_B = stance_label_prediction(one_target=tmp_target, one_text=one_item["Text"], one_original_label="N/A", two_hop_inference=False)
                    if stance_predict_B["Label"] != stance_predict_A["Label"] and stance_predict_B["Label"] != "NONE":
                        machine_annotated_sample_list.append(stance_predict_A)
                        machine_annotated_sample_list.append(stance_predict_B)
                        if len(machine_annotated_sample_list) % 50 == 0:
                            json.dump(machine_annotated_sample_list, open("tmp_res.json", "w", encoding="utf-8"), ensure_ascii=False, indent=2)
                        break
            else:
                machine_annotated_sample_list.append(stance_predict_A)
                if len(machine_annotated_sample_list) % 50 == 0:
                    json.dump(machine_annotated_sample_list, open("tmp_res.json", "w", encoding="utf-8"), ensure_ascii=False, indent=2)

    json.dump(machine_annotated_sample_list, open("tmp_res.json", "w", encoding="utf-8"), ensure_ascii=False, indent=2)
