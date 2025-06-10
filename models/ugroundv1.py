import os
import sys

# 获取当前目录
current_dir = os.path.dirname(os.path.abspath(__file__))

# 获取父目录
parent_dir = os.path.dirname(current_dir)

# 将当前目录和父目录加入到搜索路径
sys.path.append(current_dir)
sys.path.append(parent_dir)

import torch
from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from transformers.generation import GenerationConfig
import json
import re
from PIL import Image
from models.utils.region_traverser import RegionTraverser
from utils.utils import *
from typing import Tuple

from qwen_vl_utils import process_vision_info

def parse_bbox(bbox_str: str) -> Tuple[int, ...]:
    """Parse a bounding box string like '(x1, y1, x2, y2)' or '(x, y)' into a tuple of ints."""
    if not isinstance(bbox_str, str):
        bbox_str = str(bbox_str)
    return tuple(map(int, re.findall(r'\d+', bbox_str)))

def match_answer_to_candidate(candidate1: str, candidate2: str, answer: str) -> int:
    """
    Given Candidate 1, Candidate 2, and answer coordinate strings,
    return 1 if Candidate 1 is a better match, 2 if Candidate 2 is.
    """
    if answer.strip() in {'1', '2'}:
        return answer

    bbox1 = parse_bbox(candidate1)
    bbox2 = parse_bbox(candidate2)
    answer_bbox = parse_bbox(answer)

    # Compare only up to the number of values in answer_bbox
    match1 = sum(a == b for a, b in zip(answer_bbox, bbox1))
    match2 = sum(a == b for a, b in zip(answer_bbox, bbox2))

    return '1' if match1 > match2 else '2'


# bbox -> point (str)
def bbox_2_point(bbox, dig=2):
    # bbox [left, top, right, bottom]
    point = [(bbox[0]+bbox[2])/2, (bbox[1]+bbox[3])/2]
    point = [f"{item:.2f}" for item in point]
    point_str = "({},{})".format(point[0], point[1])
    return point_str

# bbox -> bbox (str)
def bbox_2_bbox(bbox, dig=2):
    bbox = [f"{item:.2f}" for item in bbox]
    bbox_str = "({},{},{},{})".format(bbox[0], bbox[1], bbox[2], bbox[3])
    return bbox_str

# point (str) -> point
def pred_2_point(s):
    floats = re.findall(r'-?\d+\.?\d*', s)
    floats = [float(num) for num in floats]
    if len(floats) == 2:
        return floats
    elif len(floats) == 4:
        return [(floats[0]+floats[2])/2, (floats[1]+floats[3])/2]
    else:
        return None

# bbox (qwen str) -> bbox
def extract_bbox(s):
    pattern = r"<\|box_start\|\>\((\d+),(\d+)\),\((\d+),(\d+)\)<\|box_end\|\>"
    matches = re.findall(pattern, s)
    if matches:
        # Get the last match and return as tuple of integers
        last_match = matches[-1]
        return (int(last_match[0]), int(last_match[1])), (int(last_match[2]), int(last_match[3]))
    return None


def image_to_temp_filename(image):
    temp_file = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
    image.save(temp_file.name)
    print(f"Image saved to temporary file: {temp_file.name}")
    return temp_file.name


class UGroundV1Model():
    def load_model(self, model_name_or_path="osunlp/UGround-V1-7B"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_name_or_path, 
            device_map="cuda", 
            trust_remote_code=True, 
            torch_dtype=torch.bfloat16,
            # attn_implementation="flash_attention_2"
        ).eval()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)
        self.processor = AutoProcessor.from_pretrained(model_name_or_path, trust_remote_code=True)

        # Setting default generation config
        # self.generation_config = GenerationConfig.from_pretrained("osunlp/UGround-V1-7B", trust_remote_code=True).to_dict()
        self.generation_config = dict()

        # self.set_generation_config(
        #     max_length=2048,
        #     do_sample=False,
        #     temperature=0.0
        # )

        # generation_config.json
        self.set_generation_config(
            max_length=2048,
            do_sample=False,
            temperature=0.0,
            bos_token_id=151643,
            eos_token_id=151645,
            use_cache=False
        )

    def set_generation_config(self, **kwargs):
        self.generation_config.update(**kwargs)
        self.model.generation_config = GenerationConfig(**self.generation_config)


    def get_model_response(self, prompt, instruction, image_path):
        # full_prompt = prompt_origin.format(instruction)
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": image_path,
                    },
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        # Preparation for inference
        text_input = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=[text_input],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to("cuda")
        generated_ids = self.model.generate(**inputs)

        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        response = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=False, clean_up_tokenization_spaces=False
        )[0]

        pred_bbox = None
        click_point = None
        
        # 提取bbox和point的坐标， 不做归一化
        if '<|box_start|>' in response and '<|box_end|>' in response:
            pred_bbox = extract_bbox(response)
            if pred_bbox is not None:
                (x1, y1), (x2, y2) = pred_bbox
                # pred_bbox = [pos / 1000 for pos in [x1, y1, x2, y2]]
                click_point = [(pred_bbox[0] + pred_bbox[2]) / 2, (pred_bbox[1] + pred_bbox[3]) / 2]
                
                # result_dict["bbox"] = pred_bbox
                # result_dict["point"] = click_point
        else:
            print('---------------')
            print(response)
            click_point = pred_2_point(response)
            # click_point = [x / 1000 for x in click_point] if click_point else None
            # result_dict["point"] = click_point  # can be none

        return response, pred_bbox, click_point

    def ground_only_positive(self, instruction, image):
        if not isinstance(image, str):
            assert isinstance(image, Image.Image)
            image_path = image_to_temp_filename(image)
        else:
            image_path = image
        assert os.path.exists(image_path) and os.path.isfile(image_path), "Invalid input image path."

        prompt_origin = f"""
        Your task is to help the user identify the precise coordinates (x, y) of a specific area/element/object on the screen based on a description.

        - Your response should aim to point to the center or a representative point within the described area/element/object as accurately as possible.
        - If the description is unclear or ambiguous, infer the most relevant area or element based on its likely context or purpose.
        - Your answer should be a single string (x, y) corresponding to the point of the interest.

        Description: {instruction}

        Answer:"""

        response, pred_bbox, pred_point = self.get_model_response(prompt_origin, instruction, image_path)

        pred_point = [x / 1000 for x in pred_point] if pred_point else None

        result_dict = {
            "result": "positive",
            "format": "x1y1x2y2",
            "raw_response": response,
            "bbox": pred_bbox,
            "point": pred_point,
            "iterations": 1  # 这里假设只有一次迭代
        }

        # # 提取并归一化bbox和point的坐标
        # if '<|box_start|>' in response and '<|box_end|>' in response:
        #     pred_bbox = extract_bbox(response)
        #     if pred_bbox is not None:
        #         (x1, y1), (x2, y2) = pred_bbox
        #         pred_bbox = [pos / 1000 for pos in [x1, y1, x2, y2]]
        #         click_point = [(pred_bbox[0] + pred_bbox[2]) / 2, (pred_bbox[1] + pred_bbox[3]) / 2]
                
        #         result_dict["bbox"] = pred_bbox
        #         result_dict["point"] = click_point
        # else:
        #     print('---------------')
        #     print(response)
        #     click_point = pred_2_point(response)
        #     click_point = [x / 1000 for x in click_point] if click_point else None
        #     result_dict["point"] = click_point  # can be none

        print("result_dict:", result_dict)
        
        return result_dict

    def calculate_relative_distance(self, normalized_point, image_width, image_height):
        import math
        """
        计算归一化坐标与图像中心的相对距离。

        参数:
        - normalized_point: tuple，归一化坐标 (x, y)，值范围在 [0, 1]。
        - image_width: int，图像的宽度。
        - image_height: int，图像的高度。

        返回:
        - relative_distance: float，相对距离，归一化到 [0, 1]。
        """
        # 图像中心的归一化坐标
        center_x = 0.5
        center_y = 0.5

        # 归一化坐标
        normalized_x, normalized_y = normalized_point

        # 计算欧几里得距离
        distance = math.sqrt((normalized_x - center_x) ** 2 + (normalized_y - center_y) ** 2)

        # 最大可能距离（图像对角线的一半，归一化后为 sqrt(0.5^2 + 0.5^2)）
        max_distance = math.sqrt(0.5 ** 2 + 0.5 ** 2)

        # 计算相对距离
        relative_distance = distance / max_distance

        return relative_distance

    def iterative_narrowing(self, prompt_origin, instruction, image_path, subset, verbose=True, max_iter=5, threshold=0.05, dim = 999):
        target = instruction
        image = Image.open(image_path)
        width, height = image.size
        current_region = image.copy()
        cropped_image = image.copy()

        # 创建输出目录
        output_dir = os.path.join('./output/ugroundv1', subset, target)
        os.makedirs(output_dir, exist_ok=True)

        traverser = RegionTraverser(current_region)
        if verbose: 
            print(target)
        k = max_iter
        threshold = threshold
        # render_crosshair_center(current_region).show()

        response = None
        bbox = None
        point = None
        prev_dis = 1.0  # 初始化前一个距离为1.0，表示初始状态
        iteration = 0

        for i in range(k):
            image_prompt = current_region.resize((dim, dim))
            # image_prompt = current_region
            response, bbox, prediction_coord = self.get_model_response(prompt_origin, target, image_prompt)
            pred_x, pred_y = prediction_coord

            print("iteration: ", i)
            
            if verbose: 
                print("middle coord:", prediction_coord)
            if i != k-1:
                traverser.consume_coordinate(pred_x, pred_y)
                result_image = traverser.get_highlighted_image()
                cropped_image = traverser.get_cropped_image()
                current_region = traverser.get_cropped_image().resize((dim, dim))
                # print("current_region:", current_region.size)
                # print("current_region:", current_region)

                # 保存中间结果图像
                result_image_path = os.path.join(output_dir, f"step_{i}_highlighted.png")
                result_image.convert("RGB").save(result_image_path)

            # 保存当前区域图像
            # current_region_path = os.path.join(output_dir, f"step_{i}_current_region.png")
            # current_region.convert("RGB").save(current_region_path)

            # 保存当前区域原尺寸图像
            cropped_image_path = os.path.join(output_dir, f"step_{i}_cropped_image.png")
            cropped_image.convert("RGB").save(cropped_image_path)


            
            # 在cropped_image上获取置信度
            prediction_coord = (pred_x / dim, pred_y / dim)  # 归一化相对坐标

            dis = self.calculate_relative_distance(prediction_coord, cropped_image.size[0], cropped_image.size[1])
            print("relative distance:", dis)

            iteration = i + 1
            
            if dis < threshold and prev_dis < threshold:
                break
            
            prev_dis = dis

        final_bbox = traverser.get_bounding_box()
        print("final_bbox:", final_bbox)

        last_porp_x = pred_x / dim
        last_porp_y = pred_y / dim
            
        # last_porp_x = pred_x / current_region.size[0]
        # last_porp_y = pred_y / current_region.size[1]
        
        delta_x = (final_bbox[2] - final_bbox[0]) * last_porp_x
        delta_y = (final_bbox[3] - final_bbox[1]) * last_porp_y

        # 绝对坐标
        x, y = final_bbox[0] + delta_x, final_bbox[1] + delta_y
        if verbose:
            # 保存最终的十字准线图像
            crosshair_current_path = os.path.join(output_dir, "final_crosshair_current_region.png")
            render_crosshair(current_region, pred_x, pred_y).convert("RGB").save(crosshair_current_path)

            crosshair_full_image_path = os.path.join(output_dir, "final_crosshair_full_image.png")
            render_crosshair(image, x, y).convert("RGB").save(crosshair_full_image_path)

        # 返回归一化的坐标
        point = (x / width, y / height)
        return response, final_bbox, point, iteration


    def ground_only_positive_iterative(self, instruction, subset, image):
        if not isinstance(image, str):
            assert isinstance(image, Image.Image)
            image_path = image_to_temp_filename(image)
        else:
            image_path = image
        assert os.path.exists(image_path) and os.path.isfile(image_path), "Invalid input image path."



        prompt_origin = f"""
        Your task is to help the user identify the precise coordinates (x, y) of a specific area/element/object on the screen based on a description.

        - Your response should aim to point to the center or a representative point within the described area/element/object as accurately as possible.
        - If the description is unclear or ambiguous, infer the most relevant area or element based on its likely context or purpose.
        - Your answer should be a single string (x, y) corresponding to the point of the interest.

        Description: {instruction}

        Answer:"""

        response, bbox, point, iterations = self.iterative_narrowing(prompt_origin, instruction, image_path, subset=subset, dim=999)

        result_dict = {
            "result": "positive",
            "format": "x1y1x2y2",
            "raw_response": response,
            "bbox": bbox,
            "point": point,
            "iterations": iterations
        }
        print("result_dict:", result_dict)

        return result_dict


    def select_answer(self, instruction, image_path, subset, decision_prompt):
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": image_path,
                    },
                    {"type": "text", "text": decision_prompt},
                ],
            }
        ]
        # Preparation for inference
        text_input = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=[text_input],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to(self.device)
        generated_ids = self.model.generate(**inputs)

        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        answer = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=False, clean_up_tokenization_spaces=False
        )[0]

        answer = answer.replace("<|im_end|>", "").strip()

        return answer
    

    def ground_only_positive_iterative_conquer(self, instruction, subset, image, max_iter, threshold):
        if not isinstance(image, str):
            assert isinstance(image, Image.Image)
            image_path = image_to_temp_filename(image)
        else:
            image_path = image
        print(image_path)
        assert os.path.exists(image_path) and os.path.isfile(image_path), "Invalid input image path."

        prompt_origin = f"""
        Your task is to help the user identify the precise coordinates (x, y) of a specific area/element/object on the screen based on a description.

        - Your response should aim to point to the center or a representative point within the described area/element/object as accurately as possible.
        - If the description is unclear or ambiguous, infer the most relevant area or element based on its likely context or purpose.
        - Your answer should be a single string (x, y) corresponding to the point of the interest.

        Description: {instruction}

        Answer:"""

        text_prompt_origin = '''
        Your task is to help the user identify the precise coordinates (x, y) of a specific TEXT element in a UI screenshot, based on a given command description.

        - Your response should point to the center or a representative point within the described **text element** as accurately as possible.
        - Only consider visible **text content** on the screen. Ignore any icons, buttons without text, or purely graphical elements.
        - If the description is unclear or ambiguous, infer the most relevant text element based on its likely context or purpose.
        - Your answer should be a single string (x, y) corresponding to the point of interest.

        In this UI screenshot, please find the TEXT element corresponding to the command "{}" (with bbox).

        Answer:
        '''

        icon_prompt_origin = '''
        Your task is to help the user identify the precise coordinates (x, y) of a specific ICON element in a UI screenshot, based on a given command description.

        - Your response should point to the center or a representative point within the described **icon element** as accurately as possible.
        - Only consider **graphical icons** on the screen. Ignore any pure text elements or labels.
        - If the description is unclear or ambiguous, infer the most relevant icon based on its likely context or purpose.
        - Your answer should be a single string (x, y) corresponding to the point of interest.

        In this UI screenshot, please find the ICON element corresponding to the command "{}" (with bbox).

        Answer:
        '''



        # todo: divide and conquer
        # text_prompt_origin = 'In this UI screenshot, please find the TEXT element corresponding to the command "{}" (with bbox). Only focus on text content, ignore icons.'
        # icon_prompt_origin = 'In this UI screenshot, please find the ICON element corresponding to the command "{}" (with bbox). Only focus on graphical icons, ignore pure text.'

        text_prompt = text_prompt_origin.format(instruction)
        icon_prompt = icon_prompt_origin.format(instruction)
        
        response_text, bbox_text, point_text, text_iteration =self.iterative_narrowing(prompt_origin=text_prompt, instruction=instruction, image_path=image_path, subset=subset + "-text", max_iter=max_iter, threshold=threshold)
        response_icon, bbox_icon, point_icon, icon_iteration =self.iterative_narrowing(prompt_origin=icon_prompt, instruction=instruction, image_path=image_path, subset=subset + "-icon", max_iter=max_iter, threshold=threshold)
        # response_text, bbox_text, point_text, text_iteration = self.get_model_response(prompt_origin=text_prompt, instruction=instruction, image_path=image_path)
        # response_icon, bbox_icon, point_icon, icon_iteration = self.get_model_response(prompt_origin=icon_prompt, instruction=instruction, image_path=image_path)


        # todo: select the best answer
        decision_prompt_template = (
            "You are given a UI screenshot and a user command: \"{instruction}\".\n"
            "There are two candidate UI elements:\n\n"
            "Candidate 1 (text-based): {point_text}\n"
            "Candidate 2 (icon-based): {point_icon}\n\n"
            "Based on the command and the description of candidates, "
            "choose which candidate (1 for text, 2 for icon) better matches the command.\n"
            "Just answer with '1' or '2'."
        )

        decision_prompt = decision_prompt_template.format(
            instruction=instruction,
            point_text = point_text, 
            point_icon = point_icon,
        )

        print(decision_prompt)
        answer = self.select_answer(instruction, image_path, subset, decision_prompt)
        print("answer:", answer)
        answer = match_answer_to_candidate(bbox_text, bbox_icon, answer)
        if answer == '1':
            response = response_text
            bbox = bbox_text
            point = point_text
        elif answer == '2':
            response = response_icon
            bbox = bbox_icon
            point = point_icon
        else:
            response = answer
            bbox = None
            point = None


        ## vanilla osatlas
        # response, bbox, point = self.get_model_response(instruction, image_path)

        result_dict = {
            "result": "positive",
            "format": "x1y1x2y2",
            "raw_response": response,
            "bbox": bbox,
            "point": point,
            "iterations": text_iteration + icon_iteration,
        }
        print("result_dict:", result_dict)
        
        return result_dict


    def ground_allow_negative(self, instruction, image):
        if not isinstance(image, str):
            assert isinstance(image, Image.Image)
            image_path = image_to_temp_filename(image)
        else:
            image_path = image
        assert os.path.exists(image_path) and os.path.isfile(image_path), "Invalid input image path."

        prompt_origin = 'Output the bounding box in the image corresponding to the instruction "{}". If the target does not exist, respond with "Target does not exist".'
        full_prompt = prompt_origin.format(instruction)
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": image_path,
                    },
                    {"type": "text", "text": full_prompt},
                ],
            }
        ]
        # Preparation for inference
        text_input = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=[text_input],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to("cuda")
        generated_ids = self.model.generate(**inputs)

        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        response = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=False, clean_up_tokenization_spaces=False
        )[0]



        result_dict = {
            "result": None,
            "format": "x1y1x2y2",
            "raw_response": response,
            "bbox": None,
            "point": None
        }

        if '<|box_start|>' in response and '<|box_end|>' in response:
            pred_bbox = extract_bbox(response)
            if pred_bbox is not None:
                (x1, y1), (x2, y2) = pred_bbox
                pred_bbox = [pos / 1000 for pos in [x1, y1, x2, y2]]
                click_point = [(pred_bbox[0] + pred_bbox[2]) / 2, (pred_bbox[1] + pred_bbox[3]) / 2]

                result_dict["bbox"] = pred_bbox
                result_dict["point"] = click_point
        else:
            print('---------------')
            print(response)
            click_point = pred_2_point(response)
            click_point = [x / 1000 for x in click_point] if click_point else None
            result_dict["point"] = click_point  # can be none

        # set result status
        if result_dict["bbox"] or result_dict["point"]:
            result_status = "positive"
        elif "Target does not exist".lower() in response.lower():
            result_status = "negative"
        else:
            result_status = "wrong_format"
        result_dict["result"] = result_status

        return result_dict

