import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(current_dir)
sys.path.append(parent_dir)

import json
import re
import tempfile
import base64
from io import BytesIO
from PIL import Image
from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from transformers.generation import GenerationConfig
import torch
from models.utils.region_traverser import RegionTraverser
from utils.utils import *
import matplotlib.pyplot as plt

import openai
from qwen_vl_utils import process_vision_info


def convert_pil_image_to_base64(image):
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode()


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
import re

def extract_bbox(s):
    # Extract text between <|box_start|> and <|box_end|> tags
    match = re.search(r'<\|box_start\|\>(.*?)<\|box_end\|\>', s)
    
    if match:
        # Get the text between the tags
        extracted_text = match.group(1)
        
        # Remove parentheses and brackets
        cleaned_text = re.sub(r'[()\[\]]', '', extracted_text)
        
        # Extract four numbers from the cleaned text
        pattern = r"(\d+),\s*(\d+),\s*(\d+),\s*(\d+)"
        numbers = re.findall(pattern, cleaned_text)
        
        if numbers:
            # Return the first match as tuples of integers
            x1, y1, x2, y2 = numbers[0]
            return (int(x1), int(y1)), (int(x2), int(y2))
    
    return None



def image_to_temp_filename(image):
    temp_file = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
    image.save(temp_file.name)
    print(f"Image saved to temporary file: {temp_file.name}")
    return temp_file.name



class OSAtlas7BModel():
    def load_model(self, model_name_or_path="OS-Copilot/OS-Atlas-Base-7B", device="cuda"):
        self.device = device
        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_name_or_path, 
            device_map=device, 
            trust_remote_code=True, 
            torch_dtype=torch.bfloat16,
            # attn_implementation="flash_attention_2"
            # output_attentions=True
        ).eval()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)
        self.processor = AutoProcessor.from_pretrained(model_name_or_path)

        # self.model.config.output_attentions = True

        # Setting default generation config
        self.generation_config = GenerationConfig.from_pretrained("OS-Copilot/OS-Atlas-Base-7B", trust_remote_code=True).to_dict()
        self.set_generation_config(
            max_length=4096,
            do_sample=False,
            temperature=0.0
        )

    def set_generation_config(self, **kwargs):
        self.generation_config.update(**kwargs)
        self.model.generation_config = GenerationConfig(**self.generation_config)



    def get_coordinate_prediction(self, image, target, prompt):

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {
                        "type": "text", 
                        # "text": f"In this UI screenshot, what is the position of the element corresponding to the command \"{target}\" (with bbox)?"
                        "text": prompt
                    },
                    #{"type": "text", "text": f"In the attached UI screenshot, what is the bbox of the element corresponding to the command \"{target}\"? Write your final answer in the following format (x1, y1, x2, y2)"},
                ]
            }
        ]
        
        # Preparation for inference
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        text += f"<|object_ref_start|>{target}<|object_ref_end|><|box_start|>("
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to(self.model.device)
        # outputs = self.model.generate(
        #     **inputs, max_new_tokens=128, output_attentions=True, return_dict_in_generate=True
        # )

        generated_ids = self.model.generate(**inputs, max_new_tokens=128)
        # generated_ids = outputs.sequences
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        
        response = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=False, clean_up_tokenization_spaces=False
        )

        # attentions = outputs.attentions
        # output_path = os.path.join("./output/attention_maps", "attention_map.png")
        # self.save_attention_map(attentions, "attention_map.png")

        # print(response)
        coords_str = response[0].split("<|box_end|>")[0]
        # print(coords_str)
        coords = coords_str.replace('(', '').replace(')', '').replace('[', '').replace(']', '').split(',')
        print(coords)
        x1, y1, x2, y2 = map(int, coords)
        bbox = [x1, y1, x2, y2]
        
        midpoint_x = (x1 + x2) / 2
        midpoint_y = (y1 + y2) / 2
        
        # point = (midpoint_x / 1000, midpoint_y / 1000)
        point = (round(midpoint_x, 4), round(midpoint_y, 4))
        print("point:", point)
        return response, bbox, point



    def get_model_response(self, instruction, image_path):
        prompt_origin = 'In this UI screenshot, what is the position of the element corresponding to the command \"{}\" (with bbox)?"'

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
        inputs = inputs.to(self.device)
        generated_ids = self.model.generate(**inputs)

        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        response = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=False, clean_up_tokenization_spaces=False
        )[0]

        bbox = None
        point = None

        # 计算归一化代码
        if '<|box_start|>' in response and '<|box_end|>' in response:
            pred_bbox = extract_bbox(response)
            if pred_bbox is not None:
                (x1, y1), (x2, y2) = pred_bbox
                pred_bbox = [pos / 1000 for pos in [x1, y1, x2, y2]]
                x1, y1, x2, y2 = pred_bbox
                if 0 < x1 < 1 and 0 < y1 < 1 and 0 < x2 < 1 and 0 < y2 < 1:
                    click_point = [(pred_bbox[0] + pred_bbox[2]) / 2, (pred_bbox[1] + pred_bbox[3]) / 2]
                    
                    bbox = pred_bbox
                    point = click_point
        else:
            print('---------------')
            print(response)
            click_point = pred_2_point(response)
            if click_point is not None:
                x, y = click_point
                if 0 < x < 1 and 0 < y < 1:
                    point = click_point  # can be none
                else :
                    point = [x / 1000, y / 1000]

        return response, bbox, point
    

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

    def iterative_narrowing(self, instruction, image_path, subset, modality, prompt,verbose=True, dim = 999, max_iter = 5, threshold=0.05):
        target = instruction
        image = Image.open(image_path)
        width, height = image.size
        current_region = image.copy()
        cropped_image = image.copy()

        # 创建输出目录
        output_dir = os.path.join('./output/ppt_windows', subset, target)
        os.makedirs(output_dir, exist_ok=True)

        traverser = RegionTraverser(current_region)
        if verbose: 
            print(target)

        # 参数设置
        k = max_iter # 最大迭代轮数
        # threshold = 0.05  # 截断阈值
        # render_crosshair_center(current_region).show()

        response = None
        bbox = None
        point = None
        prev_dis = 1.0
        iteration = 0

        log_file_path = os.path.join(output_dir, "confidence_log.txt")  # 定义日志文件路径

        with open(log_file_path, "w") as log_file:  # 打开文件以写入模式
            # log_file.write(f"prompt: {prompt}\n")  # 写入提示信息
            for i in range(k):

                # print("caption:", caption)
                print("iteration: ", i)

                # caption = self.generate_image_caption(current_region)
                # # 在 prompt 和 caption 之间添加提示语
                # full_prompt = f"{prompt} Caption of the image: {caption}"
                # print("full_prompt:", full_prompt)

                image_prompt = current_region.resize((dim, dim))


                # 使用完整的 prompt 输入模型
                response, bbox, prediction_coord = self.get_coordinate_prediction(image_prompt, target, prompt)

                # attention_path = os.path.join(output_dir, f"attention_map_step_{i}.png")
                # self.save_attention_map(attentions, attention_path)

                pred_x, pred_y = prediction_coord


                if verbose: 
                    print("middle coord:", prediction_coord)
                if i != k-1:
                    traverser.consume_coordinate(pred_x, pred_y)
                    result_image = traverser.get_highlighted_image()
                    cropped_image = traverser.get_cropped_image()
                    current_region = cropped_image.resize((dim, dim))
                    # print("current_region:", current_region.size)
                    # print("current_region:", current_region)

                    # 保存中间结果图像
                    result_image_path = os.path.join(output_dir, f"step_{i}_highlighted.png")
                    result_image.convert("RGB").save(result_image_path)
                    print("Saved highlighted image to:", result_image_path)

                # 保存当前区域图像
                # current_region_path = os.path.join(output_dir, f"step_{i}_current_region.png")
                # current_region.convert("RGB").save(current_region_path)

                
                # 保存当前区域原尺寸图像
                cropped_image_path = os.path.join(output_dir, f"step_{i}_cropped_image.png")
                cropped_image.convert("RGB").save(cropped_image_path)

                iteration = i + 1


                prediction_coord = (pred_x / dim, pred_y / dim)  # 归一化相对坐标

                dis = self.calculate_relative_distance(prediction_coord, cropped_image.size[0], cropped_image.size[1])
                print("relative distance:", dis)
                

                if dis < threshold and prev_dis < threshold:
                    break
                
                prev_dis = dis

        final_bbox = traverser.get_bounding_box()
        print("final_bbox:", final_bbox)

        last_porp_x = pred_x / dim
        last_porp_y = pred_y / dim
            
        
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
        # print("image size:", image.size)
        # point = (x / width, y / height)
        point = (round(x / width, 4), round(y / height, 4))

        return response, bbox, point, iteration


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



    def ground_only_positive_iterative_conquer(self, instruction, subset, image, max_iter=5, threshold=0.05):
        if not isinstance(image, str):
            assert isinstance(image, Image.Image)
            image_path = image_to_temp_filename(image)
        else:
            image_path = image
        print(image_path)
        assert os.path.exists(image_path) and os.path.isfile(image_path), "Invalid input image path."


        # todo: divide and conquer
        text_prompt_origin = 'In this UI screenshot, please find the TEXT element corresponding to the command "{}" (with bbox). Only focus on text content, ignore icons.'
        icon_prompt_origin = 'In this UI screenshot, please find the ICON element corresponding to the command "{}" (with bbox). Only focus on graphical icons, ignore pure text.'

        text_prompt = text_prompt_origin.format(instruction)
        icon_prompt = icon_prompt_origin.format(instruction)
        
        response_text, bbox_text, point_text, text_iteration = self.iterative_narrowing(instruction=instruction, image_path=image_path, subset=subset + "-text", prompt=text_prompt, modality="text", max_iter=max_iter, threshold=threshold)
        response_icon, bbox_icon, point_icon, icon_iteration = self.iterative_narrowing(instruction=instruction, image_path=image_path, subset=subset + "-icon", prompt=icon_prompt, modality="icon", max_iter=max_iter, threshold=threshold)


        # todo: select the best answer
        decision_prompt_template = (
            "You are given a UI screenshot and a user command: \"{instruction}\".\n"
            "There are two candidate UI elements:\n\n"
            "Candidate 1 (text-based): {bbox_text}\n"
            "Candidate 2 (icon-based): {bbox_icon}\n\n"
            "Based on the command and the description of candidates, "
            "choose which candidate (1 for text, 2 for icon) better matches the command.\n"
            "Just answer with '1' or '2'."
        )

        decision_prompt = decision_prompt_template.format(
            instruction=instruction,
            bbox_text=bbox_text,
            bbox_icon=bbox_icon
        )

        print(decision_prompt)
        answer = self.select_answer(instruction, image_path, subset, decision_prompt)
        print("answer:", answer)
        if answer == '1':
            response = response_text
            bbox = bbox_text
            point = point_text
        elif answer == '2':
            response = response_icon
            bbox = bbox_icon
            point = point_icon
        else:
            raise ValueError("Invalid answer. Answer must be either '1' or '2'.")


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

    def ground_only_positive_iterative(self, instruction, subset, image, max_iter=5, threshold=0):
        if not isinstance(image, str):
            assert isinstance(image, Image.Image)
            image_path = image_to_temp_filename(image)
        else:
            image_path = image
        assert os.path.exists(image_path) and os.path.isfile(image_path), "Invalid input image path."


        # todo: iterative narrowing
        prompt_origin = 'In this UI screenshot, what is the position of the element corresponding to the command \"{}\" (with bbox)?"'
        prompt = prompt_origin.format(instruction)
        response, bbox, point =self.iterative_narrowing(instruction=instruction, image_path=image_path, subset=subset, prompt=prompt, modality="text", max_iter=5, threshold=0)

        ## vanilla osatlas
        # response, bbox, point = self.get_model_response(instruction, image_path)

        result_dict = {
            "result": "positive",
            "format": "x1y1x2y2",
            "raw_response": response,
            "bbox": bbox,
            "point": point
        }
        print("result_dict:", result_dict)
        
        return result_dict
    


    def ground_only_positive(self, instruction, image):
        if not isinstance(image, str):
            assert isinstance(image, Image.Image)
            image_path = image_to_temp_filename(image)
        else:
            image_path = image
        assert os.path.exists(image_path) and os.path.isfile(image_path), "Invalid input image path."

        response, bbox, point = self.get_model_response(instruction, image_path)
        # response, bbox, point = self.get_coordinate_prediction(image_path, instruction)


        result_dict = {
            "result": "positive",
            "format": "x1y1x2y2",
            "raw_response": response,
            "bbox": bbox,
            "point": point,
            "iterations": 1,  # 迭代次数
        }

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


class OSAtlas7BVLLMModel():
    def load_model(self, model_name_or_path="OS-Copilot/OS-Atlas-Base-7B", server_url="http://0.0.0.0:8081/v1"):
        self.model_name_or_path = model_name_or_path
        self.client = openai.Client(
            base_url=server_url,
            api_key="token-abc123"  # arbitrary for vllm servers
        )
        # Setting default generation config
        self.generation_config = {
            "max_length": 2048,
            "do_sample": False,
            "temperature": 0.0
        }

        # ✅ Try fetching model list to verify deployment
        try:
            response = self.client.models.list()
            print("✅ Successfully connected to the model server:")
            for model in response.data:
                print(f"  - {model.id}")
        except Exception as e:
            print("❌ Failed to connect to the model server. Please check if the service is running correctly.")
            print(f"Error details: {e}")

    def set_generation_config(self, **kwargs):
        self.generation_config.update(**kwargs)


    def ground_only_positive(self, instruction, image):
        if isinstance(image, str):
            image_path = image
            assert os.path.exists(image_path) and os.path.isfile(image_path), "Invalid input image path."
            image = Image.open(image_path).convert('RGB')
        assert isinstance(image, Image.Image), "Invalid input image."

        prompt_origin = 'In this UI screenshot, what is the position of the element corresponding to the command \"{}\" (with bbox)?"'
        full_prompt = prompt_origin.format(instruction)

        chat_response = self.client.chat.completions.create(
            model=self.model_name_or_path,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": "data:image/png;base64," + convert_pil_image_to_base64(image)
                            }
                        },
                        {"type": "text", "text": full_prompt},
                    ],
                }
            ],
            temperature=self.generation_config["temperature"],
            top_p=0.8,
            max_tokens=self.generation_config["max_length"],
            extra_body={
                "skip_special_tokens": False
            }
        )
        response = chat_response.choices[0].message.content
        print(response)


        result_dict = {
            "result": "positive",
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
                x1, y1, x2, y2 = pred_bbox
                if 0 <= x1 <= 1 and 0 <= y1 <= 1 and 0 <= x2 <= 1 and 0 <= y2 <= 1:
                    click_point = [(pred_bbox[0] + pred_bbox[2]) / 2, (pred_bbox[1] + pred_bbox[3]) / 2]
                    
                    result_dict["bbox"] = pred_bbox
                    result_dict["point"] = click_point
        else:
            print('---------------')
            print(response)
            click_point = pred_2_point(response)
            if click_point is not None:
                x, y = click_point
                if 0 <= x <= 1 and 0 <= y <= 1:
                    click_point = [x / 1000, y / 1000]
                    result_dict["point"] = click_point  # can be none
        
        return result_dict