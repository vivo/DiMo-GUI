import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from transformers.generation import GenerationConfig
import json
import re
import os
from PIL import Image
import sys
import tempfile

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(current_dir)
sys.path.append(parent_dir)

from qwen_vl_utils import process_vision_info
from models.utils.region_traverser import RegionTraverser
from utils.utils import *


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


class Qwen2_5VLModel():
    def load_model(self, model_name_or_path="Qwen/Qwen2.5-VL-7B-Instruct", device="cuda"):
        print("model_name_or_path", model_name_or_path)
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_name_or_path, 
            device_map="cuda", 
            trust_remote_code=True, 
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2"
        ).eval()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)
        self.processor = AutoProcessor.from_pretrained(model_name_or_path)
        self.device = device

        # Setting default generation config
        self.generation_config = GenerationConfig.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct", trust_remote_code=True).to_dict()
        self.set_generation_config(
            max_length=2048,
            do_sample=False,
            temperature=0.0
        )

    def set_generation_config(self, **kwargs):
        self.generation_config.update(**kwargs)
        self.model.generation_config = GenerationConfig(**self.generation_config)


    def ground_only_positive(self, instruction, image):
        if not isinstance(image, str):
            assert isinstance(image, Image.Image)
            image_path = image_to_temp_filename(image)
        else:
            image_path = image
        assert os.path.exists(image_path) and os.path.isfile(image_path), "Invalid input image path."


        with Image.open(image_path) as img:
            width, height = img.size


        # prompt_origin = 'Output the bounding box in the image corresponding to the instruction "{}" with grounding.'
        prompt_origin = f"""
        Use a mouse and keyboard to interact with a computer, and take screenshots.
        * This is an interface to a desktop GUI. You do not have access to a terminal or applications menu. You must click on desktop icons to start applications.
        * Some applications may take time to start or process actions, so you may need to wait and take successive screenshots to see the results of your actions. E.g. if you click on Firefox and a window doesn't open, try wait and taking another screenshot.
        * The screen's resolution is {width}x{height}.
        * Whenever you intend to move the cursor to click on an element like an icon, you should consult a screenshot to determine the coordinates of the element before moving the cursor.
        * If you tried clicking on a program or link but it failed to load, even after waiting, try adjusting your cursor position so that the tip of the cursor visually falls on the element that you want to click.
        * Make sure to click any buttons, links, icons, etc with the cursor tip in the center of the element. Don't click boxes on their edges unless asked.
        * Output the bounding box in the image corresponding to the instruction "{instruction}" with grounding.
        * Just output the bounding box in the format of <|box_start|>(x1,y1),(x2,y2)<|box_end|> where (x1,y1) is the top-left corner and (x2,y2) is the bottom-right corner, no other information.
        """

        # full_prompt = prompt_origin.format(instruction)
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": image_path,
                    },
                    {"type": "text", "text": prompt_origin},
                ],
            }
        ]
        # Preparation for inference
        text_input = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        text_input += f"<|object_ref_start|>{instruction}<|object_ref_end|><|box_start|>("
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

        result_dict = {
            "result": "positive",
            "format": "x1y1x2y2",
            "raw_response": response,
            "bbox": None,
            "point": None
        }


        print("response:", response)
        # print("response type:", type(response))
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
        #     click_point = pred_2_point(response)
        #     click_point = [x / 1000 for x in click_point] if click_point else None
        #     result_dict["point"] = click_point  # can be none
        
        # 获取 <|box_end|> 之前的部分
        coords_str = response.split("<|box_end|>")[0]
        # print("coords_str: ", coords_str)

        # 用正则提取所有整数
        coords = re.findall(r'\d+', coords_str)
        # print("coords: ", coords)

        # 转为整数
        if len(coords) >= 4:
            x1, y1, x2, y2 = map(int, coords[:4])
            bbox = [x1, y1, x2, y2]

            midpoint_x = (x1 + x2) / 2
            midpoint_y = (y1 + y2) / 2
        else:
            raise ValueError(f"Expected at least 4 coordinates, got {len(coords)}: {coords}")
        
        # point = (midpoint_x / width, midpoint_y / height)
        point = (midpoint_x, midpoint_y)
        # point = (round(midpoint_x, 4), round(midpoint_y, 4))

        result_dict["bbox"] = bbox
        result_dict["point"] = point
        
        print("bbox:", bbox)
        print("point:", point)
        
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

    def iterative_narrowing(self, instruction, image_path, subset,  prompt, modality,verbose=True, dim = 999):
        target = instruction
        image = Image.open(image_path)
        width, height = image.size
        current_region = image.copy()

        # 创建输出目录
        output_dir = os.path.join('./output/qwen2_5', subset, target)
        os.makedirs(output_dir, exist_ok=True)

        traverser = RegionTraverser(current_region)
        if verbose: 
            print(target)
        k = 5
        # render_crosshair_center(current_region).show()

        response = None
        bbox = None
        point = None
        prev_dis = 1.0

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
                # response, bbox, prediction_coord = self.get_coordinate_prediction(image_prompt, target, prompt)
                response, bbox, prediction_coord = self.ground_only_positive(target, image_prompt)

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

                # 在cropped_image上获取置信度
                prediction_coord = (pred_x / dim, pred_y / dim)  # 归一化相对坐标

                dis = self.calculate_relative_distance(prediction_coord, cropped_image.size[0], cropped_image.size[1])
                print("relative distance:", dis)
                
                if dis < 0.1 and prev_dis < 0.1:
                    break
                
                prev_dis = dis

                # _, confidence = self.get_coordinate_confidence(cropped_image, prediction_coord, prompt)
                # print("confidence:", confidence)

                # if confidence > 0.6 and prev_confidence > 0.6:
                #     break

                # if confidence > 0.7:
                #     break
                
                # prev_confidence = confidence

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

        return response, bbox, point


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



    def ground_only_positive_iterative_conquer(self, instruction, subset, image):
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
        
        response_text, bbox_text, point_text =self.iterative_narrowing(instruction=instruction, image_path=image_path, subset=subset + "-text", prompt=text_prompt, modality="text")
        response_icon, bbox_icon, point_icon =self.iterative_narrowing(instruction=instruction, image_path=image_path, subset=subset + "-icon", prompt=icon_prompt, modality="icon")


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
            "point": point
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
    

