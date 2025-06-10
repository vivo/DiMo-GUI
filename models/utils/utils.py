import json
from PIL import Image, ImageDraw, ImageFont
import re
import os
import io
import base64

def extract_tuple_from_string(input_string):
    # Find all patterns that match float or integer x, y followed by a closing parenthesis
    matches = re.findall(r'(\d+(?:\.\d+)?),\s*(\d+(?:\.\d+)?)[\)\]]', input_string)
    if matches:
        # Select the last match and convert to floats
        x, y = float(matches[-1][0]), float(matches[-1][1])
        return (x, y)
    else:
        return None


def render_quadrants_with_numbers(image):
    """
    Divides the image into 4 quadrants with clear white borders and black outlines,
    numbers each quadrant at their midpoints, and returns the rendered image and
    the cropped/resized quadrant images with their quadrant IDs.

    Parameters:
        image (PIL.Image): The input image to render quadrants on.

    Returns:
        tuple: A new image with quadrants divided by borders and numbered,
               and a list of cropped/resized quadrant images with numbers.
    """
    image = image.resize((1024, 1024))
    rendered_image = image.copy()
    draw = ImageDraw.Draw(rendered_image)

    # Get the image dimensions
    width, height = image.size
    mid_x = width // 2
    mid_y = height // 2

    # Define the bounding boxes for each quadrant
    quadrants = [
        ((0, 0, mid_x, mid_y), 0),  # Top-left
        ((mid_x, 0, width, mid_y), 1),  # Top-right
        ((0, mid_y, mid_x, height), 2),  # Bottom-left
        ((mid_x, mid_y, width, height), 3)  # Bottom-right
    ]

    # Draw white borders with black outlines
    outline_width = 2  # Thickness of the black outline
    draw.line([(mid_x, 0), (mid_x, height)], fill="blue", width=outline_width)
    draw.line([(0, mid_y), (width, mid_y)], fill="blue", width=outline_width)

    # Number each quadrant at its midpoint with white text and black stroke
    font_size = min(width, height) // 15  # Adjust font size based on image size
    try:
        font = ImageFont.truetype("arial.ttf", font_size)
    except IOError:
        font = ImageFont.load_default()  # Fallback to default if specific font not available

    quadrant_images = []

    for (bbox, number) in quadrants:
        x1, y1, x2, y2 = bbox
        mid_point = ((x1 + x2) // 2, (y1 + y2) // 2)

        # Draw black stroke by placing text slightly offset in all directions
        for offset in [(1, 1), (-1, -1), (1, -1), (-1, 1), (1, 0), (0, 1), (-1, 0), (0, -1)]:
            draw.text((mid_point[0] + offset[0], mid_point[1] + offset[1]), str(number), font=font, fill="white", anchor="mm")

        # Draw the main white number text
        draw.text(mid_point, str(number), font=font, fill="black", anchor="mm")

        # Crop and resize the quadrant image
        quadrant_image = image.crop((x1, y1, x2, y2)).resize((1024, 1024))

        # Draw the quadrant ID in the middle of the cropped image
        quadrant_draw = ImageDraw.Draw(quadrant_image)
        quadrant_mid_point = (1024 // 2, 1024 // 2)
        for offset in [(1, 1), (-1, -1), (1, -1), (-1, 1), (1, 0), (0, 1), (-1, 0), (0, -1)]:
            quadrant_draw.text((quadrant_mid_point[0] + offset[0], quadrant_mid_point[1] + offset[1]),
                               str(number), font=font, fill="white", anchor="mm")
        quadrant_draw.text(quadrant_mid_point, str(number), font=font, fill="black", anchor="mm")

        # Add the cropped image to the list
        quadrant_images.append(quadrant_image)

    return rendered_image, quadrant_images

def load_screenspot_ds():
    result = {}
    
    for mode in ['desktop', 'mobile', 'web']:
        with open(f'./screenspot/screenspot_{mode}.json', 'r') as f:
            data = json.load(f)
            
        entries = {
            "text": [],
            "icon": []
        }
        
        for x in data:
            row = {
                "image": Image.open(f"./screenspot/images/{x['img_filename']}"),
                "target": x["instruction"],
                "bbox": x["bbox"],
                "data_type": x["data_type"]
            }
            entries[row["data_type"]].append(row)
    
        result[mode] = entries
    return result

def load_sspro_ds():
    result = {}
    
    for mode in ['desktop', 'mobile', 'web']:
        with open(f'./screenspot_pro/annotations/screenspot_{mode}.json', 'r') as f:
            data = json.load(f)
            
        entries = {
            "text": [],
            "icon": []
        }
        
        for x in data:
            row = {
                "image": Image.open(f"./screenspot_pro/images/{x['img_filename']}"),
                "target": x["instruction"],
                "bbox": x["bbox"],
                "data_type": x["data_type"]
            }
            entries[row["data_type"]].append(row)
    
        result[mode] = entries
    return result

def render_crosshair_center(image):
    rendered_image = image.copy()
    draw = ImageDraw.Draw(rendered_image)
    width, height = image.size

    # Calculate the center coordinates
    center_x = width // 2
    center_y = height // 2

    line_color = "blue"
    line_width = 3

    # Draw vertical and horizontal lines intersecting at the center
    draw.line([(center_x, 0), (center_x, height)], fill=line_color, width=line_width)
    draw.line([(0, center_y), (width, center_y)], fill=line_color, width=line_width)

    return rendered_image

def render_crosshair(image, x, y):
    rendered_image = image.copy()
    draw = ImageDraw.Draw(rendered_image)
    width, height = image.size

    line_color = "blue"
    line_width = 2

    draw.line([(x, 0), (x, height)], fill=line_color, width=line_width)
    draw.line([(0, y), (width, y)], fill=line_color, width=line_width)

    return rendered_image

def draw_bbox_on_image(image, bbox_coords, color='blue', width=3):
    image = image.copy()
    draw = ImageDraw.Draw(image)
    draw.rectangle(bbox_coords, outline=color, width=width)
    return image

def is_in_bbox(bbox, x, y):
    x_min, y_min, width, height = bbox
    x_max = x_min + width
    y_max = y_min + height
    
    return x_min <= x <= x_max and y_min <= y <= y_max

def get_bbox_midpoint(bbox):
    x_min, y_min, width, height = bbox
    x_mid = x_min + (width / 2)
    y_mid = y_min + (height / 2)
    
    return (x_mid, y_mid)


def crop(bbox, image_path):
    """
    根据给定的 bbox 裁剪图像。

    参数:
        bbox (list or tuple): 包含裁剪区域的边界框 [x1, y1, x2, y2]。
        image_path (str): 图像文件的路径。

    返回:
        PIL.Image.Image: 裁剪后的图像。
    """
    # 确保 bbox 是有效的
    assert len(bbox) == 4, "bbox 必须包含 4 个元素 [x1, y1, x2, y2]"
    assert isinstance(image_path, str), "image_path 必须是字符串"

    # 打开图像
    image = Image.open(image_path)

    # 裁剪图像
    cropped_image = image.crop(bbox)

    return cropped_image

def save_labeled_images(subset, dino_labeled_img, parsed_content_list, instruction, results_dir="output/iterative_omniparser"):
    # 将 instruction 作为子目录
    results_dir = os.path.join(results_dir)
    os.makedirs(results_dir, exist_ok=True)  # 确保目录存在

    # 定义保存路径
    output_image_path = os.path.join(results_dir, subset, instruction, "labeled_image.jpg")
    json_file_path = os.path.join(results_dir, subset, instruction, "parsed_content.json")

    # 将 Base64 字符串解码为图像
    image = Image.open(io.BytesIO(base64.b64decode(dino_labeled_img)))

    # 保存图像为 JPEG 格式
    image.save(output_image_path, format="JPEG")
    print(f"Labeled image saved to: {output_image_path}")

    # 保存解析内容到 JSON 文件
    with open(json_file_path, 'w', encoding='utf-8') as json_file:
        json.dump(parsed_content_list, json_file, ensure_ascii=False, indent=4)
    print(f"Parsed content saved to: {json_file_path}")

def save_screen_info(parsed_content_list, subset, instruction, results_dir="output/iterative_omniparser"):
    # Reformat the parsed content list
    screen_info = reformat_messages(parsed_content_list)

    # Ensure the results directory exists, with instruction as a subdirectory
    results_dir = os.path.join(results_dir, subset, instruction)
    os.makedirs(results_dir, exist_ok=True)

    # Define the file name as "screen_info.txt"
    file_name = "screen_info.txt"

    # Define the full path to save the file
    file_path = os.path.join(results_dir, file_name)

    # Save screen_info as a plain text file
    with open(file_path, 'w', encoding='utf-8') as txt_file:
        txt_file.write(json.dumps(screen_info, indent=4, ensure_ascii=False))

    print(f"Screen info has been saved to: {file_path}")

    
def reformat_messages(parsed_content_list):
    screen_info = ""
    for idx, element in enumerate(parsed_content_list):
        element['idx'] = idx
        if element['type'] == 'text':
            screen_info += f'''<p id={idx} class="text" alt="{element['content']}"> </p>\n'''
            # screen_info += f'ID: {idx}, Text: {element["content"]}\n'
        elif element['type'] == 'icon':
            screen_info += f'''<img id={idx} class="icon" alt="{element['content']}"> </img>\n'''
            # screen_info += f'ID: {idx}, Icon: {element["content"]}\n'
    return screen_info

def save_narrowed_image(subset, instruction, narrowed_image):
    # 定义保存目录，将 instruction 作为子目录
    output_dir = os.path.join("output/iterative_omniparser", subset, instruction)
    os.makedirs(output_dir, exist_ok=True)  # 确保目录存在

    # 定义保存路径，文件名统一为 "narrowed_image.png"
    file_name = "narrowed_image.png"
    output_path = os.path.join(output_dir, file_name)

    # 保存图像
    narrowed_image.save(output_path)
    print(f"Image saved to: {output_path}")