"""
LoopDVR: Looped Transformer for Efficient Dense Visual Recognition
Copyright (c) 2026 The LoopDVR Authors.
SPDX-License-Identifier: Apache-2.0

Draw predictions above the display threshold on an RGB image.
"""

import cv2
import numpy as np
from PIL import Image, ImageColor, ImageDraw, ImageFont

from .categories import COCO_CATEGORIES

SKELETON = ((15, 13), (13, 11), (16, 14), (14, 12), (11, 12), (5, 11), (6, 12),
            (5, 6), (5, 7), (6, 8), (7, 9), (8, 10), (1, 2), (0, 1), (0, 2),
            (1, 3), (2, 4), (3, 5), (4, 6))


CATEGORY_COLORS = {
    1: '#E68A35', 7: '#00BCD4', 9: '#D52BDE', 15: '#7E43F3',
    18: '#35E34C', 31: '#F7D13B', 35: '#F51C92', 37: '#FFDF00',
    39: '#05BDFF', 40: '#44EF25', 42: '#8BED25', 43: '#8A40F0',
    44: '#00CCEA', 46: '#F02DCB', 47: '#008CFF', 48: '#7B40FF',
    49: '#FF4D65', 50: '#FFE32C', 51: '#59E52B', 52: '#A6E51E',
    53: '#FF283E', 62: '#7A50FF', 67: '#00C6C8', 82: '#17D69E',
}
POSE_COLORS = {'left': '#FF8C00', 'center': '#B62CFF', 'right': '#009DFF'}
LEFT = {1, 3, 5, 7, 9, 11, 13, 15}
RIGHT = {2, 4, 6, 8, 10, 12, 14, 16}
CENTER_EDGES = ((1, 2), (5, 6), (11, 12))


def edge_side(a, b):
    joints = {a, b}
    if any(joints == set(edge) for edge in CENTER_EDGES):
        return 'center'
    if joints <= LEFT | {0}:
        return 'left'
    if joints <= RIGHT | {0}:
        return 'right'
    raise ValueError(f'Unexpected skeleton edge: {(a, b)}')


def add_label(draw, anchor, text, color, font, size, occupied, unit):
    box = draw.textbbox((0, 0), text, font=font)
    padding = round(2*unit)
    width, height = box[2]-box[0]+2*padding, box[3]-box[1]+2*padding
    limit_x, limit_y = max(0, size[0]-width), max(0, size[1]-height)
    x = min(max(0, round(anchor[0])), limit_x)
    y = min(max(0, round(anchor[1])-height), limit_y)
    candidates = [(x, y)]
    for offset in (1, -1, 2, -2, 3):
        candidates.append((x, min(max(0, y+offset*height), limit_y)))
    def overlap(point):
        px, py = point
        return sum(max(0, min(px+width, x2)-max(px, x1))*
                   max(0, min(py+height, y2)-max(py, y1))
                   for x1, y1, x2, y2 in occupied)
    x, y = min(candidates, key=overlap)
    draw.rectangle((x, y, x+width, y+height), fill=color)
    luminance = .2126*color[0]+.7152*color[1]+.0722*color[2]
    text_color = '#102030' if luminance > 140 else 'white'
    draw.text((x+padding, y+padding-box[1]), text, font=font, fill=text_color)
    occupied.append((x, y, x+width, y+height))


def _category_color(category_id):
    palette = ("#E64062", "#00A9D0", "#FF9F1C", "#7552CC", "#20AD69", "#EF6B30",
               "#4779D3", "#C948BB", "#93B523", "#05A999", "#CE9C45", "#8D62A8")
    return ImageColor.getrgb(CATEGORY_COLORS.get(category_id, palette[category_id % len(palette)]))


def _draw_masks(canvas, masks, selected, colors, unit):
    # Paint large instances first, then outline every instance (including holes).
    array = np.array(canvas)
    border = np.zeros(array.shape[:2], dtype=np.uint8)
    anchors = {}
    areas = {i: masks[i, 0].sum().item() for i in selected}
    for i in sorted(selected, key=areas.get, reverse=True):
        mask = masks[i, 0].detach().cpu().numpy().astype(np.uint8)
        x, y, w, h = cv2.boundingRect(mask)
        if w == 0 or h == 0:
            continue
        anchors[i] = (x, y)
        contours, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(border, contours, -1, 255, max(1, round(2.2 * unit)), cv2.LINE_AA)
        pixels = mask.astype(bool)
        array[pixels] = np.rint(array[pixels] * .35 + np.array(colors[i]) * .65)
    result = Image.fromarray(array)
    result.paste("white", mask=Image.fromarray(border))
    return result, anchors


def draw_predictions(image, prediction, threshold):
    canvas = image.convert("RGB")
    scores = prediction["scores"].tolist()
    selected = [i for i, score in enumerate(scores) if score > threshold]
    if not selected:
        return canvas
    unit = max(.75, min(canvas.width / 640, canvas.height / 400))
    font_size = max(1, round(11 * unit))
    try:
        font = ImageFont.truetype("DejaVuSans-Bold.ttf", font_size)
    except OSError:
        font = ImageFont.load_default(size=font_size)
    pose = "keypoints" in prediction
    labels = prediction["labels"].tolist()
    categories = {i: COCO_CATEGORIES[0 if pose else labels[i]] for i in selected}
    colors = {i: _category_color(categories[i][0]) for i in selected}
    anchors = {}
    if "masks" in prediction:
        canvas, anchors = _draw_masks(canvas, prediction["masks"], selected, colors, unit)
    draw = ImageDraw.Draw(canvas)
    if pose:
        points = prediction["keypoints"].detach().cpu().numpy().reshape(-1, 17, 3)
        for i in selected:
            xy = points[i, :, :2]
            for a, b in SKELETON:
                draw.line([tuple(xy[a]), tuple(xy[b])], fill=POSE_COLORS[edge_side(a, b)],
                          width=max(1, round(4 * unit)))
        radius = 2.3 * unit
        for i in selected:
            for x, y, _ in points[i]:
                draw.ellipse((x-radius, y-radius, x+radius, y+radius), fill="white",
                             outline="black", width=max(1, round(.8 * unit)))
            head = points[i, :5]
            text_width = draw.textlength(f"{scores[i]:.2f}", font=font)
            anchors[i] = (head[:, 0].mean() - text_width / 2 - 2 * unit,
                          head[:, 1].min() - 4 * unit)
    else:
        boxes = prediction["boxes"].tolist()
        for i in reversed(selected):
            anchors.setdefault(i, boxes[i][:2])
            if "masks" not in prediction:
                draw.rectangle(boxes[i], outline=colors[i], width=max(1, round(3.4 * unit)))
    occupied = []
    for i in selected:
        text = f"{scores[i]:.2f}" if pose else f"{categories[i][1]} {scores[i]:.2f}"
        add_label(draw, anchors[i], text, colors[i], font, canvas.size, occupied, unit)
    return canvas
