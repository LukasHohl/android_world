"""This file contains utility functions."""

# Standard Library
from pathlib import Path

# 3rd Party
from PIL import ImageDraw, ImageFont
import numpy as np
from numpy import array, ceil, floor

def toCoord(info_pool, bb):
    """Converting from pixel to Gemini's coordinate system."""
    return array([[ceil(bb[0][1]*(1000.0/info_pool.height)),ceil(bb[0][0]*(1000.0/info_pool.width))],[floor(bb[1][1]*(1000.0/info_pool.height)),floor(bb[1][0]*(1000.0/info_pool.width))]]).astype(int)

def fromCoord(info_pool, bb):
    """Converting from Gemini's coordinate system to pixel."""
    return array([[ceil(bb[0][1]*(info_pool.width/1000.0)),ceil(bb[0][0]*(info_pool.height/1000.0))],[floor(bb[1][1]*(info_pool.width/1000.0)),floor(bb[1][0]*(info_pool.height/1000.0))]]).astype(int)

def lisToBB(lis):
    return array(lis).reshape(2,2)

def play_beep():
    """This function plays a sound under Windows."""
    try:
        import winsound
        winsound.PlaySound("SystemExit", winsound.SND_ALIAS)
    except Exception as e:
        pass

# draw_boxes is not used anywhere at the moment but it is so great for debugging and generating graphics that it won't be deleted.
def draw_boxes(img, bounding_box_list, color, text_list=None, width=5):
    """This draws boxes in an image, after copying the image, returns the new image
    textlist is either none or a list of strings with the same number of entries as bounding_box_list.

    :param img: The image to copy.
    :type img: PIL.Image
    :param bounding_box_list: A list of boxes to draw.
    :type bounding_box_list: list[np.ndarray]
    :param color: The color used for drawing. (r,g,b)
    :type color: Tuple
    :param text_list: The list of texts to draw into the center of the drawn boxes.
    :type text_list: list[str]
    :param width: The width of the lines.
    :param type: int
    :return: A copy of the original images with the added boxes.
    :rtype: PIL.Image
    """

    img = img.copy()
    imgdrw = ImageDraw.Draw(img)
    fnt_size = int(min(img.size[0], img.size[1])/30)
    fnt = ImageFont.truetype(Path(__file__).resolve(
    ).parent/"Open_Sans/static/OpenSans_Condensed-Bold.ttf", fnt_size)
    for n in range(0, len(bounding_box_list)):
        imgdrw.rectangle((tuple(bounding_box_list[n][0]), tuple(
            bounding_box_list[n][1])), outline=color, width=width)
        if not (text_list is None):
            _, _, text_width, text_height = imgdrw.textbbox(
                (0, 0), text=text_list[n], font=fnt)
            imgdrw.text((0.5*(bounding_box_list[n][0]+bounding_box_list[n][1]-np.array(
                [text_width, text_height]))), text_list[n], fill=color, font=fnt)
    return img


def draw_arrow(img, start, end, angle=1/3*np.pi, length=0.2, width = 10):
    """draw_arrow draws an arrow in the given image.

    :param img: The image to draw in.
    :type img: PIL.Image
    :param start: The starting point of the arrow.
    :type start: numpy.ndarray
    :param end: The ending point of the arrow.
    :type end: numpy.ndarray
    :param angle: The angle of the smaller parts of the arrow to the main part.
    :type angle: float
    :param length: The relative length of the smaller parts to the main part.
    :type lenght: float
    """

    direction = end-start
    c = np.cos(angle)
    s = np.sin(angle)
    m1 = np.array([[c, s], [-s, c]])
    m2 = np.array([[c, -s], [s, c]])
    d1 = m1 @ direction
    d2 = m2 @ direction
    d1 = d1 / np.linalg.norm(d1) * np.linalg.norm(direction) * length
    d2 = d2 / np.linalg.norm(d2) * np.linalg.norm(direction) * length
    p1 = end - d1
    p2 = end - d2
    img_draw = ImageDraw.Draw(img)
    img_draw.line([tuple(start), tuple(end)], fill="red", width=width)
    img_draw.line([tuple(p1), tuple(end)], fill="red", width=width)
    img_draw.line([tuple(p2), tuple(end)], fill="red", width=width)


def calculate_box_area(box):
    """This function calculates the area of a given bounding box

    :param box: The box to calculate the area for.
    :type box: np.ndarray
    :return: The area of the box.
    :rtype: float
    """

    x = np.maximum(box[1]-box[0]+np.ones(2), np.zeros(2))
    return x[0]*x[1]


def calculate_intersection_over_union(box_1, box_2) -> float:
    """Given two bounding boxes in format [[x1,y1],[x2,y2]], this returns the
    area of the intersection divided by the area of union.

    :param box_1: The first box.
    :type box_1: np.ndarray
    :param box_2: The second box.
    :type box_2: np.ndarray
    :return: Intersection over union.
    :rtype: float
    """

    inter = np.array([np.maximum(box_1[0], box_2[0]),
                     np.minimum(box_1[1], box_2[1])])
    inter_area = calculate_box_area(inter)
    if inter_area <= 0.0:  # this handles bad input
        return 0.0
    return inter_area/(calculate_box_area(box_1)+calculate_box_area(box_2)-inter_area)