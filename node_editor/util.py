#!/usr/bin/env python
# -*- coding: utf-8 -*-
from typing import Any, List, Tuple, Union

import cv2
import dearpygui.dearpygui as dpg  # type: ignore
import numpy as np


def get_tag_name_list(
    node_id: Union[int, str],
    node_tag_name: str,
    input_types: List[str] = [],
    output_types: List[str] = [],
) -> Tuple[str, List[List[str]], List[List[str]]]:
    input_tag_list: List[List[str]] = []
    output_tag_list: List[List[str]] = []

    tag_node_name: str = f"{node_id}:{node_tag_name}"

    for index, input_type in enumerate(input_types):
        input_name: str = f"{tag_node_name}:{input_type}:Input{index + 1:02}"
        input_value_name: str = f"{tag_node_name}:{input_type}:Input{index + 1:02}Value"
        input_tag_list.append([input_name, input_value_name])

    for index, output_type in enumerate(output_types):
        output_name: str = f"{tag_node_name}:{output_type}:Output{index + 1:02}"
        output_value_name: str = (
            f"{tag_node_name}:{output_type}:Output{index + 1:02}Value"
        )
        output_tag_list.append([output_name, output_value_name])

    return tag_node_name, input_tag_list, output_tag_list


def dpg_set_value(tag: Union[int, str], value: Any) -> None:
    if dpg.does_item_exist(tag):
        dpg.set_value(tag, value)


def dpg_get_value(tag: Union[int, str]) -> Any:
    value: Any = None
    if dpg.does_item_exist(tag):
        value = dpg.get_value(tag)
    return value


def convert_cv_to_dpg(image: np.ndarray, width: int, height: int) -> np.ndarray:
    resize_image = cv2.resize(image, (width, height))

    data = np.flip(resize_image, 2)  # BGR → RGB
    data = data.ravel()
    data = np.asarray(data, dtype=np.float32)
    texture_data = data / 255.0

    return texture_data
