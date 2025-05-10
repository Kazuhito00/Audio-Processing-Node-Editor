#!/usr/bin/env python
# -*- coding: utf-8 -*-
from typing import Any, Dict, List, Optional

import dearpygui.dearpygui as dpg  # type: ignore
from node_editor.util import (  # type: ignore
    dpg_get_value,
    dpg_set_value,
    get_tag_name_list,
)

from node.node_abc import DpgNodeABC  # type: ignore


class Node(DpgNodeABC):
    _ver: str = "0.0.1"

    node_label: str = "Float Value"
    node_tag: str = "FloatValue"

    def __init__(self) -> None:
        pass

    def add_node(
        self,
        parent: str,
        node_id: int,
        pos: List[int] = [0, 0],
        setting_dict: Optional[Dict[str, Any]] = None,
        callback: Optional[Any] = None,
    ) -> str:
        # タグ名
        tag_name_list = get_tag_name_list(
            node_id,
            self.node_tag,
            [],
            [self.TYPE_FLOAT],
        )
        tag_node_name: str = tag_name_list[0]
        _ = tag_name_list[1]
        output_tag_list = tag_name_list[2]

        # 設定
        self._setting_dict = setting_dict or {}
        small_window_w: int = self._setting_dict.get("input_window_width", 150)

        # ノード
        with dpg.node(
            tag=tag_node_name,
            parent=parent,
            label=self.node_label,
            pos=pos,
        ):
            with dpg.node_attribute(
                tag=output_tag_list[0][0],
                attribute_type=dpg.mvNode_Attr_Output,
            ):
                dpg.add_input_float(
                    tag=output_tag_list[0][1],
                    label="Float value",
                    width=small_window_w - 94,
                    default_value=0.0,
                    callback=callback,
                )

        return tag_node_name

    def update(
        self,
        node_id: str,
        connection_list: List[Any],
        player_status_dict: Dict[str, Any],
        node_result_dict: Dict[str, Any],
    ) -> Any:
        return None

    def close(self, node_id: str) -> None:
        pass

    def get_setting_dict(self, node_id: str) -> Dict[str, Any]:
        tag_name_list = get_tag_name_list(
            node_id,
            self.node_tag,
            [],
            [self.TYPE_FLOAT],
        )
        tag_node_name: str = tag_name_list[0]
        output_tag_list = tag_name_list[2]

        output_value_tag: str = output_tag_list[0][1]

        output_value: float = round(dpg_get_value(output_value_tag), 3)
        pos: List[int] = dpg.get_item_pos(tag_node_name)

        setting_dict: Dict[str, Any] = {
            "ver": self._ver,
            "pos": pos,
            output_value_tag: output_value,
        }
        return setting_dict

    def set_setting_dict(self, node_id: int, setting_dict: Dict[str, Any]) -> None:
        tag_name_list = get_tag_name_list(
            node_id,
            self.node_tag,
            [],
            [self.TYPE_FLOAT],
        )
        output_tag_list = tag_name_list[2]

        output_value_tag: str = output_tag_list[0][1]

        output_value: float = float(setting_dict[output_value_tag])
        dpg_set_value(output_value_tag, output_value)
