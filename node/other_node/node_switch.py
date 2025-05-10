#!/usr/bin/env python
# -*- coding: utf-8 -*-
import time
from typing import Any, Dict, List, Optional

import dearpygui.dearpygui as dpg  # type: ignore
import numpy as np
from node.node_abc import DpgNodeABC  # type: ignore
from node_editor.util import (  # type: ignore
    dpg_set_value,
    get_tag_name_list,
)


class Node(DpgNodeABC):
    _ver = "0.0.1"

    node_label = "Switch"
    node_tag = "Switch"

    def __init__(self):
        self._node_data = {}

    def add_node(
        self,
        parent,
        node_id,
        pos=[0, 0],
        setting_dict=None,
        callback=None,
    ):
        # タグ名
        tag_name_list: List[Any] = get_tag_name_list(
            node_id,
            self.node_tag,
            [self.TYPE_SIGNAL_CHUNK, self.TYPE_SIGNAL_CHUNK, self.TYPE_NONE],
            [self.TYPE_SIGNAL_CHUNK, self.TYPE_TIME_MS],
        )
        tag_node_name: str = tag_name_list[0]
        input_tag_list = tag_name_list[1]
        output_tag_list = tag_name_list[2]

        self._node_data[str(node_id)] = {
            "radio_button": "Chunk 1",
        }

        # 設定
        self._setting_dict = setting_dict or {}
        self._default_sampling_rate: int = self._setting_dict.get(
            "default_sampling_rate", 16000
        )
        self._chunk_size: int = self._setting_dict.get("chunk_size", 1024)
        self._use_pref_counter: bool = self._setting_dict["use_pref_counter"]

        # ノード
        with dpg.node(
            tag=tag_node_name,
            parent=parent,
            label=self.node_label,
            pos=pos,
        ):
            # 入力端子
            with dpg.node_attribute(
                tag=input_tag_list[0][0],
                attribute_type=dpg.mvNode_Attr_Input,
            ):
                dpg.add_text(
                    tag=input_tag_list[0][1],
                    default_value="Input Chunk 1",
                )
            # 入力端子
            with dpg.node_attribute(
                tag=input_tag_list[1][0],
                attribute_type=dpg.mvNode_Attr_Input,
            ):
                dpg.add_text(
                    tag=input_tag_list[1][1],
                    default_value="Input Chunk 2",
                )
            # スイッチ
            with dpg.node_attribute(
                tag=input_tag_list[2][0],
                attribute_type=dpg.mvNode_Attr_Static,
            ):
                dpg.add_radio_button(
                    tag=input_tag_list[2][1],
                    items=["Chunk 1", "Chunk 2"],
                    default_value="OFF",
                    callback=self._callback_radio_button,
                    # horizontal=True,
                )
            # 出力端子
            with dpg.node_attribute(
                tag=output_tag_list[0][0],
                attribute_type=dpg.mvNode_Attr_Output,
            ):
                dpg.add_text(
                    tag=output_tag_list[0][1],
                    default_value="Output Chunk",
                )

            # 処理時間
            if self._use_pref_counter:
                with dpg.node_attribute(
                    tag=output_tag_list[1][0],
                    attribute_type=dpg.mvNode_Attr_Output,
                ):
                    dpg.add_text(
                        tag=output_tag_list[1][1],
                        default_value="elapsed time(ms)",
                    )

        self._prev_time = time.perf_counter()

        return tag_node_name

    def update(
        self,
        node_id,
        connection_list,
        player_status_dict,
        node_result_dict,
    ):
        # タグ名
        tag_name_list: List[Any] = get_tag_name_list(
            node_id,
            self.node_tag,
            [self.TYPE_SIGNAL_CHUNK, self.TYPE_SIGNAL_CHUNK, self.TYPE_NONE],
            [self.TYPE_SIGNAL_CHUNK, self.TYPE_TIME_MS],
        )
        tag_node_name: str = tag_name_list[0]
        output_tag_list = tag_name_list[2]

        # 計測開始
        if self._use_pref_counter:
            start_time = time.perf_counter()

        # 接続情報確認
        input_chunk_01: Optional[np.ndarray] = np.array([])
        input_chunk_02: Optional[np.ndarray] = np.array([])
        chunk_index_01: int = -1
        chunk_index_02: int = -1
        for connection_info in connection_list:
            connection_type = connection_info[0].split(":")[2]
            if connection_type == self.TYPE_SIGNAL_CHUNK:
                # 接続タグ取得
                source_node = ":".join(connection_info[0].split(":")[:2])
                destination_node = ":".join(connection_info[1].split(":")[:2])
                input_no = connection_info[1].split(":")[-1]
                if source_node in node_result_dict:
                    if tag_node_name == destination_node:
                        if input_no == "Input01":
                            chunk_index_01 = node_result_dict[source_node].get(
                                "chunk_index", -1
                            )
                            input_chunk_01 = node_result_dict[source_node].get(
                                "chunk", np.array([])
                            )
                        elif input_no == "Input02":
                            chunk_index_02 = node_result_dict[source_node].get(
                                "chunk_index", -1
                            )
                            input_chunk_02 = node_result_dict[source_node].get(
                                "chunk", np.array([])
                            )

        select_chunk: Optional[np.ndarray] = np.zeros(
            [self._chunk_size], dtype=np.float32
        )
        select_chunk_index: int = -1

        current_status = player_status_dict.get("current_status", False)
        if current_status == "play":
            if self._node_data[str(node_id)]["radio_button"] == "Chunk 1":
                if chunk_index_01 >= 0 and len(input_chunk_01) > 0:
                    select_chunk = input_chunk_01
                    select_chunk_index = chunk_index_01
            elif self._node_data[str(node_id)]["radio_button"] == "Chunk 2":
                if chunk_index_02 >= 0 and len(input_chunk_02) > 0:
                    select_chunk = input_chunk_02
                    select_chunk_index = chunk_index_02

        result_dict = {
            "chunk_index": select_chunk_index,
            "chunk": select_chunk,
        }

        # 計測終了
        if self._use_pref_counter:
            elapsed_time = time.perf_counter() - start_time
            elapsed_time = int(elapsed_time * 1000)
            dpg_set_value(output_tag_list[1][1], f"{elapsed_time:04d}ms")

        return result_dict

    def close(self, node_id):
        pass

    def get_setting_dict(self, node_id):
        # タグ名
        tag_name_list: List[Any] = get_tag_name_list(
            node_id,
            self.node_tag,
            [self.TYPE_SIGNAL_CHUNK, self.TYPE_SIGNAL_CHUNK, self.TYPE_NONE],
            [self.TYPE_SIGNAL_CHUNK, self.TYPE_TIME_MS],
        )
        tag_node_name: str = tag_name_list[0]

        pos: List[int] = dpg.get_item_pos(tag_node_name)

        setting_dict: Dict[str, Any] = {
            "ver": self._ver,
            "pos": pos,
        }
        return setting_dict

    def set_setting_dict(self, node_id, setting_dict):
        pass

    #
    def _callback_radio_button(self, sender, app_data, user_data):
        node_id = sender.split(":")[0]
        self._node_data[str(node_id)]["radio_button"] = app_data
