#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import time
from datetime import datetime
from typing import Any, Dict, List, Optional

import dearpygui.dearpygui as dpg  # type: ignore
import numpy as np
import soundfile as sf
from node_editor.util import dpg_set_value, get_tag_name_list  # type: ignore

from node.node_abc import DpgNodeABC  # type: ignore


class Node(DpgNodeABC):
    _ver = "0.0.1"

    node_label = "Write Wav File"
    node_tag = "WriteWavFile"

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
            [self.TYPE_SIGNAL_CHUNK],
            [self.TYPE_TIME_MS],
        )
        tag_node_name: str = tag_name_list[0]
        input_tag_list = tag_name_list[1]
        output_tag_list = tag_name_list[2]

        # 設定
        self._setting_dict = setting_dict or {}
        self._default_sampling_rate: int = self._setting_dict.get(
            "default_sampling_rate", 16000
        )
        self._chunk_size: int = self._setting_dict.get("chunk_size", 1024)
        self._use_pref_counter: bool = self._setting_dict["use_pref_counter"]
        self._output_directory: str = self._setting_dict.get("output_directory", "./")

        # 出力ディレクトリを作成（存在しない場合）
        os.makedirs(self._output_directory, exist_ok=True)

        # Wav保存準備
        self._node_data[str(node_id)] = {
            "current_chunk_index": -1,
        }

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
                    default_value="Input Chunk",
                )
            # 処理時間
            if self._use_pref_counter:
                with dpg.node_attribute(
                    tag=output_tag_list[0][0],
                    attribute_type=dpg.mvNode_Attr_Output,
                ):
                    dpg.add_text(
                        tag=output_tag_list[0][1],
                        default_value="elapsed time(ms)",
                    )

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
            [self.TYPE_SIGNAL_CHUNK],
            [self.TYPE_TIME_MS],
        )
        output_tag_list = tag_name_list[2]

        # 計測開始
        if self._use_pref_counter:
            start_time = time.perf_counter()

        chunk: Optional[np.ndarray] = np.array([])
        chunk_index: int = -1

        # 接続情報確認
        for connection_info in connection_list:
            connection_type = connection_info[0].split(":")[2]
            if connection_type == self.TYPE_SIGNAL_CHUNK:
                # 接続タグ取得
                source_node = ":".join(connection_info[0].split(":")[:2])
                chunk_index = node_result_dict[source_node].get("chunk_index", -1)
                chunk = node_result_dict[source_node].get("chunk", np.array([]))
                if chunk is None:
                    chunk = np.array([])

        # 再生
        current_status = player_status_dict.get("current_status", False)
        if current_status == "play" and chunk_index >= 0 and len(chunk) > 0:
            if "sound_file" not in self._node_data[str(node_id)]:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_") + str(node_id)
                self._node_data[str(node_id)]["sound_file"] = sf.SoundFile(
                    os.path.join(self._output_directory, f"{timestamp}.wav"),
                    mode="w",
                    samplerate=self._default_sampling_rate,
                    channels=1,
                    subtype="PCM_16",
                )

            if len(chunk) < self._chunk_size:
                chunk = np.pad(
                    chunk, (0, self._chunk_size - len(chunk)), constant_values=0
                )
            if self._node_data[str(node_id)]["current_chunk_index"] < chunk_index:
                if (
                    chunk_index
                    != self._node_data[str(node_id)]["current_chunk_index"] + 1
                ):
                    print(
                        f"    [Warning] Speaker Node Chunk Index Gap: {chunk_index - self._node_data[str(node_id)]['current_chunk_index']} (Index: {self._node_data[str(node_id)]['current_chunk_index']} -> {chunk_index})"
                    )

                # WAV 書き込み（同期）
                self._node_data[str(node_id)]["sound_file"].write(chunk.reshape(-1, 1))
                self._node_data[str(node_id)]["current_chunk_index"] = chunk_index

                self._node_data[str(node_id)]["current_chunk_index"] = chunk_index
        elif current_status == "stop":
            self._node_data[str(node_id)]["current_chunk_index"] = -1

            if "sound_file" in self._node_data[str(node_id)]:
                self._node_data[str(node_id)]["sound_file"].close()
                del self._node_data[str(node_id)]["sound_file"]

        # 計測終了
        if self._use_pref_counter:
            elapsed_time = time.perf_counter() - start_time
            elapsed_time = int(elapsed_time * 1000)
            dpg_set_value(output_tag_list[0][1], str(elapsed_time).zfill(4) + "ms")

        return None

    def close(self, node_id):
        if "sound_file" in self._node_data[str(node_id)]:
            self._node_data[str(node_id)]["sound_file"].close()
            del self._node_data[str(node_id)]["sound_file"]

    def get_setting_dict(self, node_id):
        # タグ名
        tag_name_list: List[Any] = get_tag_name_list(
            node_id,
            self.node_tag,
            [self.TYPE_SIGNAL_CHUNK],
            [self.TYPE_TIME_MS],
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
