#!/usr/bin/env python
# -*- coding: utf-8 -*-
import time
from typing import Any, Dict, List, Optional

import dearpygui.dearpygui as dpg  # type: ignore
import numpy as np
from node.node_abc import DpgNodeABC  # type: ignore
from node_editor.util import (  # type: ignore
    dpg_get_value,
    dpg_set_value,
    get_tag_name_list,
)


class PeakCompressor:
    def __init__(
        self,
        threshold: float = 0.5,
        ratio: float = 4.0,
        attack_ms: int = 0,
        release_ms: int = 0,
        sample_rate: int = 16000,
    ):
        self.threshold = threshold
        self.ratio = ratio
        self.sample_rate = sample_rate
        self.alpha_attack = (
            0.0
            if attack_ms == 0
            else np.exp(-1.0 / (sample_rate * (attack_ms / 1000.0)))
        )
        self.alpha_release = (
            0.0
            if release_ms == 0
            else np.exp(-1.0 / (sample_rate * (release_ms / 1000.0)))
        )

        self.env = 1.0

    def __call__(self, chunk: np.ndarray) -> np.ndarray:
        peak = np.max(np.abs(chunk))
        if peak > self.threshold:
            over = peak / self.threshold
            gain = (1 / over) ** (1 - 1 / self.ratio)
        else:
            gain = 1.0

        target_env = gain

        if target_env < self.env:
            self.env = (
                self.alpha_attack * self.env + (1 - self.alpha_attack) * target_env
            )
        else:
            self.env = (
                self.alpha_release * self.env + (1 - self.alpha_release) * target_env
            )

        return chunk * self.env


class RMSCompressor:
    def __init__(
        self,
        threshold: float = 0.3,
        ratio: float = 4.0,
        attack_ms: int = 0,
        release_ms: int = 0,
        sample_rate: int = 16000,
    ):
        self.threshold = threshold
        self.ratio = ratio
        self.sample_rate = sample_rate

        self.alpha_attack = (
            0.0
            if attack_ms == 0
            else np.exp(-1.0 / (sample_rate * (attack_ms / 1000.0)))
        )
        self.alpha_release = (
            0.0
            if release_ms == 0
            else np.exp(-1.0 / (sample_rate * (release_ms / 1000.0)))
        )

        self.env = 1.0

    def __call__(self, chunk: np.ndarray) -> np.ndarray:
        rms = np.sqrt(np.mean(chunk**2))
        if rms > self.threshold:
            over = rms / self.threshold
            gain = (1 / over) ** (1 - 1 / self.ratio)
        else:
            gain = 1.0

        target_env = gain

        if target_env < self.env:
            self.env = (
                self.alpha_attack * self.env + (1 - self.alpha_attack) * target_env
            )
        else:
            self.env = (
                self.alpha_release * self.env + (1 - self.alpha_release) * target_env
            )

        return chunk * self.env


class Node(DpgNodeABC):
    _ver = "0.0.1"

    node_label = "Dynamic Range Compression"
    node_tag = "DynamicRangeCompressio"

    def __init__(self):
        self._node_data = {}
        self._type_names: List[str] = [
            "Peak Compression",
            "RMS Compression",
        ]

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
            [
                self.TYPE_SIGNAL_CHUNK,
                self.TYPE_FLOAT,
                self.TYPE_FLOAT,
                self.TYPE_INT,
                self.TYPE_INT,
                self.TYPE_INT,
                self.TYPE_TEXT,
            ],
            [self.TYPE_SIGNAL_CHUNK, self.TYPE_TIME_MS],
        )
        tag_node_name: str = tag_name_list[0]
        input_tag_list = tag_name_list[1]
        output_tag_list = tag_name_list[2]

        self._node_data[str(node_id)] = {
            "display_x_buffer": np.array([]),
            "display_y_buffer": np.array([]),
            "current_chunk_index": 0,
            "compressed_chunk": np.array([]),
            "compression_type": self._type_names[0],
            "compressor": PeakCompressor(),
            "threshold": 0.5,
            "ratio": 4.0,
            "attack_ms": 0,
            "release_ms": 0,
        }

        # 設定
        self._setting_dict = setting_dict or {}
        waveform_w: int = self._setting_dict.get("waveform_width", 200)
        waveform_h: int = self._setting_dict.get("waveform_height", 400)
        self._default_sampling_rate: int = self._setting_dict.get(
            "default_sampling_rate", 16000
        )
        self._chunk_size: int = self._setting_dict.get("chunk_size", 1024)
        self._use_pref_counter: bool = self._setting_dict["use_pref_counter"]

        # 表示用バッファ用意
        buffer_len: int = self._default_sampling_rate * 5
        self._node_data[str(node_id)]["display_y_buffer"] = np.zeros(
            buffer_len, dtype=np.float32
        )
        self._node_data[str(node_id)]["display_x_buffer"] = (
            np.arange(len(self._node_data[str(node_id)]["display_y_buffer"]))
            / self._default_sampling_rate
        )

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
            # プロットエリア
            with dpg.node_attribute(
                tag=output_tag_list[0][0],
                attribute_type=dpg.mvNode_Attr_Output,
            ):
                with dpg.plot(
                    height=waveform_h,
                    width=waveform_w,
                    no_inputs=False,
                    tag=f"{node_id}:audio_plot_area",
                ):
                    dpg.add_plot_axis(
                        dpg.mvXAxis,
                        label="Time(s)",
                        no_label=True,
                        no_tick_labels=True,
                        tag=f"{node_id}:xaxis",
                    )
                    dpg.add_plot_axis(
                        dpg.mvYAxis,
                        label="Amplitude",
                        no_label=True,
                        no_tick_labels=True,
                        tag=f"{node_id}:yaxis",
                    )
                    dpg.set_axis_limits(f"{node_id}:xaxis", 0.0, 5.0)
                    dpg.set_axis_limits(f"{node_id}:yaxis", -1.0, 1.0)

                    dpg.add_line_series(
                        self._node_data[str(node_id)]["display_x_buffer"].tolist(),
                        list(self._node_data[str(node_id)]["display_y_buffer"]),
                        parent=f"{node_id}:yaxis",
                        tag=f"{node_id}:audio_line_series",
                    )
            # Threshold
            with dpg.node_attribute(
                tag=input_tag_list[1][0],
                attribute_type=dpg.mvNode_Attr_Input,
            ):
                dpg.add_input_float(
                    tag=input_tag_list[1][1],
                    label="Threshold",
                    width=waveform_w - 135,
                    default_value=0.5,
                    callback=None,
                )
            # Ratio
            with dpg.node_attribute(
                tag=input_tag_list[2][0],
                attribute_type=dpg.mvNode_Attr_Input,
            ):
                dpg.add_input_float(
                    tag=input_tag_list[2][1],
                    label="Ratio",
                    width=waveform_w - 135,
                    default_value=4.0,
                    callback=None,
                )
            # Attack_ms
            with dpg.node_attribute(
                tag=input_tag_list[3][0],
                attribute_type=dpg.mvNode_Attr_Input,
            ):
                dpg.add_input_int(
                    tag=input_tag_list[3][1],
                    label="Attack(ms)",
                    width=waveform_w - 135,
                    default_value=0,
                    callback=None,
                )
            # Release_ms
            with dpg.node_attribute(
                tag=input_tag_list[4][0],
                attribute_type=dpg.mvNode_Attr_Input,
            ):
                dpg.add_input_int(
                    tag=input_tag_list[4][1],
                    label="Release(ms)",
                    width=waveform_w - 135,
                    default_value=0,
                    callback=None,
                )
            # タイプ選択
            with dpg.node_attribute(
                tag=input_tag_list[5][0],
                attribute_type=dpg.mvNode_Attr_Static,
            ):
                dpg.add_combo(
                    self._type_names,
                    default_value=self._type_names[0],
                    width=waveform_w,
                    tag=input_tag_list[5][1],
                    callback=self._on_compression_type_select,
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
            [
                self.TYPE_SIGNAL_CHUNK,
                self.TYPE_FLOAT,
                self.TYPE_FLOAT,
                self.TYPE_INT,
                self.TYPE_INT,
                self.TYPE_INT,
                self.TYPE_TEXT,
            ],
            [self.TYPE_SIGNAL_CHUNK, self.TYPE_TIME_MS],
        )
        tag_node_name: str = tag_name_list[0]
        input_tag_list = tag_name_list[1]
        output_tag_list = tag_name_list[2]

        # 計測開始
        if self._use_pref_counter:
            start_time = time.perf_counter()

        chunk: Optional[np.ndarray] = np.array([])
        chunk_index: int = -1

        # 接続情報確認
        for connection_info in connection_list:
            connection_type = connection_info[0].split(":")[2]
            if connection_type == self.TYPE_INT:
                # 接続タグ取得
                source_tag = connection_info[0] + "Value"
                destination_tag = connection_info[1] + "Value"
                # 値更新
                input_value = int(dpg_get_value(source_tag))
                if input_value < 0:
                    input_value = 0
                dpg_set_value(destination_tag, input_value)
            if connection_type == self.TYPE_FLOAT:
                # 接続タグ取得
                source_tag = connection_info[0] + "Value"
                destination_tag = connection_info[1] + "Value"
                # 値更新
                input_value = float(dpg_get_value(source_tag))
                if input_value < 0:
                    input_value = 0
                dpg_set_value(destination_tag, input_value)
            if connection_type == self.TYPE_SIGNAL_CHUNK:
                # 接続タグ取得
                source_node = ":".join(connection_info[0].split(":")[:2])
                destination_node = ":".join(connection_info[1].split(":")[:2])
                if tag_node_name == destination_node:
                    chunk_index = node_result_dict[source_node].get("chunk_index", -1)
                    chunk = node_result_dict[source_node].get("chunk", np.array([]))
                    break

        # パラメータ取得
        threshold = float(dpg_get_value(input_tag_list[1][1]))
        threshold = float(np.clip(threshold, 0.0, 1.0))
        dpg_set_value(input_tag_list[1][1], threshold)

        ratio = float(dpg_get_value(input_tag_list[2][1]))
        ratio = float(np.clip(ratio, 0.1, 10))
        dpg_set_value(input_tag_list[2][1], ratio)

        attack_ms = int(dpg_get_value(input_tag_list[3][1]))
        if attack_ms < 0:
            attack_ms = 0
        dpg_set_value(input_tag_list[3][1], attack_ms)

        release_ms = int(dpg_get_value(input_tag_list[4][1]))
        if release_ms < 0:
            release_ms = 0
        dpg_set_value(input_tag_list[4][1], release_ms)

        compression_type = dpg_get_value(input_tag_list[5][1])

        # パラメータ変更時
        prev_threshold = self._node_data[str(node_id)]["threshold"]
        prev_ratio = self._node_data[str(node_id)]["ratio"]
        prev_attack_ms = self._node_data[str(node_id)]["attack_ms"]
        prev_release_ms = self._node_data[str(node_id)]["release_ms"]
        prev_compression_type = self._node_data[str(node_id)]["compression_type"]
        if (
            prev_threshold != threshold
            or prev_ratio != ratio
            or prev_attack_ms != attack_ms
            or prev_release_ms != release_ms
            or prev_compression_type != compression_type
        ):
            self._node_data[str(node_id)]["threshold"] = threshold
            self._node_data[str(node_id)]["ratio"] = ratio
            self._node_data[str(node_id)]["attack_ms"] = attack_ms
            self._node_data[str(node_id)]["release_ms"] = release_ms
            self._node_data[str(node_id)]["compression_type"] = compression_type

            if compression_type == "Peak Compression":
                self._node_data[str(node_id)]["compressor"] = PeakCompressor(
                    threshold=threshold,
                    ratio=ratio,
                    attack_ms=attack_ms,
                    release_ms=release_ms,
                    sample_rate=self._default_sampling_rate,
                )
            elif compression_type == "RMS Compression":
                self._node_data[str(node_id)]["compressor"] = RMSCompressor(
                    threshold=threshold,
                    ratio=ratio,
                    attack_ms=attack_ms,
                    release_ms=release_ms,
                    sample_rate=self._default_sampling_rate,
                )

        # プロット
        compressed_chunk: Optional[np.ndarray] = np.zeros(
            [self._chunk_size], dtype=np.float32
        )
        current_status = player_status_dict.get("current_status", False)
        if current_status == "play" and chunk_index >= 0 and len(chunk) > 0:
            if len(chunk) < self._chunk_size:
                chunk = np.pad(
                    chunk, (0, self._chunk_size - len(chunk)), constant_values=0
                )
            if self._node_data[str(node_id)]["current_chunk_index"] < chunk_index:
                if (
                    chunk_index
                    != self._node_data[str(node_id)]["current_chunk_index"] + 1
                    and self._node_data[str(node_id)]["current_chunk_index"] != -1
                ):
                    print(
                        f"    [Warning] Scale Node Chunk Index Gap: {chunk_index - self._node_data[str(node_id)]['current_chunk_index']} (Index: {self._node_data[str(node_id)]['current_chunk_index']} -> {chunk_index})"
                    )

                # チャンク生成
                compressed_chunk = self._node_data[str(node_id)]["compressor"](chunk)
                self._node_data[str(node_id)]["compressed_chunk"] = compressed_chunk

                # プロット
                temp_display_y_buffer = self._node_data[str(node_id)][
                    "display_y_buffer"
                ]
                temp_display_y_buffer = np.roll(
                    temp_display_y_buffer, -self._chunk_size
                )
                temp_display_y_buffer[-self._chunk_size :] = compressed_chunk
                self._node_data[str(node_id)]["display_y_buffer"] = (
                    temp_display_y_buffer
                )
                dpg.set_value(
                    f"{node_id}:audio_line_series",
                    [
                        self._node_data[str(node_id)]["display_x_buffer"],
                        temp_display_y_buffer,
                    ],
                )
                self._node_data[str(node_id)]["current_chunk_index"] = chunk_index
        elif current_status == "stop":
            # バッファ初期化
            self._node_data[str(node_id)]["buffer"]: np.ndarray = np.zeros(
                [], dtype=np.float32
            )

            # プロットエリア初期化
            self._node_data[str(node_id)]["current_chunk_index"] = -1

            buffer_len: int = self._default_sampling_rate * 5
            self._node_data[str(node_id)]["display_y_buffer"] = np.zeros(
                buffer_len, dtype=np.float32
            )
            self._node_data[str(node_id)]["display_x_buffer"] = (
                np.arange(len(self._node_data[str(node_id)]["display_y_buffer"]))
                / self._default_sampling_rate
            )
            dpg.set_value(
                f"{node_id}:audio_line_series",
                [
                    self._node_data[str(node_id)]["display_x_buffer"].tolist(),
                    list(self._node_data[str(node_id)]["display_y_buffer"]),
                ],
            )

        result_dict = {
            "chunk_index": self._node_data[str(node_id)]["current_chunk_index"],
            "chunk": self._node_data[str(node_id)]["compressed_chunk"],
        }

        # 計測終了
        if self._use_pref_counter:
            elapsed_time = time.perf_counter() - start_time
            elapsed_time = int(elapsed_time * 1000)
            dpg_set_value(output_tag_list[1][1], str(elapsed_time).zfill(4) + "ms")

        return result_dict

    def close(self, node_id):
        pass

    def get_setting_dict(self, node_id):
        # タグ名
        tag_name_list: List[Any] = get_tag_name_list(
            node_id,
            self.node_tag,
            [
                self.TYPE_SIGNAL_CHUNK,
                self.TYPE_FLOAT,
                self.TYPE_FLOAT,
                self.TYPE_INT,
                self.TYPE_INT,
                self.TYPE_INT,
                self.TYPE_TEXT,
            ],
            [self.TYPE_SIGNAL_CHUNK, self.TYPE_TIME_MS],
        )
        tag_node_name: str = tag_name_list[0]
        input_tag_list = tag_name_list[1]

        pos: List[int] = dpg.get_item_pos(tag_node_name)
        threshold: float = float(dpg_get_value(input_tag_list[1][1]))
        ratio: float = float(dpg_get_value(input_tag_list[2][1]))
        attack_ms = int(dpg_get_value(input_tag_list[3][1]))
        release_ms = int(dpg_get_value(input_tag_list[4][1]))
        compression_type = dpg_get_value(input_tag_list[5][1])

        setting_dict: Dict[str, Any] = {
            "ver": self._ver,
            "pos": pos,
            "threshold": threshold,
            "ratio": ratio,
            "attack_ms": attack_ms,
            "release_ms": release_ms,
            "compression_type": compression_type,
        }
        return setting_dict

    def set_setting_dict(self, node_id, setting_dict):
        # タグ名
        tag_name_list: List[Any] = get_tag_name_list(
            node_id,
            self.node_tag,
            [
                self.TYPE_SIGNAL_CHUNK,
                self.TYPE_FLOAT,
                self.TYPE_FLOAT,
                self.TYPE_INT,
                self.TYPE_INT,
                self.TYPE_INT,
                self.TYPE_TEXT,
            ],
            [self.TYPE_SIGNAL_CHUNK, self.TYPE_TIME_MS],
        )
        input_tag_list = tag_name_list[1]

        threshold = float(setting_dict["threshold"])
        ratio = float(setting_dict["ratio"])
        attack_ms = int(setting_dict["attack_ms"])
        release_ms = int(setting_dict["release_ms"])
        compression_type = setting_dict["compression_type"]

        dpg_set_value(input_tag_list[1][1], threshold)
        dpg_set_value(input_tag_list[2][1], ratio)
        dpg_set_value(input_tag_list[3][1], attack_ms)
        dpg_set_value(input_tag_list[4][1], release_ms)
        dpg_set_value(input_tag_list[5][1], compression_type)

        if compression_type == "Peak Compression":
            self._node_data[str(node_id)]["compressor"] = PeakCompressor(
                threshold=threshold,
                ratio=ratio,
                attack_ms=attack_ms,
                release_ms=release_ms,
                sample_rate=self._default_sampling_rate,
            )
        elif compression_type == "RMS Compression":
            self._node_data[str(node_id)]["compressor"] = RMSCompressor(
                threshold=threshold,
                ratio=ratio,
                attack_ms=attack_ms,
                release_ms=release_ms,
                sample_rate=self._default_sampling_rate,
            )

    def _on_compression_type_select(self, sender, app_data, user_data):
        node_id = sender.split(":")[0]
        self._node_data[str(node_id)]["compression_type"] = app_data
        self._node_data[str(node_id)]["buffer"] = np.zeros(0, dtype=np.float32)

        # タグ名
        tag_name_list: List[Any] = get_tag_name_list(
            node_id,
            self.node_tag,
            [
                self.TYPE_SIGNAL_CHUNK,
                self.TYPE_FLOAT,
                self.TYPE_FLOAT,
                self.TYPE_INT,
                self.TYPE_INT,
                self.TYPE_INT,
                self.TYPE_TEXT,
            ],
            [self.TYPE_SIGNAL_CHUNK, self.TYPE_TIME_MS],
        )
        input_tag_list = tag_name_list[1]

        # パラメータ取得
        threshold = float(dpg_get_value(input_tag_list[1][1]))
        threshold = float(np.clip(threshold, 0, int(self._default_sampling_rate / 2)))
        dpg_set_value(input_tag_list[1][1], threshold)

        ratio = float(dpg_get_value(input_tag_list[2][1]))
        ratio = float(np.clip(ratio, 0, int(self._default_sampling_rate / 2)))
        dpg_set_value(input_tag_list[2][1], ratio)

        attack_ms = int(dpg_get_value(input_tag_list[3][1]))
        if attack_ms < 0:
            attack_ms = 0
        dpg_set_value(input_tag_list[3][1], attack_ms)

        release_ms = int(dpg_get_value(input_tag_list[4][1]))
        if release_ms < 0:
            release_ms = 0
        dpg_set_value(input_tag_list[4][1], release_ms)

        compression_type = dpg_get_value(input_tag_list[5][1])

        if compression_type == "Peak Compression":
            self._node_data[str(node_id)]["compressor"] = PeakCompressor(
                threshold=threshold,
                ratio=ratio,
                attack_ms=attack_ms,
                release_ms=release_ms,
                sample_rate=self._default_sampling_rate,
            )
        elif compression_type == "RMS Compression":
            self._node_data[str(node_id)]["compressor"] = RMSCompressor(
                threshold=threshold,
                ratio=ratio,
                attack_ms=attack_ms,
                release_ms=release_ms,
                sample_rate=self._default_sampling_rate,
            )
