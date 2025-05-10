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

import numpy as np


class NoiseGate:
    def __init__(
        self,
        threshold: float,
        sr: int,
        attack_ms: float = 10.0,
        release_ms: float = 100.0,
        hold_ms: float = 50.0,
    ):
        """
        状態付きノイズゲート

        Args:
            threshold (float): ノイズゲート開始のしきい値（振幅）
            sr (int): サンプリングレート
            attack_ms (float): ゲイン回復にかける時間（ms）
            release_ms (float): ゲイン低下にかける時間（ms）
            hold_ms (float): ゲイン低下前に保持する時間（ms）
        """
        self.threshold = threshold
        self.sr = sr
        self.attack_samples = int((attack_ms / 1000) * sr)
        self.release_samples = int((release_ms / 1000) * sr)
        self.hold_samples = int((hold_ms / 1000) * sr)

        self.gain = 0.0
        self.gate_open = False
        self.hold_counter = 0

    def process(self, chunk: np.ndarray) -> np.ndarray:
        """
        音声チャンクを処理して、ノイズゲートを適用する
        """
        output = np.zeros_like(chunk)
        for i, sample in enumerate(chunk):
            level = abs(sample)

            if level > self.threshold:
                self.gate_open = True
                self.hold_counter = self.hold_samples
            else:
                if self.hold_counter > 0:
                    self.hold_counter -= 1
                else:
                    self.gate_open = False

            # ゲイン制御（フェードイン/アウト）
            if self.gate_open:
                if self.attack_samples == 0:
                    self.gain = 1.0
                else:
                    self.gain += 1.0 / self.attack_samples
            else:
                if self.release_samples == 0:
                    self.gain = 0.0
                else:
                    self.gain -= 1.0 / self.release_samples

            # ゲインを0.0〜1.0の範囲にクリップ
            self.gain = np.clip(self.gain, 0.0, 1.0)

            # ゲインを適用
            output[i] = sample * self.gain

        return output


class Node(DpgNodeABC):
    _ver = "0.0.1"

    node_label = "Noise Gate"
    node_tag = "NoiseGate"

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
            [
                self.TYPE_SIGNAL_CHUNK,
                self.TYPE_FLOAT,
                self.TYPE_INT,
                self.TYPE_INT,
                self.TYPE_INT,
            ],
            [self.TYPE_SIGNAL_CHUNK, self.TYPE_TIME_MS],
        )
        tag_node_name: str = tag_name_list[0]
        input_tag_list = tag_name_list[1]
        output_tag_list = tag_name_list[2]

        # 設定
        self._setting_dict = setting_dict or {}
        waveform_w: int = self._setting_dict.get("waveform_width", 200)
        waveform_h: int = self._setting_dict.get("waveform_height", 400)
        self._default_sampling_rate: int = self._setting_dict.get(
            "default_sampling_rate", 16000
        )
        self._chunk_size: int = self._setting_dict.get("chunk_size", 1024)
        self._use_pref_counter: bool = self._setting_dict["use_pref_counter"]

        self._node_data[str(node_id)] = {
            "display_x_buffer": np.array([]),
            "display_y_buffer": np.array([]),
            "current_chunk_index": 0,
            "gated_chunk": np.array([]),
            "threshold": 0.02,
            "attack_ms": 0,
            "release_ms": 0,
            "hold_ms": 0,
            "noise_gate": NoiseGate(
                threshold=0.02,
                sr=self._default_sampling_rate,
                attack_ms=0,
                release_ms=0,
                hold_ms=0,
            ),
        }

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
                    width=waveform_w - 110,
                    default_value=0.02,
                    callback=None,
                )
            # Attack_ms
            with dpg.node_attribute(
                tag=input_tag_list[2][0],
                attribute_type=dpg.mvNode_Attr_Input,
            ):
                dpg.add_input_int(
                    tag=input_tag_list[2][1],
                    label="Attack(ms)",
                    width=waveform_w - 110,
                    default_value=0,
                    callback=None,
                )
            # Release_ms
            with dpg.node_attribute(
                tag=input_tag_list[3][0],
                attribute_type=dpg.mvNode_Attr_Input,
            ):
                dpg.add_input_int(
                    tag=input_tag_list[3][1],
                    label="Release(ms)",
                    width=waveform_w - 110,
                    default_value=0,
                    callback=None,
                )
            # Hold_ms
            with dpg.node_attribute(
                tag=input_tag_list[4][0],
                attribute_type=dpg.mvNode_Attr_Input,
            ):
                dpg.add_input_int(
                    tag=input_tag_list[4][1],
                    label="Hold(ms)",
                    width=waveform_w - 110,
                    default_value=0,
                    callback=None,
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
                self.TYPE_INT,
                self.TYPE_INT,
                self.TYPE_INT,
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
                input_value = float(np.clip(input_value, 0, 1.0))
                dpg_set_value(destination_tag, input_value)
            if connection_type == self.TYPE_SIGNAL_CHUNK:
                # 接続タグ取得
                source_node = ":".join(connection_info[0].split(":")[:2])
                destination_node = ":".join(connection_info[1].split(":")[:2])
                if tag_node_name == destination_node:
                    chunk_index = node_result_dict[source_node].get("chunk_index", -1)
                    chunk = node_result_dict[source_node].get("chunk", np.array([]))
                    break

        threshold = float(dpg_get_value(input_tag_list[1][1]))
        threshold = float(np.clip(threshold, 0.0, 1.0))
        dpg_set_value(input_tag_list[1][1], threshold)
        attack_ms = int(dpg_get_value(input_tag_list[2][1]))
        if attack_ms < 0:
            attack_ms = 0
        dpg_set_value(input_tag_list[2][1], attack_ms)
        release_ms = int(dpg_get_value(input_tag_list[3][1]))
        if release_ms < 0:
            release_ms = 0
        dpg_set_value(input_tag_list[3][1], release_ms)
        hold_ms = int(dpg_get_value(input_tag_list[4][1]))
        if hold_ms < 0:
            hold_ms = 0
        dpg_set_value(input_tag_list[4][1], hold_ms)

        # プロット
        gated_chunk: Optional[np.ndarray] = np.zeros(
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
                        f"    [Warning] Noise Gate Node Chunk Index Gap: {chunk_index - self._node_data[str(node_id)]['current_chunk_index']} (Index: {self._node_data[str(node_id)]['current_chunk_index']} -> {chunk_index})"
                    )

                # ノイズゲート準備
                prev_threshold = self._node_data[str(node_id)]["threshold"]
                prev_attack_ms = self._node_data[str(node_id)]["attack_ms"]
                prev_release_ms = self._node_data[str(node_id)]["release_ms"]
                prev_hold_ms = self._node_data[str(node_id)]["hold_ms"]
                if (
                    threshold != prev_threshold
                    or attack_ms != prev_attack_ms
                    or release_ms != prev_release_ms
                    or hold_ms != prev_hold_ms
                ):
                    self._node_data[str(node_id)]["noise_gate"] = NoiseGate(
                        threshold=threshold,
                        sr=self._default_sampling_rate,
                        attack_ms=attack_ms,
                        release_ms=prev_release_ms,
                        hold_ms=hold_ms,
                    )
                    self._node_data[str(node_id)]["threshold"] = threshold
                    self._node_data[str(node_id)]["attack_ms"] = attack_ms
                    self._node_data[str(node_id)]["release_ms"] = release_ms
                    self._node_data[str(node_id)]["hold_ms"] = hold_ms

                # チャンク生成
                gated_chunk = self._node_data[str(node_id)]["noise_gate"].process(chunk)
                self._node_data[str(node_id)]["gated_chunk"] = gated_chunk

                # プロット
                temp_display_y_buffer = self._node_data[str(node_id)][
                    "display_y_buffer"
                ]
                temp_display_y_buffer = np.roll(
                    temp_display_y_buffer, -self._chunk_size
                )
                temp_display_y_buffer[-self._chunk_size :] = gated_chunk
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
            "chunk": self._node_data[str(node_id)]["gated_chunk"],
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
                self.TYPE_INT,
                self.TYPE_INT,
                self.TYPE_INT,
            ],
            [self.TYPE_SIGNAL_CHUNK, self.TYPE_TIME_MS],
        )
        tag_node_name: str = tag_name_list[0]
        input_tag_list = tag_name_list[1]

        pos: List[int] = dpg.get_item_pos(tag_node_name)

        threshold = float(dpg_get_value(input_tag_list[1][1]))
        attack_ms = int(dpg_get_value(input_tag_list[2][1]))
        release_ms = int(dpg_get_value(input_tag_list[3][1]))
        hold_ms = int(dpg_get_value(input_tag_list[4][1]))

        setting_dict: Dict[str, Any] = {
            "ver": self._ver,
            "pos": pos,
            "threshold": threshold,
            "attack_ms": attack_ms,
            "release_ms": release_ms,
            "hold_ms": hold_ms,
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
                self.TYPE_INT,
                self.TYPE_INT,
                self.TYPE_INT,
            ],
            [self.TYPE_SIGNAL_CHUNK, self.TYPE_TIME_MS],
        )
        input_tag_list = tag_name_list[1]

        threshold = float(setting_dict["threshold"])
        attack_ms = int(setting_dict["attack_ms"])
        release_ms = int(setting_dict["release_ms"])
        hold_ms = int(setting_dict["hold_ms"])

        dpg_set_value(input_tag_list[1][1], threshold)
        dpg_set_value(input_tag_list[2][1], attack_ms)
        dpg_set_value(input_tag_list[3][1], release_ms)
        dpg_set_value(input_tag_list[4][1], hold_ms)
