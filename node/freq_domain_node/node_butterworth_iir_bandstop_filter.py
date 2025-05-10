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
from scipy.signal import butter, lfilter, lfilter_zi


def design_filter(stopband_start: float, stopband_end: float, fs: int, order: int = 5):
    nyq = 0.5 * fs
    min_freq = 1.0  # 1Hz未満は無視
    min_bandwidth = 2.0  # 2Hz幅を最低限確保

    # 下限・上限の正規化
    stopband_start = max(min_freq, stopband_start)
    stopband_end = max(stopband_start + min_bandwidth, stopband_end)

    low = stopband_start / nyq
    high = stopband_end / nyq

    # high が Nyquist を超えないようにする
    if high >= 1.0:
        high = 0.9999
    if low >= high:
        low = high - (min_bandwidth / nyq)
        if low <= 0:
            low = min_freq / nyq  # 最低1Hz
            high = low + (min_bandwidth / nyq)

    # 最終チェック
    if low <= 0 or high >= 1.0 or low >= high:
        raise ValueError(f"Invalid filter frequency range: low={low}, high={high}")

    return butter(order, [low, high], btype="bandstop")


def apply_filter(
    data: np.ndarray, ff_coef: np.ndarray, fb_coef: np.ndarray, zi: np.ndarray
) -> np.ndarray:
    return lfilter(ff_coef, fb_coef, data, zi=zi)


class Node(DpgNodeABC):
    _ver = "0.0.1"

    node_label = "Bandstop Filter(Butterworth IIR)"
    node_tag = "BandstopFilter"

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
            [self.TYPE_SIGNAL_CHUNK, self.TYPE_INT, self.TYPE_INT, self.TYPE_INT],
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
            "chunk": np.array([]),
            "current_chunk_index": 0,
            "display_x_buffer": np.array([]),
            "display_y_buffer": np.array([]),
            "lowcut": 0,
            "highcut": int(self._default_sampling_rate / 2),
            "order": 5,
            "ff_coef": None,
            "fb_coef": None,
            "zi": None,
            "change_filter_flag": False,
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
            # highcut
            with dpg.node_attribute(
                tag=input_tag_list[2][0],
                attribute_type=dpg.mvNode_Attr_Input,
            ):
                dpg.add_input_int(
                    tag=input_tag_list[2][1],
                    label="Stopband End(Hz)",
                    width=waveform_w - 175,
                    default_value=61,
                )
            # lowcut
            with dpg.node_attribute(
                tag=input_tag_list[1][0],
                attribute_type=dpg.mvNode_Attr_Input,
            ):
                dpg.add_input_int(
                    tag=input_tag_list[1][1],
                    label="Stopband Start(Hz)",
                    width=waveform_w - 175,
                    default_value=49,
                )
            # order
            with dpg.node_attribute(
                tag=input_tag_list[3][0],
                attribute_type=dpg.mvNode_Attr_Input,
            ):
                dpg.add_input_int(
                    tag=input_tag_list[3][1],
                    label="Filter Order",
                    width=waveform_w - 175,
                    default_value=3,
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
            [self.TYPE_SIGNAL_CHUNK, self.TYPE_INT, self.TYPE_INT, self.TYPE_INT],
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
                input_value = int(
                    np.clip(input_value, 0, int(self._default_sampling_rate / 2))
                )
                dpg_set_value(destination_tag, input_value)
            if connection_type == self.TYPE_SIGNAL_CHUNK:
                # 接続タグ取得
                source_node = ":".join(connection_info[0].split(":")[:2])
                destination_node = ":".join(connection_info[1].split(":")[:2])
                if tag_node_name == destination_node:
                    if source_node in node_result_dict:
                        chunk_index = node_result_dict[source_node].get(
                            "chunk_index", -1
                        )
                        chunk = node_result_dict[source_node].get("chunk", np.array([]))
                        break

        # フィルター設定値
        lowcut = int(dpg_get_value(input_tag_list[1][1]))
        highcut = int(dpg_get_value(input_tag_list[2][1]))
        order = int(dpg_get_value(input_tag_list[3][1]))
        if lowcut <= 0:
            lowcut = 0
            dpg_set_value(input_tag_list[1][1], lowcut)
        if highcut >= int(self._default_sampling_rate / 2):
            highcut = int(self._default_sampling_rate / 2)
            dpg_set_value(input_tag_list[2][1], highcut)
        if highcut <= 0:
            highcut = 1
            dpg_set_value(input_tag_list[2][1], highcut)
        if lowcut >= highcut:
            lowcut = highcut - 1
            dpg_set_value(input_tag_list[1][1], lowcut)

        # フィルター初期化
        if self._node_data[str(node_id)]["ff_coef"] is None:
            self._node_data[str(node_id)]["change_filter_flag"] = True

        # フィルター設定変更時
        prev_lowcut = self._node_data[str(node_id)]["lowcut"]
        prev_highcut = self._node_data[str(node_id)]["highcut"]
        prev_order = self._node_data[str(node_id)]["order"]
        if prev_lowcut != lowcut or prev_highcut != highcut or prev_order != order:
            self._node_data[str(node_id)]["change_filter_flag"] = True

            self._node_data[str(node_id)]["lowcut"] = lowcut
            self._node_data[str(node_id)]["highcut"] = highcut
            self._node_data[str(node_id)]["order"] = order

        # プロット
        filtered_chunk: Optional[np.ndarray] = np.zeros(
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
                ):
                    print(
                        f"    [Warning] bandstop Node Chunk Index Gap: {chunk_index - self._node_data[str(node_id)]['current_chunk_index']} (Index: {self._node_data[str(node_id)]['current_chunk_index']} -> {chunk_index})"
                    )

                if self._node_data[str(node_id)]["change_filter_flag"]:
                    ff_coef, fb_coef = design_filter(
                        lowcut, highcut, self._default_sampling_rate, order
                    )
                    if ff_coef is None or fb_coef is None:
                        self._node_data[str(node_id)]["ff_coef"] = None
                        self._node_data[str(node_id)]["fb_coef"] = None
                        self._node_data[str(node_id)]["zi"] = None
                    else:
                        zi = lfilter_zi(ff_coef, fb_coef) * chunk[0]
                        self._node_data[str(node_id)]["ff_coef"] = ff_coef
                        self._node_data[str(node_id)]["fb_coef"] = fb_coef
                        self._node_data[str(node_id)]["zi"] = zi

                    self._node_data[str(node_id)]["change_filter_flag"] = False

                # フィルター適用
                ff_coef = self._node_data[str(node_id)]["ff_coef"]
                fb_coef = self._node_data[str(node_id)]["fb_coef"]
                zi = self._node_data[str(node_id)]["zi"]
                if ff_coef is None or fb_coef is None:
                    filtered_chunk = chunk
                else:
                    filtered_chunk, zi = apply_filter(chunk, ff_coef, fb_coef, zi)
                    self._node_data[str(node_id)]["zi"] = zi

                self._node_data[str(node_id)]["chunk"] = filtered_chunk

                # プロット
                temp_display_y_buffer = self._node_data[str(node_id)][
                    "display_y_buffer"
                ]
                temp_display_y_buffer = np.roll(
                    temp_display_y_buffer, -self._chunk_size
                )
                temp_display_y_buffer[-self._chunk_size :] = filtered_chunk
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
            self._node_data[str(node_id)]["buffer"] = np.zeros([], dtype=np.float32)

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
            "chunk": self._node_data[str(node_id)]["chunk"],
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
            [self.TYPE_SIGNAL_CHUNK, self.TYPE_INT, self.TYPE_INT, self.TYPE_INT],
            [self.TYPE_SIGNAL_CHUNK, self.TYPE_TIME_MS],
        )
        tag_node_name: str = tag_name_list[0]
        input_tag_list = tag_name_list[1]

        pos: List[int] = dpg.get_item_pos(tag_node_name)
        lowcut = int(dpg_get_value(input_tag_list[1][1]))
        highcut = int(dpg_get_value(input_tag_list[2][1]))
        order = int(dpg_get_value(input_tag_list[3][1]))

        setting_dict: Dict[str, Any] = {
            "ver": self._ver,
            "pos": pos,
            "lowcut": lowcut,
            "highcut": highcut,
            "order": order,
        }
        return setting_dict

    def set_setting_dict(self, node_id, setting_dict):
        # タグ名
        tag_name_list: List[Any] = get_tag_name_list(
            node_id,
            self.node_tag,
            [self.TYPE_SIGNAL_CHUNK, self.TYPE_INT, self.TYPE_INT, self.TYPE_INT],
            [self.TYPE_SIGNAL_CHUNK, self.TYPE_TIME_MS],
        )
        input_tag_list = tag_name_list[1]

        lowcut = int(setting_dict["lowcut"])
        highcut = int(setting_dict["highcut"])
        order = int(setting_dict["order"])

        dpg_set_value(input_tag_list[1][1], lowcut)
        dpg_set_value(input_tag_list[2][1], highcut)
        dpg_set_value(input_tag_list[3][1], order)
