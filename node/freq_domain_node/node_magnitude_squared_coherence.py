#!/usr/bin/env python
# -*- coding: utf-8 -*-
import time
from typing import Any, List, Optional

import dearpygui.dearpygui as dpg
import numpy as np
from node_editor.util import dpg_set_value, get_tag_name_list

from node.node_abc import DpgNodeABC


class Node(DpgNodeABC):
    _ver = "0.0.1"

    node_label = "Magnitude Squared Coherence"
    node_tag = "MagnitudeSquaredCoherence"

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
            [self.TYPE_SIGNAL_CHUNK, self.TYPE_SIGNAL_CHUNK],
            [self.TYPE_TIME_MS],
        )
        tag_node_name: str = tag_name_list[0]
        input_tag_list = tag_name_list[1]
        output_tag_list = tag_name_list[2]

        # 設定
        self._setting_dict = setting_dict or {}
        self._waveform_w: int = self._setting_dict.get("waveform_width", 200)
        self._waveform_h: int = self._setting_dict.get("waveform_height", 200)
        self._default_sampling_rate: int = self._setting_dict.get(
            "default_sampling_rate", 16000
        )
        self._chunk_size: int = self._setting_dict.get("chunk_size", 1024)
        self._use_pref_counter: bool = self._setting_dict["use_pref_counter"]

        self._fft_window_size: int = self._setting_dict.get("fft_window_size", 1024)
        self._alpha: float = self._setting_dict.get(
            "alpha", 0.2
        )  # EWMA smoothing factor

        # 保持用バッファなど
        freq_data = np.linspace(
            0, self._default_sampling_rate / 2, self._fft_window_size // 2 + 1
        )
        coherence_data = np.zeros(self._fft_window_size // 2 + 1, dtype=np.float32)

        self._node_data[str(node_id)] = {
            "current_chunk_index": -1,
            "frame_buffer_1": np.zeros(0, dtype=np.float32),
            "frame_buffer_2": np.zeros(0, dtype=np.float32),
            "freq_data": freq_data,
            "coherence_data": coherence_data,
            "avg_pxx": np.zeros(self._fft_window_size // 2 + 1, dtype=np.float32),
            "avg_pyy": np.zeros(self._fft_window_size // 2 + 1, dtype=np.float32),
            "avg_pxy": np.zeros(self._fft_window_size // 2 + 1, dtype=np.complex64),
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
                dpg.add_text(tag=input_tag_list[0][1], default_value="Input 1")
            with dpg.node_attribute(
                tag=input_tag_list[1][0],
                attribute_type=dpg.mvNode_Attr_Input,
            ):
                dpg.add_text(tag=input_tag_list[1][1], default_value="Input 2")

            # 描画エリア
            plot_tag = dpg.generate_uuid()
            xaxis_tag = dpg.generate_uuid()
            yaxis_tag = dpg.generate_uuid()
            series_tag = dpg.generate_uuid()

            self._node_data[str(node_id)]["series_tag"] = series_tag

            with dpg.node_attribute(attribute_type=dpg.mvNode_Attr_Static):
                with dpg.plot(
                    height=self._waveform_h,
                    width=self._waveform_w,
                    tag=plot_tag,
                    no_inputs=False,
                ):
                    dpg.add_plot_axis(
                        dpg.mvXAxis, tag=xaxis_tag, no_label=True, no_tick_labels=True
                    )
                    dpg.add_plot_axis(
                        dpg.mvYAxis, tag=yaxis_tag, no_label=True, no_tick_labels=True
                    )
                    dpg.set_axis_limits(xaxis_tag, 0, self._default_sampling_rate / 2)
                    dpg.set_axis_limits(yaxis_tag, 0, 1)
                    dpg.add_line_series(
                        list(freq_data),
                        list(coherence_data),
                        tag=series_tag,
                        parent=yaxis_tag,
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
            [self.TYPE_SIGNAL_CHUNK, self.TYPE_SIGNAL_CHUNK],
            [self.TYPE_TIME_MS],
        )
        input_tag_list = tag_name_list[1]
        output_tag_list = tag_name_list[2]

        # 計測開始
        if self._use_pref_counter:
            start_time = time.perf_counter()

        # 入力データ取得
        chunk_1: Optional[np.ndarray] = None
        chunk_2: Optional[np.ndarray] = None
        chunk_index: int = -1

        for connection_info in connection_list:
            source_node_id = connection_info[0]
            destination_node_id = connection_info[1]

            if destination_node_id == input_tag_list[0][0]:  # Input 1
                source_node = ":".join(source_node_id.split(":")[:2])
                if source_node in node_result_dict:
                    chunk_index = node_result_dict[source_node].get("chunk_index", -1)
                    chunk_1 = node_result_dict[source_node].get("chunk", None)

            if destination_node_id == input_tag_list[1][0]:  # Input 2
                source_node = ":".join(source_node_id.split(":")[:2])
                if source_node in node_result_dict:
                    # 2番目の入力はchunk_indexを上書きしない
                    chunk_2 = node_result_dict[source_node].get("chunk", None)

        # 再生
        current_status = player_status_dict.get("current_status", False)
        if (
            current_status == "play"
            and chunk_index >= 0
            and chunk_1 is not None
            and chunk_2 is not None
        ):
            if self._node_data[str(node_id)]["current_chunk_index"] < chunk_index:
                n_fft = self._fft_window_size

                # バッファ更新
                buffer_1 = np.concatenate(
                    [self._node_data[str(node_id)]["frame_buffer_1"], chunk_1]
                )
                buffer_2 = np.concatenate(
                    [self._node_data[str(node_id)]["frame_buffer_2"], chunk_2]
                )

                if len(buffer_1) > n_fft:
                    buffer_1 = buffer_1[-n_fft:]
                if len(buffer_2) > n_fft:
                    buffer_2 = buffer_2[-n_fft:]

                self._node_data[str(node_id)]["frame_buffer_1"] = buffer_1
                self._node_data[str(node_id)]["frame_buffer_2"] = buffer_2

                # FFTに十分なデータがあるか確認
                if len(buffer_1) >= n_fft and len(buffer_2) >= n_fft:
                    frame_1 = buffer_1
                    frame_2 = buffer_2

                    # 窓関数 + FFT
                    window = np.hanning(n_fft)
                    X = np.fft.rfft(frame_1 * window)
                    Y = np.fft.rfft(frame_2 * window)

                    # スペクトル密度計算
                    Pxx = np.abs(X) ** 2
                    Pyy = np.abs(Y) ** 2
                    Pxy = X * np.conj(Y)

                    # EWMA更新
                    alpha = self._alpha
                    self._node_data[str(node_id)]["avg_pxx"] = (
                        alpha * Pxx
                        + (1 - alpha) * self._node_data[str(node_id)]["avg_pxx"]
                    )
                    self._node_data[str(node_id)]["avg_pyy"] = (
                        alpha * Pyy
                        + (1 - alpha) * self._node_data[str(node_id)]["avg_pyy"]
                    )
                    self._node_data[str(node_id)]["avg_pxy"] = (
                        alpha * Pxy
                        + (1 - alpha) * self._node_data[str(node_id)]["avg_pxy"]
                    )

                    # コヒーレンス計算
                    # ゼロ除算を避ける
                    denominator = (
                        self._node_data[str(node_id)]["avg_pxx"]
                        * self._node_data[str(node_id)]["avg_pyy"]
                    )
                    coherence = (
                        np.abs(self._node_data[str(node_id)]["avg_pxy"]) ** 2
                    ) / (denominator + 1e-10)
                    coherence = np.nan_to_num(coherence)  # nanを0に置換

                    # プロットデータ更新
                    series_tag = self._node_data[str(node_id)]["series_tag"]
                    dpg_set_value(
                        series_tag,
                        [
                            list(self._node_data[str(node_id)]["freq_data"]),
                            list(coherence),
                        ],
                    )

                self._node_data[str(node_id)]["current_chunk_index"] = chunk_index

        elif current_status == "stop":
            # 状態リセット
            self._node_data[str(node_id)]["current_chunk_index"] = -1
            self._node_data[str(node_id)]["frame_buffer_1"] = np.zeros(
                0, dtype=np.float32
            )
            self._node_data[str(node_id)]["frame_buffer_2"] = np.zeros(
                0, dtype=np.float32
            )
            self._node_data[str(node_id)]["avg_pxx"].fill(0)
            self._node_data[str(node_id)]["avg_pyy"].fill(0)
            self._node_data[str(node_id)]["avg_pxy"].fill(0)

            # プロットをリセット
            series_tag = self._node_data[str(node_id)]["series_tag"]
            dpg_set_value(
                series_tag,
                [
                    list(self._node_data[str(node_id)]["freq_data"]),
                    list(np.zeros_like(self._node_data[str(node_id)]["freq_data"])),
                ],
            )

        # 計測終了
        if self._use_pref_counter:
            elapsed_time = time.perf_counter() - start_time
            elapsed_time = int(elapsed_time * 1000)
            dpg_set_value(output_tag_list[0][1], str(elapsed_time).zfill(4) + "ms")

        return None

    def close(self, node_id):
        self._node_data.pop(str(node_id), None)

    def get_setting_dict(self, node_id):
        tag_name_list: List[Any] = get_tag_name_list(
            node_id,
            self.node_tag,
            [self.TYPE_SIGNAL_CHUNK, self.TYPE_SIGNAL_CHUNK],
            [self.TYPE_TIME_MS],
        )
        tag_node_name: str = tag_name_list[0]
        pos = dpg.get_item_pos(tag_node_name)

        setting_dict: Dict[str, Any] = {
            "ver": self._ver,
            "pos": pos,
        }
        return setting_dict

    def set_setting_dict(self, node_id, setting_dict):
        pass
