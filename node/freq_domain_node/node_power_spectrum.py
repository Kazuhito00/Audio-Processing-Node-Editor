#!/usr/bin/env python
# -*- coding: utf-8 -*-
import time
from typing import Any, Dict, List, Optional

import dearpygui.dearpygui as dpg  # type: ignore
import numpy as np
from node.node_abc import DpgNodeABC  # type: ignore
from node_editor.util import (
    dpg_set_value,
    get_tag_name_list,
)


class Node(DpgNodeABC):
    _ver = "0.0.1"

    node_label = "Power Spectrum"
    node_tag = "PowerSpectrum"

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
            [self.TYPE_SIGNAL_CHUNK, self.TYPE_TIME_MS],
        )
        tag_node_name: str = tag_name_list[0]
        input_tag_list = tag_name_list[1]
        output_tag_list = tag_name_list[2]

        # 設定
        self._setting_dict = setting_dict or {}
        self._waveform_w: int = self._setting_dict.get("waveform_width", 200)
        self._waveform_h: int = self._setting_dict.get("waveform_height", 400)
        self._default_sampling_rate: int = self._setting_dict.get(
            "default_sampling_rate", 16000
        )
        self._chunk_size: int = self._setting_dict.get("chunk_size", 1024)
        self._use_pref_counter: bool = self._setting_dict["use_pref_counter"]

        self._fft_window_size: int = self._setting_dict.get("fft_window_size", 1024)
        self._power_spectrum_scaling = self._setting_dict.get(
            "power_spectrum_scaling",
            [-80, 40],  # dBスケールの一般的な範囲
        )

        # 保持用バッファなど
        self._node_data[str(node_id)] = {
            "current_chunk_index": -1,
            "frame_buffer": np.zeros(0, dtype=np.float32),
            "power_spectrum_data": np.full(
                self._fft_window_size // 2 + 1, -80, dtype=np.float32
            ),  # 初期値を-80dBに設定
            "freq_data": np.linspace(
                0, self._default_sampling_rate / 2, self._fft_window_size // 2 + 1
            ),
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
            # 描画エリア
            with dpg.node_attribute(
                tag=output_tag_list[0][0],
                attribute_type=dpg.mvNode_Attr_Static,
            ):
                with dpg.plot(
                    height=self._waveform_h,
                    width=self._waveform_w,
                    tag=output_tag_list[0][1] + "_plot",
                    no_inputs=False,
                ):
                    dpg.add_plot_axis(
                        dpg.mvXAxis,
                        no_label=True,
                        no_tick_labels=True,
                        tag=output_tag_list[0][1] + "_xaxis",
                    )
                    dpg.add_plot_axis(
                        dpg.mvYAxis,
                        no_label=True,
                        no_tick_labels=True,
                        tag=output_tag_list[0][1] + "_yaxis",
                    )
                    dpg.set_axis_limits(
                        output_tag_list[0][1] + "_xaxis",
                        0,
                        self._default_sampling_rate / 2,
                    )
                    dpg.set_axis_limits(
                        output_tag_list[0][1] + "_yaxis",
                        self._power_spectrum_scaling[0],
                        self._power_spectrum_scaling[1],
                    )
                    dpg.add_line_series(
                        list(self._node_data[str(node_id)]["freq_data"]),
                        list(self._node_data[str(node_id)]["power_spectrum_data"]),
                        tag=output_tag_list[0][1] + "_series",
                        parent=output_tag_list[0][1] + "_yaxis",
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
            [self.TYPE_SIGNAL_CHUNK, self.TYPE_TEXT],
            [self.TYPE_SIGNAL_CHUNK, self.TYPE_TIME_MS],
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
                if source_node in node_result_dict:
                    chunk_index = node_result_dict[source_node].get("chunk_index", -1)
                    chunk = node_result_dict[source_node].get("chunk", np.array([]))
                    if chunk is None:
                        chunk = np.array([])

        # 再生
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
                        f"    [Warning] Power Spectrum Node Chunk Index Gap: {chunk_index - self._node_data[str(node_id)]['current_chunk_index']} (Index: {self._node_data[str(node_id)]['current_chunk_index']} -> {chunk_index})"
                    )

                n_fft = self._fft_window_size
                sampling_rate = self._default_sampling_rate

                frame_buffer = self._node_data[str(node_id)]["frame_buffer"]
                frame_buffer = np.concatenate([frame_buffer, chunk])

                # frame_bufferがn_fftより大きい場合、最新のn_fftサンプルを保持
                if len(frame_buffer) > n_fft:
                    frame_buffer = frame_buffer[-n_fft:]

                # FFTウィンドウサイズ以上のデータがある場合のみ処理
                # 不足している場合はゼロパディング
                if len(frame_buffer) < n_fft:
                    frame = np.pad(
                        frame_buffer, (0, n_fft - len(frame_buffer)), "constant"
                    )
                else:
                    frame = frame_buffer

                # 窓関数 + FFT
                window = np.hamming(n_fft)
                spectrum = np.fft.rfft(frame * window)
                power_spectrum = np.abs(spectrum) ** 2  # パワースペクトル
                
                # dBスケールに変換
                # ノイズフロアを設定してゼロ除算を回避
                power_spectrum_db = 10 * np.log10(np.maximum(power_spectrum, 1e-10))
                power_spectrum_db = np.clip(power_spectrum_db, self._power_spectrum_scaling[0], self._power_spectrum_scaling[1])

                series_tag = output_tag_list[0][1] + "_series"

                # Verify lengths before plotting
                if len(self._node_data[str(node_id)]["freq_data"]) != len(
                    power_spectrum_db
                ):
                    pass
                else:
                    # プロットデータ更新
                    dpg_set_value(
                        series_tag,
                        [
                            list(self._node_data[str(node_id)]["freq_data"]),
                            list(power_spectrum_db),
                        ],
                    )

                # バッファ保存
                self._node_data[str(node_id)]["frame_buffer"] = frame_buffer
                self._node_data[str(node_id)]["current_chunk_index"] = chunk_index
        elif current_status == "stop":
            self._node_data[str(node_id)]["current_chunk_index"] = -1
            # プロットをリセット
            dpg_set_value(
                output_tag_list[0][1] + "_series",
                [
                    list(self._node_data[str(node_id)]["freq_data"]),
                    list(np.full(self._fft_window_size // 2 + 1, self._power_spectrum_scaling[0], dtype=np.float32)),
                ],
            )  # リセット時も0に

        # 計測終了
        if self._use_pref_counter:
            elapsed_time = time.perf_counter() - start_time
            elapsed_time = int(elapsed_time * 1000)
            dpg_set_value(output_tag_list[1][1], str(elapsed_time).zfill(4) + "ms")

        return None

    def close(self, node_id):
        pass

    def get_setting_dict(self, node_id):
        # タグ名
        tag_name_list: List[Any] = get_tag_name_list(
            node_id,
            self.node_tag,
            [self.TYPE_SIGNAL_CHUNK, self.TYPE_TEXT],
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
