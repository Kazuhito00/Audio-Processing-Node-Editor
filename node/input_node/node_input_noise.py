#!/usr/bin/env python
# -*- coding: utf-8 -*-
import time
from typing import Any, Dict, List, Optional

import dearpygui.dearpygui as dpg  # type: ignore
import numpy as np
from node_editor.util import dpg_set_value, dpg_get_value, get_tag_name_list  # type: ignore
from scipy.signal import butter, lfilter

from node.node_abc import DpgNodeABC  # type: ignore


def generate_pink_noise_voss_mccartney(
    duration: float, sr: int = 16000, rows: int = 16
) -> np.ndarray:
    N = int(sr * duration)
    out = np.zeros(N, dtype=np.float32)
    for j in range(rows):
        interval = 2**j
        nseg = (N + interval - 1) // interval
        seg_vals = np.random.uniform(-1.0, 1.0, size=nseg).astype(np.float32)
        row_wave = np.repeat(seg_vals, interval)[:N]
        out += row_wave

    return out / np.max(np.abs(out))


def generate_noise(
    noise_type: str, duration: float = 1.0, sr: int = 16000
) -> np.ndarray:
    samples = int(sr * duration)

    if noise_type == "White Noise":
        # ホワイトノイズ
        white_noise = np.random.normal(0, 1, samples)
        return white_noise / np.max(np.abs(white_noise))

    elif noise_type == "Pink Noise":
        # 簡易的なピンクノイズ生成
        return generate_pink_noise_voss_mccartney(duration=duration, sr=sr)

    elif noise_type == "Hiss Noise":
        # ホワイトノイズを高域強調（ハイパスフィルタ）
        white_noise = np.random.normal(0, 1, samples)
        hiss_noise = highpass_filter(white_noise, cutoff=3000.0, sr=sr)
        return hiss_noise / np.max(np.abs(hiss_noise))

    elif noise_type == "Hum Noise(50Hz)":
        # 50Hzハムノイズ
        t = np.linspace(0, duration, samples, endpoint=False)
        return 0.1 * np.sin(2 * np.pi * 50 * t)  # 50Hz

    elif noise_type == "Hum Noise(60Hz)":
        # 60Hzハムノイズ
        t = np.linspace(0, duration, samples, endpoint=False)
        return 0.1 * np.sin(2 * np.pi * 60 * t)  # 60Hz

    elif noise_type == "Pulse Noise":
        # パルスノイズ
        pulse_noise = np.zeros(samples)

        num_pulses = np.random.randint(
            0, int(duration * 10) + 1
        )  # 最大10個/秒の範囲で調整
        pulse_indices = np.random.randint(0, samples, size=num_pulses)
        pulse_amplitudes = np.random.uniform(-1.0, 1.0, size=num_pulses)
        pulse_noise[pulse_indices] = pulse_amplitudes

        return pulse_noise

    else:
        raise ValueError(f"Unknown noise type: {noise_type}")


def highpass_filter(
    data: np.ndarray, cutoff: float, sr: int, order: int = 5
) -> np.ndarray:
    nyq = 0.5 * sr
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype="high", analog=False)
    return lfilter(b, a, data)


class Node(DpgNodeABC):
    _ver: str = "0.0.1"

    node_label: str = "Noise"
    node_tag: str = "Noise"

    def __init__(self) -> None:
        self._node_data = {}
        self._noise_names: List[str] = [
            "White Noise",
            "Pink Noise",
            "Hiss Noise",
            "Hum Noise(50Hz)",
            "Hum Noise(60Hz)",
            "Pulse Noise",
        ]
        self._start_time: Optional[float] = None
        self._paused_elapsed: float = 0.0

    def add_node(
        self,
        parent: str,
        node_id: int,
        pos: List[int] = [0, 0],
        setting_dict: Optional[Dict[str, Any]] = None,
        callback: Optional[Any] = None,
    ) -> str:
        # タグ名
        tag_name_list: List[Any] = get_tag_name_list(
            node_id,
            self.node_tag,
            [self.TYPE_TEXT],
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
            "buffer": np.zeros(0, dtype=np.float32),
            "noize_chunk": np.array([]),
            "chunk_index": -1,
            "display_x_buffer": np.array([]),
            "display_y_buffer": np.array([]),
            "noise_type": self._noise_names[0],
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
                        [],
                        [],
                        parent=f"{node_id}:yaxis",
                        tag=f"{node_id}:audio_line_series",
                    )
            # ノイズタイプ選択
            with dpg.node_attribute(
                tag=input_tag_list[0][0],
                attribute_type=dpg.mvNode_Attr_Static,
            ):
                dpg.add_combo(
                    self._noise_names,
                    default_value=self._noise_names[0],
                    width=waveform_w,
                    tag=input_tag_list[0][1],
                    callback=self._on_noise_select,
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
        node_id: str,
        connection_list: List[Any],
        player_status_dict: Dict[str, Any],
        node_result_dict: Dict[str, Any],
    ) -> Any:
        tag_name_list: List[Any] = get_tag_name_list(
            node_id,
            self.node_tag,
            [self.TYPE_TEXT],
            [self.TYPE_SIGNAL_CHUNK, self.TYPE_TIME_MS],
        )
        output_tag_list = tag_name_list[2]

        # 計測開始
        if self._use_pref_counter:
            start_time = time.perf_counter()

        # 再生に合わせてスクロールし、チャンク設定を行う
        chunk_index = 0
        current_status = player_status_dict.get("current_status", False)
        if current_status == "play":
            if self._start_time is None:
                self._start_time = time.time() - self._paused_elapsed

            elapsed = time.time() - self._start_time
            sr = self._default_sampling_rate
            chunk_time = self._chunk_size / sr
            chunk_index = int(elapsed / chunk_time)

            # ノイズ生成
            min_buffer_len = self._chunk_size * 2
            if len(self._node_data[str(node_id)]["buffer"]) < min_buffer_len:
                noise_buffer = generate_noise(
                    self._node_data[str(node_id)]["noise_type"], duration=0.3, sr=sr
                )
                self._node_data[str(node_id)]["buffer"] = np.concatenate(
                    (self._node_data[str(node_id)]["buffer"], noise_buffer)
                )

            # ノイズを生成して、チャンク取り出し
            if self._node_data[str(node_id)]["chunk_index"] < chunk_index:
                # ノイズチャンク取り出し
                self._node_data[str(node_id)]["noise_chunk"] = self._node_data[
                    str(node_id)
                ]["buffer"][: self._chunk_size]
                # バッファの先頭を削除
                self._node_data[str(node_id)]["buffer"] = self._node_data[str(node_id)][
                    "buffer"
                ][self._chunk_size :]

                # プロット更新
                temp_display_y_buffer = self._node_data[str(node_id)][
                    "display_y_buffer"
                ]
                temp_display_y_buffer = np.roll(
                    temp_display_y_buffer, -self._chunk_size
                )
                temp_display_y_buffer[-self._chunk_size :] = self._node_data[
                    str(node_id)
                ]["noise_chunk"]
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

            # チャンクインデックス更新
            self._node_data[str(node_id)]["chunk_index"] = chunk_index

        elif current_status == "pause":
            if self._start_time is not None:
                self._paused_elapsed = time.time() - self._start_time
                self._start_time = None

        elif current_status == "stop":
            self._start_time = None
            self._paused_elapsed = 0.0
            # バッファ初期化
            self._node_data[str(node_id)]["buffer"] = np.zeros(0, dtype=np.float32)
            self._node_data[str(node_id)]["noise_chunk"] = np.array([])

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
            "chunk_index": self._node_data[str(node_id)]["chunk_index"],
            "chunk": self._node_data[str(node_id)]["noise_chunk"],
        }

        # 計測終了
        if self._use_pref_counter:
            elapsed_time = time.perf_counter() - start_time
            elapsed_time = int(elapsed_time * 1000)
            dpg_set_value(output_tag_list[1][1], str(elapsed_time).zfill(4) + "ms")

        return result_dict

    def close(self, node_id: str) -> None:
        pass

    def get_setting_dict(self, node_id: str) -> Dict[str, Any]:
        tag_name_list: List[Any] = get_tag_name_list(
            node_id,
            self.node_tag,
            [self.TYPE_TEXT],
            [self.TYPE_SIGNAL_CHUNK, self.TYPE_TIME_MS],
        )
        tag_node_name: str = tag_name_list[0]
        input_tag_list = tag_name_list[1]

        pos: List[int] = dpg.get_item_pos(tag_node_name)
        noise_type: str = dpg_get_value(input_tag_list[0][1])

        setting_dict: Dict[str, Any] = {
            "ver": self._ver,
            "pos": pos,
            "noise_type": noise_type,
        }
        return setting_dict

    def set_setting_dict(self, node_id: int, setting_dict: Dict[str, Any]) -> None:
        # タグ名
        tag_name_list: List[Any] = get_tag_name_list(
            node_id,
            self.node_tag,
            [self.TYPE_TEXT],
            [self.TYPE_SIGNAL_CHUNK, self.TYPE_TIME_MS],
        )
        input_tag_list = tag_name_list[1]

        noise_type = setting_dict["noise_type"]

        dpg_set_value(input_tag_list[0][1], noise_type)

        self._node_data[str(node_id)]["noise_type"] = noise_type
        self._node_data[str(node_id)]["buffer"] = np.zeros(0, dtype=np.float32)

    def _on_noise_select(self, sender, app_data, user_data):
        node_id = sender.split(":")[0]
        self._node_data[str(node_id)]["noise_type"] = app_data
        self._node_data[str(node_id)]["buffer"] = np.zeros(0, dtype=np.float32)
