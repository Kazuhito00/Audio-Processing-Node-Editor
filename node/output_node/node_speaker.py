#!/usr/bin/env python
# -*- coding: utf-8 -*-
import queue
import time
from typing import Any, Dict, List, Optional, Tuple

import dearpygui.dearpygui as dpg  # type: ignore
import numpy as np
import sounddevice as sd  # type: ignore
from node_editor.util import dpg_set_value, get_tag_name_list  # type: ignore

from node.node_abc import DpgNodeABC  # type: ignore


class Node(DpgNodeABC):
    _ver = "0.0.1"

    node_label = "Speaker"
    node_tag = "Speaker"

    def __init__(self):
        self._default_output_id: int = sd.default.device[1]  # (input_id, output_id)

        # スピーカーリスト作成（[デバイス名, ID]のリスト）
        self._speaker_list: List[Tuple[str, int]] = []
        for output_id, device in enumerate(sd.query_devices()):
            if device["max_output_channels"] > 0:
                if output_id == self._default_output_id:
                    self._speaker_list.append([device["name"], output_id])
        for output_id, device in enumerate(sd.query_devices()):
            if device["max_output_channels"] > 0:
                if output_id != self._default_output_id:
                    self._speaker_list.append([device["name"], output_id])

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
            [self.TYPE_SIGNAL_CHUNK, self.TYPE_TEXT],
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

        # 再生用ストリーム準備
        self._node_data[str(node_id)] = {
            "audio_queue": queue.Queue(),
            "stream": sd.OutputStream(
                samplerate=self._default_sampling_rate,
                channels=1,
                blocksize=self._chunk_size,
                device=self._default_output_id,
                callback=lambda outdata,
                frames,
                time_info,
                status: self._audio_callback(
                    outdata, frames, time_info, status, node_id=str(node_id)
                ),
            ),
            "display_x_buffer": np.array([]),
            "display_y_buffer": np.array([]),
            "current_chunk_index": -1,
        }
        self._node_data[str(node_id)]["stream"].start()

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
                attribute_type=dpg.mvNode_Attr_Static,
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
            # スピーカー選択
            with dpg.node_attribute(
                tag=input_tag_list[1][0],
                attribute_type=dpg.mvNode_Attr_Static,
            ):
                speaker_names: List[str] = [name for name, _ in self._speaker_list]
                dpg.add_combo(
                    speaker_names,
                    default_value=speaker_names[0],
                    width=waveform_w,
                    tag=input_tag_list[1][1],
                    callback=self._on_speaker_select,
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
                        f"    [Warning] Speaker Node Chunk Index Gap: {chunk_index - self._node_data[str(node_id)]['current_chunk_index']} (Index: {self._node_data[str(node_id)]['current_chunk_index']} -> {chunk_index})"
                    )
                # プロット
                temp_display_y_buffer = self._node_data[str(node_id)][
                    "display_y_buffer"
                ]
                temp_display_y_buffer = np.roll(
                    temp_display_y_buffer, -self._chunk_size
                )
                temp_display_y_buffer[-self._chunk_size :] = chunk
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
                # 再生デバイス
                self._node_data[str(node_id)]["audio_queue"].put(
                    chunk.astype(np.float32)
                )
                self._node_data[str(node_id)]["current_chunk_index"] = chunk_index
        elif current_status == "stop":
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

        # 計測終了
        if self._use_pref_counter:
            elapsed_time = time.perf_counter() - start_time
            elapsed_time = int(elapsed_time * 1000)
            dpg_set_value(output_tag_list[1][1], str(elapsed_time).zfill(4) + "ms")

        return None

    def close(self, node_id):
        if self._node_data[str(node_id)]["stream"] is not None:
            if self._node_data[str(node_id)]["stream"].active:
                self._node_data[str(node_id)]["stream"].stop()
            self._node_data[str(node_id)]["stream"].close()
            self._node_data[str(node_id)]["stream"] = None

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

    def _on_speaker_select(self, sender, app_data, user_data):
        node_id = sender.split(":")[0]

        if self._node_data[str(node_id)]["stream"] is not None:
            if self._node_data[str(node_id)]["stream"].active:
                self._node_data[str(node_id)]["stream"].stop()
            self._node_data[str(node_id)]["stream"].close()
            self._node_data[str(node_id)]["stream"] = None

        for device_name, output_id in self._speaker_list:
            if device_name == app_data:
                self._node_data[str(node_id)]["stream"] = sd.OutputStream(
                    samplerate=self._default_sampling_rate,
                    channels=1,
                    blocksize=self._chunk_size,
                    device=output_id,
                    callback=lambda outdata,
                    frames,
                    time_info,
                    status: self._audio_callback(
                        outdata, frames, time_info, status, node_id=str(node_id)
                    ),
                )
                self._node_data[str(node_id)]["stream"].start()
                break

    def _audio_callback(self, outdata, frames, time_info, status, *, node_id):
        try:
            data = self._node_data[node_id]["audio_queue"].get_nowait()
            outdata[: len(data)] = data.reshape(-1, 1)
        except queue.Empty:
            outdata.fill(0)  # 無音を出力
