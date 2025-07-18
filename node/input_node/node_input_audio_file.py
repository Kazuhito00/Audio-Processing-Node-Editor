#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import time
from typing import Any, Dict, List, Optional

import dearpygui.dearpygui as dpg  # type: ignore
import librosa
import numpy as np
from node_editor.util import dpg_set_value, get_tag_name_list  # type: ignore
from node.node_abc import DpgNodeABC  # type: ignore


class Node(DpgNodeABC):
    _ver: str = "0.0.1"

    node_label: str = "Audio File"
    node_tag: str = "AudioFile"

    def __init__(self) -> None:
        self._node_data = {}
        self._setting_dict: Dict[str, Any] = {}
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

        self._node_data[str(node_id)] = {
            "buffer": np.array([]),
            "chunk": np.array([]),
        }

        # 設定
        self._setting_dict = setting_dict or {}
        small_window_w: int = self._setting_dict.get("input_window_width", 200)
        small_window_h: int = self._setting_dict.get("input_window_height", 400)
        waveform_w: int = self._setting_dict.get("waveform_width", 200)
        waveform_h: int = self._setting_dict.get("waveform_height", 400)
        self._default_sampling_rate: int = self._setting_dict.get(
            "default_sampling_rate", 16000
        )
        self._chunk_size: int = self._setting_dict.get("chunk_size", 1024)
        self._use_pref_counter: bool = self._setting_dict.get("use_pref_counter", False)

        # ファイルダイアログ
        with dpg.file_dialog(
            directory_selector=False,
            show=False,
            modal=True,
            width=int(small_window_w * 3),
            height=int(small_window_h * 3),
            callback=self._callback_file_select,
            tag=f"{node_id}:audio_file_select",
        ):
            dpg.add_file_extension(
                "Audio Files (*.wav *.mp3 *.ogg *.m4a){.wav,.mp3,.ogg,.m4a}"
            )
            dpg.add_file_extension("", color=(150, 255, 150, 255))

        # ノード
        with dpg.node(
            tag=tag_node_name,
            parent=parent,
            label=self.node_label,
            pos=pos,
        ):
            # ファイル選択ボタン
            with dpg.node_attribute(
                tag=input_tag_list[0][0],
                attribute_type=dpg.mvNode_Attr_Static,
            ):
                dpg.add_button(
                    label="Select Audio File",
                    width=waveform_w,
                    callback=lambda: dpg.show_item(f"{node_id}:audio_file_select"),
                )
                dpg.add_loading_indicator(
                    tag=f"{node_id}:audio_file_loading",
                    show=False,
                    style=0,
                    radius=3.0,
                    parent=input_tag_list[0][0],
                    color=(255, 255, 255, 255),
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
                        no_label=True,
                        no_tick_labels=True,
                        tag=f"{node_id}:xaxis",
                    )
                    dpg.add_plot_axis(
                        dpg.mvYAxis,
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
        tag_name_list = get_tag_name_list(
            node_id,
            self.node_tag,
            [self.TYPE_TEXT],
            [self.TYPE_SIGNAL_CHUNK, self.TYPE_TIME_MS],
        )
        output_tag_list = tag_name_list[2]

        # 計測開始
        if self._use_pref_counter:
            start_time = time.perf_counter()

        chunk = np.array([])
        chunk_index = 0
        current_status = player_status_dict.get("current_status", "stop")

        # 再生に合わせてスクロールし、チャンク取り出しを行う
        if current_status == "play":
            if self._start_time is None:
                self._start_time = time.time() - self._paused_elapsed

            elapsed = time.time() - self._start_time
            sr = self._default_sampling_rate
            chunk_time = self._chunk_size / sr
            chunk_index = int(elapsed / chunk_time)

            start = chunk_index * self._chunk_size
            end = start + self._chunk_size

            buffer = self._node_data[str(node_id)]["buffer"]
            # チャンクサイズより短い場合0パディング
            if end > len(buffer):
                buffer = np.pad(buffer, (0, end - len(buffer)))

            # 終了位置がバッファを越えた場合終了
            if len(buffer) <= end:
                player_status_dict["current_status"] = "stop"

            chunk = buffer[start:end]
            self._node_data[str(node_id)]["chunk"] = chunk

            buffer_with_offset = self._node_data[str(node_id)].get(
                "buffer_with_offset", None
            )
            if buffer_with_offset is not None:
                # temp_start_time = time.perf_counter()

                offset = int(sr * 5)
                plot_start = int(elapsed * sr)
                plot_end = plot_start + offset

                y_display = buffer_with_offset[plot_start:plot_end]
                x_base = self._node_data[str(node_id)]["x_display"]
                x_display = x_base + (plot_start / sr)

                # print(int((time.perf_counter() - temp_start_time) * 1000))

                dpg.set_value(f"{node_id}:audio_line_series", [x_display, y_display])
                dpg.set_axis_limits(f"{node_id}:xaxis", x_display[0], x_display[-1])

        elif current_status == "pause":
            if self._start_time is not None:
                self._paused_elapsed = time.time() - self._start_time
                self._start_time = None
        elif current_status == "stop":
            self._start_time = None
            self._paused_elapsed = 0.0

            # スライド表示のため、先頭に5秒の空白を常に足して表示
            duration_sec = 5
            offset = int(self._default_sampling_rate * duration_sec)
            y_display = np.zeros(offset)
            x_display = np.arange(0.0, 5.0) / self._default_sampling_rate

            dpg.set_axis_limits(f"{node_id}:xaxis", 0.0, 5.0)
            dpg.set_value(f"{node_id}:audio_line_series", [x_display, y_display])

        # 計測終了
        if self._use_pref_counter:
            elapsed_time = time.perf_counter() - start_time
            elapsed_time = int(elapsed_time * 1000)
            dpg_set_value(output_tag_list[1][1], f"{elapsed_time:04d}ms")

        result_dict = {
            "chunk_index": chunk_index,
            "chunk": self._node_data[str(node_id)]["chunk"],
        }

        return result_dict

    def _callback_file_select(self, sender: int, data: Dict[str, Any]) -> None:
        node_id = sender.split(":")[0]

        # ノードラベル名更新
        tag_name_list = get_tag_name_list(
            node_id,
            self.node_tag,
            [self.TYPE_TEXT],
            [self.TYPE_SIGNAL_CHUNK, self.TYPE_TIME_MS],
        )
        tag_node_name = tag_name_list[0]
        base_filename = data['file_name']
        filename_without_ext, file_extension = os.path.splitext(base_filename)

        display_name = base_filename
        if len(filename_without_ext) >= 23:
            display_name = f"{filename_without_ext[:20]}...{file_extension}"

        dpg.set_item_label(tag_node_name, f"{self.node_label} ({display_name})")

        # ローディングアイコン
        loading_tag = f"{node_id}:audio_file_loading"
        if dpg.does_item_exist(loading_tag):
            dpg.configure_item(loading_tag, show=True)

        # 読み込み
        y, sr = librosa.load(data["file_path_name"], sr=self._default_sampling_rate)
        self._node_data[str(node_id)]["buffer"] = y
        self._node_data[str(node_id)]["sr"] = sr

        offset = int(sr * 5)
        buffer_with_offset = np.concatenate([np.zeros(offset), y])
        self._node_data[str(node_id)]["buffer_with_offset"] = buffer_with_offset
        self._node_data[str(node_id)]["x_display"] = np.arange(offset) / sr

        # 初期表示は空白
        y_display = np.zeros(offset)
        x_display = self._node_data[str(node_id)]["x_display"]
        dpg.set_value(f"{node_id}:audio_line_series", [x_display, y_display])
        dpg.set_axis_limits(f"{node_id}:xaxis", 0, 5)
        dpg.set_axis_limits(f"{node_id}:yaxis", -1.0, 1.0)

        if dpg.does_item_exist(loading_tag):
            dpg.configure_item(loading_tag, show=False)

    def close(self, node_id: str) -> None:
        pass

    def get_setting_dict(self, node_id: str) -> Dict[str, Any]:
        tag_name_list = get_tag_name_list(
            node_id,
            self.node_tag,
            [self.TYPE_TEXT],
            [self.TYPE_SIGNAL_CHUNK, self.TYPE_TIME_MS],
        )
        tag_node_name: str = tag_name_list[0]
        pos: List[int] = dpg.get_item_pos(tag_node_name)
        return {"ver": self._ver, "pos": pos}

    def set_setting_dict(self, node_id: int, setting_dict: Dict[str, Any]) -> None:
        pass
