#!/usr/bin/env python
# -*- coding: utf-8 -*-
import time
from typing import Any, Dict, List, Optional

import cv2  # type: ignore
import dearpygui.dearpygui as dpg  # type: ignore
import numpy as np
from node.node_abc import DpgNodeABC  # type: ignore
from node_editor.util import (  # type: ignore
    convert_cv_to_dpg,
    dpg_set_value,
    get_tag_name_list,
)


class Node(DpgNodeABC):
    _ver = "0.0.1"

    node_label = "Simple Spectrogram"
    node_tag = "Spectrogram"

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

        self._spectrogram_window_size: int = self._setting_dict.get(
            "spectrogram_window_size", 512
        )
        self._spectrogram_hop_size: int = self._setting_dict.get(
            "spectrogram_hop_size", 128
        )
        self._spectrogram_window_name: int = self._setting_dict.get(
            "spectrogram_window", "hamming"
        )
        self._spectrogram_scaling = self._setting_dict.get(
            "spectrogram_scaling", [-80, 0]
        )
        self._spectrogram_smooth_order = self._setting_dict.get(
            "spectrogram_smooth_order", 5
        )

        # 保持用バッファなど
        self._node_data[str(node_id)] = {
            "current_chunk_index": -1,
            "spectrogram": np.full(
                (
                    self._spectrogram_window_size // 2 + 1,
                    int(
                        (self._default_sampling_rate * 5) // self._spectrogram_hop_size
                    ),
                ),
                fill_value=255,
                dtype=np.uint8,
            ),
            "spec_history": [],
            "frame_buffer": np.zeros(0, dtype=np.float32),
        }

        # 初期化用画像
        temp_image = np.full(
            (self._waveform_h, self._waveform_w, 3),
            fill_value=(128, 0, 0),
            dtype=np.uint8,
        )
        temp_texture = convert_cv_to_dpg(
            temp_image,
            self._waveform_w,
            self._waveform_h,
        )

        # テクスチャ登録
        with dpg.texture_registry(show=False):
            dpg.add_raw_texture(
                self._waveform_w,
                self._waveform_h,
                temp_texture,
                tag=output_tag_list[0][1],
                format=dpg.mvFormat_Float_rgb,
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
            # 描画エリア
            with dpg.node_attribute(
                tag=output_tag_list[0][0],
                attribute_type=dpg.mvNode_Attr_Output,
            ):
                dpg.add_image(output_tag_list[0][1])
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
                        f"    [Warning] Spectrogram Node Chunk Index Gap: {chunk_index - self._node_data[str(node_id)]['current_chunk_index']} (Index: {self._node_data[str(node_id)]['current_chunk_index']} -> {chunk_index})"
                    )

                # スペクトログラム
                n_fft = self._spectrogram_window_size
                hop_size = self._spectrogram_hop_size
                sampling_rate = self._default_sampling_rate

                frame_buffer = self._node_data[str(node_id)]["frame_buffer"]
                frame_buffer = np.concatenate([frame_buffer, chunk])

                # フレーム処理
                while len(frame_buffer) >= n_fft:
                    frame = frame_buffer[:n_fft]

                    # 窓関数 + FFT
                    window = np.hamming(n_fft)
                    spectrum = np.fft.rfft(frame * window)
                    magnitude = np.abs(spectrum)

                    # dBスケーリング: 20 * log10(|A| / max(|A|))
                    ref = np.max(magnitude)
                    if ref < 1e-3:
                        db = np.full_like(magnitude, -80.0)
                    else:
                        db = 20 * np.log10(np.maximum(magnitude, 1e-10) / ref)
                        db = np.clip(db, -80.0, 0.0)

                    # カラーマップ用のスケーリング（0〜255）
                    column = np.interp(db, [-80, 0], [255, 0]).astype(np.uint8)

                    # 平滑化（任意）
                    spec_history = self._node_data[str(node_id)]["spec_history"]
                    spec_history.append(column)
                    if len(spec_history) > self._spectrogram_smooth_order:
                        spec_history.pop(0)
                    column = np.mean(spec_history, axis=0).astype(np.uint8)
                    self._node_data[str(node_id)]["spec_history"] = spec_history

                    # スペクトログラムバッファ更新
                    spec_buf = self._node_data[str(node_id)]["spectrogram"]
                    spec_buf = np.roll(spec_buf, -1, axis=1)
                    spec_buf[:, -1] = column
                    self._node_data[str(node_id)]["spectrogram"] = spec_buf

                    # バッファを進める
                    frame_buffer = frame_buffer[hop_size:]

                # バッファ保存
                self._node_data[str(node_id)]["frame_buffer"] = frame_buffer

                # 表示する周波数帯域を制限
                f_max = 8000
                k_max = int((f_max * n_fft) / sampling_rate)

                # 最新のスペクトログラムを取得（上下反転して0Hzを下に）
                display_buf = self._node_data[str(node_id)]["spectrogram"]
                display_buf = display_buf[display_buf.shape[0] - k_max :, :]
                display_buf = np.flipud(display_buf)

                # OpenCVでリサイズ＆カラー変換
                display_buf = cv2.resize(
                    display_buf, dsize=(self._waveform_w, self._waveform_h)
                )
                color_img = cv2.applyColorMap(display_buf, cv2.COLORMAP_JET)
                color_img = cv2.cvtColor(color_img, cv2.COLOR_BGR2RGB)

                # 描画
                texture = convert_cv_to_dpg(
                    color_img,
                    self._waveform_w,
                    self._waveform_h,
                )
                dpg_set_value(output_tag_list[0][1], texture)

                self._node_data[str(node_id)]["current_chunk_index"] = chunk_index
        elif current_status == "stop":
            self._node_data[str(node_id)]["current_chunk_index"] = -1

            self._node_data[str(node_id)]["spectrogram"] = np.full(
                (
                    512 // 2 + 1,
                    int((self._default_sampling_rate * 5) // 128),
                ),
                fill_value=255,
                dtype=np.uint8,
            )

            temp_image = np.full(
                (self._waveform_h, self._waveform_w, 3),
                fill_value=(128, 0, 0),
                dtype=np.uint8,
            )
            temp_texture = convert_cv_to_dpg(
                temp_image,
                self._waveform_w,
                self._waveform_h,
            )
            dpg_set_value(output_tag_list[0][1], temp_texture)

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
