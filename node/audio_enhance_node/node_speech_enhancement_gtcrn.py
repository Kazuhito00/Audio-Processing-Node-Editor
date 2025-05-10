#!/usr/bin/env python
# -*- coding: utf-8 -*-
import time
from typing import Any, Dict, List, Optional

import dearpygui.dearpygui as dpg  # type: ignore
import numpy as np
import onnxruntime  # type: ignore
from node.node_abc import DpgNodeABC  # type: ignore
from node_editor.util import (  # type: ignore
    dpg_set_value,
    get_tag_name_list,
)


class Node(DpgNodeABC):
    _ver = "0.0.1"

    node_label = "Speech Enhancement(GTCRN)"
    node_tag = "SpeechEnhancementGTCRN"

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
        waveform_w: int = self._setting_dict.get("waveform_width", 200)
        waveform_h: int = self._setting_dict.get("waveform_height", 400)
        self._default_sampling_rate: int = self._setting_dict.get(
            "default_sampling_rate", 16000
        )
        self._chunk_size: int = self._setting_dict.get("chunk_size", 1024)
        self._use_pref_counter: bool = self._setting_dict["use_pref_counter"]

        # モデル読み込み
        self._node_data[str(node_id)] = {
            "buffer": np.zeros(0, dtype=np.float32),
            "enhanced_chunk": np.array([]),
            "display_x_buffer": np.array([]),
            "display_y_buffer": np.array([]),
            "current_chunk_index": 0,
            "model": onnxruntime.InferenceSession(
                "node/audio_enhance_node/model/gtcrn_simple.onnx",
                providers=["CPUExecutionProvider"],
            ),
            "conv_cache": np.zeros([2, 1, 16, 16, 33], dtype=np.float32),
            "tra_cache": np.zeros([2, 3, 1, 1, 16], dtype=np.float32),
            "inter_cache": np.zeros([2, 1, 33, 16], dtype=np.float32),
            "enhanced_data_init_flag": False,
            "temp_enhanced_buffer": np.zeros(512 + 256),
            "temp_norm_buffer": np.zeros(512 + 256),
            "enhanced_buffer": np.array([]),
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
            [self.TYPE_SIGNAL_CHUNK, self.TYPE_INT],
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
            if connection_type == self.TYPE_SIGNAL_CHUNK:
                # 接続タグ取得
                source_node = ":".join(connection_info[0].split(":")[:2])
                destination_node = ":".join(connection_info[1].split(":")[:2])
                if tag_node_name == destination_node:
                    chunk_index = node_result_dict[source_node].get("chunk_index", -1)
                    chunk = node_result_dict[source_node].get("chunk", np.array([]))
                    break

        # プロット
        enhanced_chunk: Optional[np.ndarray] = np.zeros(
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
                        f"    [Warning] GTCRN Node Chunk Index Gap: {chunk_index - self._node_data[str(node_id)]['current_chunk_index']} (Index: {self._node_data[str(node_id)]['current_chunk_index']} -> {chunk_index})"
                    )

                # バッファ更新
                self._node_data[str(node_id)]["buffer"] = np.concatenate(
                    (self._node_data[str(node_id)]["buffer"], chunk)
                )

                sqrt_hanning_window = np.sqrt(np.hanning(512))  # √ハン窓
                while len(self._node_data[str(node_id)]["buffer"]) > 512:
                    temp_frame = self._node_data[str(node_id)]["buffer"][:512]

                    # STFT (1 frame)
                    frame_windowed = temp_frame * sqrt_hanning_window
                    frame_spec = np.fft.rfft(frame_windowed, n=512)
                    real = np.real(frame_spec)
                    imag = np.imag(frame_spec)
                    frame_input = np.stack([real, imag], axis=-1)[
                        None, :, None, :
                    ]  # (1, 257, 1, 2)

                    # ONNX inference
                    out_i, conv_cache, tra_cache, inter_cache = self._node_data[
                        str(node_id)
                    ]["model"].run(
                        None,
                        {
                            "mix": frame_input.astype(np.float32),
                            "conv_cache": self._node_data[str(node_id)]["conv_cache"],
                            "tra_cache": self._node_data[str(node_id)]["tra_cache"],
                            "inter_cache": self._node_data[str(node_id)]["inter_cache"],
                        },
                    )
                    self._node_data[str(node_id)]["conv_cache"] = conv_cache
                    self._node_data[str(node_id)]["tra_cache"] = tra_cache
                    self._node_data[str(node_id)]["inter_cache"] = inter_cache

                    # IRFFT
                    out_real = out_i[0][:, 0, 0]
                    out_imag = out_i[0][:, 0, 1]
                    spec_enh = out_real + 1j * out_imag
                    time_frame = np.fft.irfft(spec_enh, n=512)[:512]
                    time_frame *= sqrt_hanning_window

                    # オーバーラップ加算
                    if not self._node_data[str(node_id)]["enhanced_data_init_flag"]:
                        self._node_data[str(node_id)]["temp_enhanced_buffer"][
                            0:512
                        ] += time_frame
                        self._node_data[str(node_id)]["temp_norm_buffer"][0:512] += (
                            sqrt_hanning_window**2
                        )
                        self._node_data[str(node_id)]["enhanced_data_init_flag"] = True
                    else:
                        self._node_data[str(node_id)]["temp_enhanced_buffer"][
                            0 + 256 : 512 + 256
                        ] += time_frame
                        self._node_data[str(node_id)]["temp_norm_buffer"][
                            0 + 256 : 512 + 256
                        ] += sqrt_hanning_window**2

                        # 正規化
                        normalized_frame = np.zeros(512)
                        nonzero = self._node_data[str(node_id)]["temp_enhanced_buffer"][
                            :512
                        ]
                        denominator = self._node_data[str(node_id)]["temp_norm_buffer"][
                            :512
                        ]
                        nonzero = denominator > 1e-8
                        normalized_frame[nonzero] = (
                            self._node_data[str(node_id)]["temp_enhanced_buffer"][:512][
                                nonzero
                            ]
                            / self._node_data[str(node_id)]["temp_norm_buffer"][:512][
                                nonzero
                            ]
                        )

                        # ECバッファ更新
                        self._node_data[str(node_id)]["enhanced_buffer"] = (
                            np.concatenate(
                                [
                                    self._node_data[str(node_id)]["enhanced_buffer"],
                                    normalized_frame[:256],
                                ]
                            )
                        )

                        # 256サンプル分前に詰める
                        self._node_data[str(node_id)]["temp_enhanced_buffer"][:-256] = (
                            self._node_data[str(node_id)]["temp_enhanced_buffer"][256:]
                        )
                        self._node_data[str(node_id)]["temp_norm_buffer"][:-256] = (
                            self._node_data[str(node_id)]["temp_norm_buffer"][256:]
                        )
                        # 最後256サンプルを0埋めする
                        self._node_data[str(node_id)]["temp_enhanced_buffer"][-256:] = (
                            0.0
                        )
                        self._node_data[str(node_id)]["temp_norm_buffer"][-256:] = 0.0

                    # 先頭を256ポイントを削除
                    self._node_data[str(node_id)]["buffer"] = self._node_data[
                        str(node_id)
                    ]["buffer"][256:]

                # チャンク生成
                if (
                    len(self._node_data[str(node_id)]["enhanced_buffer"])
                    > self._chunk_size
                ):
                    enhanced_chunk = self._node_data[str(node_id)]["enhanced_buffer"][
                        : self._chunk_size
                    ]
                    self._node_data[str(node_id)]["enhanced_chunk"] = enhanced_chunk

                    # 位置更新
                    self._node_data[str(node_id)]["enhanced_buffer"] = self._node_data[
                        str(node_id)
                    ]["enhanced_buffer"][self._chunk_size :]

                    self._node_data[str(node_id)]["current_chunk_index"] += 1

                # プロット
                temp_display_y_buffer = self._node_data[str(node_id)][
                    "display_y_buffer"
                ]
                temp_display_y_buffer = np.roll(
                    temp_display_y_buffer, -self._chunk_size
                )
                temp_display_y_buffer[-self._chunk_size :] = enhanced_chunk
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
        elif current_status == "stop":
            # バッファ初期化
            self._node_data[str(node_id)]["buffer"] = np.zeros(0, dtype=np.float32)

            self._node_data[str(node_id)]["conv_cache"] = np.zeros(
                [2, 1, 16, 16, 33], dtype=np.float32
            )
            self._node_data[str(node_id)]["tra_cache"] = np.zeros(
                [2, 3, 1, 1, 16], dtype=np.float32
            )
            self._node_data[str(node_id)]["inter_cache"] = np.zeros(
                [2, 1, 33, 16], dtype=np.float32
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
            "chunk": self._node_data[str(node_id)]["enhanced_chunk"],
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
            [self.TYPE_SIGNAL_CHUNK, self.TYPE_INT],
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
