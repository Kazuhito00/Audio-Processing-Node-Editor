#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import tempfile
import time
from typing import Any, Dict, List, Optional

import dearpygui.dearpygui as dpg  # type: ignore
import numpy as np
import soundfile as sf
from moviepy import VideoFileClip
from node_editor.util import dpg_set_value, get_tag_name_list  # type: ignore
from scipy.signal import resample

from node.node_abc import DpgNodeABC  # type: ignore


class Node(DpgNodeABC):
    _ver: str = "0.0.1"

    node_label: str = "Video File"
    node_tag: str = "VideoFile"

    def __init__(self) -> None:
        self._node_data = {}
        self._setting_dict: Dict[str, Any] = {}
        self._start_time: Optional[float] = None
        self._paused_elapsed: float = 0.0
        self._default_sampling_rate: int = 16000

    def close(self, node_id: str) -> None:
        clip = self._node_data[str(node_id)].get("clip")
        if clip:
            clip.close()
        if str(node_id) in self._node_data:
            del self._node_data[str(node_id)]

    def get_setting_dict(self, node_id: str) -> Dict[str, Any]:
        return self._setting_dict

    def set_setting_dict(self, node_id: str, setting_dict: Dict[str, Any]) -> None:
        self._setting_dict.update(setting_dict)

    def _update_audio_display(self, node_id: str) -> None:
        full_audio_buffer = self._node_data[str(node_id)].get("audio_buffer", np.array([]))
        sr = self._node_data[str(node_id)].get("sr", self._default_sampling_rate)
        audio_duration = len(full_audio_buffer) / sr

        plot_duration = 5.0  # 5秒間の表示ウィンドウ

        # For initial display, the plot should show a 5-second blank leading section
        # and the audio should start appearing from the right edge of this window.
        # So, the plot range should be from -plot_duration to 0.0 (or audio_duration if it's shorter than 5s)
        
        # The right edge of the plot should be 0.0 (initial playback time)
        plot_end_time = 0.0
        # The left edge of the plot should be 5.0 seconds before the right edge
        plot_start_time = plot_end_time - plot_duration

        # Calculate sample indices for the full_audio_buffer
        # If plot_start_time is negative, the actual_start_sample will be 0
        actual_start_sample = int(max(0.0, plot_start_time) * sr)
        actual_end_sample = int(min(audio_duration, plot_end_time) * sr) # Ensure we don't go beyond audio_duration

        y_display_raw = full_audio_buffer[actual_start_sample:actual_end_sample]

        # Calculate how many leading zeros are needed for the blank space
        leading_zeros_samples = 0
        if plot_start_time < 0:
            leading_zeros_samples = int(abs(plot_start_time) * sr)
        
        # Pad y_display with leading zeros if necessary
        y_display = np.pad(y_display_raw, (leading_zeros_samples, 0), 'constant')

        # If the total length is still less than expected, pad with trailing zeros
        expected_display_samples = int(plot_duration * sr)
        if len(y_display) < expected_display_samples:
            y_display = np.pad(y_display, (0, expected_display_samples - len(y_display)), 'constant')
        
        # Trim if it's longer than expected (shouldn't happen with correct padding logic)
        y_display = y_display[:expected_display_samples]

        # Calculate x_display values
        # x_display should start from plot_start_time
        x_display = np.arange(len(y_display)) / sr + plot_start_time

        if len(y_display) > 0:
            dpg.set_value(
                f"{node_id}:audio_line_series",
                [x_display.tolist(), y_display.tolist()],
            )
        else:
            dpg.set_value(f"{node_id}:audio_line_series", [[], []])

        dpg.set_axis_limits(f"{node_id}:xaxis", plot_start_time, plot_end_time)
        dpg.set_axis_limits(f"{node_id}:yaxis", -1.0, 1.0)

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
        self._default_sampling_rate = self._setting_dict.get(
            "default_sampling_rate", 16000
        )
        self._chunk_size: int = self._setting_dict.get("chunk_size", 2048)
        self._use_pref_counter: bool = self._setting_dict.get("use_pref_counter", False)

        # ファイルダイアログ
        with dpg.file_dialog(
            directory_selector=False,
            show=False,
            modal=True,
            width=int(small_window_w * 3),
            height=int(small_window_h * 3),
            callback=self._callback_file_select,
            tag=f"{node_id}:video_file_select",
        ):
            dpg.add_file_extension("Video Files (*.mp4 *.avi *.webm){.mp4,.avi,.webm}")
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
                    label="Select Video File",
                    width=waveform_w,
                    callback=lambda: dpg.show_item(f"{node_id}:video_file_select"),
                )
                # 初期表示用のプレースホルダーテクスチャを作成
                placeholder_texture_tag = f"{node_id}:video_texture_placeholder"
                if not dpg.does_item_exist(placeholder_texture_tag):
                    with dpg.texture_registry(show=False):
                        # 1x1の黒いテクスチャ (RGBA)
                        dpg.add_raw_texture(
                            1,
                            1,
                            np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32).flatten(),
                            format=dpg.mvFormat_Float_rgba,
                            tag=placeholder_texture_tag,
                        )

                dpg.add_image(
                    texture_tag=placeholder_texture_tag,  # 最初はプレースホルダーテクスチャを使用
                    tag=f"{node_id}:video_preview",
                    width=waveform_w,
                    height=int(waveform_h / 2),
                    uv_min=[0, 0],
                    uv_max=[1, 1],
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

                dpg.add_loading_indicator(
                    tag=f"{node_id}:audio_file_loading",
                    show=False,
                    style=0,
                    radius=3.0,
                    parent=output_tag_list[0][0],
                    color=(255, 255, 255, 255),
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

        elapsed = 0.0  # elapsedを初期化
        if current_status == "play":
            if self._start_time is None:
                self._start_time = time.time() - self._paused_elapsed
            elapsed = time.time() - self._start_time
        elif current_status == "pause":
            elapsed = self._paused_elapsed  # ポーズ時の経過時間を使用

        # 再生に合わせてスクロールし、チャンク取り出しを行う
        if current_status == "play":
            # 動画フレームの更新
            clip = self._node_data[str(node_id)].get("clip")
            texture_tag = f"{node_id}:video_texture"

            if clip:
                # 描画負荷軽減のため、一定間隔でフレームを更新
                frame_update_interval = (
                    1.0 / clip.fps
                )  # 動画のFPSに基づいて更新間隔を設定
                if not hasattr(self, "_last_frame_update_time"):
                    self._last_frame_update_time = 0.0

                if (
                    time.time() - self._last_frame_update_time
                ) >= frame_update_interval:
                    self._last_frame_update_time = time.time()

                    try:
                        # 現在の経過時間に基づいてフレームを取得
                        frame = clip.get_frame(elapsed)
                        # MoviePyのフレームはRGB、DearPyGuiはRGBAを期待するため変換
                        frame_rgb = frame.astype(np.float32) / 255.0
                        # Add alpha channel (1.0 for opaque)
                        alpha_channel = np.ones(
                            (*frame_rgb.shape[:2], 1), dtype=np.float32
                        )
                        texture_data = np.concatenate(
                            (frame_rgb, alpha_channel), axis=-1
                        )
                        texture_data = texture_data.flatten()

                        video_width = self._node_data[str(node_id)].get("video_width")
                        video_height = self._node_data[str(node_id)].get("video_height")
                        aspect_ratio = video_width / video_height

                        display_width = self._setting_dict.get(
                            "waveform_width", 200
                        )  # 波形の幅に合わせる
                        display_height = int(
                            display_width / aspect_ratio
                        )  # アスペクト比を維持して高さを調整

                        dpg.set_value(texture_tag, texture_data)
                        dpg.configure_item(
                            f"{node_id}:video_preview",
                            width=display_width,
                            height=display_height,
                        )
                    except Exception as e:
                        print(f"[WARNING] Could not get video frame at {elapsed}s: {e}")
                        player_status_dict["current_status"] = "stop"
            else:
                # クリップがない場合、再生を停止
                player_status_dict["current_status"] = "stop"

            sr = self._default_sampling_rate
            chunk_time = self._chunk_size / sr
            chunk_index = int(elapsed / chunk_time)

            full_audio_buffer = self._node_data[str(node_id)][
                "audio_buffer"
            ]  # 表示用の全処理済みオーディオデータ

            # --- 出力用のチャンク抽出 ---
            chunk_start = chunk_index * self._chunk_size
            chunk_end = chunk_start + self._chunk_size

            # 全オーディオバッファからチャンクを抽出
            current_chunk = full_audio_buffer[chunk_start:chunk_end]

            # チャンクが期待より短い場合、ゼロでパディング
            if len(current_chunk) < self._chunk_size:
                current_chunk = np.pad(
                    current_chunk, (0, self._chunk_size - len(current_chunk))
                )
                # 実際のオーディオの終わりに達した場合、再生を停止
                if chunk_end >= len(full_audio_buffer):
                    player_status_dict["current_status"] = "stop"

            self._node_data[str(node_id)]["chunk"] = current_chunk
            chunk = current_chunk  # 返される'chunk'変数に割り当て

            # --- 波形表示の更新 ---
            plot_duration = 5.0  # 5秒間の表示ウィンドウ

            # The current playback time 'elapsed' should be at the right edge of the plot
            plot_end_time = elapsed
            plot_start_time = elapsed - plot_duration

            # Calculate sample indices for the full_audio_buffer
            sr = self._default_sampling_rate
            
            # Calculate the number of samples to display
            expected_display_samples = int(plot_duration * sr)

            # Determine the actual start and end samples from the audio buffer
            # If plot_start_time is negative, the actual_start_sample will be 0
            actual_start_sample = int(max(0.0, plot_start_time) * sr)
            actual_end_sample = int(plot_end_time * sr)

            y_display_raw = full_audio_buffer[actual_start_sample:actual_end_sample]

            # Calculate how many leading zeros are needed for the blank space
            leading_zeros_samples = 0
            if plot_start_time < 0:
                leading_zeros_samples = int(abs(plot_start_time) * sr)
            
            # Pad y_display with leading zeros if necessary
            y_display = np.pad(y_display_raw, (leading_zeros_samples, 0), 'constant')

            # If the total length is still less than expected, pad with trailing zeros
            if len(y_display) < expected_display_samples:
                y_display = np.pad(y_display, (0, expected_display_samples - len(y_display)), 'constant')
            
            # Trim if it's longer than expected (shouldn't happen with correct padding logic)
            y_display = y_display[:expected_display_samples]

            # Calculate x_display values
            # x_display should start from plot_start_time
            x_display = np.arange(len(y_display)) / sr + plot_start_time

            if len(y_display) > 0:
                dpg.set_value(
                    f"{node_id}:audio_line_series",
                    [x_display.tolist(), y_display.tolist()],
                )
            else:
                dpg.set_value(f"{node_id}:audio_line_series", [[], []])
            dpg.set_axis_limits(f"{node_id}:xaxis", plot_start_time, plot_end_time)

        elif current_status == "pause":
            if self._start_time is not None:
                self._paused_elapsed = time.time() - self._start_time
                self._start_time = None
        elif current_status == "stop":
            self._start_time = None
            self._paused_elapsed = 0.0

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
        video_path = data["file_path_name"]

        # ノードラベル名更新
        tag_name_list = get_tag_name_list(
            node_id,
            self.node_tag,
            [self.TYPE_TEXT],
            [self.TYPE_SIGNAL_CHUNK, self.TYPE_TIME_MS],
        )
        tag_node_name = tag_name_list[0]
        base_filename = os.path.basename(video_path)
        filename_without_ext, file_extension = os.path.splitext(base_filename)

        display_name = base_filename
        if len(filename_without_ext) >= 23:
            display_name = f"{filename_without_ext[:20]}...{file_extension}"

        dpg.set_item_label(
            tag_node_name, f"{self.node_label} ({display_name})"
        )

        # ローディングアイコン表示
        loading_tag = f"{node_id}:audio_file_loading"
        if dpg.does_item_exist(loading_tag):
            dpg.configure_item(loading_tag, show=True)

        try:
            # 既存のクリップがあれば閉じる
            existing_clip = self._node_data[str(node_id)].get("clip")
            if existing_clip:
                existing_clip.close()

            clip = VideoFileClip(video_path)
            self._node_data[str(node_id)]["clip"] = clip
            self._node_data[str(node_id)]["duration"] = clip.duration
            self._node_data[str(node_id)]["fps"] = clip.fps
            self._node_data[str(node_id)]["video_width"] = clip.w
            self._node_data[str(node_id)]["video_height"] = clip.h

            # オーディオの抽出とバッファリング
            audio_clip = clip.audio
            if audio_clip:
                # 一時ファイルにオーディオを書き出す
                temp_audio_file = tempfile.NamedTemporaryFile(
                    suffix=".wav", delete=False
                )
                temp_audio_file.close()
                audio_clip.write_audiofile(
                    temp_audio_file.name, fps=self._default_sampling_rate
                )

                # soundfileでオーディオを読み込む
                audio_array, original_sr = sf.read(temp_audio_file.name)
                os.unlink(temp_audio_file.name)  # 一時ファイルを削除

                # ここでのモノラル変換は不要になるが、念のため残しておく
                if audio_array.ndim == 2:
                    audio_array = np.mean(audio_array, axis=1)

                # リサンプリングは不要になるはずだが、念のため条件分岐は残しておく
                if original_sr != self._default_sampling_rate:
                    num_samples = int(
                        len(audio_array) * self._default_sampling_rate / original_sr
                    )
                    audio_array = resample(audio_array, num_samples)

                # 正規化と増幅
                abs_max = np.max(np.abs(audio_array))
                if abs_max > 1e-6:
                    audio_array = (
                        audio_array / abs_max * 0.8
                    )  # -1.0から1.0に正規化し、80%に増幅

                self._node_data[str(node_id)]["audio_buffer"] = audio_array
                self._node_data[str(node_id)]["sr"] = self._default_sampling_rate
            else:
                self._node_data[str(node_id)]["audio_buffer"] = np.array([])
                self._node_data[str(node_id)]["sr"] = self._default_sampling_rate

            # UIの更新
            self._update_ui_after_loading(node_id, video_path)

        except Exception as e:
            print(f"[ERROR] Failed to load video or audio: {e}")
            if dpg.does_item_exist(loading_tag):
                dpg.configure_item(loading_tag, show=False)
            # エラー時はプレースホルダーに戻す
            dpg.configure_item(
                f"{node_id}:video_preview",
                texture_tag=f"{node_id}:video_texture_placeholder",
                width=self._setting_dict.get("waveform_width", 200),
                height=int(self._setting_dict.get("waveform_height", 400) / 2),
            )
            dpg.configure_item(
                f"{node_id}:audio_channel_combo", items=[], default_value="None"
            )
            dpg.set_value(f"{node_id}:audio_line_series", [[], []])
            dpg.set_axis_limits(f"{node_id}:xaxis", 0.0, 5.0)
            dpg.set_axis_limits(f"{node_id}:yaxis", -1.0, 1.0)
        finally:
            if dpg.does_item_exist(loading_tag):
                dpg.configure_item(loading_tag, show=False)

    def _update_ui_after_loading(self, node_id: str, video_path: str) -> None:
        clip = self._node_data[str(node_id)].get("clip")
        if not clip:
            return

        # 動画の初期フレームをテクスチャとして設定
        first_frame = clip.get_frame(0)  # 最初のフレームを取得
        # MoviePyのフレームはRGB、DearPyGuiはRGBAを期待するため変換
        frame_rgb = first_frame.astype(np.float32) / 255.0
        # Add alpha channel (1.0 for opaque)
        alpha_channel = np.ones((*frame_rgb.shape[:2], 1), dtype=np.float32)
        texture_data = np.concatenate((frame_rgb, alpha_channel), axis=-1)
        texture_data = texture_data.flatten()

        video_width = clip.w
        video_height = clip.h
        texture_tag = f"{node_id}:video_texture"
        placeholder_texture_tag = f"{node_id}:video_texture_placeholder"

        # Before deleting, ensure the image widget is not using the texture
        if dpg.does_item_exist(f"{node_id}:video_preview"):
            dpg.configure_item(
                f"{node_id}:video_preview", texture_tag=placeholder_texture_tag
            )

        if dpg.does_item_exist(texture_tag):
            dpg.delete_item(texture_tag)

        with dpg.texture_registry(show=False):
            dpg.add_raw_texture(
                video_width,
                video_height,
                texture_data,
                format=dpg.mvFormat_Float_rgba,
                tag=texture_tag,
            )

        # アスペクト比を維持して表示サイズを計算
        display_width = self._setting_dict.get("waveform_width", 200)
        aspect_ratio = video_width / video_height
        display_height = int(display_width / aspect_ratio)

        dpg.configure_item(
            f"{node_id}:video_preview",
            texture_tag=texture_tag,
            width=display_width,
            height=display_height,
        )

        # Clear the audio line series and reset axis limits for initial blank display
        dpg.set_value(f"{node_id}:audio_line_series", [[], []])
        dpg.set_axis_limits(f"{node_id}:xaxis", 0.0, 5.0)

        loading_tag = f"{node_id}:audio_file_loading"
        if dpg.does_item_exist(loading_tag):
            dpg.configure_item(loading_tag, show=False)
