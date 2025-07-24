#!/usr/bin/env python
# -*- coding: utf-8 -*-
import datetime
import os
import queue
import textwrap
import threading
import time
from typing import Any, Dict, List, Optional

import dearpygui.dearpygui as dpg  # type: ignore
import numpy as np

from node.node_abc import DpgNodeABC  # type: ignore
from node_editor.util import dpg_set_value, get_tag_name_list  # type: ignore


class Node(DpgNodeABC):
    _ver = "0.0.1"

    node_label = "Speech Recognition(Google Speech-to-Text)"
    node_tag = "SpeechRecognitionGoogleSTT"

    def __init__(self):
        self._node_data = {}

        self._audio_queue: Dict[int, queue.Queue] = {}
        self._stt_running_flag: Dict[int, bool] = {}
        self._node_languages: Dict[int, str] = {}

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
            [self.TYPE_TEXT, self.TYPE_TIME_MS],
        )
        tag_node_name: str = tag_name_list[0]
        input_tag_list = tag_name_list[1]
        output_tag_list = tag_name_list[2]

        # 設定
        self._setting_dict = setting_dict or {}
        small_window_w: int = self._setting_dict.get("waveform_width", 200)
        small_window_h: int = self._setting_dict.get("waveform_height", 400)
        self._default_sampling_rate: int = self._setting_dict.get(
            "default_sampling_rate", 16000
        )
        self._chunk_size: int = self._setting_dict.get("chunk_size", 1024)
        self._use_pref_counter: bool = self._setting_dict["use_pref_counter"]

        self._google_application_credentials_json: str = self._setting_dict[
            "google_application_credentials_json"
        ]

        # 言語設定を保持
        self._language_codes = {"日本語": "ja-JP", "English": "en-US"}

        # 指定なしの場合はノード追加無し
        if not self._google_application_credentials_json or not os.path.isfile(
            self._google_application_credentials_json
        ):
            print(
                "    [Warning] STT Node:A valid credentials JSON file must be specified to use this node."
            )
            return None

        # Googleクレデンシャル設定
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = (
            self._google_application_credentials_json
        )

        # 言語選択用のタグ名を生成
        language_combo_tag = f"{node_id}:{self.node_tag}:language_combo"

        # ノードごとの言語設定を保持（保存された設定があればそれを使用）
        saved_language = (
            setting_dict.get("selected_language", "日本語")
            if setting_dict
            else "日本語"
        )
        self._node_languages[node_id] = saved_language

        self._stt_running_flag[node_id] = True

        # 音声認識準備
        def generator(node_id: int):
            from google.cloud import speech

            timeout_time = self._chunk_size / self._default_sampling_rate * 2
            silent_chunk = np.zeros(self._chunk_size, dtype=np.int16).tobytes()

            while self._stt_running_flag:
                try:
                    item = self._audio_queue[str(node_id)].get(timeout=timeout_time)
                except queue.Empty:
                    yield speech.StreamingRecognizeRequest(audio_content=silent_chunk)
                    continue

                if item is None:
                    break
                recv_node_id, chunk_bytes = item
                if int(recv_node_id) != node_id:
                    continue
                yield speech.StreamingRecognizeRequest(audio_content=chunk_bytes)

        def run_stt(node_id: int):
            from google.cloud import speech

            # タグ名
            tag_name_list: List[Any] = get_tag_name_list(
                node_id,
                self.node_tag,
                [self.TYPE_SIGNAL_CHUNK],
                [self.TYPE_TEXT, self.TYPE_TIME_MS],
            )
            output_tag_list = tag_name_list[2]

            # 現在選択されている言語を取得
            selected_language = self._node_languages.get(node_id, "日本語")
            language_code = self._language_codes.get(selected_language, "ja-JP")

            # 言語に応じたSTT設定を作成
            stt_config = speech.RecognitionConfig(
                encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
                sample_rate_hertz=self._default_sampling_rate,
                language_code=language_code,
                use_enhanced=True,
                model="default",
            )

            streaming_config = speech.StreamingRecognitionConfig(
                config=stt_config,
                interim_results=True,
            )

            speech_client = self._node_data[str(node_id)]["speech_client"]
            responses = speech_client.streaming_recognize(
                streaming_config, generator(node_id)
            )

            start_time = datetime.datetime.now()

            for response in responses:
                try:
                    # 最大ストリーム時間（Googleの仕様: 305秒）を超えたら終了
                    if (datetime.datetime.now() - start_time).total_seconds() > 300:
                        print("🔁 [STT] Reconnecting after 300 seconds")
                        break

                    for result in response.results:
                        transcript = result.alternatives[0].transcript
                        # 言語に応じて折り返し文字数を変更
                        wrap_width = 20 if selected_language == "日本語" else 40
                        wrapped_text = "\n".join(textwrap.wrap(transcript, wrap_width))
                        dpg_set_value(output_tag_list[0][1], wrapped_text)
                except Exception as e:
                    print(f"[STT Error] during response loop: {e}")
                    break

        def loop_run_stt(node_id: int):
            from google.cloud import speech

            while self._stt_running_flag.get(node_id, False):
                try:
                    # 新しいクライアントを生成（セッションごとに）
                    speech_client = speech.SpeechClient()
                    self._node_data[str(node_id)]["speech_client"] = speech_client
                    run_stt(node_id)
                except Exception as e:
                    print(f"[Error] STT loop error: {e}")
                time.sleep(1)  # クールダウンして再接続

        self._node_data[str(node_id)] = {
            "current_chunk_index": -1,
            "speech_client": None,
            "stt_thread": threading.Thread(
                target=loop_run_stt, args=(node_id,), daemon=True
            ),
        }
        self._node_data[str(node_id)]["stt_thread"].start()

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
            # テキストエリア
            if self._use_pref_counter:
                with dpg.node_attribute(
                    tag=output_tag_list[0][0],
                    attribute_type=dpg.mvNode_Attr_Output,
                ):
                    dpg.add_input_text(
                        tag=output_tag_list[0][1],
                        multiline=True,
                        width=small_window_w,
                        height=small_window_h,
                        readonly=False,
                        default_value="",
                    )

            # 言語選択ドロップダウン
            with dpg.node_attribute(
                tag=f"{node_id}:{self.node_tag}:language_attr",
                attribute_type=dpg.mvNode_Attr_Static,
            ):
                dpg.add_combo(
                    tag=language_combo_tag,
                    items=list(self._language_codes.keys()),
                    default_value=saved_language,
                    width=small_window_w,
                    callback=lambda s, a, u: self._on_language_change(u[0], a),
                    user_data=(node_id,),
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
            [self.TYPE_SIGNAL_CHUNK],
            [self.TYPE_TEXT, self.TYPE_TIME_MS],
        )
        output_tag_list = tag_name_list[2]

        # 計測開始
        if self._use_pref_counter:
            start_time = time.perf_counter()

        if node_id not in self._audio_queue:
            self._audio_queue[str(node_id)] = queue.Queue()

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

        # 音声認識
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
                        f"    [Warning] STT Node Chunk Index Gap: {chunk_index - self._node_data[str(node_id)]['current_chunk_index']} (Index: {self._node_data[str(node_id)]['current_chunk_index']} -> {chunk_index})"
                    )

                # chunk（numpy）→ bytes に変換
                chunk_bytes = (chunk * 32767).astype(np.int16).tobytes()
                self._audio_queue[str(node_id)].put((node_id, chunk_bytes))

                self._node_data[str(node_id)]["current_chunk_index"] = chunk_index
        elif current_status == "stop":
            self._node_data[str(node_id)]["current_chunk_index"] = -1

        # 計測終了
        if self._use_pref_counter:
            elapsed_time = time.perf_counter() - start_time
            elapsed_time = int(elapsed_time * 1000)
            dpg_set_value(output_tag_list[1][1], str(elapsed_time).zfill(4) + "ms")

        return None

    def close(self, node_id):
        self._stt_running_flag[node_id] = False
        if str(node_id) in self._audio_queue:
            self._audio_queue[str(node_id)].put(None)  # ジェネレーター終了

    def _on_language_change(self, node_id, selected_language):
        """言語が変更されたときの処理"""
        self._node_languages[node_id] = selected_language

    def get_setting_dict(self, node_id):
        # タグ名
        tag_name_list: List[Any] = get_tag_name_list(
            node_id,
            self.node_tag,
            [self.TYPE_SIGNAL_CHUNK],
            [self.TYPE_TEXT, self.TYPE_TIME_MS],
        )
        tag_node_name: str = tag_name_list[0]

        pos: List[int] = dpg.get_item_pos(tag_node_name)

        # 現在選択されている言語を取得
        language_combo_tag = f"{node_id}:{self.node_tag}:language_combo"
        selected_language = dpg.get_value(language_combo_tag)

        setting_dict: Dict[str, Any] = {
            "ver": self._ver,
            "pos": pos,
            "selected_language": selected_language,
        }
        return setting_dict

    def set_setting_dict(self, node_id, setting_dict):
        # 保存された言語設定を復元
        selected_language = setting_dict.get("selected_language", "日本語")
        self._node_languages[node_id] = selected_language

        # UIのドロップダウンも更新
        language_combo_tag = f"{node_id}:{self.node_tag}:language_combo"
        if dpg.does_item_exist(language_combo_tag):
            dpg.set_value(language_combo_tag, selected_language)
