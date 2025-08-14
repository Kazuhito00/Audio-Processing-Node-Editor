#!/usr/bin/env python
# -*- coding: utf-8 -*-
import platform
import queue
import time
from typing import Any, Dict, List, Optional

import dearpygui.dearpygui as dpg
import numpy as np
import sounddevice as sd
from node_editor.util import dpg_set_value, get_tag_name_list

from node.node_abc import DpgNodeABC


class Node(DpgNodeABC):
    _ver: str = "0.0.1"

    node_label: str = "Loopback (Windows)"
    node_tag: str = "Loopback"

    def __init__(self) -> None:
        self._node_data = {}
        self._system = platform.system()
        
        # Windowsでない場合は警告
        if self._system != "Windows":
            print("[WARNING] Loopback node is designed for Windows only.")
            
        self._loopback_device_id: Optional[int] = None
        self._candidate_devices: List[int] = []
        self._current_device_index: int = 0

        # データ受け渡し用のスレッドセーフなキュー
        self.queue = queue.Queue(maxsize=200)  # キューサイズを増加
        self.stream_started: bool = False
        self._previous_chunk: Optional[np.ndarray] = None  # 前回のチャンクを保持

        self._find_loopback_device()

    def _find_loopback_device(self) -> None:
        """
        Windowsのループバック入力デバイスを検索し、候補リストを作成する。
        """
        self._candidate_devices = []
        
        # Windows以外では処理をスキップ
        if self._system != "Windows":
            print("[WARNING] Loopback device search is only supported on Windows.")
            return
            
        try:
            devices = sd.query_devices()
        except Exception as e:
            print(f"[ERROR] Failed to query audio devices: {e}")
            return

        # Windows専用のループバックデバイス検索
        priority_levels = {1: [], 2: [], 3: []}
        for device_id, device in enumerate(devices):
            if device.get("max_input_channels", 0) > 0:
                device_name = device.get("name", "").lower()
                # 最高優先度: ステレオミキサー系
                if any(k in device_name for k in ["stereo mix", "ステレオ ミキサー", "what u hear"]):
                    priority_levels[1].append(device_id)
                # 中優先度: PCスピーカー系
                elif "pc スピーカー" in device_name or "pc speaker" in device_name:
                    priority_levels[2].append(device_id)
                # 低優先度: Steam系
                elif "steam streaming speakers" in device_name and "input" in device_name:
                    priority_levels[3].append(device_id)

        for level in sorted(priority_levels.keys()):
            self._candidate_devices.extend(priority_levels[level])
        self._candidate_devices = list(dict.fromkeys(self._candidate_devices))

        print("[INFO] Found Windows loopback devices:")
        if self._candidate_devices:
            for i, dev_id in enumerate(self._candidate_devices):
                print(f"  - Candidate {i}: ID {dev_id}, Name: {sd.query_devices(dev_id)['name']}")
            self._loopback_device_id = self._candidate_devices[0]
            print(f"[INFO] Selected initial device: ID {self._loopback_device_id}")
        else:
            print("[WARNING] No Windows loopback device found. Please enable 'Stereo Mix' in Windows sound settings.")

    def add_node(self, parent: str, node_id: int, pos: List[int] = [0, 0], setting_dict: Optional[Dict[str, Any]] = None, callback: Optional[Any] = None) -> str:
        tag_name_list = get_tag_name_list(node_id, self.node_tag, [], [self.TYPE_SIGNAL_CHUNK, self.TYPE_TIME_MS])
        tag_node_name, _, output_tag_list = tag_name_list[0], tag_name_list[1], tag_name_list[2]

        self._setting_dict = setting_dict or {}
        waveform_w = self._setting_dict.get("waveform_width", 200)
        waveform_h = self._setting_dict.get("waveform_height", 400)
        self._default_sampling_rate = self._setting_dict.get("default_sampling_rate", 16000)
        self._chunk_size = self._setting_dict.get("chunk_size", 1024)
        self._use_pref_counter = self._setting_dict.get("use_pref_counter", False)

        self._node_data[str(node_id)] = {
            "buffer": np.array([]),
            "chunk": np.zeros(self._chunk_size, dtype=np.float32),
            "chunk_index": -1,
            "stream": None,
            "device_sampling_rate": self._default_sampling_rate,
        }

        buffer_len = self._default_sampling_rate * 5
        display_y_buffer = np.zeros(buffer_len, dtype=np.float32)
        display_x_buffer = np.arange(buffer_len) / self._default_sampling_rate
        self._node_data[str(node_id)]["display_y_buffer"] = display_y_buffer
        self._node_data[str(node_id)]["display_x_buffer"] = display_x_buffer

        with dpg.node(tag=tag_node_name, parent=parent, label=self.node_label, pos=pos):
            with dpg.node_attribute(tag=output_tag_list[0][0], attribute_type=dpg.mvNode_Attr_Output):
                with dpg.plot(height=waveform_h, width=waveform_w, no_inputs=False, tag=f"{node_id}:audio_plot_area"):
                    dpg.add_plot_axis(dpg.mvXAxis, tag=f"{node_id}:xaxis", no_tick_labels=True)
                    dpg.add_plot_axis(dpg.mvYAxis, tag=f"{node_id}:yaxis", no_tick_labels=True)
                    dpg.set_axis_limits(f"{node_id}:xaxis", 0.0, 5.0)
                    dpg.set_axis_limits(f"{node_id}:yaxis", -1.0, 1.0)
                    dpg.add_line_series([], [], parent=f"{node_id}:yaxis", tag=f"{node_id}:audio_line_series")
            if self._use_pref_counter:
                with dpg.node_attribute(tag=output_tag_list[1][0], attribute_type=dpg.mvNode_Attr_Output):
                    dpg.add_text(tag=output_tag_list[1][1], default_value="elapsed time(ms)")
        return tag_node_name

    def update(self, node_id: str, connection_list: List[Any], player_status_dict: Dict[str, Any], node_result_dict: Dict[str, Any]) -> Any:
        node_data = self._node_data[str(node_id)]
        current_status = player_status_dict.get("current_status", "stop")

        if current_status == "play":
            if not self.stream_started:
                if self._start_loopback(node_id):
                    print("[INFO] Stream started.")
                else:
                    if self._switch_to_next_device():
                        print("[INFO] Retrying with next device...")
                    else:
                        print("[ERROR] All candidate devices failed.")

            # キューからデータを取得し、バッファに追加
            temp_buffer = []
            while not self.queue.empty():
                try:
                    temp_buffer.append(self.queue.get_nowait())
                except queue.Empty:
                    break
            if temp_buffer:
                new_data = np.concatenate(temp_buffer)
                # モノラル化
                if new_data.ndim > 1 and new_data.shape[1] > 1:
                    new_data = np.mean(new_data, axis=1)
                else:
                    new_data = new_data.flatten()
                # リサンプリング
                device_rate = node_data["device_sampling_rate"]
                if device_rate != self._default_sampling_rate:
                    ratio = self._default_sampling_rate / device_rate
                    new_length = int(len(new_data) * ratio)
                    if new_length > 0:
                        indices = np.linspace(0, len(new_data) - 1, new_length)
                        new_data = np.interp(indices, np.arange(len(new_data)), new_data)
                node_data["buffer"] = np.concatenate((node_data["buffer"], new_data))

            # バッファからチャンクを取り出して処理（マイクノードと同じ方式）
            if len(node_data["buffer"]) >= self._chunk_size:
                chunk = node_data["buffer"][:self._chunk_size]
                node_data["chunk"] = chunk
                node_data["buffer"] = node_data["buffer"][self._chunk_size:]
                
                # チャンクインデックス更新
                node_data["chunk_index"] += 1
                
                # プロット更新（マイクノードと同じ方式）
                temp_display_y_buffer = node_data["display_y_buffer"]
                temp_display_y_buffer = np.roll(temp_display_y_buffer, -self._chunk_size)
                temp_display_y_buffer[-self._chunk_size:] = chunk
                node_data["display_y_buffer"] = temp_display_y_buffer
                dpg.set_value(f"{node_id}:audio_line_series", [node_data["display_x_buffer"], temp_display_y_buffer])
                
                self._previous_chunk = chunk.copy()  # 前回のチャンクを保存
            else:
                # バッファが不足している場合、前回のチャンクを使用して継続性を保つ
                if self._previous_chunk is not None:
                    node_data["chunk"] = self._previous_chunk.copy()
                else:
                    node_data["chunk"] = np.zeros(self._chunk_size, dtype=np.float32)

        elif current_status in ["pause", "stop"] and self.stream_started:
            self._stop_loopback(node_id)
            if current_status == "stop":
                # UIリセット
                display_y_buffer = np.zeros_like(node_data["display_y_buffer"])
                node_data["display_y_buffer"] = display_y_buffer
                dpg.set_value(f"{node_id}:audio_line_series", [node_data["display_x_buffer"], display_y_buffer])

        return {"chunk_index": node_data["chunk_index"], "chunk": node_data["chunk"]}

    def _start_loopback(self, node_id: str) -> bool:
        if self._system != "Windows":
            print("[ERROR] Loopback is only supported on Windows.")
            return False
        if self._loopback_device_id is None: 
            print("[ERROR] No Windows loopback device available.")
            return False

        def callback_loopback(indata, frames, time_info, status):
            if status:
                print(f"[WARNING] Stream callback status: {status}")
            try:
                self.queue.put_nowait(indata.copy())
            except queue.Full:
                # キューが満杯の場合、古いデータを破棄して新しいデータを追加
                try:
                    self.queue.get_nowait()  # 最古のデータを破棄
                    self.queue.put_nowait(indata.copy())  # 新しいデータを追加
                except queue.Empty:
                    pass

        try:
            device_info = sd.query_devices(self._loopback_device_id)
            device_rate = int(device_info["default_samplerate"])
            channels = min(2, device_info["max_input_channels"])
            self._node_data[node_id]["device_sampling_rate"] = device_rate

            stream = sd.InputStream(
                samplerate=device_rate, channels=channels,
                device=self._loopback_device_id, dtype="float32",
                callback=callback_loopback,
                blocksize=512, # より小さなブロックサイズでバッファリング改善
                latency='low' # 低遅延設定
            )
            stream.start()
            self._node_data[node_id]["stream"] = stream
            self.stream_started = True
            return True
        except Exception as e:
            print(f"[ERROR] Failed to start stream on device ID {self._loopback_device_id}: {e}")
            self.stream_started = False
            return False

    def _stop_loopback(self, node_id: str):
        if not self.stream_started: return
        print("[INFO] Stopping loopback...")
        stream = self._node_data[node_id].get("stream")
        if stream:
            try:
                if not stream.closed:
                    stream.stop()
                    stream.close()
                print("[INFO] Stream stopped and closed.")
            except Exception as e:
                print(f"[ERROR] Error closing stream: {e}")
        self._node_data[node_id]["stream"] = None
        self.stream_started = False
        self._previous_chunk = None  # 前回のチャンクもリセット
        # キューをクリア
        while not self.queue.empty():
            try: self.queue.get_nowait()
            except queue.Empty: break

    def _switch_to_next_device(self) -> bool:
        self._current_device_index += 1
        if self._current_device_index < len(self._candidate_devices):
            self._loopback_device_id = self._candidate_devices[self._current_device_index]
            print(f"[INFO] Switched to next candidate device: ID {self._loopback_device_id}")
            return True
        else:
            print("[WARNING] No more candidate devices to try.")
            self._loopback_device_id = None
            return False

    def close(self, node_id: str):
        self._stop_loopback(node_id)

    def get_setting_dict(self, node_id: str) -> Dict[str, Any]:
        pos = dpg.get_item_pos(get_tag_name_list(node_id, self.node_tag, [], [])[0])
        return {"ver": self._ver, "pos": pos}

    def set_setting_dict(self, node_id: int, setting_dict: Dict[str, Any]):
        pass
