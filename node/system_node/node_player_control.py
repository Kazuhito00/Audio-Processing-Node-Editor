#!/usr/bin/env python
# -*- coding: utf-8 -*-
from typing import Any, Dict, List, Optional

import dearpygui.dearpygui as dpg  # type: ignore
from node.node_abc import DpgNodeABC  # type: ignore
from node_editor.util import get_tag_name_list  # type: ignore


# ボタンのデフォルトテーマ（非選択）
def create_default_button_theme():
    with dpg.theme() as theme:
        with dpg.theme_component(dpg.mvButton):
            dpg.add_theme_color(
                dpg.mvThemeCol_Button, (51, 51, 55), category=dpg.mvThemeCat_Core
            )
    return theme


# ボタンのアクティブテーマ（選択中）
def create_active_button_theme():
    with dpg.theme() as theme:
        with dpg.theme_component(dpg.mvButton):
            dpg.add_theme_color(
                dpg.mvThemeCol_Button, (0, 180, 255), category=dpg.mvThemeCat_Core
            )
    return theme


class Node(DpgNodeABC):
    _ver: str = "0.0.1"

    node_label: str = "Audio Control"
    node_tag: str = "AudioControl"

    def __init__(self) -> None:
        self._add_node_flag: bool = False
        self._player_status: str = "stop"
        self._status_change_flag: bool = False

        self._input_tag_list: List[List[str]] = []
        self._output_tag_list: List[List[str]] = []

        self._setting_dict: Dict[str, Any] = {}

        self._tag_loading_indicator: str = ""
        self._tag_plot_area: str = ""
        self._tag_line_series: str = ""

        self._play_theme = create_active_button_theme()
        self._default_theme = create_default_button_theme()

    def add_node(
        self,
        parent: str,
        node_id: int,
        pos: List[int] = [0, 0],
        setting_dict: Optional[Dict[str, Any]] = None,
        callback: Optional[Any] = None,
    ) -> Optional[str]:
        if self._add_node_flag:
            return None
        self._add_node_flag = True

        # タグ名
        tag_name_list: List[Any] = get_tag_name_list(
            node_id,
            self.node_tag,
            [self.TYPE_NONE],
            [],
        )
        tag_node_name: str = tag_name_list[0]
        input_tag_list = tag_name_list[1]

        # 設定
        self._setting_dict = setting_dict or {}
        small_window_w: int = self._setting_dict.get("input_window_width", 200)

        # ノード
        with dpg.node(
            tag=tag_node_name,
            parent=parent,
            label=self.node_label,
            pos=pos,
        ):
            with dpg.node_attribute(
                tag=input_tag_list[0][0],
                attribute_type=dpg.mvNode_Attr_Static,
            ):
                with dpg.group(horizontal=True):
                    dpg.add_button(
                        label="再生",
                        tag=f"{node_id}:play_button",
                        width=int(small_window_w / 3),
                        height=int(small_window_w / 3),
                        callback=self._on_play,
                        user_data=node_id,
                    )
                    dpg.add_button(
                        label="一時停止",
                        tag=f"{node_id}:pause_button",
                        width=int(small_window_w / 3),
                        height=int(small_window_w / 3),
                        callback=self._on_pause,
                        user_data=node_id,
                    )
                    dpg.add_button(
                        label="停止",
                        tag=f"{node_id}:stop_button",
                        width=int(small_window_w / 3),
                        height=int(small_window_w / 3),
                        callback=self._on_stop,
                        user_data=node_id,
                    )
        self._apply_button_theme(node_id, "stop")

        return tag_node_name

    def update(
        self,
        node_id: str,
        connection_list: List[Any],
        player_status_dict: Dict[str, Any],
        node_result_dict: Dict[str, Any],
    ) -> Optional[Any]:
        if self._status_change_flag:
            player_status_dict["current_status"] = self._player_status
            self._status_change_flag = False

        if player_status_dict["current_status"] != self._player_status:
            self._player_status = player_status_dict["current_status"]
            self._apply_button_theme(node_id, self._player_status)
        return None

    def close(self, node_id: str) -> None:
        pass

    def get_setting_dict(self, node_id: str) -> Dict[str, Any]:
        tag_name_list: List[Any] = get_tag_name_list(
            node_id,
            self.node_tag,
            [],
            [],
        )
        tag_node_name: str = tag_name_list[0]
        pos: List[int] = dpg.get_item_pos(tag_node_name)
        return {
            "ver": self._ver,
            "pos": pos,
        }

    def set_setting_dict(self, node_id: int, setting_dict: Dict[str, Any]) -> None:
        pass

    def _apply_button_theme(self, node_id: int, selected: str):
        for state in ["play", "pause", "stop"]:
            tag = f"{node_id}:{state}_button"
            theme = self._play_theme if state == selected else self._default_theme
            dpg.bind_item_theme(tag, theme)

    def _on_play(self, sender: int, app_data: Any, user_data: Any) -> None:
        self._player_status = "play"
        self._status_change_flag = True
        self._apply_button_theme(user_data, "play")

    def _on_pause(self, sender: int, app_data: Any, user_data: Any) -> None:
        self._player_status = "pause"
        self._status_change_flag = True
        self._apply_button_theme(user_data, "pause")

    def _on_stop(self, sender: int, app_data: Any, user_data: Any) -> None:
        self._player_status = "stop"
        self._status_change_flag = True
        self._apply_button_theme(user_data, "stop")
