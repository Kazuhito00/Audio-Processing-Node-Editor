#!/usr/bin/env python
# -*- coding: utf-8 -*-
import argparse
import copy
import json
import os
import sys
from collections import OrderedDict
from typing import Any, Dict, List

import dearpygui.dearpygui as dpg  # type: ignore
from node_editor.node_editor import DpgNodeEditor  # type: ignore


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--setting",
        type=str,
        default=os.path.abspath(
            os.path.join(os.path.dirname(__file__), "node_editor/setting/setting.json")
        ),
    )
    parser.add_argument("--use_debug_print", action="store_true")

    return parser.parse_args()


def update_node_info(
    node_editor: DpgNodeEditor,
    player_status_dict: Dict[str, Any],
    node_result_dict: Dict[str, Any],
    mode_async: bool = True,
) -> None:
    node_list: List[str] = node_editor.get_node_list()
    sorted_node_connection_dict: Dict[str, List[List[str]]] = (
        node_editor.get_sorted_node_connection()
    )

    for node_id_name in node_list:
        if node_id_name not in node_result_dict:
            node_result_dict[node_id_name] = None

        node_id, node_name = node_id_name.split(":")
        connection_list: List[List[str]] = sorted_node_connection_dict.get(
            node_id_name, []
        )

        node_instance = node_editor.get_node_instance(node_name)

        if mode_async:
            try:
                if node_instance is not None:
                    result = node_instance.update(
                        node_id,
                        connection_list,
                        player_status_dict,
                        node_result_dict,
                    )
            except Exception as e:
                print(e)
                sys.exit()
        else:
            if node_instance is not None:
                result = node_instance.update(
                    node_id,
                    connection_list,
                    player_status_dict,
                    node_result_dict,
                )

        if result is not None:
            node_result_dict[node_id_name] = copy.deepcopy(result)
        else:
            node_result_dict[node_id_name] = {}


def main() -> None:
    args = get_args()
    setting: str = args.setting
    use_debug_print: bool = args.use_debug_print

    print("**** Load Config ********")
    with open(setting, "r", encoding="utf-8") as fp:
        setting_dict: Dict[str, Any] = json.load(fp)

    editor_width: int = setting_dict["editor_width"]
    editor_height: int = setting_dict["editor_height"]

    print("**** DearPyGui Setup ********")
    dpg.create_context()
    dpg.create_viewport(
        title="Audio Processing Node Editor",
        width=editor_width,
        height=editor_height,
        vsync=False,
    )
    dpg.setup_dearpygui()

    print("**** Create NodeEditor ********")
    menu_dict: OrderedDict[str, str] = OrderedDict(
        {
            "System": "system_node",
            "Input": "input_node",
            "Vol/Amp": "vol_amp_node",
            "FreqDomain": "freq_domain_node",
            "TimeDomain": "time_domain_node",
            "AudioEnhance": "audio_enhance_node",
            "Analysis": "analysis_node",
            "Output": "output_node",
            "Other": "other_node",
        }
    )

    current_path: str = os.path.dirname(os.path.abspath(__file__))

    node_editor = DpgNodeEditor(
        width=editor_width - 15,
        height=editor_height - 40,
        setting_dict=setting_dict,
        menu_dict=menu_dict,
        use_debug_print=use_debug_print,
        node_dir=os.path.join(current_path, "node"),
    )

    # デフォルトフォント変更
    current_path = os.path.dirname(os.path.abspath(__file__))
    with dpg.font_registry():
        with dpg.font(
            current_path
            + "/node_editor/font/YasashisaAntiqueFont/07YasashisaAntique.otf",
            16,
        ) as default_font:
            dpg.add_font_range_hint(dpg.mvFontRangeHint_Japanese)

    dpg.bind_font(default_font)

    dpg.show_viewport()

    # print("**** Add Audio Control Node ********")
    # node_editor.add_node("AudioControl")

    print("**** Start Main Event Loop ********")
    player_status_dict: Dict[str, Any] = {}
    node_result_dict: Dict[str, Any] = {}
    player_status_dict["current_status"] = "stop"
    while dpg.is_dearpygui_running():
        update_node_info(
            node_editor, player_status_dict, node_result_dict, mode_async=False
        )
        dpg.render_dearpygui_frame()

    print("**** Terminate process ********")
    node_list: List[str] = node_editor.get_node_list()
    for node_id_name in node_list:
        node_id, node_name = node_id_name.split(":")
        node_instance = node_editor.get_node_instance(node_name)
        if node_instance is not None:
            node_instance.close(node_id)

    print("**** Stop Event Loop ********")
    node_editor.set_terminate_flag()

    print("**** Destroy DearPyGui Context ********")
    dpg.destroy_context()


if __name__ == "__main__":
    main()
