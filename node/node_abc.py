from abc import ABCMeta, abstractmethod
from typing import Any, Dict, List


class DpgNodeABC(metaclass=ABCMeta):
    _ver: str = "0.0.0"

    node_label: str = ""
    node_tag: str = ""

    TYPE_NONE: str = "None"
    TYPE_INT: str = "Int"
    TYPE_FLOAT: str = "Float"
    TYPE_SIGNAL_CHUNK: str = "SignalChunk"
    TYPE_FFT_CHUNK: str = "FftChunk"
    TYPE_TIME_MS: str = "TimeMS"
    TYPE_TEXT: str = "Text"

    @abstractmethod
    def add_node(
        self,
        parent: str,
        node_id: int,
        pos: List[int],
        width: int,
        height: int,
        setting_dict: Dict[str, Any],
    ) -> str:
        pass

    @abstractmethod
    def update(
        self,
        node_id: str,
        connection_list: List[Any],
        player_status_dict: Dict[str, Any],
        node_result_dict: Dict[str, Any],
    ) -> Any:
        pass

    @abstractmethod
    def get_setting_dict(self, node_id: str) -> Dict[str, Any]:
        pass

    @abstractmethod
    def set_setting_dict(self, node_id: int, setting_dict: Dict[str, Any]) -> None:
        pass

    @abstractmethod
    def close(self, node_id: str) -> None:
        pass
