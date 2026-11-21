import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(__file__))


def pytest_configure(config):
    config.addinivalue_line("markers", "noise: task02 分形值噪声用例")


def pytest_collection_modifyitems(items):
    # 让验收命令 `pytest -k noise` 能选中本任务全部用例（不改变用例本身行为）
    for item in items:
        item.add_marker(pytest.mark.noise)
