"""补充用例：覆盖边界要求（性能、并发、非有限深度、重复键、负图层）。"""

import math
import threading
import time

import pytest

import sorter


def _sprites(n):
    return [
        {"id": i, "layer": i % 7, "depth": (i % 13) / 13.0}
        for i in range(n)
    ]


def test_sort_100k_under_200ms():
    items = _sprites(100_000)
    started = time.perf_counter()
    out = sorter.Sorter().sort(items)
    elapsed = time.perf_counter() - started
    assert len(out) == 100_000
    assert elapsed < 0.200, "10 万项耗时 {:.3f}s".format(elapsed)


def test_sort_is_concurrency_safe():
    shared = sorter.Sorter()
    items = _sprites(20_000)
    expected = [x["id"] for x in shared.sort(items)]
    results = []
    errors = []

    def run():
        try:
            results.append([x["id"] for x in shared.sort(items)])
        except Exception as exc:  # pragma: no cover - 失败时记录
            errors.append(exc)

    threads = [threading.Thread(target=run) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not errors
    assert len(results) == 8
    assert all(r == expected for r in results)


@pytest.mark.parametrize("bad", [math.nan, math.inf, -math.inf])
def test_sort_rejects_non_finite_depth(bad):
    with pytest.raises(ValueError):
        sorter.Sorter().sort([{"id": "x", "layer": 0, "depth": bad}])


def test_sort_keeps_insertion_order_for_duplicate_keys():
    items = [
        {"id": "a", "layer": 1, "depth": 0.5},
        {"id": "b", "layer": 1, "depth": 0.5},
        {"id": "c", "layer": 1, "depth": 0.5},
        {"id": "d", "layer": 0, "depth": 0.5},
    ]
    out = sorter.Sorter().sort(items)
    assert [x["id"] for x in out] == ["d", "a", "b", "c"]


@pytest.mark.parametrize("layer", [-1, -100, True, 1.0, "1", None])
def test_sort_rejects_illegal_layer(layer):
    with pytest.raises(ValueError):
        sorter.Sorter().sort([{"id": "x", "layer": layer, "depth": 0.0}])


def test_sort_does_not_mutate_input():
    items = _sprites(100)
    snapshot = list(items)
    sorter.Sorter().sort(items)
    assert items == snapshot
