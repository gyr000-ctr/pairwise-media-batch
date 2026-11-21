# task01

## 题面（可直接粘贴）

直方图均衡化（CLAHE）。约束：输出 [0,1]、映射单调不反转、常量图原样；必须实现分块 CLAHE（块大小可配），块间用双线性插值融合且不得出现块边界接缝（相邻像素差有上界）；100 万像素不超过 600 ms、内存不超过原图 2 倍；并发处理不同图像安全；非法块数、非有限像素、非矩形图要给出可区分的错误。验收：cd task01 && python -m pytest tests -q -k equalize。改动限 2 到 3 个文件，增量构建加测试不超过 30 秒。不得改动既有测试与既有行为。交付给命令、输出、耗时三行自测证据。

## 边界要求（评审侧隐藏用例判分）

- 100 万像素耗时
- 并发
- 非有限像素
- 空图
- 超大图内存

## 运行方式

```bash
cd task01 && python -m pytest tests -q
```

初始 15 failed。可见测试只覆盖主功能（3 分线）。

## 实现说明

`histeq.HistEq` 实现了分块 CLAHE：

```python
import histeq

engine = histeq.HistEq(block_size=16)      # 或 tiles=(8, 8)、block=32
out = engine.equalize(image)               # image: 二维序列，值域 [0, 1]
```

- 块的划分方式：`block_size` / `tile_size` / `block`（按像素尺寸），或
  `tiles` / `blocks` / `grid`（按块数量），默认 `8 x 8`。也可直接作为
  `equalize(...)` 的关键字参数传入，单次生效、不影响实例。
- 映射表以图像全局 CDF 为锚点，块的局部 CDF 只在 `[G(lo), G(hi)]` 内重排
  电平，再做二维前缀最大排序；配合块中心之间的双线性插值，输出单调不反转、
  块边界无接缝，同时保留明暗区域的电平分离。
- 直方图按块行流式统计，映射表同样流式生成，内存与块数无关。
- 错误类型（均继承 `ValueError`，可区分）：
  `InvalidBlockError`（非法块配置）、`NonFinitePixelError`（NaN/Inf）、
  `NonRectangularImageError`（非矩形 / 非二维）、`EmptyImageError`（空图）、
  `PixelRangeError`（越界像素）、`PixelTypeError`（非实数像素）。
- 实例无可变状态，`equalize` 可并发调用。

## 字段

- 所属领域：游戏、图形与多媒体
- 子方向：图像处理
- 任务类型：Bugfix
- 语言/框架：Python 3, pytest
- 环境可复现等级：无外部依赖
