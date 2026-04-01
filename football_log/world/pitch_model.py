"""标准足球场平面世界坐标定义（米）。"""

from dataclasses import dataclass


@dataclass(frozen=True)
class PitchSpec:
    """
    矩形球场在平面上的尺寸。

    坐标约定（与 homography 的「世界点」顺序一致即可）：
    - 原点 (0, 0) 在**一条底线与一侧边线的交点**（由标定时 4 点顺序决定）。
    - world_x 沿**边线方向**从 0 到 width_m（通常对应画面「横向」场地宽度）。
    - world_y 沿**底线到对面底线**从 0 到 length_m（通常对应画面「纵向」进攻方向）。

    默认采用 FIFA 推荐尺寸 105m × 68m；可通过构造参数覆盖。
    """

    length_m: float = 105.0
    width_m: float = 68.0

    def validate(self) -> None:
        if self.length_m <= 0 or self.width_m <= 0:
            raise ValueError("length_m and width_m must be positive")
