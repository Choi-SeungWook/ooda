"""그림 공통 설정: 설치된 한글 글꼴을 찾아 지정 (Windows: 맑은 고딕, Docker/Linux: 나눔고딕 등)."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import font_manager  # noqa: E402

_CANDIDATES = ["Malgun Gothic", "NanumGothic", "Noto Sans CJK KR", "AppleGothic"]


def setup_korean_font() -> str:
    installed = {f.name for f in font_manager.fontManager.ttflist}
    for name in _CANDIDATES:
        if name in installed:
            plt.rcParams["font.family"] = name
            break
    else:
        name = "DejaVu Sans"   # 한글 글꼴이 없으면 한글이 네모로 나옴 (Docker 이미지에는 나눔 설치)
    plt.rcParams["axes.unicode_minus"] = False
    return name
