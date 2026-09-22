import unicodedata


def normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFKC", text).casefold()
    return "".join(char for char in text if not char.isspace() and not unicodedata.category(char).startswith("P"))


def edit_distance(reference: str, hypothesis: str) -> int:
    previous = list(range(len(hypothesis) + 1))
    for i, left in enumerate(reference, 1):
        current = [i]
        for j, right in enumerate(hypothesis, 1):
            current.append(min(current[-1] + 1, previous[j] + 1, previous[j - 1] + (left != right)))
        previous = current
    return previous[-1]


def character_errors(reference: str, hypothesis: str) -> tuple[int, int]:
    reference = normalize_text(reference)
    hypothesis = normalize_text(hypothesis)
    if not reference:
        raise ValueError("CER 的参考文本不能为空或仅包含标点。")
    return edit_distance(reference, hypothesis), len(reference)
