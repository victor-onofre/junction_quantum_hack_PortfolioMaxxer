import galois


def construct_binary(x: int) -> str:
    ret: list[str] = []
    n_iterations = x.bit_length()
    for _ in range(n_iterations):
        ret.append(str(x & 1))
        x //= 2
    return "".join(ret)


def parse_binary(text: str) -> int:
    x = 0
    for c in text[::-1]:
        if c == "0":
            x = x << 1
        if c == "1":
            x = (x << 1) | 1
    return x


def n_iterations_divstep(d):
    return (49 * d + 80) // 17 if d < 46 else (49 * d + 57) // 17


def construct_dialog_int_divstep(f: int, g: int) -> tuple[int, list[int]]:
    delta = 1
    d = max(f.bit_length(), g.bit_length())
    dialog = []
    for _ in range(n_iterations_divstep(d)):
        is_swap = delta > 0 and g & 1
        dialog.append(g & 1)
        if is_swap:
            delta, f, g = -delta, g, -f
        if g & 1:
            g = g + f
        delta = 2 + delta
        g = g >> 1
    assert g == 0
    return f, dialog


def _get_delta(dialog: list[int]):
    delta = 1
    for c in dialog:
        is_swap = delta > 0 and int(c)
        if is_swap:
            delta = -delta
        delta = 2 + delta
    return delta


def craig_dialog_from_int(val: int, *, mod: int, min_length: int = 0) -> str:
    import math

    assert mod & 1
    assert 1 <= val < mod
    assert math.gcd(val, mod) == 1

    u = val
    v = mod
    branches = []
    delta = 0
    while u or min_length > 0:
        min_length -= 1
        branch = 'DA'[u & 1]
        branches.append(branch)
        if branch == 'A' and delta >= 0:
            u, v = v, u
            u *= -1
            delta ^= -1
        if branch == 'A':
            u += v
        u >>= 1
        delta += 1

    sign = 1
    if v < 0:
        sign = -1
        v *= -1
    assert u == 0
    assert v == 1
    return branches


def multiply_int_with_divstep_dialog(
    a: int | galois.FieldArray, dialog: list[int]
) -> tuple[int, int]:
    f, g, f_s, g_s = a, type(a)(0), 1, 1
    for i, c in [*enumerate(dialog)][::-1]:
        is_swap = int(_get_delta(dialog[:i]) > 0) and int(c)
        g = 2 * g
        if c:
            g = g - f
        if is_swap:
            f, g = g, -f
    return abs(f), abs(g)


def mod_divide_int_with_divstep_dialog(
    a: int | galois.FieldArray, dialog: list[int]
) -> tuple[int, int]:
    f, g = type(a)(0), a
    for i, c in enumerate(dialog):
        is_swap = _get_delta(dialog[:i]) > 0 and int(c)
        if is_swap:
            f, g = g, -f
        if int(c):
            g = g + f
        g = g // type(a)(2)
    return abs(f), abs(g)


def n_iterations_bin(d: int):
    return 2 * d


def construct_dialog_int_bin(f: int, g: int) -> tuple[int, list[int]]:
    d = max(f.bit_length(), g.bit_length())
    dialog = []
    for _ in range(n_iterations_bin(d)):
        is_g_odd, is_f_gt_g = int(g & 1), int(f > g)
        dialog.append((is_g_odd << 1) | is_f_gt_g)
        if is_g_odd and is_f_gt_g:
            f, g = g, f
        if is_g_odd:
            g = g - f
        g >>= 1
    assert g == 0
    return f, dialog


def multiply_int_with_bin_dialog(a: int | galois.FieldArray, dialog: list[int]) -> tuple[int, int]:
    f, g = a, type(a)(0)
    for c in dialog[::-1]:
        is_g_odd, is_f_gt_g = c & 2, c & 1
        g = type(a)(2) * g
        if is_g_odd:
            g = g + f
        if is_g_odd and is_f_gt_g:
            f, g = g, f
    return f, g


def mod_divide_int_with_bin_dialog(
    a: int | galois.FieldArray, dialog: list[int]
) -> tuple[int, int]:
    f, g = type(a)(0), a
    for c in dialog:
        is_g_odd, is_f_gt_g = c & 2, c & 1
        if is_g_odd and is_f_gt_g:
            f, g = g, f
        if is_g_odd:
            g = g - f
        g = g // type(a)(2)
    return f, g
