"""
Построение эпюр поперечных сил Q и изгибающих моментов M для статически
определимых балок (сопромат). Аналитический расчёт, график — matplotlib.

Поддерживаются опоры:
    fixed_left   - заделка на левом конце (x = 0), правый конец свободен
    fixed_right  - заделка на правом конце (x = L), левый конец свободен
    two_supports - шарнирно-неподвижная + шарнирно-подвижная опоры в xa и xb
                   (допускаются консоли: xa > 0 и/или xb < L)

Поддерживаются нагрузки:
    сосредоточенная сила       add_force(x, P, 'down'|'up')
    сосредоточенный момент     add_moment(x, M, 'cw'|'ccw')   (cw - по часовой)
    распределённая нагрузка    add_distributed(x1, x2, q, 'up'|'down', q_end=None)
                               (q_end задан -> трапеция/треугольник)

Правила знаков (внутренние усилия считаются по левой отсечённой части):
    Q(x) = сумма вертикальных сил слева от сечения (вверх = +)
    M(x) = сумма моментов сил слева от сечения относительно сечения;
           момент, вращающий отсечённую часть по часовой стрелке, даёт +
           (растянуты нижние волокна). На графике M, как в вашей работе,
           положительные значения откладываются вверх.
           Параметр tension_side=True -> эпюра M строится на растянутом волокне
           (положительные значения вниз), как требует ГОСТ/большинство кафедр.

Результат не зависит от того, с какой стороны идти (слева или справа):
для консоли «от свободного конца» даёт то же самое, что и «от заделки с
реакциями» - это эквивалентные записи одного и того же равновесия.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass

import numpy as np
import matplotlib.pyplot as plt

EPS = 1e-9


# --------------------------------------------------------------------------
#  Нагрузки (внутреннее представление)
# --------------------------------------------------------------------------
@dataclass
class PointForce:
    x: float
    up: float          # вверх = +


@dataclass
class PointMoment:
    x: float
    cw: float          # по часовой = +


@dataclass
class Distributed:
    x1: float
    x2: float
    q1: float          # интенсивность в x1, вверх = +
    q2: float          # интенсивность в x2, вверх = +

    @property
    def k(self) -> float:
        return (self.q2 - self.q1) / (self.x2 - self.x1)

    def resultant(self) -> float:
        l = self.x2 - self.x1
        return self.q1 * l + self.k * l ** 2 / 2

    def moment_ccw_about(self, xp: float) -> float:
        """Момент равнодействующей относительно точки xp (против часовой = +)."""
        l = self.x2 - self.x1
        i0 = self.resultant()
        i1 = self.q1 * l ** 2 / 2 + self.k * l ** 3 / 3   # ∫ w(s)(s-x1) ds
        return i1 + i0 * (self.x1 - xp)


# --------------------------------------------------------------------------
#  Балка
# --------------------------------------------------------------------------
class Beam:
    def __init__(self, length: float, support: str = "fixed_left",
                 xa: float = 0.0, xb: float | None = None):
        if length <= 0:
            raise ValueError("Длина балки должна быть положительной")
        if support not in ("fixed_left", "fixed_right", "two_supports"):
            raise ValueError("support: fixed_left | fixed_right | two_supports")
        self.L = float(length)
        self.support = support
        self.xa = float(xa)
        self.xb = float(self.L if xb is None else xb)
        if support == "two_supports" and not (0 <= self.xa < self.xb <= self.L):
            raise ValueError("Нужно 0 <= xa < xb <= L")
        self.forces: list[PointForce] = []
        self.moments: list[PointMoment] = []
        self.dists: list[Distributed] = []
        self.reactions: dict[str, float] = {}
        self._react_forces: list[PointForce] = []
        self._react_moments: list[PointMoment] = []

    # ---- ввод нагрузок -----------------------------------------------------
    def _check_x(self, *xs):
        for x in xs:
            if x < -EPS or x > self.L + EPS:
                raise ValueError(f"Координата {x} вне балки [0; {self.L}]")

    def add_force(self, x, value, direction="down"):
        self._check_x(x)
        sign = -1 if direction == "down" else 1
        self.forces.append(PointForce(float(x), sign * abs(value)))

    def add_moment(self, x, value, direction="cw"):
        self._check_x(x)
        sign = 1 if direction == "cw" else -1
        self.moments.append(PointMoment(float(x), sign * abs(value)))

    def add_distributed(self, x1, x2, q, direction="up", q_end=None):
        self._check_x(x1, x2)
        if x2 <= x1:
            raise ValueError("Для распределённой нагрузки нужно x2 > x1")
        sign = 1 if direction == "up" else -1
        q2 = q if q_end is None else q_end
        self.dists.append(Distributed(float(x1), float(x2), sign * q, sign * q2))

    # ---- статика: реакции ------------------------------------------------
    def _applied_sums(self, xp):
        """Сумма вертикальных сил (вверх +) и моментов относительно xp (ccw +)."""
        fy = sum(f.up for f in self.forces) + sum(d.resultant() for d in self.dists)
        m = sum(f.up * (f.x - xp) for f in self.forces)
        m += sum(-mm.cw for mm in self.moments)
        m += sum(d.moment_ccw_about(xp) for d in self.dists)
        return fy, m

    def solve(self):
        self._react_forces, self._react_moments, self.reactions = [], [], {}
        if self.support in ("fixed_left", "fixed_right"):
            xf = 0.0 if self.support == "fixed_left" else self.L
            fy, m = self._applied_sums(xf)
            ry, m_ccw = -fy, -m                      # ΣFy = 0, ΣM_f = 0
            self._react_forces.append(PointForce(xf, ry))
            self._react_moments.append(PointMoment(xf, -m_ccw))
            self.reactions = {"R_y (вверх +)": ry,
                              "M_заделки (против часовой +)": m_ccw}
        else:
            fy, m = self._applied_sums(self.xa)
            r2 = -m / (self.xb - self.xa)            # ΣM_A = 0
            r1 = -fy - r2                            # ΣFy = 0
            self._react_forces += [PointForce(self.xa, r1), PointForce(self.xb, r2)]
            self.reactions = {"R_A (вверх +)": r1, "R_B (вверх +)": r2}
        self._check_equilibrium()
        return self.reactions

    def _check_equilibrium(self):
        allf = self.forces + self._react_forces
        fy = sum(f.up for f in allf) + sum(d.resultant() for d in self.dists)
        m = sum(f.up * f.x for f in allf)
        m += sum(-mm.cw for mm in self.moments + self._react_moments)
        m += sum(d.moment_ccw_about(0.0) for d in self.dists)
        scale = 1 + sum(abs(f.up) for f in allf)
        if abs(fy) > 1e-7 * scale or abs(m) > 1e-7 * scale * self.L:
            raise RuntimeError(f"Нарушено равновесие: ΣFy={fy:.3g}, ΣM={m:.3g}")

    # ---- внутренние усилия ----------------------------------------------
    def _sections(self):
        pts = [0.0, self.L]
        pts += [f.x for f in self.forces + self._react_forces]
        pts += [m.x for m in self.moments + self._react_moments]
        for d in self.dists:
            pts += [d.x1, d.x2]
        pts.sort()
        out = [pts[0]]
        for p in pts[1:]:
            if p - out[-1] > EPS:
                out.append(p)
        return out

    def internal(self, x, xmid):
        """Q и M в точках x, лежащих в интервале с серединой xmid
        (учитываются нагрузки, расположенные левее xmid)."""
        x = np.asarray(x, dtype=float)
        q = np.zeros_like(x)
        m = np.zeros_like(x)
        for f in self.forces + self._react_forces:
            if f.x <= xmid:
                q += f.up
                m += f.up * (x - f.x)
        for mm in self.moments + self._react_moments:
            if mm.x <= xmid:
                m += mm.cw
        for d in self.dists:
            if d.x1 < xmid:
                u = np.clip(x, d.x1, d.x2)
                l = u - d.x1
                f_res = d.q1 * l + d.k * l ** 2 / 2
                q += f_res
                m += (x - d.x1) * f_res - (d.q1 * l ** 2 / 2 + d.k * l ** 3 / 3)
        return q, m

    def diagrams(self, n=400):
        """Возвращает данные эпюр по интервалам и характерные значения."""
        if not self.reactions:
            self.solve()
        pts = self._sections()
        segs = []
        for xa, xb in zip(pts[:-1], pts[1:]):
            xs = np.linspace(xa, xb, n)
            q, m = self.internal(xs, 0.5 * (xa + xb))
            segs.append((xs, q, m))
        # экстремумы M (там, где Q = 0 внутри участка)
        extrema = []
        for (xa, xb) in zip(pts[:-1], pts[1:]):
            xmid = 0.5 * (xa + xb)
            xs = np.linspace(xa, xb, 2001)
            q, _ = self.internal(xs, xmid)
            for i in range(len(xs) - 1):
                if q[i] * q[i + 1] < 0:
                    lo, hi = xs[i], xs[i + 1]
                    for _ in range(60):
                        mid = 0.5 * (lo + hi)
                        qm, _ = self.internal(np.array([mid]), xmid)
                        if qm[0] * q[i] > 0:
                            lo = mid
                        else:
                            hi = mid
                    xr = 0.5 * (lo + hi)
                    _, mr = self.internal(np.array([xr]), xmid)
                    extrema.append((xr, float(mr[0])))
        return pts, segs, extrema

    def table(self):
        """Значения Q и M слева/справа от каждой характерной точки."""
        pts, segs, _ = self.diagrams()
        rows = []
        for i, x in enumerate(pts):
            ql = segs[i - 1][1][-1] if i > 0 else 0.0
            ml = segs[i - 1][2][-1] if i > 0 else 0.0
            qr = segs[i][1][0] if i < len(segs) else 0.0
            mr = segs[i][2][0] if i < len(segs) else 0.0
            rows.append((x, ql, qr, ml, mr))
        return rows

    # ---- вывод и график -------------------------------------------------
    def report(self):
        print("Реакции опор:")
        for k, v in self.reactions.items():
            print(f"  {k} = {v:.4g}")
        print("\n   x   |  Q слева  Q справа |  M слева  M справа")
        for x, ql, qr, ml, mr in self.table():
            print(f"{x:6.3g} | {ql:8.4g} {qr:9.4g} | {ml:8.4g} {mr:9.4g}")
        _, _, ex = self.diagrams()
        for xr, mr in ex:
            print(f"Экстремум M: x = {xr:.4g}, M = {mr:.4g}")

    def plot(self, filename=None, tension_side=False, title=None, show=True,
             fig=None):
        """fig - готовый matplotlib.figure.Figure (для встраивания в окно GUI)."""
        pts, segs, extrema = self.diagrams()
        sgn = -1 if tension_side else 1
        ratios = {"height_ratios": [1.1, 1.4, 1.6]}
        if fig is None:
            fig, (a0, a1, a2) = plt.subplots(3, 1, figsize=(11, 10), sharex=True,
                                             gridspec_kw=ratios)
        else:
            fig.clf()
            a0, a1, a2 = fig.subplots(3, 1, sharex=True, gridspec_kw=ratios)
        self._draw_scheme(a0)
        a0.set_title(title or "Расчётная схема и эпюры Q, M")

        # --- Q ---
        self._draw_diagram(a1, segs, 1, 1, "Q", pts)
        # --- M ---
        self._draw_diagram(a2, segs, 2, sgn, "M", pts)
        for xr, mr in extrema:
            a2.plot([xr], [sgn * mr], "ko", ms=4)
            a2.annotate(f"{mr:.4g}\n(x={xr:.3g})", (xr, sgn * mr),
                        textcoords="offset points", xytext=(6, -22 if mr * sgn > 0 else 10),
                        fontsize=8, color="darkred")
        a2.set_xlabel("x")
        for ax in (a1, a2):
            ax.axhline(0, color="k", lw=1)
            for p in pts:
                ax.axvline(p, color="gray", lw=0.4, ls=":")
        fig.tight_layout()
        if filename:
            fig.savefig(filename, dpi=150)
        if show:
            plt.show()
        return fig

    def _draw_diagram(self, ax, segs, idx, sgn, name, pts):
        xs_all, ys_all = [0.0], [0.0]
        for s in segs:
            xs_all += list(s[0])
            ys_all += list(sgn * s[idx])
        xs_all.append(self.L)
        ys_all.append(0.0)
        xs_all, ys_all = np.array(xs_all), np.array(ys_all)
        ax.fill_between(xs_all, ys_all, 0, facecolor="none", edgecolor="tab:blue",
                        hatch="|", lw=0)
        ax.plot(xs_all, ys_all, color="k", lw=1.6)
        ymax = max(1e-9, np.max(np.abs(ys_all)))
        ax.set_ylim(-1.3 * ymax, 1.3 * ymax)
        ax.set_ylabel(f"{name}" + (" (на растянутом волокне)" if (idx == 2 and sgn < 0) else ""))
        # знаки в кружках
        for xs, q, m in segs:
            v = sgn * (q if idx == 1 else m)
            if abs(v).max() < 1e-9 * ymax or v.min() * v.max() < 0:
                continue
            xm = 0.5 * (xs[0] + xs[-1])
            ym = 0.5 * np.sign(v.mean()) * ymax * 0.35
            val_sign = np.sign((q if idx == 1 else m).mean())
            ax.text(xm, ym, "⊕" if val_sign > 0 else "⊖", ha="center", va="center",
                    fontsize=13, color="tab:red")
        # подписи значений в характерных точках (с обеих сторон от скачка)
        lab = set()
        for i, x in enumerate(pts):
            vals = []          # (значение, сторона: -1 слева от точки, +1 справа)
            if i > 0:
                vals.append((segs[i - 1][idx][-1], -1))
            if i < len(segs):
                vals.append((segs[i][idx][0], +1))
            for v, side in vals:
                key = (round(x, 6), round(v, 6))
                if key in lab or abs(v) < 1e-9:
                    continue
                lab.add(key)
                ax.annotate(f"{v:.4g}", (x, sgn * v), textcoords="offset points",
                            xytext=(5 * side, 5 if sgn * v >= 0 else -12),
                            ha="left" if side > 0 else "right", fontsize=8)

    def _draw_scheme(self, ax):
        L = self.L
        ax.plot([0, L], [0, 0], "k-", lw=4, solid_capstyle="butt")
        h = 0.55 * max(1.0, L / 8)
        ax.set_ylim(-1.6 * h, 2.4 * h)
        ax.set_yticks([])
        # опоры
        if self.support in ("fixed_left", "fixed_right"):
            xf = 0 if self.support == "fixed_left" else L
            d = -1 if xf == 0 else 1
            ax.plot([xf, xf], [-h, h], "k-", lw=3)
            for yy in np.linspace(-h, h, 9):
                ax.plot([xf, xf + d * 0.12 * h * 2], [yy, yy - 0.15 * h], "k-", lw=0.8)
        else:
            for x, kind in ((self.xa, "pin"), (self.xb, "roller")):
                ax.plot([x, x - 0.18 * h, x + 0.18 * h, x],
                        [0, -0.5 * h, -0.5 * h, 0], "k-", lw=1.5)
                if kind == "roller":
                    ax.plot([x], [-0.62 * h], "ko", ms=5, mfc="white")
        # нагрузки
        for d in self.dists:
            n = max(4, int((d.x2 - d.x1) / L * 20))
            xs = np.linspace(d.x1, d.x2, n)
            ys = d.q1 + (d.q2 - d.q1) * (xs - d.x1) / (d.x2 - d.x1)
            qmax = max(abs(d.q1), abs(d.q2), 1e-12)
            hh = 0.8 * h * ys / qmax
            # вверх направленная нагрузка рисуется под балкой (стрелки к балке),
            # вниз направленная - над балкой
            env = np.where(hh > 0, -hh - 0.05, -hh + 0.05)
            for x, y, e in zip(xs, hh, env):
                if abs(y) < 1e-9:
                    continue
                head = -0.05 if y > 0 else 0.05
                ax.annotate("", (x, head), (x, e),
                            arrowprops=dict(arrowstyle="->", color="tab:green"))
            ax.plot(xs, env, color="tab:green", lw=1)
            up = d.q1 + d.q2 > 0
            ty = -(abs(hh).max() + 0.3 * h) if up else abs(hh).max() + 0.25 * h
            ax.text(0.5 * (d.x1 + d.x2), ty,
                    f"q={abs(d.q1):.4g}" + ("" if d.q1 == d.q2 else f"…{abs(d.q2):.4g}"),
                    ha="center", color="tab:green", fontsize=9)
        for f in self.forces:
            y0, y1 = (1.5 * h, 0.05) if f.up < 0 else (-1.0 * h, -0.05)
            ax.annotate("", (f.x, y1), (f.x, y0),
                        arrowprops=dict(arrowstyle="-|>", color="tab:red", lw=2))
            ax.text(f.x, 1.65 * h if f.up < 0 else -1.2 * h, f"{abs(f.up):.4g}",
                    ha="center", color="tab:red", fontsize=10)
        for m in self.moments:
            ax.text(m.x, 0.9 * h, "↻" if m.cw > 0 else "↺", ha="center", va="center",
                    fontsize=22, color="tab:purple")
            ax.text(m.x, 1.65 * h, f"{abs(m.cw):.4g}", ha="center", color="tab:purple")
        ax.set_xlim(-0.08 * L, 1.08 * L)
        ax.axis("off")
        ax.text(0, -1.5 * h, "↻ — по часовой, ↺ — против часовой; q: зелёные стрелки",
                fontsize=7, color="gray")


# --------------------------------------------------------------------------
#  Интерактивный ввод
# --------------------------------------------------------------------------
def _num(prompt, default=None):
    while True:
        s = input(prompt).strip().replace(",", ".")
        if not s and default is not None:
            return default
        try:
            return float(s)
        except ValueError:
            print("  Введите число.")


def interactive() -> Beam:
    print("Тип балки: 1 - заделка слева, 2 - заделка справа, 3 - две шарнирные опоры")
    t = int(_num("Выбор: "))
    L = _num("Полная длина балки L: ")
    if t == 3:
        beam = Beam(L, "two_supports", _num("Координата опоры A (шарнир): "),
                    _num("Координата опоры B (каток): "))
    else:
        beam = Beam(L, "fixed_left" if t == 1 else "fixed_right")
    print("Нагрузки (координата x отсчитывается от левого конца). Пустой ввод — конец.")
    while True:
        k = input("Добавить: [f]сила, [m]момент, [q]распр. нагрузка, Enter - готово: ").strip().lower()
        if not k:
            break
        try:
            if k == "f":
                x = _num("  x = ")
                p = _num("  величина P = ")
                d = input("  направление (down/up) [down]: ").strip() or "down"
                beam.add_force(x, p, d)
            elif k == "m":
                x = _num("  x = ")
                m = _num("  величина M = ")
                d = input("  направление (cw/ccw) [cw]: ").strip() or "cw"
                beam.add_moment(x, m, d)
            elif k == "q":
                x1, x2 = _num("  начало x1 = "), _num("  конец x2 = ")
                q = _num("  q (в начале) = ")
                qe = input("  q в конце (Enter — равномерная): ").strip().replace(",", ".")
                d = input("  направление (up/down) [up]: ").strip() or "up"
                beam.add_distributed(x1, x2, q, d, float(qe) if qe else None)
        except ValueError as e:
            print("  Ошибка:", e)
    return beam


# --------------------------------------------------------------------------
#  Пример: задача №1 (заделка слева, a=2, b=2, c=4)
# --------------------------------------------------------------------------
def example_task1(M=10, P=100, q=10, a=2, b=2, c=4) -> Beam:
    beam = Beam(a + b + c, "fixed_left")
    beam.add_moment(a, M, "cw")                   # M в точке B (по часовой)
    beam.add_distributed(a, a + b, q, "up")       # q на участке BC вверх
    beam.add_moment(a + b, 2 * M, "ccw")          # 2M в точке C (против часовой)
    beam.add_force(a + b + c, P, "down")          # P в точке D вниз
    return beam


if __name__ == "__main__":
    if "--interactive" in sys.argv:
        beam = interactive()
        beam.solve()
        beam.report()
        beam.plot()
    else:
        beam = example_task1()
        beam.solve()
        beam.report()
        beam.plot("epury_example.png", show="--noshow" not in sys.argv)
