"""
Графический интерфейс для beam_diagrams.py (CustomTkinter + matplotlib).
Запуск:  python epury_gui.py
Сборка:  pyinstaller --onefile --windowed --name Epury epury_gui.py
"""
from __future__ import annotations

import io
from contextlib import redirect_stdout
from tkinter import filedialog, messagebox

import customtkinter as ctk
import matplotlib
matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

from beam_diagrams import Beam

ctk.set_appearance_mode("light")
ctk.set_default_color_theme("blue")

SUPPORTS = {
    "Заделка слева": "fixed_left",
    "Заделка справа": "fixed_right",
    "Две опоры (шарнир + каток)": "two_supports",
}
KINDS = ["Сила", "Момент", "Распределённая нагрузка"]
DIRS = {
    "Сила": ["вниз", "вверх"],
    "Момент": ["по часовой", "против часовой"],
    "Распределённая нагрузка": ["вверх", "вниз"],
}
DIR_CODE = {"вниз": "down", "вверх": "up", "по часовой": "cw", "против часовой": "ccw"}


def num(s: str, name: str) -> float:
    try:
        return float(s.strip().replace(",", "."))
    except ValueError:
        raise ValueError(f"Поле «{name}»: введите число") from None


class App(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("Эпюры Q и M для балок")
        self.geometry("1380x860")
        self.minsize(1100, 700)
        self.loads: list[dict] = []
        self.fig = Figure(figsize=(8, 8), dpi=100)

        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)
        self._build_left()
        self._build_right()
        self._on_support()
        self._on_kind(KINDS[0])
        self.load_example()

    # ------------------------------------------------------------------ UI
    def _build_left(self):
        left = ctk.CTkScrollableFrame(self, width=340, label_text="Исходные данные")
        left.grid(row=0, column=0, sticky="nsew", padx=(10, 5), pady=10)
        self.left = left

        ctk.CTkLabel(left, text="Тип балки").pack(anchor="w")
        self.sup_var = ctk.StringVar(value=list(SUPPORTS)[0])
        ctk.CTkOptionMenu(left, values=list(SUPPORTS), variable=self.sup_var,
                          command=lambda _: self._on_support()).pack(fill="x", pady=(0, 6))

        g = ctk.CTkFrame(left, fg_color="transparent")
        g.pack(fill="x")
        g.grid_columnconfigure((1, 3), weight=1)
        self.e_L = self._entry(g, "L", 0, 0, "8")
        self.lbl_xa, self.e_xa = self._entry(g, "xA", 1, 0, "0", ret_label=True)
        self.lbl_xb, self.e_xb = self._entry(g, "xB", 1, 2, "8", ret_label=True)

        ctk.CTkLabel(left, text="Нагрузки (x — от левого конца балки)",
                     font=ctk.CTkFont(weight="bold")).pack(anchor="w", pady=(12, 2))
        box = ctk.CTkFrame(left)
        box.pack(fill="x")
        box.grid_columnconfigure((1, 3), weight=1)
        self.kind_var = ctk.StringVar(value=KINDS[0])
        ctk.CTkOptionMenu(box, values=KINDS, variable=self.kind_var,
                          command=self._on_kind).grid(row=0, column=0, columnspan=4,
                                                      sticky="ew", padx=6, pady=6)
        self.lbl_x1, self.e_x1 = self._entry(box, "x", 1, 0, "", ret_label=True)
        self.lbl_x2, self.e_x2 = self._entry(box, "x2", 1, 2, "", ret_label=True)
        self.lbl_v, self.e_v = self._entry(box, "P", 2, 0, "", ret_label=True)
        self.lbl_v2, self.e_v2 = self._entry(box, "q кон.", 2, 2, "", ret_label=True)
        ctk.CTkLabel(box, text="Направл.").grid(row=3, column=0, padx=6, pady=4, sticky="w")
        self.dir_var = ctk.StringVar()
        self.dir_menu = ctk.CTkOptionMenu(box, values=DIRS["Сила"], variable=self.dir_var)
        self.dir_menu.grid(row=3, column=1, columnspan=3, sticky="ew", padx=6, pady=4)
        ctk.CTkButton(box, text="＋ Добавить нагрузку", command=self.add_load
                      ).grid(row=4, column=0, columnspan=4, sticky="ew", padx=6, pady=(4, 8))

        ctk.CTkLabel(left, text="Список нагрузок").pack(anchor="w", pady=(10, 2))
        self.list_frame = ctk.CTkFrame(left)
        self.list_frame.pack(fill="x")

        self.tension_var = ctk.BooleanVar(value=False)
        ctk.CTkCheckBox(left, text="Эпюра M на растянутом волокне\n(положительные значения вниз)",
                        variable=self.tension_var, command=self.calculate
                        ).pack(anchor="w", pady=(12, 6))

        ctk.CTkButton(left, text="▶ Рассчитать и построить", height=38,
                      command=self.calculate).pack(fill="x", pady=(4, 4))
        row = ctk.CTkFrame(left, fg_color="transparent")
        row.pack(fill="x")
        row.grid_columnconfigure((0, 1), weight=1)
        ctk.CTkButton(row, text="Пример (задача 1)", fg_color="gray40",
                      command=self.load_example).grid(row=0, column=0, sticky="ew", padx=(0, 3))
        ctk.CTkButton(row, text="Очистить", fg_color="gray40",
                      command=self.clear_loads).grid(row=0, column=1, sticky="ew", padx=(3, 0))
        ctk.CTkButton(left, text="💾 Сохранить график (PNG)", fg_color="seagreen",
                      command=self.save_png).pack(fill="x", pady=(6, 0))

    def _entry(self, parent, label, row, col, default, ret_label=False):
        lbl = ctk.CTkLabel(parent, text=label)
        lbl.grid(row=row, column=col, padx=(6, 2), pady=4, sticky="w")
        e = ctk.CTkEntry(parent, width=80)
        e.grid(row=row, column=col + 1, padx=(0, 6), pady=4, sticky="ew")
        e.insert(0, default)
        return (lbl, e) if ret_label else e

    def _build_right(self):
        right = ctk.CTkFrame(self)
        right.grid(row=0, column=1, sticky="nsew", padx=(5, 10), pady=10)
        right.grid_columnconfigure(0, weight=1)
        right.grid_rowconfigure(0, weight=4)
        right.grid_rowconfigure(1, weight=1)
        self.canvas = FigureCanvasTkAgg(self.fig, master=right)
        self.canvas.get_tk_widget().grid(row=0, column=0, sticky="nsew", padx=4, pady=4)
        self.out = ctk.CTkTextbox(right, height=170, font=ctk.CTkFont(family="Consolas", size=13))
        self.out.grid(row=1, column=0, sticky="nsew", padx=4, pady=(0, 4))

    # ------------------------------------------------------- динамика формы
    def _on_support(self):
        two = SUPPORTS[self.sup_var.get()] == "two_supports"
        for w in (self.lbl_xa, self.e_xa, self.lbl_xb, self.e_xb):
            w.grid() if two else w.grid_remove()

    def _on_kind(self, kind):
        self.dir_menu.configure(values=DIRS[kind])
        self.dir_var.set(DIRS[kind][0])
        dist = kind == KINDS[2]
        self.lbl_x1.configure(text="x1" if dist else "x")
        self.lbl_v.configure(text={"Сила": "P", "Момент": "M"}.get(kind, "q нач."))
        for w in (self.lbl_x2, self.e_x2, self.lbl_v2, self.e_v2):
            w.grid() if dist else w.grid_remove()

    # ------------------------------------------------------------ нагрузки
    def add_load(self):
        try:
            kind = self.kind_var.get()
            d = {"kind": kind, "dir": self.dir_var.get(),
                 "x1": num(self.e_x1.get(), "x"), "v": num(self.e_v.get(), "величина")}
            if kind == KINDS[2]:
                d["x2"] = num(self.e_x2.get(), "x2")
                s = self.e_v2.get().strip()
                d["v2"] = num(s, "q кон.") if s else None
            self._validate_load(d)
        except ValueError as e:
            messagebox.showerror("Ошибка ввода", str(e))
            return
        self.loads.append(d)
        self._refresh_list()
        self.calculate()

    def _validate_load(self, d):
        L = num(self.e_L.get(), "L")
        xs = [d["x1"]] + ([d["x2"]] if "x2" in d else [])
        if any(x < 0 or x > L for x in xs):
            raise ValueError(f"Координата должна быть в пределах 0…{L:g}")
        if "x2" in d and d["x2"] <= d["x1"]:
            raise ValueError("Для распределённой нагрузки нужно x2 > x1")

    def _describe(self, d):
        if d["kind"] == "Сила":
            return f"P = {d['v']:g} {d['dir']} при x = {d['x1']:g}"
        if d["kind"] == "Момент":
            return f"M = {d['v']:g} {d['dir']} при x = {d['x1']:g}"
        q2 = "" if d["v2"] is None else f"…{d['v2']:g}"
        return f"q = {d['v']:g}{q2} {d['dir']}, x = {d['x1']:g}…{d['x2']:g}"

    def _refresh_list(self):
        for w in self.list_frame.winfo_children():
            w.destroy()
        if not self.loads:
            ctk.CTkLabel(self.list_frame, text="(пусто)", text_color="gray").pack(pady=4)
        for i, d in enumerate(self.loads):
            r = ctk.CTkFrame(self.list_frame, fg_color="transparent")
            r.pack(fill="x")
            ctk.CTkLabel(r, text=self._describe(d), anchor="w", wraplength=240,
                         justify="left").pack(side="left", padx=4, pady=2)
            ctk.CTkButton(r, text="✕", width=28, fg_color="firebrick",
                          command=lambda k=i: self.remove_load(k)).pack(side="right", padx=4)

    def remove_load(self, i):
        self.loads.pop(i)
        self._refresh_list()
        self.calculate()

    def clear_loads(self):
        self.loads.clear()
        self._refresh_list()
        self.calculate()

    def load_example(self):
        """Задача 1: заделка слева, a=2, b=2, c=4, M=10, P=100, q=10."""
        self.sup_var.set(list(SUPPORTS)[0])
        self._on_support()
        self.e_L.delete(0, "end")
        self.e_L.insert(0, "8")
        self.loads = [
            {"kind": "Момент", "dir": "по часовой", "x1": 2, "v": 10},
            {"kind": KINDS[2], "dir": "вверх", "x1": 2, "x2": 4, "v": 10, "v2": None},
            {"kind": "Момент", "dir": "против часовой", "x1": 4, "v": 20},
            {"kind": "Сила", "dir": "вниз", "x1": 8, "v": 100},
        ]
        self._refresh_list()
        self.calculate()

    # --------------------------------------------------------------- расчёт
    def build_beam(self) -> Beam:
        L = num(self.e_L.get(), "L")
        sup = SUPPORTS[self.sup_var.get()]
        if sup == "two_supports":
            beam = Beam(L, sup, num(self.e_xa.get(), "xA"), num(self.e_xb.get(), "xB"))
        else:
            beam = Beam(L, sup)
        for d in self.loads:
            code = DIR_CODE[d["dir"]]
            if d["kind"] == "Сила":
                beam.add_force(d["x1"], d["v"], code)
            elif d["kind"] == "Момент":
                beam.add_moment(d["x1"], d["v"], code)
            else:
                beam.add_distributed(d["x1"], d["x2"], d["v"], code, d["v2"])
        return beam

    def calculate(self):
        try:
            beam = self.build_beam()
            beam.solve()
            buf = io.StringIO()
            with redirect_stdout(buf):
                beam.report()
            beam.plot(show=False, tension_side=self.tension_var.get(), fig=self.fig)
        except Exception as e:                     # noqa: BLE001
            messagebox.showerror("Ошибка расчёта", str(e))
            return
        self.canvas.draw()
        self.out.configure(state="normal")
        self.out.delete("1.0", "end")
        self.out.insert("1.0", buf.getvalue())
        self.out.configure(state="disabled")

    def save_png(self):
        path = filedialog.asksaveasfilename(defaultextension=".png",
                                            filetypes=[("PNG", "*.png")],
                                            initialfile="epury.png")
        if path:
            self.fig.savefig(path, dpi=200)


if __name__ == "__main__":
    App().mainloop()
