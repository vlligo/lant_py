"""
Port of QInt64SpinBox: Qt's built-in QSpinBox is capped at 32-bit ints,
but step counts can legitimately be huge (the original app defaults its
"auto-run" burst to 10,000,000 steps at a time). Python ints have no
overflow limit, so this spin box just needs its own line-edit-backed
widget instead of relying on QSpinBox's internal int range.
"""
from PySide6.QtCore import Signal
from PySide6.QtGui import QValidator
from PySide6.QtWidgets import QAbstractSpinBox


class IntSpinBox(QAbstractSpinBox):
    valueChanged = Signal(object)  # emits a Python int

    def __init__(self, parent=None):
        super().__init__(parent)
        self._value = 0
        self._minimum = 0
        self._maximum = 2**63 - 1
        self._single_step = 1
        self._suffix = ""

        self.lineEdit().editingFinished.connect(self._on_editing_finished)
        self._sync_text()

    # -- public API (mirrors QSpinBox) ---------------------------------- #
    def value(self) -> int:
        return self._value

    def setValue(self, val: int):
        val = max(self._minimum, min(int(val), self._maximum))
        if val != self._value:
            self._value = val
            self._sync_text()
            self.valueChanged.emit(self._value)

    def setRange(self, minimum: int, maximum: int):
        self._minimum, self._maximum = int(minimum), int(maximum)

    def setMinimum(self, minimum: int):
        self._minimum = int(minimum)

    def setMaximum(self, maximum: int):
        self._maximum = int(maximum)

    def setSingleStep(self, step: int):
        self._single_step = int(step)

    def setSuffix(self, suffix: str):
        self._suffix = suffix
        self._sync_text()

    # -- internals -------------------------------------------------------#
    def _sync_text(self):
        self.lineEdit().setText(f"{self._value:,}{self._suffix}")

    def _on_editing_finished(self):
        text = self.lineEdit().text().replace(self._suffix, "").replace(",", "").strip()
        try:
            self.setValue(int(text))
        except ValueError:
            self._sync_text()

    def stepBy(self, steps: int):
        self.setValue(self._value + steps * self._single_step)

    def stepEnabled(self):
        flags = QAbstractSpinBox.StepEnabledFlag.StepNone
        if self._value < self._maximum:
            flags |= QAbstractSpinBox.StepEnabledFlag.StepUpEnabled
        if self._value > self._minimum:
            flags |= QAbstractSpinBox.StepEnabledFlag.StepDownEnabled
        return flags

    def validate(self, text: str, pos: int):
        cleaned = text.replace(self._suffix, "").replace(",", "").strip()
        if cleaned in ("", "-"):
            return QValidator.State.Intermediate, text, pos
        try:
            int(cleaned)
            return QValidator.State.Acceptable, text, pos
        except ValueError:
            return QValidator.State.Invalid, text, pos
