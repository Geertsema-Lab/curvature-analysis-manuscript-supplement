"""
Napari widget for creating a mask based on:
  - Two normalized intensity channels (select which one to threshold)
  - Principal curvatures k1 and k2
  - Min/max thresholds for intensity and Gaussian curvature (k1 * k2).

Built with qtpy only (no magicgui, no superqt).
"""

from __future__ import annotations

import traceback
from pathlib import Path

import napari
import numpy as np
from qtpy.QtCore import Qt
from qtpy.QtWidgets import (
    QApplication,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

import imcurvpy.io as io

# ---------------------------------------------------------------------------
# Shared last-directory state
# ---------------------------------------------------------------------------

_last_dir: dict[str, str] = {"path": str(Path.home())}


# ---------------------------------------------------------------------------
# Small reusable widgets
# ---------------------------------------------------------------------------


class FilePickerRow(QWidget):
    """A QLineEdit + Browse button that remembers the last opened directory."""

    def __init__(
        self, tooltip: str = "", file_filter: str = "All files (*)", parent=None
    ):
        super().__init__(parent)
        self._filter = file_filter

        self.line = QLineEdit()
        self.line.setPlaceholderText("No file selected…")
        self.line.setToolTip(tooltip)

        self.btn = QPushButton("Browse…")
        self.btn.setFixedWidth(80)
        self.btn.clicked.connect(self._browse)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.line)
        layout.addWidget(self.btn)

    def _browse(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Select file", _last_dir["path"], self._filter
        )
        if path:
            _last_dir["path"] = str(Path(path).parent)
            self.line.setText(path)

    @property
    def value(self) -> str:
        return self.line.text().strip()


class SectionLabel(QLabel):
    """Bold section header label."""

    def __init__(self, text: str, parent=None):
        super().__init__(f"<b>{text}</b>", parent)
        self.setTextFormat(Qt.RichText)


# ---------------------------------------------------------------------------
# Main widget
# ---------------------------------------------------------------------------


class MaskWidget(QWidget):
    IMG_FILTER = "Images (*.tif *.tiff);;All files (*)"

    def __init__(self, viewer: napari.Viewer, parent=None):
        super().__init__(parent)
        self.viewer = viewer

        # Internal state
        self._ch1: np.ndarray | None = None
        self._ch2: np.ndarray | None = None
        self._gaussian_curvature: np.ndarray | None = None
        self.result_mask: np.ndarray | None = None

        self._build_ui()

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setSpacing(4)

        def add(widget):
            root.addWidget(widget)

        # File pickers
        add(SectionLabel("Path to normalized intensity - channel 1"))
        self.pick_ch1 = FilePickerRow(
            tooltip="Channel-1 intensity", file_filter=self.IMG_FILTER
        )
        add(self.pick_ch1)

        add(SectionLabel("Path to normalized intensity - channel 2 (optional)"))
        self.pick_ch2 = FilePickerRow(
            tooltip="Channel-2 intensity", file_filter=self.IMG_FILTER
        )
        add(self.pick_ch2)

        add(SectionLabel("Path to k1"))
        self.pick_k1 = FilePickerRow(
            tooltip="k1 (first principal curvature)", file_filter=self.IMG_FILTER
        )
        add(self.pick_k1)

        add(SectionLabel("Path to k2"))
        self.pick_k2 = FilePickerRow(
            tooltip="k2 (second principal curvature)", file_filter=self.IMG_FILTER
        )
        add(self.pick_k2)

        self.btn_load = QPushButton("Load images")
        self.btn_load.clicked.connect(self._on_load_images)
        add(self.btn_load)

        # Channel selector
        add(SectionLabel("Intensity channel for thresholding"))
        self.combo_channel = QComboBox()
        self.combo_channel.addItems(["Channel 1", "Channel 2"])
        self.combo_channel.currentIndexChanged.connect(self._refresh_intensity_slider)
        add(self.combo_channel)

        # Intensity range spinboxes
        add(SectionLabel("Min / max normalized intensity"))
        row_int, self.spin_int_min, self.spin_int_max = self._make_spinbox_row(0.0, 1.0)
        add(row_int)

        # Curvature range spinboxes
        add(SectionLabel("Min / max Gaussian curvature (k1 x k2)"))
        row_curv, self.spin_curv_min, self.spin_curv_max = self._make_spinbox_row(
            -1.0, 1.0
        )
        add(row_curv)

        # Create mask button
        self.btn_create = QPushButton("Create mask")
        self.btn_create.clicked.connect(self._on_create_mask)
        add(self.btn_create)

        # Status label
        self.lbl_status = QLabel("Ready.")
        self.lbl_status.setWordWrap(True)
        add(self.lbl_status)

        root.addStretch()

    @staticmethod
    def _make_spinbox_row(
        lo: float, hi: float, decimals: int = 4
    ) -> tuple[QWidget, QDoubleSpinBox, QDoubleSpinBox]:
        """Return (row_widget, spin_min, spin_max)."""
        row = QWidget()
        layout = QHBoxLayout(row)
        layout.setContentsMargins(0, 0, 0, 0)

        def _spinbox(value: float) -> QDoubleSpinBox:
            sb = QDoubleSpinBox()
            sb.setDecimals(decimals)
            sb.setRange(-1e9, 1e9)
            sb.setSingleStep(10**-decimals)
            sb.setValue(value)
            sb.setMinimumWidth(90)
            return sb

        lbl_min = QLabel("min")
        spin_min = _spinbox(lo)
        lbl_max = QLabel("max")
        spin_max = _spinbox(hi)

        layout.addWidget(lbl_min)
        layout.addWidget(spin_min)
        layout.addWidget(lbl_max)
        layout.addWidget(spin_max)
        return row, spin_min, spin_max

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _set_status(self, msg: str, color: str | None = None) -> None:
        print(f"[mask_widget] {msg}")
        if color:
            self.lbl_status.setText(f"<span style='color:{color}'>{msg}</span>")
        else:
            self.lbl_status.setText(msg)
        QApplication.processEvents()

    def _add_or_update_image(self, data: np.ndarray, name: str, **kwargs) -> None:
        if name in self.viewer.layers:
            self.viewer.layers[name].data = data
        else:
            self.viewer.add_image(data, name=name, **kwargs)

    def _add_or_update_labels(self, data: np.ndarray, name: str) -> None:
        if name in self.viewer.layers:
            self.viewer.layers[name].data = data
        else:
            self.viewer.add_labels(data, name=name)

    def _refresh_intensity_slider(self) -> None:
        arr = self._ch1 if self.combo_channel.currentIndex() == 0 else self._ch2
        if arr is None:
            self._set_status("Channel 2 not loaded.", color="orange")
            return
        lo, hi = float(np.nanmin(arr)), float(np.nanmax(arr))
        if lo == hi:
            lo -= 0.001
            hi += 0.001
        self.spin_int_min.setValue(lo)
        self.spin_int_max.setValue(hi)

    # ------------------------------------------------------------------
    # Load images
    # ------------------------------------------------------------------

    def _on_load_images(self) -> None:
        paths = {
            "ch1": self.pick_ch1.value,
            "ch2": self.pick_ch2.value,  # optional
            "k1": self.pick_k1.value,
            "k2": self.pick_k2.value,
        }
        # ch2 is optional; ch1, k1, k2 are required
        missing = [k for k in ("ch1", "k1", "k2") if not paths[k]]
        if missing:
            self._set_status(
                f"Please select files for: {', '.join(missing)}", color="orange"
            )
            return

        self._set_status("Loading images...")

        try:
            ch1, _, _ = io.read_tiff_with_spacing(paths["ch1"])
            print(f"[mask_widget] ch1 shape={ch1.shape} dtype={ch1.dtype}")
            ch2 = None
            if paths["ch2"]:
                ch2, _, _ = io.read_tiff_with_spacing(paths["ch2"])
                print(f"[mask_widget] ch2 shape={ch2.shape} dtype={ch2.dtype}")
            else:
                print("[mask_widget] ch2 not provided, skipping")
            k1, _, _ = io.read_tiff_with_spacing(paths["k1"])
            print(f"[mask_widget] k1  shape={k1.shape}  dtype={k1.dtype}")
            k2, _, _ = io.read_tiff_with_spacing(paths["k2"])
            print(f"[mask_widget] k2  shape={k2.shape}  dtype={k2.dtype}")
        except Exception as exc:
            self._set_status(f"Error loading: {exc}", color="red")
            traceback.print_exc()
            return

        # Validate shapes for loaded arrays only
        shapes = {"ch1": ch1.shape, "k1": k1.shape, "k2": k2.shape}
        if ch2 is not None:
            shapes["ch2"] = ch2.shape
        if len(set(shapes.values())) > 1:
            detail = ", ".join(f"{k}={v}" for k, v in shapes.items())
            self._set_status(f"Shape mismatch: {detail}", color="red")
            return

        # Update channel combo: only show ch2 option if loaded
        self.combo_channel.blockSignals(True)
        current = self.combo_channel.currentIndex()
        self.combo_channel.clear()
        self.combo_channel.addItem("Channel 1")
        if ch2 is not None:
            self.combo_channel.addItem("Channel 2")
        self.combo_channel.setCurrentIndex(min(current, self.combo_channel.count() - 1))
        self.combo_channel.blockSignals(False)

        try:
            print("[mask_widget] Computing Gaussian curvature...")
            gaussian_curvature = k1.astype(float) * k2.astype(float)

            nan_count = int(np.isnan(gaussian_curvature).sum())
            total = gaussian_curvature.size
            if nan_count:
                print(
                    f"[mask_widget] NaNs in curvature: {nan_count}/{total} ({100 * nan_count / total:.1f}%) — excluded from mask"
                )

            curv_lo = float(np.nanmin(gaussian_curvature))
            curv_hi = float(np.nanmax(gaussian_curvature))
            print(
                f"[mask_widget] Curvature range (NaN-safe): [{curv_lo:.4f}, {curv_hi:.4f}]"
            )

            self._ch1 = ch1
            self._ch2 = ch2
            self._gaussian_curvature = gaussian_curvature

            # Update sliders to data ranges (NaN-safe)
            self._refresh_intensity_slider()

            if curv_lo == curv_hi:
                curv_lo -= 1.0
                curv_hi += 1.0
            self.spin_curv_min.setValue(curv_lo)
            self.spin_curv_max.setValue(curv_hi)

            print("[mask_widget] Adding layers to viewer...")
            self._add_or_update_image(
                ch1, "intensity ch1", colormap="gray", blending="additive"
            )
            if ch2 is not None:
                self._add_or_update_image(
                    ch2, "intensity ch2", colormap="gray", blending="additive"
                )
            self._add_or_update_image(
                k1, "k1", colormap="bwr", blending="additive", visible=False
            )
            self._add_or_update_image(
                k2, "k2", colormap="bwr", blending="additive", visible=False
            )
            self._add_or_update_image(
                gaussian_curvature,
                "gaussian_curvature",
                colormap="magma",
                blending="additive",
                visible=False,
            )

            self._set_status(f"Loaded \u2014 shape {ch1.shape}")

        except Exception as exc:
            self._set_status(f"Error processing: {exc}", color="red")
            traceback.print_exc()

    # ------------------------------------------------------------------
    # Create mask
    # ------------------------------------------------------------------

    def _on_create_mask(self) -> None:
        if self._ch1 is None:
            self._set_status("Please load images first.", color="orange")
            return

        try:
            if self.combo_channel.currentIndex() == 1 and self._ch2 is None:
                self._set_status("Channel 2 not loaded.", color="orange")
                return
            intensity = (
                self._ch1 if self.combo_channel.currentIndex() == 0 else self._ch2
            )
            gaussian_curvature = self._gaussian_curvature

            int_lo, int_hi = self.spin_int_min.value(), self.spin_int_max.value()
            curv_lo, curv_hi = self.spin_curv_min.value(), self.spin_curv_max.value()

            print(
                f"[mask_widget] intensity [{int_lo:.4f}, {int_hi:.4f}]  curvature [{curv_lo:.4f}, {curv_hi:.4f}]"
            )

            # NaN voxels in curvature or intensity are excluded (treated as False)
            mask = (
                (intensity >= int_lo)
                & (intensity <= int_hi)
                & (gaussian_curvature >= curv_lo)
                & (gaussian_curvature <= curv_hi)
                & ~np.isnan(gaussian_curvature)
                & ~np.isnan(intensity.astype(float))
            ).astype(np.uint8) * 255

            n_true = int((mask == 255).sum())
            self._set_status(
                f"Mask created \u2014 {n_true:,} voxels selected ({100 * n_true / mask.size:.2g}%)"
            )

            self._add_or_update_image(
                mask, "mask", colormap="green", blending="additive"
            )
            self.result_mask = mask

        except Exception as exc:
            self._set_status(f"Error creating mask: {exc}", color="red")
            traceback.print_exc()


if __name__ == "__main__":
    viewer = napari.Viewer()
    widget = MaskWidget(viewer)
    viewer.window.add_dock_widget(widget, area="right", name="Create Mask")
    napari.run()
