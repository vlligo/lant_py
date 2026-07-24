"""Port of MainWindow (PySide6). Close 1:1 layout/behavior match to the
original Qt C++ app, plus a TensorBoard toggle for logging statistics."""
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QAction, QCursor
from PySide6.QtWidgets import (QCheckBox, QComboBox, QFileDialog, QGridLayout,
                                QGroupBox, QHBoxLayout, QHeaderView, QInputDialog,
                                QLabel, QLineEdit, QMainWindow, QMenu, QMessageBox,
                                QPushButton, QScrollArea, QSpinBox, QTableWidget,
                                QTableWidgetItem, QVBoxLayout, QWidget)

from engine.common import DIRECTION_SYMBOLS, PRESETS, expand_rule_shorthand
from engine.highway import (HighwayStatus, KNOWN_HIGHWAYS, get_highway_status,
                            supports_highway_detection)
from gui.antfieldwidget import AntFieldWidget, DisplayStyle
from gui.highway_worker import HighwayScanner
from gui.int_spinbox import IntSpinBox
from stats.tb_logger import TensorBoardLogger, TENSORBOARD_AVAILABLE

STEP_COUNT = 10_000_000  # steps per auto-run timer tick, matches the original


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.presets = PRESETS
        self.last_custom_steps = 1000
        self.current_style_index = 0
        self.tb_logger: TensorBoardLogger | None = None

        # Cached highway result. The onset is an absolute step number, so
        # this is computed once per configuration change and the live
        # countdown is then just subtraction — no rescanning per frame.
        self.highway_status: HighwayStatus = HighwayStatus()
        self.highway_scanner = HighwayScanner(self)

        self._setup_ui()
        self._setup_connections()

        self.rules_edit.setText("LR")
        self._update_rules()

        self.setWindowTitle("Langton's Ant with Statistics (Python)")
        self.resize(1200, 800)

    # ------------------------------------------------------------------ #
    def _setup_ui(self):
        central = QWidget(self)
        self.setCentralWidget(central)
        main_layout = QVBoxLayout(central)

        file_menu = self.menuBar().addMenu("&File")
        self.save_action = file_menu.addAction("&Save State")
        self.load_action = file_menu.addAction("&Load State")
        file_menu.addSeparator()
        self.exit_action = file_menu.addAction("E&xit")

        sim_menu = self.menuBar().addMenu("&Simulation")
        self.reset_stats_action = sim_menu.addAction("&Reset Statistics")

        stats_menu = self.menuBar().addMenu("&Stats Logging")
        self.tb_action = QAction("Enable TensorBoard Logging", self, checkable=True)
        if not TENSORBOARD_AVAILABLE:
            self.tb_action.setEnabled(False)
            self.tb_action.setText("Enable TensorBoard Logging (torch not installed)")
        stats_menu.addAction(self.tb_action)

        self.stats_label = QLabel()
        self.coordinate_label = QLabel("Mouse: (0, 0)")
        self.statusBar().addPermanentWidget(self.stats_label)
        self.statusBar().addPermanentWidget(self.coordinate_label)
        self.statusBar().showMessage("Ready")

        control_group = QGroupBox("Controls")
        grid = QGridLayout(control_group)
        row = 0

        grid.addWidget(QLabel("Rules:"), row, 0)
        self.rules_edit = QLineEdit()
        grid.addWidget(self.rules_edit, row, 1)
        self.rules_button = QPushButton("Update Rules")
        grid.addWidget(self.rules_button, row, 2)

        self.preset_combo = QComboBox()
        self.preset_combo.addItems([name for _, name in
                                    sorted(((k, v[0]) for k, v in self.presets.items()))])
        grid.addWidget(self.preset_combo, row, 3)

        self.rules_label = QLabel("Rules: LR")
        grid.addWidget(self.rules_label, row, 4, 1, 2)
        self.save_button = QPushButton("Save State")
        grid.addWidget(self.save_button, row, 6)
        self.load_button = QPushButton("Load State")
        grid.addWidget(self.load_button, row, 7)
        row += 1

        self.step_button = QPushButton("Step (1)")
        grid.addWidget(self.step_button, row, 0)

        self.quick_steps_spin = IntSpinBox()
        self.quick_steps_spin.setRange(1, 2**63 - 1)
        self.quick_steps_spin.setValue(self.last_custom_steps)
        self.quick_steps_spin.setSingleStep(1000)
        self.quick_steps_spin.setSuffix(" steps")
        grid.addWidget(self.quick_steps_spin, row, 1)

        self.quick_steps_button = QPushButton("Run")
        grid.addWidget(self.quick_steps_button, row, 2)

        self.start_button = QPushButton("Start")
        grid.addWidget(self.start_button, row, 3)
        self.stop_button = QPushButton("Stop")
        self.stop_button.setEnabled(False)
        grid.addWidget(self.stop_button, row, 4)

        self.reset_button = QPushButton("Reset")
        grid.addWidget(self.reset_button, row, 5)

        self.stats_checkbox = QCheckBox("Track Statistics")
        self.stats_checkbox.setChecked(True)
        grid.addWidget(self.stats_checkbox, row, 6)
        row += 1

        grid.addWidget(QLabel("Radius:"), row, 0)
        self.radius_spin = QSpinBox()
        self.radius_spin.setRange(1, 100_000)
        self.radius_spin.setValue(10)
        self.radius_spin.setSuffix(" cells")
        self.radius_spin.setGroupSeparatorShown(True)
        grid.addWidget(self.radius_spin, row, 1)

        self.randomize_button = QPushButton("Randomize Area")
        grid.addWidget(self.randomize_button, row, 2)
        row += 1

        self.center_button = QPushButton("Center on Ant")
        grid.addWidget(self.center_button, row, 0)
        self.zoom_out_button = QPushButton("Zoom Out")
        grid.addWidget(self.zoom_out_button, row, 1)
        self.zoom_in_button = QPushButton("Zoom In")
        grid.addWidget(self.zoom_in_button, row, 2)
        self.zoom_label = QLabel("Zoom: 1.0x")
        grid.addWidget(self.zoom_label, row, 3)
        self.cell_size_button = QPushButton("Cell Size...")
        grid.addWidget(self.cell_size_button, row, 4)
        self.toggle_panel_button = QPushButton("Show Statistics Panel")
        grid.addWidget(self.toggle_panel_button, row, 5)
        self.center_most_visited_button = QPushButton("Center Most Visited")
        grid.addWidget(self.center_most_visited_button, row, 6)
        self.center_coordinates_button = QPushButton("Center Coordinates...")
        grid.addWidget(self.center_coordinates_button, row, 7)
        row += 1

        self.left_button = QPushButton("\u2190")
        self.left_button.setFixedSize(40, 30)
        grid.addWidget(self.left_button, row, 0)
        self.up_button = QPushButton("\u2191")
        self.up_button.setFixedSize(40, 30)
        grid.addWidget(self.up_button, row, 1)
        self.down_button = QPushButton("\u2193")
        self.down_button.setFixedSize(40, 30)
        grid.addWidget(self.down_button, row, 2)
        self.right_button = QPushButton("\u2192")
        self.right_button.setFixedSize(40, 30)
        grid.addWidget(self.right_button, row, 3)
        self.center_table_button = QPushButton("Center Selected")
        grid.addWidget(self.center_table_button, row, 4)
        self.center_stats_button = QPushButton("Center Stats...")
        grid.addWidget(self.center_stats_button, row, 5)
        self.steps_label = QLabel("Total steps: 0")
        grid.addWidget(self.steps_label, row, 6, 1, 2)
        row += 1

        stats_group = QGroupBox("Quick Statistics")
        stats_layout = QGridLayout(stats_group)
        self.unique_cells_label = QLabel("Unique cells: 0")
        self.most_visited_label = QLabel("Most visited: (0,0)")
        self.max_visits_label = QLabel("Max visits: 0")
        self.average_visits_label = QLabel("Average: 0.0")
        stats_layout.addWidget(self.unique_cells_label, 0, 0)
        stats_layout.addWidget(self.most_visited_label, 0, 1)
        stats_layout.addWidget(self.max_visits_label, 0, 2)
        stats_layout.addWidget(self.average_visits_label, 0, 3)
        self.cell_details_button = QPushButton("Show Cell Details")
        stats_layout.addWidget(self.cell_details_button, 0, 4)
        self.stats_dialog_button = QPushButton("Detailed Statistics")
        stats_layout.addWidget(self.stats_dialog_button, 0, 5)

        self.highway_label = QLabel("Highway: n/a")
        stats_layout.addWidget(self.highway_label, 1, 0, 1, 4)
        self.rescan_highway_button = QPushButton("Rescan Highway")
        self.rescan_highway_button.setToolTip(
            "Simulate ahead from the current state to locate the highway.")
        stats_layout.addWidget(self.rescan_highway_button, 1, 4, 1, 2)
        grid.addWidget(stats_group, row, 0, 1, 6)

        self.style_button = QPushButton("Next Style")
        grid.addWidget(self.style_button, row, 7)

        self.ant_field = AntFieldWidget()
        scroll_area = QScrollArea()
        scroll_area.setWidget(self.ant_field)
        scroll_area.setWidgetResizable(True)
        scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)

        main_layout.addWidget(control_group)
        main_layout.addWidget(scroll_area, 1)

        self.stats_table_group = QGroupBox("Top 20 Most Visited Cells")
        table_layout = QVBoxLayout(self.stats_table_group)
        self.stats_table = QTableWidget()
        self.stats_table.setColumnCount(2)
        self.stats_table.setHorizontalHeaderLabels(["Cell (X,Y)", "Visits"])
        self.stats_table.horizontalHeader().setStretchLastSection(True)
        self.stats_table.setMaximumHeight(200)
        table_layout.addWidget(self.stats_table)
        self.stats_table_group.setVisible(False)
        main_layout.addWidget(self.stats_table_group)

        self.statusBar().showMessage("Ready")

    # ------------------------------------------------------------------ #
    def _setup_connections(self):
        self.save_action.triggered.connect(self._save_state)
        self.load_action.triggered.connect(self._load_state)
        self.save_button.clicked.connect(self._save_state)
        self.load_button.clicked.connect(self._load_state)
        self.reset_stats_action.triggered.connect(self._reset_statistics_action)
        self.exit_action.triggered.connect(self.close)
        self.tb_action.toggled.connect(self._toggle_tensorboard)

        self.ant_field.antMoved.connect(self._on_ant_moved)
        self.ant_field.zoomChanged.connect(self._on_zoom_changed)
        self.ant_field.stepsChanged.connect(
            lambda steps: self.steps_label.setText(f"Total steps: {steps:,}"))
        self.ant_field.mouseOverCell.connect(self._on_mouse_over_cell)

        self.auto_run_timer = QTimer(self)
        self.auto_run_timer.timeout.connect(self._auto_step)

        self.start_button.clicked.connect(self._start_simulation)
        self.stop_button.clicked.connect(self._stop_simulation)
        self.rules_button.clicked.connect(self._update_rules)
        self.step_button.clicked.connect(self._take_step)
        self.quick_steps_button.clicked.connect(self._on_quick_steps_clicked)
        self.reset_button.clicked.connect(self._reset_simulation)
        self.center_button.clicked.connect(self.ant_field.center_on_ant)
        self.center_most_visited_button.clicked.connect(self._center_on_most_visited)
        self.center_coordinates_button.clicked.connect(self._center_on_coordinates)
        self.center_table_button.clicked.connect(self._center_on_table_cell)
        self.center_stats_button.clicked.connect(self._center_on_selected_statistic)
        self.zoom_in_button.clicked.connect(lambda: self.ant_field.set_zoom(self.ant_field.get_zoom() * 1.2))
        self.zoom_out_button.clicked.connect(lambda: self.ant_field.set_zoom(self.ant_field.get_zoom() / 1.2))
        self.cell_size_button.clicked.connect(self._change_cell_size)
        self.randomize_button.clicked.connect(self._randomize_area)

        self.left_button.clicked.connect(lambda: self.ant_field.move_view(50, 0))
        self.right_button.clicked.connect(lambda: self.ant_field.move_view(-50, 0))
        self.up_button.clicked.connect(lambda: self.ant_field.move_view(0, 50))
        self.down_button.clicked.connect(lambda: self.ant_field.move_view(0, -50))

        self.style_button.clicked.connect(self._change_style)
        self.preset_combo.activated.connect(self._load_preset)
        self.rules_edit.returnPressed.connect(self._update_rules)

        self.cell_details_button.clicked.connect(self._show_cell_details)
        self.stats_dialog_button.clicked.connect(self._show_statistics)
        self.toggle_panel_button.clicked.connect(self._toggle_stats_panel)
        self.stats_checkbox.toggled.connect(self._toggle_statistics)

        self.highway_scanner.started.connect(self._on_highway_scan_started)
        self.highway_scanner.resultReady.connect(self._on_highway_result)
        self.rescan_highway_button.clicked.connect(
            lambda: self._refresh_highway(force_scan=True))

        self.stats_timer = QTimer(self)
        self.stats_timer.timeout.connect(self._periodic_stats_update)
        self.stats_timer.start(500)

    # ------------------------------------------------------------------ #
    # Slots
    # ------------------------------------------------------------------ #
    def _change_style(self):
        self.current_style_index = (self.current_style_index + 1) % 5
        self.ant_field.set_display_style(DisplayStyle(self.current_style_index))

    def _randomize_area(self):
        radius = self.radius_spin.value()
        estimated_bytes = self.ant_field.estimate_randomize_area_bytes(radius)
        warn_threshold = 10_000 * 1024 * 1024

        if estimated_bytes > warn_threshold:
            estimated_gb = estimated_bytes / (1024.0 ** 3)
            reply = QMessageBox.warning(
                self, "Large Randomize Area",
                f"A radius of {radius:,} needs roughly {estimated_gb:.1f} GB of memory to "
                "fill, and may exhaust your system's RAM or take a long time.\n\nContinue anyway?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No)
            if reply != QMessageBox.StandardButton.Yes:
                self.statusBar().showMessage("Randomize cancelled.", 3000)
                return

        self.ant_field.randomize_area(radius)
        # The randomized cells change where (and whether) the highway forms,
        # so the cached onset is now meaningless — find the new one.
        self._refresh_highway()
        side = 2 * radius + 1
        self.statusBar().showMessage(f"Randomized a {side:,}x{side:,} area around the ant.", 3000)

    def _on_mouse_over_cell(self, x, y):
        pattern = "Vertical" if (x + y) % 2 == 0 else "Horizontal"
        self.coordinate_label.setText(f"Mouse: ({x:,}; {y:,}) [{pattern}]")

    def _update_quick_statistics(self):
        if not self.stats_checkbox.isChecked():
            self.unique_cells_label.setText("Unique cells: 0")
            self.most_visited_label.setText("Most visited: (0,0)")
            self.max_visits_label.setText("Max visits: 0")
            self.average_visits_label.setText("Average: 0.0")
            self.stats_label.clear()
            return

        summary = self.ant_field.get_statistics_summary()
        self.unique_cells_label.setText(f"Unique cells: {summary.unique_cells_visited:,}")
        mv = summary.most_visited_cell
        self.most_visited_label.setText(f"Most visited: ({mv[0]:,}, {mv[1]:,})")
        self.max_visits_label.setText(f"Max visits: {summary.max_visits_per_cell:,}")
        self.average_visits_label.setText(f"Average: {summary.average_visits:.2f}")

        if summary.max_visits_per_cell > 0:
            self.stats_label.setText(
                f"Most visited: ({mv[0]:,}, {mv[1]:,}) = {summary.max_visits_per_cell:,} times")

        self._update_highway_label()

        if self.tb_logger is not None and self.tb_logger.enabled:
            self.tb_logger.log_summary(summary, self.ant_field.get_rules())

    # -- highway countdown ---------------------------------------------- #
    def _refresh_highway(self, force_scan: bool = False):
        """Decide how to obtain the highway onset, and update the label.

        Three cases:
          * unsupported rule      -> nothing to show
          * pristine grid         -> known constant, instant, no scan
          * randomized / loaded   -> no constant exists, so scan ahead on
                                     a background thread
        """
        engine = self.ant_field.engine
        rules = engine.rules

        if not supports_highway_detection(rules):
            self.highway_status = HighwayStatus(
                reason=f"No known highway period for rule '{rules}'. "
                       f"Supported: {', '.join(sorted(KNOWN_HIGHWAYS))}.")
            self.rescan_highway_button.setEnabled(False)
            self._update_highway_label()
            return

        self.rescan_highway_button.setEnabled(True)
        pristine = getattr(engine, "grid_pristine", False)

        if pristine and not force_scan:
            # Deterministic run from an empty grid: the onset is a constant.
            self.highway_status = get_highway_status(rules, engine.step_count, True)
            self._update_highway_label()
            return

        # Perturbed grid: the onset depends on every randomized cell, so it
        # has to be found by simulating ahead.
        self.highway_scanner.request(engine)

    def _on_highway_scan_started(self):
        self.highway_label.setText("Highway: scanning ahead\u2026")
        self.highway_label.setToolTip(
            "Simulating forward on a background thread to locate the highway.")

    def _on_highway_result(self, status: HighwayStatus):
        self.highway_status = status
        self._update_highway_label()
        if status.known:
            self.statusBar().showMessage(
                f"Highway located after simulating {status.steps_scanned:,} steps ahead.", 4000)

    def _update_highway_label(self):
        current = self.highway_status.for_step(self.ant_field.engine.step_count)
        self.highway_label.setText(current.format())
        self.highway_label.setToolTip(current.tooltip())

    def _update_rules(self):
        expanded, compressed, error = expand_rule_shorthand(self.rules_edit.text())
        if error:
            QMessageBox.warning(self, "Invalid Rules", error)
            return

        if self.rules_edit.text() != compressed:
            self.rules_edit.setText(compressed)

        self.ant_field.set_rules(expanded)
        self.rules_label.setText(f"Rules: {compressed}")
        self._refresh_highway()
        self._update_quick_statistics()

    def _on_ant_moved(self, x, y, direction, steps):
        dir_str = DIRECTION_SYMBOLS[direction] if 0 <= direction < 4 else "?"
        self.statusBar().showMessage(f"Ant: ({x:,}, {y:,}) {dir_str} | Steps: {steps:,}")

    def _update_statistics_table(self):
        if not self.stats_checkbox.isChecked():
            self.stats_table.clearContents()
            self.stats_table.setRowCount(0)
            return

        top_cells = self.ant_field.get_top_visited_cells(20)
        self.stats_table.setRowCount(len(top_cells))
        for i, ((x, y), visits) in enumerate(top_cells):
            self.stats_table.setItem(i, 0, QTableWidgetItem(f"({x}, {y})"))
            self.stats_table.setItem(i, 1, QTableWidgetItem(str(visits)))

    def _take_step(self):
        self.ant_field.next_step(1)
        self._update_quick_statistics()

    def _on_quick_steps_clicked(self):
        steps = self.quick_steps_spin.value()
        self.last_custom_steps = steps
        self.ant_field.next_step(steps)
        self._update_quick_statistics()

    def _reset_simulation(self):
        self.ant_field.reset()
        self._refresh_highway()
        self._update_quick_statistics()
        self._update_statistics_table()

    def _center_on_most_visited(self):
        x, y = self.ant_field.get_most_visited_cell()
        self.ant_field.center_on_point(x, y)

    def _center_on_coordinates(self):
        text, ok = QInputDialog.getText(self, "Center on Coordinates",
                                        "Enter coordinates to center on (x,y):",
                                        QLineEdit.EchoMode.Normal, "0,0")
        if ok and text:
            self._parse_and_center(text)

    def _parse_and_center(self, text: str):
        coords = text.split(",")
        if len(coords) == 2:
            try:
                x, y = int(coords[0].strip()), int(coords[1].strip())
                self.ant_field.center_on_point(x, y)
                return
            except ValueError:
                pass
            QMessageBox.warning(self, "Invalid Input", "Please enter valid integers for coordinates.")
        else:
            QMessageBox.warning(self, "Invalid Format", "Please enter coordinates in the format: x,y")

    def _center_on_table_cell(self):
        row = self.stats_table.currentRow()
        if row < 0:
            QMessageBox.information(self, "No Selection",
                                    "Please select a cell from the statistics table first.")
            return
        item = self.stats_table.item(row, 0)
        if item:
            text = item.text()[1:-1]  # strip parentheses
            coords = text.split(",")
            if len(coords) == 2:
                try:
                    x, y = int(coords[0].strip()), int(coords[1].strip())
                    self.ant_field.center_on_point(x, y)
                except ValueError:
                    pass

    def _center_on_selected_statistic(self):
        top_cells = self.ant_field.get_top_visited_cells(5)
        if not top_cells:
            QMessageBox.information(self, "No Statistics", "No statistics available yet.")
            return

        menu = QMenu(self)
        actions = {}
        for i, ((x, y), visits) in enumerate(top_cells):
            action = menu.addAction(f"{i + 1}. ({x:,}, {y:,}) - {visits:,} visits")
            actions[action] = (x, y)

        selected = menu.exec(QCursor.pos())
        if selected in actions:
            self.ant_field.center_on_point(*actions[selected])

    def _change_cell_size(self):
        current = self.ant_field.get_cell_size()
        size, ok = QInputDialog.getInt(self, "Cell Size", "Enter cell size (pixels):",
                                       current, 1, 50, 1)
        if ok:
            self.ant_field.set_cell_size(size)

    def _load_preset(self, index):
        if index in self.presets:
            _, rules = self.presets[index]
            self.rules_edit.setText(rules)
            self._update_rules()

    def _show_statistics(self):
        summary = self.ant_field.get_statistics_summary()
        top_cells = self.ant_field.get_top_visited_cells(20)

        lines = [
            "Simulation Statistics", "====================",
            f"Total Steps: {summary.total_cells_visited:,}",
            f"Unique Cells Visited: {summary.unique_cells_visited:,}",
            f"Most Visited Cell: ({summary.most_visited_cell[0]:,}, "
            f"{summary.most_visited_cell[1]:,}) [{summary.max_visits_per_cell:,} times]",
            f"Average Visits per Cell: {summary.average_visits:.2f}",
            f"Simulation Time: {summary.simulation_time_ms} ms",
            "", "Top 20 Most Visited Cells:",
        ]
        for i, ((x, y), visits) in enumerate(top_cells):
            lines.append(f"{i + 1}. ({x:,}, {y:,}): {visits:,} visits")

        QMessageBox.information(self, "Detailed Statistics", "\n".join(lines))

    def _reset_statistics_action(self):
        self.ant_field.reset_statistics()
        self._update_quick_statistics()
        self._update_statistics_table()

    def _toggle_statistics(self, enabled):
        self.ant_field.set_statistics_enabled(enabled)
        self._update_quick_statistics()
        if not enabled:
            self.stats_label.clear()
            if self.stats_table_group.isVisible():
                self.stats_table.clearContents()
                self.stats_table.setRowCount(0)

    def _toggle_stats_panel(self):
        visible = not self.stats_table_group.isVisible()
        self.stats_table_group.setVisible(visible)
        self.toggle_panel_button.setText("Hide Statistics Panel" if visible else "Show Statistics Panel")

    def _toggle_tensorboard(self, enabled):
        if enabled:
            self.tb_logger = TensorBoardLogger(run_name=f"rules_{self.ant_field.get_rules()}")
            if self.tb_logger.enabled:
                self.statusBar().showMessage(
                    f"TensorBoard logging to {self.tb_logger.run_path} "
                    f"(run: tensorboard --logdir runs)", 5000)
        elif self.tb_logger is not None:
            self.tb_logger.close()
            self.tb_logger = None

    def _save_state(self):
        default_name = f"{self.ant_field.get_rules()}.ant"
        filename, _ = QFileDialog.getSaveFileName(self, "Save Simulation State", default_name,
                                                  "Ant State Files (*.ant);;All Files (*)")
        if filename:
            if self.ant_field.save_state(filename):
                self.statusBar().showMessage(f"State saved to {filename}", 3000)
            else:
                QMessageBox.critical(self, "Save Error", "Failed to save the simulation state.")

    def _load_state(self):
        filename, _ = QFileDialog.getOpenFileName(self, "Load Simulation State", "",
                                                  "Ant State Files (*.ant);;All Files (*)")
        if filename:
            if self.ant_field.load_state(filename):
                loaded_rules = self.ant_field.get_rules()
                self.rules_edit.setText(loaded_rules)
                self.rules_label.setText(f"Rules: {loaded_rules}")
                self.stats_checkbox.setChecked(self.ant_field.is_statistics_enabled())
                self._refresh_highway()
                self._update_quick_statistics()
                if self.stats_table_group.isVisible():
                    self._update_statistics_table()
                self.statusBar().showMessage(f"State loaded from {filename}", 3000)
            else:
                QMessageBox.critical(self, "Load Error", "Failed to load the simulation state.")

    def _show_cell_details(self):
        text, ok = QInputDialog.getText(self, "Check Cell", "Enter cell coordinates (x,y):",
                                        QLineEdit.EchoMode.Normal, "0,0")
        if not (ok and text):
            return
        coords = text.split(",")
        if len(coords) != 2:
            QMessageBox.warning(self, "Invalid Format", "Please enter coordinates in the format: x,y")
            return
        try:
            x, y = int(coords[0].strip()), int(coords[1].strip())
        except ValueError:
            QMessageBox.warning(self, "Invalid Input", "Please enter valid integers for coordinates.")
            return
        visits = self.ant_field.get_visit_count(x, y)
        QMessageBox.information(self, "Cell Details", f"Cell ({x:,}, {y:,}) has been visited {visits:,} times")

    def _on_zoom_changed(self, zoom):
        self.zoom_label.setText(f"Zoom: {zoom:.1f}x")

    def _start_simulation(self):
        self.start_button.setEnabled(False)
        self.stop_button.setEnabled(True)
        self.auto_run_timer.start(0)

    def _stop_simulation(self):
        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        self.auto_run_timer.stop()

    def _auto_step(self):
        self.ant_field.next_step(STEP_COUNT)
        self._update_quick_statistics()

    def _periodic_stats_update(self):
        if self.stats_checkbox.isChecked():
            self._update_quick_statistics()
            if self.stats_table_group.isVisible():
                self._update_statistics_table()

    def closeEvent(self, event):
        self.highway_scanner.shutdown()
        if self.tb_logger is not None:
            self.tb_logger.close()
        super().closeEvent(event)
