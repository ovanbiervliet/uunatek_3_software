"""
grbl_streamer.py -- Threaded GRBL sender for the iDraw/DrawCore board.

Streaming uses GRBL's character-counting protocol: we track how many
bytes are sitting in the controller's 128-byte serial receive buffer
and keep it full instead of waiting for an 'ok' after every line.
This keeps the motion planner's lookahead queue populated so the
machine can actually sustain high feed rates through short segments,
instead of decelerating between every line.

Also supports:
  * live pause / resume (GRBL feed hold '!' and cycle start '~')
  * abort (Ctrl-X soft reset, pen-up, re-home so carriage returns Home)
  * blocking waits between color layers for manual pen swaps
  * firmware settings query ($$) so the GUI can report speed caps
"""

from __future__ import annotations

import re
import threading
import time
from collections import deque

import serial
import serial.tools.list_ports
from PySide6.QtCore import QThread, Signal

BAUD = 115200
RX_BUFFER = 128            # GRBL 1.1 serial receive buffer size
RT_FEED_HOLD = b"!"
RT_CYCLE_START = b"~"
RT_SOFT_RESET = b"\x18"
RT_STATUS = b"?"


def list_serial_ports() -> list:
    """[(device, description), ...] for the port picker."""
    return [(p.device, p.description) for p in serial.tools.list_ports.comports()]


class GrblConnection:
    """Thin wrapper over the serial port shared by GUI and stream thread."""

    def __init__(self):
        self.ser: serial.Serial | None = None
        self.lock = threading.Lock()   # guards writes (realtime vs stream)

    @property
    def connected(self) -> bool:
        return self.ser is not None and self.ser.is_open

    def open(self, port: str) -> str:
        self.close()
        self.ser = serial.Serial(port, BAUD, timeout=2)
        self.ser.write(b"\r\n\r\n")            # wake GRBL
        time.sleep(2.0)
        banner = self.ser.read_all().decode(errors="replace").strip()
        self.ser.reset_input_buffer()
        return banner or "(no banner received -- check the port)"

    def close(self):
        if self.ser is not None:
            try:
                self.ser.close()
            except Exception:
                pass
        self.ser = None

    def send_realtime(self, code: bytes):
        with self.lock:
            if self.connected:
                self.ser.write(code)

    def _write_line(self, line: str):
        with self.lock:
            self.ser.write((line.strip() + "\n").encode())

    def send_line_wait_ok(self, line: str, timeout: float = 30.0) -> str:
        """Send one line, block until 'ok' or 'error'. For short commands."""
        if not self.connected:
            raise RuntimeError("Not connected")
        self._write_line(line)
        deadline = time.time() + timeout
        while time.time() < deadline:
            resp = self.ser.readline().decode(errors="replace").strip()
            if resp == "ok":
                return "ok"
            if resp.startswith(("error", "ALARM")):
                return resp
        raise TimeoutError(f"No response to: {line}")

    def query(self, line: str, timeout: float = 10.0) -> list:
        """Send a $-command and return the response lines before 'ok'."""
        if not self.connected:
            raise RuntimeError("Not connected")
        self.ser.reset_input_buffer()
        self._write_line(line)
        lines = []
        deadline = time.time() + timeout
        while time.time() < deadline:
            resp = self.ser.readline().decode(errors="replace").strip()
            if resp == "ok" or resp.startswith("error"):
                return lines
            if resp:
                lines.append(resp)
        return lines

    def read_settings(self) -> dict:
        """Return {setting_number: float_value} from '$$'."""
        settings = {}
        for line in self.query("$$", timeout=5):
            m = re.match(r"\$(\d+)=([-\d.]+)", line)
            if m:
                settings[int(m.group(1))] = float(m.group(2))
        return settings


class StreamJob(QThread):
    """Streams a multi-layer job with character-counting flow control."""

    progress = Signal(int, int)          # lines_done, lines_total
    layer_started = Signal(int, str)     # layer index, color hex
    need_pen_change = Signal(int, str)   # layer index, color hex
    console = Signal(str)
    finished_ok = Signal()
    failed = Signal(str)

    def __init__(self, conn: GrblConnection, layer_jobs: list,
                 header: list, footer: list,
                 pen_up_z: float = 0.0, z_feed: float = 2500.0, parent=None):
        super().__init__(parent)
        self.conn = conn
        self.layer_jobs = layer_jobs
        self.header = header
        self.footer = footer
        self.pen_up_z = pen_up_z
        self.z_feed = z_feed
        self._abort = threading.Event()
        self._paused = threading.Event()
        self._pen_ready = threading.Event()
        self._done = 0
        self._total = 1

    # ---- controls called from the GUI thread ----
    def pause(self):
        self.conn.send_realtime(RT_FEED_HOLD)
        self._paused.set()
        self.console.emit(">> paused (feed hold)")

    def resume(self):
        self.conn.send_realtime(RT_CYCLE_START)
        self._paused.clear()
        self.console.emit(">> resumed")

    def abort(self):
        """Request stop. Feed-hold decelerates the machine smoothly; the
        soft reset is sent by the recovery routine only after motion has
        actually stopped (resetting mid-motion slams the steppers and
        loses position -- the 'grinding' failure mode)."""
        self._abort.set()
        self._pen_ready.set()
        self._paused.clear()
        self.conn.send_realtime(RT_FEED_HOLD)

    def confirm_pen_changed(self):
        self._pen_ready.set()

    # ---- streaming internals ----
    def _check_abort(self):
        if self._abort.is_set():
            raise InterruptedError

    def _stream_block(self, lines: list):
        """Send lines keeping GRBL's RX buffer full (character counting)."""
        pending = deque()        # byte length of each un-acked line
        payloads = deque()       # the lines themselves, for error reports
        i, n = 0, len(lines)
        while i < n or pending:
            self._check_abort()
            while self._paused.is_set() and not self._abort.is_set():
                time.sleep(0.05)     # hold: stop feeding, keep motion held
            self._check_abort()

            # Stuff the receive buffer with as many lines as fit.
            while i < n:
                data = (lines[i].strip() + "\n").encode()
                if pending and sum(pending) + len(data) > RX_BUFFER - 1:
                    break
                with self.conn.lock:
                    self.conn.ser.write(data)
                pending.append(len(data))
                payloads.append(lines[i])
                i += 1

            # Collect one response (short serial timeout keeps this live).
            resp = self.conn.ser.readline().decode(errors="replace").strip()
            if not resp:
                continue
            if resp == "ok":
                if pending:
                    pending.popleft()
                    payloads.popleft()
                self._done += 1
                self.progress.emit(self._done, self._total)
            elif resp.startswith(("error", "ALARM")):
                culprit = payloads[0] if payloads else "?"
                raise RuntimeError(f"GRBL said '{resp}' for: {culprit}")
            # anything else (status reports, banners) is ignored

    def _wait_for_idle(self, timeout: float = 120.0):
        """Poll '?' until the machine reports Idle (motion finished)."""
        deadline = time.time() + timeout
        while time.time() < deadline:
            self._check_abort()
            self.conn.send_realtime(RT_STATUS)
            resp = self.conn.ser.readline().decode(errors="replace").strip()
            if resp.startswith("<Idle"):
                return
            if not resp.startswith("<"):
                time.sleep(0.2)

    # ---- worker ----
    def run(self):
        self._total = (len(self.header) + len(self.footer) + 1
                       + sum(len(g) + 1 for _, g in self.layer_jobs))
        try:
            old_timeout = self.conn.ser.timeout
            self.conn.ser.timeout = 0.1     # keep the stream loop responsive

            self._stream_block(self.header)

            for idx, (color, gcode) in enumerate(self.layer_jobs):
                self._check_abort()
                # Park at Home and wait for motion to stop before prompting.
                self._stream_block(["G0 X0 Y0"])
                self._wait_for_idle()
                self._pen_ready.clear()
                self.need_pen_change.emit(idx, color)
                self.console.emit(f">> waiting for pen {color} "
                                  f"(layer {idx + 1}/{len(self.layer_jobs)})")
                while not self._pen_ready.wait(timeout=0.2):
                    self._check_abort()
                self._check_abort()
                self.layer_started.emit(idx, color)
                self._stream_block(gcode)

            self._stream_block(self.footer + ["G4 P0"])   # G4 = sync/flush
            self._wait_for_idle()
            self.conn.ser.timeout = old_timeout
            self.finished_ok.emit()

        except InterruptedError:
            self._recover_after_abort()
            self.failed.emit("Plot aborted by user.")
        except Exception as exc:
            self.failed.emit(str(exc))

    def _status_line(self, wait: float = 0.4) -> str:
        """Request and return one <...> status report ('' on timeout)."""
        self.conn.send_realtime(RT_STATUS)
        deadline = time.time() + wait
        while time.time() < deadline:
            line = self.conn.ser.readline().decode(errors="replace").strip()
            if line.startswith("<"):
                return line
        return ""

    def _recover_after_abort(self):
        """
        Stop sequence: the feed hold sent by abort() is decelerating the
        machine. Wait until motion has fully stopped (and grab the work
        coordinate offset from the status reports while we do), THEN
        soft-reset, clear the alarm, lift the pen, and return to Home --
        via $H if the firmware supports homing, otherwise by moving to
        the recorded origin in machine coordinates (G53).
        """
        try:
            if not self.conn.connected:
                return
            self.console.emit(">> stopping (decelerating)...")
            wco = None
            stopped = False
            for _ in range(60):                     # up to ~6 s
                line = self._status_line()
                if not line:
                    continue
                m = re.search(r"WCO:(-?[\d.]+),(-?[\d.]+)", line)
                if m:
                    wco = (float(m.group(1)), float(m.group(2)))
                state = line[1:].split("|", 1)[0]
                if state.startswith(("Hold:0", "Idle", "Alarm")):
                    stopped = True
                    if wco is not None:
                        break
                time.sleep(0.1)
            if not stopped:
                self.console.emit(">> machine did not report a stop; "
                                  "resetting anyway")

            self.conn.send_realtime(RT_SOFT_RESET)
            time.sleep(1.2)
            self.conn.ser.timeout = 2
            self.conn.ser.reset_input_buffer()
            self.conn.send_line_wait_ok("$X", timeout=5)
            self.conn.send_line_wait_ok("G90", timeout=5)
            self.conn.send_line_wait_ok(
                f"G1 Z{self.pen_up_z:g} F{self.z_feed:g}", timeout=10)

            self.console.emit(">> returning to Home...")
            resp = self.conn.send_line_wait_ok("$H", timeout=120)
            if resp == "ok":
                self.conn.send_line_wait_ok("G92 X0 Y0", timeout=5)
                self.console.emit(">> homed; parked at Home, origin restored")
            elif wco is not None:
                self.console.emit(f">> homing unavailable ({resp}); "
                                  "returning via machine coordinates")
                self.conn.send_line_wait_ok(
                    f"G53 G0 X{wco[0]:g} Y{wco[1]:g}", timeout=10)
                for _ in range(300):                # wait for the move
                    line = self._status_line()
                    if line.startswith("<Idle"):
                        break
                    time.sleep(0.1)
                self.conn.send_line_wait_ok("G92 X0 Y0", timeout=5)
                self.console.emit(">> parked at Home, origin restored")
            else:
                self.console.emit(f">> could not return to Home: homing "
                                  f"said '{resp}' and no origin offset was "
                                  "captured. Move the carriage and use "
                                  "'Set origin here' before the next plot.")
        except Exception as exc:
            self.console.emit(f">> stop recovery problem: {exc}")
