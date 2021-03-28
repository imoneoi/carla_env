import os
from fcntl import ioctl
import array
import threading
import select
import struct


class Joystick:
    @staticmethod
    def list_devices():
        dirname = "/dev/input/"
        return [dirname + fn for fn in os.listdir(dirname) if fn.startswith("js")]

    def __init__(self, device_name):
        self.dev = open(device_name, "rb")

        # initialize axes and buttons
        buf = array.array('B', [0])
        ioctl(self.dev, 0x80016a11, buf)  # JSIOCGAXES
        self.num_axes = buf[0]

        buf = array.array('B', [0])
        ioctl(self.dev, 0x80016a12, buf)  # JSIOCGBUTTONS
        self.num_buttons = buf[0]

        # Get the axis map.
        buf = array.array('B', [0] * 0x40)
        ioctl(self.dev, 0x80406a32, buf)  # JSIOCGAXMAP

        self.axes = {}
        self.axis_map = []
        for axis in buf[:self.num_axes]:
            axis_name = self.AXIS_NAME_MAP.get(axis, 'unknown(0x%02x)' % axis)
            self.axis_map.append(axis_name)
            self.axes[axis_name] = 0.0

        # Get the button map.
        buf = array.array('H', [0] * 200)
        ioctl(self.dev, 0x80406a34, buf)  # JSIOCGBTNMAP

        self.buttons = {}
        self.button_map = []
        for btn in buf[:self.num_buttons]:
            btn_name = self.BUTTON_NAME_MAP.get(btn, 'unknown(0x%03x)' % btn)
            self.button_map.append(btn_name)
            self.buttons[btn_name] = 0

        # start thread
        self.thread = threading.Thread(target=self._worker_thread, daemon=True)
        self.thread.start()

    def _worker_thread(self):
        dev = self.dev
        while True:
            # read event
            select.select([dev.fileno()], [], [])

            ev_buf = dev.read(8)
            if not ev_buf:
                continue

            # update
            time, value, ev_type, number = struct.unpack('IhBB', ev_buf)
            if ev_type & 0x01:
                self.buttons[self.button_map[number]] = value

            if ev_type & 0x02:
                self.axes[self.axis_map[number]] = float(value) / 32767.0

    # Constant Table
    AXIS_NAME_MAP = {
        0x00: 'x',
        0x01: 'y',
        0x02: 'z',
        0x03: 'rx',
        0x04: 'ry',
        0x05: 'rz',
        0x06: 'trottle',
        0x07: 'rudder',
        0x08: 'wheel',
        0x09: 'gas',
        0x0a: 'brake',
        0x10: 'hat0x',
        0x11: 'hat0y',
        0x12: 'hat1x',
        0x13: 'hat1y',
        0x14: 'hat2x',
        0x15: 'hat2y',
        0x16: 'hat3x',
        0x17: 'hat3y',
        0x18: 'pressure',
        0x19: 'distance',
        0x1a: 'tilt_x',
        0x1b: 'tilt_y',
        0x1c: 'tool_width',
        0x20: 'volume',
        0x28: 'misc',
    }

    BUTTON_NAME_MAP = {
        0x120: 'trigger',
        0x121: 'thumb',
        0x122: 'thumb2',
        0x123: 'top',
        0x124: 'top2',
        0x125: 'pinkie',
        0x126: 'base',
        0x127: 'base2',
        0x128: 'base3',
        0x129: 'base4',
        0x12a: 'base5',
        0x12b: 'base6',
        0x12f: 'dead',
        0x130: 'a',
        0x131: 'b',
        0x132: 'c',
        0x133: 'x',
        0x134: 'y',
        0x135: 'z',
        0x136: 'tl',
        0x137: 'tr',
        0x138: 'tl2',
        0x139: 'tr2',
        0x13a: 'select',
        0x13b: 'start',
        0x13c: 'mode',
        0x13d: 'thumbl',
        0x13e: 'thumbr',

        0x220: 'dpad_up',
        0x221: 'dpad_down',
        0x222: 'dpad_left',
        0x223: 'dpad_right',

        # XBox 360 controller uses these codes.
        0x2c0: 'dpad_left',
        0x2c1: 'dpad_right',
        0x2c2: 'dpad_up',
        0x2c3: 'dpad_down',
    }
