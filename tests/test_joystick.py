from scripts.util.joystick import Joystick

import time


def main():
    joy_list = [Joystick(fn) for fn in Joystick.list_devices()]
    while True:
        print([(x.axes, x.buttons) for x in joy_list])
        time.sleep(0.1)


if __name__ == "__main__":
    main()
