# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import carb
import omni
import numpy as np


from omni.ext.mobility_gen.common import Module, Buffer

    
#=========================================================
#  IMPLEMENTATION
#=========================================================


class KeyboardButton:

    def __init__(self,
            key: carb.input.KeyboardInput
        ):
        self._key = key
        self._value = False

    @property
    def value(self) -> bool:
        return self._value
    
    def _event_callback(self, event: carb.input.KeyboardEvent, *args, **kwargs):
        if event.input == self._key:
            if event.type == carb.input.KeyboardEventType.KEY_PRESS \
                or event.type == carb.input.KeyboardEventType.KEY_REPEAT:
                self._value = True
            elif event.type == carb.input.KeyboardEventType.KEY_RELEASE:
                self._value = False


class KeyboardDriver(object):
    
    _instance = None

    def __init__(self):
        if self._instance is not None:
            raise RuntimeError("Keyboard singleton already instantiated.  Please call Keyboard.instance() instead.")
        self._appwindow = None
        self._input = None
        self._keyboard = None
        self._event_handle = None

        key_input_types = [
            carb.input.KeyboardInput.W,
            carb.input.KeyboardInput.A,
            carb.input.KeyboardInput.S,
            carb.input.KeyboardInput.D,
            carb.input.KeyboardInput.SPACE,
            carb.input.KeyboardInput.C,
        ]

        self.buttons = [KeyboardButton(key) for key in key_input_types]

    def _event_callback(self, event: carb.input.KeyboardEvent, *args, **kwargs):
        for button in self.buttons:
            button._event_callback(event, *args, **kwargs)

    def _refresh_interfaces(self):
        self._appwindow = omni.appwindow.get_default_app_window()
        self._input = carb.input.acquire_input_interface()
        self._keyboard = self._appwindow.get_keyboard() if self._appwindow is not None else None

    def _needs_reconnect(self) -> bool:
        current_appwindow = omni.appwindow.get_default_app_window()
        current_input = carb.input.acquire_input_interface()
        current_keyboard = current_appwindow.get_keyboard() if current_appwindow is not None else None
        return (
            self._event_handle is None
            or self._appwindow is not current_appwindow
            or self._input is not current_input
            or self._keyboard is not current_keyboard
            or self._keyboard is None
        )

    def _connect(self):
        if self._event_handle is not None:
            self._disconnect()
        self._refresh_interfaces()
        if self._keyboard is None or self._input is None:
            self._event_handle = None
            return False
        self._event_handle = self._input.subscribe_to_keyboard_events(
            self._keyboard,
            self._event_callback
        )
        return self._event_handle is not None

    def _disconnect(self):
        if self._event_handle is None:
            return
        if self._input is None or self._keyboard is None:
            self._refresh_interfaces()
        if self._input is None or self._keyboard is None:
            self._event_handle = None
            return
        self._input.unsubscribe_to_keyboard_events(
            self._keyboard,
            self._event_handle
        )
        self._event_handle = None

    @staticmethod
    def connect():
        instance = KeyboardDriver.instance()
        instance._connect()
        return instance

    @staticmethod
    def ensure_connected():
        instance = KeyboardDriver.instance()
        if instance._needs_reconnect():
            instance._connect()
        return instance
    
    @staticmethod
    def disconnect():
        if KeyboardDriver._instance is None:
            return
        KeyboardDriver.instance()._disconnect()

    @staticmethod
    def instance():
        if KeyboardDriver._instance is None:
            KeyboardDriver._instance = KeyboardDriver()
        return KeyboardDriver._instance

    def get_button_values(self) -> np.ndarray:
        if self._needs_reconnect():
            self._connect()
        return np.array([b.value for b in self.buttons])
    

class GamepadAxis:

    def __init__(self,
            gamepad: "Gamepad",
            carb_pos_input: carb.input.GamepadInput,
            carb_neg_input: carb.input.GamepadInput,
            deadzone: bool = 0.01
        ):
        self.carb_pos_input = carb_pos_input
        self.carb_neg_input = carb_neg_input
        self.deadzone = deadzone
        self._gamepad = gamepad
        self._pos_val = 0.
        self._neg_val = 0.

    @property
    def value(self):
        if self._pos_val > self._neg_val:
            return self._pos_val if self._pos_val > self.deadzone else 0.
        else:
            return -self._neg_val if self._neg_val > self.deadzone else 0.
        
    def _event_callback(self, event: carb.input.GamepadEvent, *args, **kwargs):
        cur_val = event.value
        if event.input == self.carb_pos_input:
            self._pos_val = cur_val
        if event.input == self.carb_neg_input:
            self._neg_val = cur_val


class GamepadDriver(object):
    
    _instance = None

    def __init__(self):
        if self._instance is not None:
            raise RuntimeError("Gamepad singleton already instantiated.  Please call Gamepad.instance() instead.")
        self._appwindow = omni.appwindow.get_default_app_window()
        self._input = carb.input.acquire_input_interface()
        self._gamepad = self._appwindow.get_gamepad(0)
        self.axes = [
            GamepadAxis(
                gamepad=self,
                carb_pos_input=carb.input.GamepadInput.LEFT_STICK_UP,
                carb_neg_input=carb.input.GamepadInput.LEFT_STICK_DOWN,
            ),
            GamepadAxis(
                gamepad=self,
                carb_pos_input=carb.input.GamepadInput.LEFT_STICK_RIGHT,
                carb_neg_input=carb.input.GamepadInput.LEFT_STICK_LEFT,
            ),
            GamepadAxis(
                gamepad=self,
                carb_pos_input=carb.input.GamepadInput.RIGHT_STICK_UP,
                carb_neg_input=carb.input.GamepadInput.RIGHT_STICK_DOWN,
            ),
            GamepadAxis(
                gamepad=self,
                carb_pos_input=carb.input.GamepadInput.RIGHT_STICK_RIGHT,
                carb_neg_input=carb.input.GamepadInput.RIGHT_STICK_LEFT,
            )
        ]
    
    def _event_callback(self, event: carb.input.GamepadEvent, *args, **kwargs):
        for axis in self.axes:
            axis._event_callback(event, *args, **kwargs)
        # carb.log_warn(f"{self.axes[0].value}, {self.axes[1].value}, {self.axes[2].value}, {self.axes[3].value}")

    def _connect(self):
        self._event_handle = self._input.subscribe_to_gamepad_events(
            self._gamepad,
            self._event_callback
        )

    def _disconnect(self):
        self._input.unsubscribe_to_gamepad_events(
            self._gamepad,
            self._event_handle
        )
        self._event_handle = None

    @staticmethod
    def connect():
        instance = GamepadDriver.instance()
        instance._connect()
        return instance
    
    @staticmethod
    def disconnect():
        if GamepadDriver._instance is None:
            return
        GamepadDriver.instance()._disconnect()

    @staticmethod
    def instance():
        if GamepadDriver._instance is None:
            GamepadDriver._instance = GamepadDriver()
        return GamepadDriver._instance

    def get_axis_values(self) -> np.ndarray:
        return np.array([axis.value for axis in self.axes])
    
    def get_button_values(self) -> np.ndarray:
        return np.ndarray([])
    
    
#=========================================================
#  MODULES
#=========================================================

class Keyboard(Module):

    def __init__(self):
        self._keyboard = KeyboardDriver.instance()
        self.buttons = Buffer()

    def update_state(self):
        self.buttons.set_value(self._keyboard.get_button_values())
        return super().update_state()


class Gamepad(Module):
    def __init__(self):
        self._gamepad = GamepadDriver.instance()
        self.buttons = Buffer()
        self.axes = Buffer()

    def update_state(self):
        self.buttons.set_value(self._gamepad.get_button_values())
        self.axes.set_value(self._gamepad.get_axis_values())
        return super().update_state()
