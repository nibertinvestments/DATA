"""
Command Design Pattern in Python
Encapsulates requests as objects for flexible execution
"""

from abc import ABC, abstractmethod
from typing import List

# Command interface
class Command(ABC):
    @abstractmethod
    def execute(self):
        pass
    
    @abstractmethod
    def undo(self):
        pass

# Receiver class
class Light:
    def __init__(self, location: str):
        self.location = location
        self.is_on = False
    
    def turn_on(self):
        self.is_on = True
        print(f"{self.location} light is ON")
    
    def turn_off(self):
        self.is_on = False
        print(f"{self.location} light is OFF")

# Concrete commands
class LightOnCommand(Command):
    def __init__(self, light: Light):
        self.light = light
    
    def execute(self):
        self.light.turn_on()
    
    def undo(self):
        self.light.turn_off()

class LightOffCommand(Command):
    def __init__(self, light: Light):
        self.light = light
    
    def execute(self):
        self.light.turn_off()
    
    def undo(self):
        self.light.turn_on()

# Invoker
class RemoteControl:
    def __init__(self):
        self.commands: List[Command] = []
        self.history: List[Command] = []
    
    def set_command(self, slot: int, command: Command):
        if len(self.commands) <= slot:
            self.commands.extend([None] * (slot - len(self.commands) + 1))
        self.commands[slot] = command
    
    def press_button(self, slot: int):
        if slot < len(self.commands) and self.commands[slot]:
            self.commands[slot].execute()
            self.history.append(self.commands[slot])
    
    def press_undo(self):
        if self.history:
            command = self.history.pop()
            command.undo()

# Macro command
class MacroCommand(Command):
    def __init__(self, commands: List[Command]):
        self.commands = commands
    
    def execute(self):
        for command in self.commands:
            command.execute()
    
    def undo(self):
        for command in reversed(self.commands):
            command.undo()

def main():
    print("Command Pattern Demonstration")
    print("============================\n")
    
    # Create receivers
    living_room_light = Light("Living Room")
    bedroom_light = Light("Bedroom")
    
    # Create commands
    living_on = LightOnCommand(living_room_light)
    living_off = LightOffCommand(living_room_light)
    bedroom_on = LightOnCommand(bedroom_light)
    bedroom_off = LightOffCommand(bedroom_light)
    
    # Set up remote
    remote = RemoteControl()
    remote.set_command(0, living_on)
    remote.set_command(1, living_off)
    remote.set_command(2, bedroom_on)
    remote.set_command(3, bedroom_off)
    
    # Test commands
    print("Pressing button 0:")
    remote.press_button(0)
    
    print("\nPressing button 2:")
    remote.press_button(2)
    
    print("\nUndoing last command:")
    remote.press_undo()
    
    print("\nTesting macro command:")
    all_on = MacroCommand([living_on, bedroom_on])
    all_on.execute()
    
    print("\nUndoing macro:")
    all_on.undo()

if __name__ == "__main__":
    main()
