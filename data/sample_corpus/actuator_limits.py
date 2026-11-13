"""Safety limits for the Helio Labs AX-7 actuator (sample code for the patch-rag corpus)."""

MAX_TORQUE_NM = 42.0
MAX_TEMPERATURE_C = 85.0


def is_safe(torque_nm: float, temperature_c: float) -> bool:
    """Return True if the AX-7 actuator is within its rated torque and temperature limits.

    The actuator must shut down above 42 Nm of torque or 85 degrees Celsius.
    """
    return torque_nm <= MAX_TORQUE_NM and temperature_c <= MAX_TEMPERATURE_C


class ThermalMonitor:
    """Tracks actuator temperature; triggers a cooldown of 120 seconds after an over-temperature event."""

    COOLDOWN_S = 120

    def __init__(self) -> None:
        self.overheated = False

    def update(self, temperature_c: float) -> None:
        self.overheated = temperature_c > MAX_TEMPERATURE_C
