"""
UART packet protocol helpers.

This module contains:
- command definitions
- Fletcher checksum calculation
- packet building
- packet validation/parsing helpers

Packet format:
[SOF][CMD][LEN][DATA_1]...[DATA_N][CHECKSUM_1][CHECKSUM_2]
"""

from typing import List, Tuple, Optional

# -----------------------------
# Protocol constants
# -----------------------------

SOF = 0xFF  # 170 decimal

# Pi -> FPGA commands (must match master.sv localparam values)
CMD_INIT = 0
CMD_SPIKE = 1
CMD_DOPAMINE = 2
CMD_STOP = 3
CMD_RESET = 4
CMD_ERR = 5

# FPGA -> Pi commands
CMD_AFFIRM = 0
CMD_OUT = 1
CMD_ERR = 2
CMD_UPDATE = 3
CMD_RESEND_REPLY = 4  # Same numeric value as CMD_RESEND, named for clarity


# -----------------------------
# Checksum
# -----------------------------

def fletcher_checksum(data: List[int]) -> Tuple[int, int]:
    """
    Compute Fletcher checksum over a list of byte values.

    Args:
        data: List of integers in range 0-255.

    Returns:
        (sum_1, sum_2) as a tuple of ints.
    """
    sum_1 = 0
    sum_2 = 0

    for value in data:
        sum_1 = (sum_1 + value) % 255
        sum_2 = (sum_2 + sum_1) % 255

    return sum_1, sum_2


# -----------------------------
# Packet building
# -----------------------------

def build_packet(cmd: int, payload: List[int]) -> bytes:
    """
    Build a UART packet.

    Packet format:
    [SOF][CMD][LEN][DATA...][CHECKSUM_1][CHECKSUM_2]

    Args:
        cmd: Command byte.
        payload: List of payload bytes.

    Returns:
        Packet as bytes.

    Raises:
        ValueError: If payload is too large or contains invalid byte values.
    """
    if len(payload) > 255:
        raise ValueError("Payload too large. LEN field is one byte, max 255.")

    for value in payload:
        if not 0 <= value <= 255:
            raise ValueError(f"Invalid payload byte: {value}")

    header_and_data = [SOF, cmd, len(payload)] + payload
    sum_1, sum_2 = fletcher_checksum(header_and_data)

    full_packet = header_and_data + [sum_1, sum_2]
    return bytes(full_packet)


# -----------------------------
# Packet parsing / validation
# -----------------------------

def expected_packet_length(length_field: int) -> int:
    """
    Return total packet length from LEN.

    Total packet size = 1(SOF) + 1(CMD) + 1(LEN) + LEN(DATA) + 2(CHECKSUM)
                      = LEN + 5
    """
    return length_field + 5


def validate_packet(packet: bytes) -> bool:
    """
    Validate a complete packet.

    Args:
        packet: Full packet bytes.

    Returns:
        True if packet structure and checksum are valid, else False.
    """
    if len(packet) < 5:
        return False

    if packet[0] != SOF:
        return False

    payload_length = packet[2]
    if len(packet) != expected_packet_length(payload_length):
        return False

    data_without_checksum = list(packet[:-2])
    rx_sum_1 = packet[-2]
    rx_sum_2 = packet[-1]

    calc_sum_1, calc_sum_2 = fletcher_checksum(data_without_checksum)

    return (rx_sum_1 == calc_sum_1) and (rx_sum_2 == calc_sum_2)


def parse_packet(packet: bytes) -> Optional[Tuple[int, List[int]]]:
    """
    Parse a complete packet after validating it.

    Args:
        packet: Full packet bytes.

    Returns:
        (cmd, payload) if valid, else None.
    """
    if not validate_packet(packet):
        return None

    cmd = packet[1]
    payload_length = packet[2]
    payload = list(packet[3:3 + payload_length])

    return cmd, payload