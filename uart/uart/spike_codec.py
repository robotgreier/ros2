from typing import List


def pack_input_spikes(spikes: List[int]) -> List[int]:
    """
    Pack input spikes using:
    [a,b,c] -> 00 aa bb cc

    Example:
    [1,0,1] -> 0b00110011 -> 51
    """
    packed = []

    for i in range(0, len(spikes), 3):
        a = spikes[i] if i < len(spikes) else 0
        b = spikes[i + 1] if i + 1 < len(spikes) else 0
        c = spikes[i + 2] if i + 2 < len(spikes) else 0

        byte = (
            (a << 5) | (a << 4) |
            (b << 3) | (b << 2) |
            (c << 1) | c
        )

        packed.append(byte)

    return packed


def unpack_input_spikes(data_bytes: List[int], expected_len: int) -> List[int]:
    """
    Reverse of pack_input_spikes.
    Mainly useful for testing/debugging.
    """
    spikes = []

    for byte in data_bytes:
        a = (byte >> 5) & 1
        b = (byte >> 3) & 1
        c = (byte >> 1) & 1

        spikes.extend([a, b, c])

    return spikes[:expected_len]


def unpack_output_spikes(byte: int, expected_len: int = 4) -> List[int]:
    """
    FPGA output format:
    one byte contains one-hot output bits.

    Example:
    0b00000010 -> [0,1,0,0]
    """
    return [(byte >> i) & 1 for i in range(expected_len)]