"""
BOLT 11 — Bitcoin Lightning payment invoice encoder/decoder.

Specification: https://github.com/lightning/bolts/blob/master/11-payment-encoding.md

This implementation handles:
  - decode(invoice_str): parses hrp, timestamp, payment_hash, description,
    expiry, amount_msat, and feature bits.
  - encode(fields): basic invoice construction (without real signature).

Bech32 internals are pure Python; no external dependencies.
"""

from __future__ import annotations

import hashlib
import struct
import time
from dataclasses import dataclass, field
from typing import Optional

# ---------------------------------------------------------------------------
# Bech32 constants & helpers
# ---------------------------------------------------------------------------

_CHARSET = "qpzry9x8gf2tvdw0s3jn54khce6mua7l"
_GENERATOR = [0x3B6A57B2, 0x26508E6D, 0x1EA119FA, 0x3D4233DD, 0x2A1462B3]


def _bech32_polymod(values: list[int]) -> int:
    chk = 1
    for value in values:
        top = chk >> 25
        chk = (chk & 0x1FFFFFF) << 5 ^ value
        for i in range(5):
            chk ^= _GENERATOR[i] if ((top >> i) & 1) else 0
    return chk


def _bech32_hrp_expand(hrp: str) -> list[int]:
    return [ord(x) >> 5 for x in hrp] + [0] + [ord(x) & 31 for x in hrp]


def _bech32_verify_checksum(hrp: str, data: list[int]) -> bool:
    return _bech32_polymod(_bech32_hrp_expand(hrp) + data) == 1


def _bech32_decode(bech: str) -> tuple[str, list[int]]:
    """Decode a bech32 string.  Returns (hrp, data_5bit_list)."""
    if any(ord(c) < 33 or ord(c) > 126 for c in bech):
        raise ValueError("Invalid character in bech32 string.")
    if bech.lower() != bech and bech.upper() != bech:
        raise ValueError("Mixed case in bech32 string.")
    bech = bech.lower()
    pos = bech.rfind("1")
    if pos < 1 or pos + 7 > len(bech):
        raise ValueError("Invalid separator position in bech32 string.")
    hrp = bech[:pos]
    data: list[int] = []
    for c in bech[pos + 1 :]:
        d = _CHARSET.find(c)
        if d < 0:
            raise ValueError(f"Invalid bech32 character: {c!r}")
        data.append(d)
    if not _bech32_verify_checksum(hrp, data):
        raise ValueError("Invalid bech32 checksum.")
    return hrp, data[:-6]


def _convert_bits(data: list[int], from_bits: int, to_bits: int, pad: bool = True) -> list[int]:
    """General power-of-2 base conversion."""
    acc = 0
    bits = 0
    result: list[int] = []
    maxv = (1 << to_bits) - 1
    for value in data:
        acc = ((acc << from_bits) | value) & 0xFFFFFF
        bits += from_bits
        while bits >= to_bits:
            bits -= to_bits
            result.append((acc >> bits) & maxv)
    if pad and bits:
        result.append((acc << (to_bits - bits)) & maxv)
    elif bits >= from_bits or ((acc << (to_bits - bits)) & maxv):
        pass  # ignore trailing bits in decode mode
    return result


# ---------------------------------------------------------------------------
# Amount parsing
# ---------------------------------------------------------------------------

_MULTIPLIER_MSAT: dict[str, int] = {
    "m": 100_000_000,      # milli-bitcoin → msat  (1 BTC = 100_000_000_000 msat)
    "u": 100_000,          # micro-bitcoin → msat
    "n": 100,              # nano-bitcoin  → msat
    "p": 1,                # pico-bitcoin  → msat (1 pBTC = 0.1 msat, round down)
}


def _parse_amount(hrp_amount: str) -> Optional[int]:
    """Parse the amount from the HRP suffix.  Returns millisatoshis or None."""
    if not hrp_amount:
        return None
    multiplier_char = hrp_amount[-1]
    if multiplier_char in _MULTIPLIER_MSAT:
        digits = hrp_amount[:-1]
    else:
        # No multiplier — amount is in whole BTC
        digits = hrp_amount
        multiplier_char = ""
    if not digits.isdigit():
        raise ValueError(f"Invalid amount digits in HRP: {digits!r}")
    amount_int = int(digits)
    if multiplier_char:
        return amount_int * _MULTIPLIER_MSAT[multiplier_char]
    # Whole BTC to msat: 1 BTC = 100_000_000_000 msat
    return amount_int * 100_000_000_000


# ---------------------------------------------------------------------------
# Tagged fields
# ---------------------------------------------------------------------------

_TAG_PAYMENT_HASH = 1
_TAG_DESCRIPTION = 13
_TAG_PAYEE_PUBKEY = 19
_TAG_DESCRIPTION_HASH = 23
_TAG_EXPIRY = 6
_TAG_MIN_FINAL_CLTV = 24
_TAG_FALLBACK_ADDRESS = 9
_TAG_ROUTE_HINT = 3
_TAG_FEATURE_BITS = 5
_TAG_PAYMENT_SECRET = 16
_TAG_PAYMENT_METADATA = 27


def _read_tagged_fields(data5: list[int]) -> dict[int, bytes]:
    """Parse tagged fields from 5-bit data array.  Returns {tag: raw_bytes}."""
    fields: dict[int, bytes] = {}
    pos = 0
    while pos + 3 <= len(data5):
        tag = data5[pos]
        length = (data5[pos + 1] << 5) | data5[pos + 2]
        pos += 3
        if pos + length > len(data5):
            break
        field_data = data5[pos : pos + length]
        field_bytes = bytes(_convert_bits(field_data, 5, 8, pad=False))
        fields[tag] = field_bytes
        pos += length
    return fields


def _5bit_to_int(data5: list[int]) -> int:
    result = 0
    for b in data5:
        result = (result << 5) | b
    return result


# ---------------------------------------------------------------------------
# Dataclass
# ---------------------------------------------------------------------------

@dataclass
class Invoice:
    """Decoded BOLT 11 payment invoice."""

    hrp: str                        # full hrp, e.g. "lnbc2500u"
    network: str                    # "bc", "tb", "bcrt", etc.
    amount_msat: Optional[int]      # amount in millisatoshis, None if unspecified
    timestamp: int                  # Unix timestamp
    payment_hash: bytes             # 32 bytes
    description: str                # UTF-8 description or ""
    expiry: int                     # seconds until expiry (default 3600)
    features: int                   # feature bits as integer
    payment_secret: Optional[bytes] # 32 bytes or None
    signature: bytes                # 65-byte compact DER or empty


# ---------------------------------------------------------------------------
# Decode
# ---------------------------------------------------------------------------

def decode(invoice_str: str) -> Invoice:
    """
    Decode a BOLT 11 invoice string.

    Parameters
    ----------
    invoice_str: bech32-encoded invoice, optionally prefixed with "lightning:".

    Returns
    -------
    Invoice dataclass with all parsed fields.

    Raises
    ------
    ValueError: if the invoice is malformed.
    """
    s = invoice_str.strip()
    if s.lower().startswith("lightning:"):
        s = s[len("lightning:"):]

    hrp, data5 = _bech32_decode(s)

    # Validate HRP prefix
    if not (hrp.startswith("lnbc") or hrp.startswith("lntb") or hrp.startswith("lnbcrt") or hrp.startswith("lntbs")):
        raise ValueError(f"Unrecognised HRP prefix: {hrp!r}")

    # Extract network and optional amount from HRP
    if hrp.startswith("lnbcrt"):
        network = "bcrt"
        hrp_amount = hrp[6:]
    elif hrp.startswith("lntbs"):
        network = "tbs"
        hrp_amount = hrp[5:]
    elif hrp.startswith("lntb"):
        network = "tb"
        hrp_amount = hrp[4:]
    else:
        network = "bc"
        hrp_amount = hrp[4:]

    amount_msat = _parse_amount(hrp_amount) if hrp_amount else None

    # First 7 x 5-bit groups = timestamp (35 bits)
    if len(data5) < 7:
        raise ValueError("Invoice data too short for timestamp.")
    timestamp = _5bit_to_int(data5[:7])

    # Remaining data before the 104-byte (832 bits → ceil(832/5)=167 groups) signature
    # Signature occupies the last 104 bytes = 832 bits → 167 five-bit groups (rounded up)
    _SIG_5BIT_LEN = 104  # 65 bytes → 104 five-bit groups (65*8/5 = 104)
    tagged_data5 = data5[7:-_SIG_5BIT_LEN]
    sig_data5 = data5[-_SIG_5BIT_LEN:]

    signature = bytes(_convert_bits(sig_data5, 5, 8, pad=False))

    tagged = _read_tagged_fields(tagged_data5)

    # Payment hash (tag 1) — 32 bytes
    payment_hash_bytes = tagged.get(_TAG_PAYMENT_HASH, b"\x00" * 32)
    if len(payment_hash_bytes) >= 32:
        payment_hash = payment_hash_bytes[:32]
    else:
        payment_hash = payment_hash_bytes.ljust(32, b"\x00")

    # Description (tag 13)
    description_bytes = tagged.get(_TAG_DESCRIPTION, b"")
    description = description_bytes.decode("utf-8", errors="replace")

    # Expiry (tag 6)
    expiry_bytes = tagged.get(_TAG_EXPIRY, b"")
    if expiry_bytes:
        expiry = int.from_bytes(expiry_bytes, "big")
    else:
        expiry = 3600  # BOLT 11 default

    # Feature bits (tag 5)
    feature_bytes = tagged.get(_TAG_FEATURE_BITS, b"")
    features = int.from_bytes(feature_bytes, "big") if feature_bytes else 0

    # Payment secret (tag 16)
    payment_secret_bytes = tagged.get(_TAG_PAYMENT_SECRET)
    payment_secret: Optional[bytes] = payment_secret_bytes[:32] if payment_secret_bytes else None

    return Invoice(
        hrp=hrp,
        network=network,
        amount_msat=amount_msat,
        timestamp=timestamp,
        payment_hash=payment_hash,
        description=description,
        expiry=expiry,
        features=features,
        payment_secret=payment_secret,
        signature=signature,
    )


# ---------------------------------------------------------------------------
# Encode (basic — scaffold without real signature)
# ---------------------------------------------------------------------------

def _int_to_5bits(value: int, n_bits: int) -> list[int]:
    """Encode *value* as a sequence of *n_bits* 5-bit groups (big-endian)."""
    result: list[int] = []
    for _ in range(n_bits):
        result.append(value & 0x1F)
        value >>= 5
    return list(reversed(result))


def _bech32_encode(hrp: str, data: list[int]) -> str:
    """Encode a bech32 string from hrp and 5-bit data (without checksum appended yet)."""
    combined = data + [0, 0, 0, 0, 0, 0]
    polymod = _bech32_polymod(_bech32_hrp_expand(hrp) + combined) ^ 1
    checksum = [(polymod >> 5 * (5 - i)) & 31 for i in range(6)]
    return hrp + "1" + "".join([_CHARSET[d] for d in data + checksum])


def encode(
    payment_hash: bytes,
    amount_msat: Optional[int] = None,
    description: str = "",
    expiry: int = 3600,
    network: str = "bc",
    timestamp: Optional[int] = None,
) -> str:
    """
    Encode a basic BOLT 11 invoice (scaffold — uses zeroed signature).

    Parameters
    ----------
    payment_hash: 32-byte payment hash.
    amount_msat:  Amount in millisatoshis, or None for unspecified amount.
    description:  Human-readable invoice description.
    expiry:       Seconds until the invoice expires.
    network:      "bc" (mainnet), "tb" (testnet), "bcrt" (regtest).
    timestamp:    Unix timestamp; defaults to now.

    Returns
    -------
    Bech32-encoded BOLT 11 invoice string.
    """
    if len(payment_hash) != 32:
        raise ValueError("payment_hash must be exactly 32 bytes.")

    ts = timestamp if timestamp is not None else int(time.time())

    # Build HRP
    prefix = {"bc": "lnbc", "tb": "lntb", "bcrt": "lnbcrt"}.get(network, "lnbc")
    if amount_msat is not None:
        # Express in smallest unit with multiplier
        if amount_msat % 100_000_000_000 == 0:
            amount_str = str(amount_msat // 100_000_000_000)
        elif amount_msat % 100_000_000 == 0:
            amount_str = str(amount_msat // 100_000_000) + "m"
        elif amount_msat % 100_000 == 0:
            amount_str = str(amount_msat // 100_000) + "u"
        elif amount_msat % 100 == 0:
            amount_str = str(amount_msat // 100) + "n"
        else:
            amount_str = str(amount_msat) + "p"
        hrp = prefix + amount_str
    else:
        hrp = prefix

    # Timestamp: 7 × 5-bit groups
    data5: list[int] = _int_to_5bits(ts, 7)

    # Tagged field helper
    def _add_field(tag: int, value_bytes: bytes) -> None:
        bits5 = _convert_bits(list(value_bytes), 8, 5, pad=True)
        length = len(bits5)
        data5.append(tag)
        data5.append(length >> 5)
        data5.append(length & 0x1F)
        data5.extend(bits5)

    # Payment hash (tag 1)
    _add_field(_TAG_PAYMENT_HASH, payment_hash)

    # Description (tag 13)
    _add_field(_TAG_DESCRIPTION, description.encode("utf-8"))

    # Expiry (tag 6)
    expiry_bytes = expiry.to_bytes((expiry.bit_length() + 7) // 8 or 1, "big")
    _add_field(_TAG_EXPIRY, expiry_bytes)

    # Zeroed 65-byte signature placeholder
    sig_bytes = b"\x00" * 65
    data5.extend(_convert_bits(list(sig_bytes), 8, 5, pad=True))

    return _bech32_encode(hrp, data5)


# ---------------------------------------------------------------------------
# Convenience
# ---------------------------------------------------------------------------

def is_expired(invoice: Invoice) -> bool:
    """Return True if the invoice has expired."""
    return int(time.time()) > invoice.timestamp + invoice.expiry
