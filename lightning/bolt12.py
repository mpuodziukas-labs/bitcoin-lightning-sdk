"""
BOLT 12 — Offers, Invoice Requests, and Invoices.

Specification: https://github.com/lightning/bolts/blob/master/12-offer-encoding.md

This implementation provides:
  - TLV encoding/decoding (Type-Length-Value)
  - Offer, InvoiceRequest, Invoice dataclasses
  - Basic signature scaffold (stub — real Schnorr signing requires secp256k1)
"""

from __future__ import annotations

import hashlib
import struct
import time
from dataclasses import dataclass, field
from typing import Optional


# ---------------------------------------------------------------------------
# TLV primitives
# ---------------------------------------------------------------------------

def _encode_varint(value: int) -> bytes:
    """Encode a value as a Bitcoin-style variable-length integer."""
    if value < 0xFD:
        return bytes([value])
    elif value <= 0xFFFF:
        return b"\xfd" + struct.pack("<H", value)
    elif value <= 0xFFFFFFFF:
        return b"\xfe" + struct.pack("<I", value)
    else:
        return b"\xff" + struct.pack("<Q", value)


def _decode_varint(data: bytes, offset: int) -> tuple[int, int]:
    """
    Decode a variable-length integer at *offset*.

    Returns (value, new_offset).
    """
    if offset >= len(data):
        raise ValueError("Buffer too short for varint.")
    first = data[offset]
    if first < 0xFD:
        return first, offset + 1
    elif first == 0xFD:
        if offset + 3 > len(data):
            raise ValueError("Buffer too short for 2-byte varint.")
        return struct.unpack_from("<H", data, offset + 1)[0], offset + 3
    elif first == 0xFE:
        if offset + 5 > len(data):
            raise ValueError("Buffer too short for 4-byte varint.")
        return struct.unpack_from("<I", data, offset + 1)[0], offset + 5
    else:
        if offset + 9 > len(data):
            raise ValueError("Buffer too short for 8-byte varint.")
        return struct.unpack_from("<Q", data, offset + 1)[0], offset + 9


def tlv_encode(records: dict[int, bytes]) -> bytes:
    """
    Encode a dict of {type_int: value_bytes} into a TLV byte stream.

    Types must be unique and are encoded in ascending order as required
    by BOLT 12.
    """
    out = bytearray()
    for typ in sorted(records.keys()):
        value = records[typ]
        out += _encode_varint(typ)
        out += _encode_varint(len(value))
        out += value
    return bytes(out)


def tlv_decode(data: bytes) -> dict[int, bytes]:
    """
    Decode a TLV byte stream into a dict of {type_int: value_bytes}.

    Raises ValueError on malformed input.
    """
    records: dict[int, bytes] = {}
    offset = 0
    last_type = -1
    while offset < len(data):
        typ, offset = _decode_varint(data, offset)
        if typ <= last_type:
            raise ValueError(
                f"TLV types must be strictly ascending; got {typ} after {last_type}."
            )
        last_type = typ
        length, offset = _decode_varint(data, offset)
        if offset + length > len(data):
            raise ValueError(
                f"TLV record type {typ}: length {length} exceeds remaining data."
            )
        records[typ] = data[offset : offset + length]
        offset += length
    return records


# ---------------------------------------------------------------------------
# TLV type constants (BOLT 12)
# ---------------------------------------------------------------------------

# Offer TLV types
_OFFER_CHAINS = 2
_OFFER_METADATA = 4
_OFFER_CURRENCY = 6
_OFFER_AMOUNT = 8
_OFFER_DESCRIPTION = 10
_OFFER_FEATURES = 12
_OFFER_ABSOLUTE_EXPIRY = 14
_OFFER_ISSUER = 18
_OFFER_QUANTITY_MAX = 20
_OFFER_NODE_ID = 22

# Invoice Request TLV types
_INVREQ_METADATA = 0
_INVREQ_CHAIN = 80
_INVREQ_AMOUNT = 88
_INVREQ_FEATURES = 90
_INVREQ_QUANTITY = 92
_INVREQ_PAYER_ID = 98
_INVREQ_PAYER_NOTE = 99
_INVREQ_SIGNATURE = 240

# Invoice TLV types
_INV_CREATED_AT = 8
_INV_RELATIVE_EXPIRY = 10
_INV_PAYMENT_HASH = 26
_INV_AMOUNT = 30
_INV_FALLBACKS = 33
_INV_FEATURES = 38
_INV_NODE_ID = 30241  # non-standard placeholder
_INV_SIGNATURE = 240


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class Offer:
    """BOLT 12 Offer — published by a merchant to receive payments."""

    description: str
    amount_msat: Optional[int] = None
    currency: Optional[str] = None         # ISO-4217 if non-BTC
    issuer: Optional[str] = None
    node_id: Optional[bytes] = None        # 33-byte compressed pubkey
    features: int = 0
    absolute_expiry: Optional[int] = None  # Unix timestamp
    quantity_max: Optional[int] = None     # 0 = unlimited
    metadata: bytes = field(default_factory=bytes)

    def to_tlv(self) -> bytes:
        records: dict[int, bytes] = {}
        records[_OFFER_DESCRIPTION] = self.description.encode("utf-8")
        if self.amount_msat is not None:
            records[_OFFER_AMOUNT] = _encode_varint(self.amount_msat)
        if self.currency:
            records[_OFFER_CURRENCY] = self.currency.encode("utf-8")
        if self.issuer:
            records[_OFFER_ISSUER] = self.issuer.encode("utf-8")
        if self.node_id:
            records[_OFFER_NODE_ID] = self.node_id
        if self.features:
            bit_length = (self.features.bit_length() + 7) // 8 or 1
            records[_OFFER_FEATURES] = self.features.to_bytes(bit_length, "big")
        if self.absolute_expiry is not None:
            records[_OFFER_ABSOLUTE_EXPIRY] = _encode_varint(self.absolute_expiry)
        if self.quantity_max is not None:
            records[_OFFER_QUANTITY_MAX] = _encode_varint(self.quantity_max)
        if self.metadata:
            records[_OFFER_METADATA] = self.metadata
        return tlv_encode(records)

    @classmethod
    def from_tlv(cls, data: bytes) -> "Offer":
        records = tlv_decode(data)
        description = records.get(_OFFER_DESCRIPTION, b"").decode("utf-8", errors="replace")
        amount_msat: Optional[int] = None
        if _OFFER_AMOUNT in records:
            amount_msat, _ = _decode_varint(records[_OFFER_AMOUNT], 0)
        currency: Optional[str] = None
        if _OFFER_CURRENCY in records:
            currency = records[_OFFER_CURRENCY].decode("utf-8", errors="replace")
        issuer: Optional[str] = None
        if _OFFER_ISSUER in records:
            issuer = records[_OFFER_ISSUER].decode("utf-8", errors="replace")
        node_id = records.get(_OFFER_NODE_ID)
        features = 0
        if _OFFER_FEATURES in records:
            features = int.from_bytes(records[_OFFER_FEATURES], "big")
        absolute_expiry: Optional[int] = None
        if _OFFER_ABSOLUTE_EXPIRY in records:
            absolute_expiry, _ = _decode_varint(records[_OFFER_ABSOLUTE_EXPIRY], 0)
        quantity_max: Optional[int] = None
        if _OFFER_QUANTITY_MAX in records:
            quantity_max, _ = _decode_varint(records[_OFFER_QUANTITY_MAX], 0)
        metadata = records.get(_OFFER_METADATA, b"")
        return cls(
            description=description,
            amount_msat=amount_msat,
            currency=currency,
            issuer=issuer,
            node_id=node_id,
            features=features,
            absolute_expiry=absolute_expiry,
            quantity_max=quantity_max,
            metadata=metadata,
        )


@dataclass
class InvoiceRequest:
    """BOLT 12 Invoice Request — sent by the payer to the merchant."""

    offer_id: bytes                        # SHA-256 of the offer TLV
    payer_id: bytes                        # 33-byte compressed pubkey
    metadata: bytes = field(default_factory=bytes)
    chain: Optional[bytes] = None          # 32-byte chain hash or None (default: bitcoin)
    amount_msat: Optional[int] = None
    features: int = 0
    quantity: Optional[int] = None
    payer_note: Optional[str] = None
    signature: bytes = field(default_factory=lambda: b"\x00" * 64)  # Schnorr stub

    def to_tlv(self) -> bytes:
        records: dict[int, bytes] = {}
        records[_INVREQ_METADATA] = self.metadata or self.offer_id  # use offer_id as metadata
        if self.chain:
            records[_INVREQ_CHAIN] = self.chain
        if self.amount_msat is not None:
            records[_INVREQ_AMOUNT] = _encode_varint(self.amount_msat)
        if self.features:
            records[_INVREQ_FEATURES] = self.features.to_bytes(
                (self.features.bit_length() + 7) // 8 or 1, "big"
            )
        if self.quantity is not None:
            records[_INVREQ_QUANTITY] = _encode_varint(self.quantity)
        records[_INVREQ_PAYER_ID] = self.payer_id
        if self.payer_note:
            records[_INVREQ_PAYER_NOTE] = self.payer_note.encode("utf-8")
        records[_INVREQ_SIGNATURE] = self.signature
        return tlv_encode(records)

    @classmethod
    def from_tlv(cls, data: bytes, offer_id: Optional[bytes] = None) -> "InvoiceRequest":
        records = tlv_decode(data)
        metadata = records.get(_INVREQ_METADATA, b"")
        effective_offer_id = offer_id or metadata or b""
        payer_id = records.get(_INVREQ_PAYER_ID, b"")
        chain = records.get(_INVREQ_CHAIN)
        amount_msat: Optional[int] = None
        if _INVREQ_AMOUNT in records:
            amount_msat, _ = _decode_varint(records[_INVREQ_AMOUNT], 0)
        features = 0
        if _INVREQ_FEATURES in records:
            features = int.from_bytes(records[_INVREQ_FEATURES], "big")
        quantity: Optional[int] = None
        if _INVREQ_QUANTITY in records:
            quantity, _ = _decode_varint(records[_INVREQ_QUANTITY], 0)
        payer_note: Optional[str] = None
        if _INVREQ_PAYER_NOTE in records:
            payer_note = records[_INVREQ_PAYER_NOTE].decode("utf-8", errors="replace")
        signature = records.get(_INVREQ_SIGNATURE, b"\x00" * 64)
        return cls(
            offer_id=effective_offer_id,
            payer_id=payer_id,
            metadata=metadata,
            chain=chain,
            amount_msat=amount_msat,
            features=features,
            quantity=quantity,
            payer_note=payer_note,
            signature=signature,
        )


@dataclass
class Invoice:
    """BOLT 12 Invoice — returned by the merchant in response to an InvoiceRequest."""

    payment_hash: bytes                    # 32 bytes
    amount_msat: int
    created_at: int                        # Unix timestamp
    relative_expiry: int = 7200            # seconds (default 2h)
    features: int = 0
    node_id: Optional[bytes] = None        # 33-byte compressed pubkey
    signature: bytes = field(default_factory=lambda: b"\x00" * 64)  # Schnorr stub

    def to_tlv(self) -> bytes:
        records: dict[int, bytes] = {}
        records[_INV_CREATED_AT] = _encode_varint(self.created_at)
        records[_INV_RELATIVE_EXPIRY] = _encode_varint(self.relative_expiry)
        records[_INV_PAYMENT_HASH] = self.payment_hash
        records[_INV_AMOUNT] = _encode_varint(self.amount_msat)
        if self.features:
            records[_INV_FEATURES] = self.features.to_bytes(
                (self.features.bit_length() + 7) // 8 or 1, "big"
            )
        records[_INV_SIGNATURE] = self.signature
        return tlv_encode(records)

    @classmethod
    def from_tlv(cls, data: bytes) -> "Invoice":
        records = tlv_decode(data)
        payment_hash = records.get(_INV_PAYMENT_HASH, b"\x00" * 32)[:32]
        amount_msat = 0
        if _INV_AMOUNT in records:
            amount_msat, _ = _decode_varint(records[_INV_AMOUNT], 0)
        created_at = 0
        if _INV_CREATED_AT in records:
            created_at, _ = _decode_varint(records[_INV_CREATED_AT], 0)
        relative_expiry = 7200
        if _INV_RELATIVE_EXPIRY in records:
            relative_expiry, _ = _decode_varint(records[_INV_RELATIVE_EXPIRY], 0)
        features = 0
        if _INV_FEATURES in records:
            features = int.from_bytes(records[_INV_FEATURES], "big")
        signature = records.get(_INV_SIGNATURE, b"\x00" * 64)
        return cls(
            payment_hash=payment_hash,
            amount_msat=amount_msat,
            created_at=created_at,
            relative_expiry=relative_expiry,
            features=features,
            signature=signature,
        )

    def is_expired(self) -> bool:
        return int(time.time()) > self.created_at + self.relative_expiry


# ---------------------------------------------------------------------------
# Signature scaffold
# ---------------------------------------------------------------------------

def sign_stub(message: bytes, private_key: bytes) -> bytes:
    """
    Stub Schnorr signature using SHA-256 HMAC (not a real signature).

    Real implementation requires secp256k1 (e.g. coincurve/cryptography).
    This scaffold preserves the 64-byte format for protocol conformance testing.
    """
    import hmac
    return hmac.new(private_key, message, hashlib.sha256).digest() * 2  # 64 bytes


def verify_stub(message: bytes, signature: bytes, public_key: bytes) -> bool:
    """Stub — always returns False (no real secp256k1 available)."""
    return False
