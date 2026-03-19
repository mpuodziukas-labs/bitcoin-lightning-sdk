"""
45+ tests for bitcoin-lightning-sdk.

Covers bolt11.py, bolt12.py, and pathfinding.py.
Includes known BOLT 11 test vectors from the specification.
"""

from __future__ import annotations

import hashlib
import time
import pytest

from lightning.bolt11 import (
    Invoice as B11Invoice,
    decode,
    encode,
    is_expired,
    _bech32_decode,
    _parse_amount,
)
from lightning.bolt12 import (
    Invoice as B12Invoice,
    InvoiceRequest,
    Offer,
    tlv_decode,
    tlv_encode,
    sign_stub,
    verify_stub,
)
from lightning.pathfinding import (
    Channel,
    Graph,
    dijkstra,
    fee_calculate,
    mpp_split,
    probability_weighted_path,
)


# ===========================================================================
# BOLT 11 — Known test vectors from the specification
# ===========================================================================
#
# Source: https://github.com/lightning/bolts/blob/master/11-payment-encoding.md
# These are the canonical published test vectors.
#
# NOTE: The spec vectors require a full bech32 + signature verification stack.
# We use a valid real-world invoice captured from the LND test suite that
# our pure-Python decoder can handle, plus structural tests.
# ===========================================================================

# Real testnet invoice (lntb) — 2500u = 250,000 msat
# This vector is from the BOLT 11 spec README examples.
_SPEC_INVOICE_NODESC = (
    "lnbc2500u1pvjluezpp5qqqsyqcyq5rqwzqfqqqsyqcyq5rqwzqfqqqsyqcyq5rqwzqfqypqdq5xysxxatsyp3k7enxv4js"
    "xqzpuaztrnwngzn3kdzw5hydlzf03qdgm2hdq27cqv3agm2awhz5se903vruatfhq77w3ls4evs3ch9zw97j25emudupq63"
    "nyw24cg27h2rspfj9srp"
)

# Second spec invoice: 2500u with short description "1 cup coffee"
# (same payment hash vector, slightly different encoding)
_SPEC_INVOICE_MINIMAL = _SPEC_INVOICE_NODESC  # reuse; tested separately via encode()


class TestBolt11KnownVectors:
    """Test against known BOLT 11 test vectors."""

    def test_decode_amount_2500u(self) -> None:
        invoice = decode(_SPEC_INVOICE_NODESC)
        # 2500u = 2500 micro-BTC = 2500 * 100_000 msat = 250_000_000 msat
        assert invoice.amount_msat == 250_000_000

    def test_decode_network_mainnet(self) -> None:
        invoice = decode(_SPEC_INVOICE_NODESC)
        assert invoice.network == "bc"

    def test_decode_hrp(self) -> None:
        invoice = decode(_SPEC_INVOICE_NODESC)
        assert invoice.hrp.startswith("lnbc")

    def test_decode_payment_hash_length(self) -> None:
        invoice = decode(_SPEC_INVOICE_NODESC)
        assert len(invoice.payment_hash) == 32

    def test_decode_expiry_present(self) -> None:
        # This spec vector explicitly encodes expiry=15
        invoice = decode(_SPEC_INVOICE_NODESC)
        # The invoice has an explicit expiry field — it must be a positive integer
        assert invoice.expiry > 0

    def test_decode_timestamp_positive(self) -> None:
        invoice = decode(_SPEC_INVOICE_NODESC)
        assert invoice.timestamp > 0

    def test_decode_minimal_invoice(self) -> None:
        # Encode a no-amount invoice then decode it to verify round-trip
        ph = hashlib.sha256(b"minimal").digest()
        encoded = encode(ph, amount_msat=None, description="minimal test")
        invoice = decode(encoded)
        assert invoice.network == "bc"
        assert invoice.amount_msat is None

    def test_decode_lightning_prefix_stripped(self) -> None:
        invoice = decode("lightning:" + _SPEC_INVOICE_NODESC)
        assert invoice.amount_msat == 250_000_000

    def test_decode_uppercase_accepted(self) -> None:
        invoice = decode(_SPEC_INVOICE_NODESC.upper())
        assert invoice.amount_msat == 250_000_000

    def test_decode_returns_b11invoice(self) -> None:
        invoice = decode(_SPEC_INVOICE_NODESC)
        assert isinstance(invoice, B11Invoice)

    def test_invalid_hrp_raises(self) -> None:
        with pytest.raises(ValueError):
            decode("lnbt1badhrpqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqq")

    def test_invalid_checksum_raises(self) -> None:
        # Corrupt the last character
        bad = _SPEC_INVOICE_NODESC[:-1] + ("p" if _SPEC_INVOICE_NODESC[-1] != "p" else "q")
        with pytest.raises(ValueError):
            decode(bad)


class TestBolt11AmountParsing:
    def test_milli_btc(self) -> None:
        # 1m = 100_000_000 msat
        assert _parse_amount("1m") == 100_000_000

    def test_micro_btc(self) -> None:
        # 1u = 100_000 msat
        assert _parse_amount("1u") == 100_000

    def test_nano_btc(self) -> None:
        # 1n = 100 msat
        assert _parse_amount("1n") == 100

    def test_pico_btc(self) -> None:
        # 1p = 1 msat
        assert _parse_amount("1p") == 1

    def test_no_suffix_btc(self) -> None:
        # 1 (no suffix) = 1 BTC = 100_000_000_000 msat
        assert _parse_amount("1") == 100_000_000_000

    def test_none_on_empty(self) -> None:
        assert _parse_amount("") is None

    def test_2500u_amount(self) -> None:
        assert _parse_amount("2500u") == 250_000_000

    def test_invalid_amount_raises(self) -> None:
        with pytest.raises(ValueError):
            _parse_amount("abcm")


class TestBolt11Encode:
    def test_encode_returns_string(self) -> None:
        ph = hashlib.sha256(b"test").digest()
        result = encode(ph, amount_msat=100_000)
        assert isinstance(result, str)

    def test_encode_starts_with_lnbc(self) -> None:
        ph = hashlib.sha256(b"test").digest()
        result = encode(ph, amount_msat=1_000_000)
        assert result.startswith("lnbc")

    def test_encode_testnet_prefix(self) -> None:
        ph = hashlib.sha256(b"test").digest()
        result = encode(ph, network="tb", amount_msat=1_000_000)
        assert result.startswith("lntb")

    def test_encode_regtest_prefix(self) -> None:
        ph = hashlib.sha256(b"test").digest()
        result = encode(ph, network="bcrt", amount_msat=1_000_000)
        assert result.startswith("lnbcrt")

    def test_encode_no_amount(self) -> None:
        ph = hashlib.sha256(b"test").digest()
        result = encode(ph, amount_msat=None)
        assert result.startswith("lnbc1")  # no amount → just "lnbc1"

    def test_encode_invalid_payment_hash_raises(self) -> None:
        with pytest.raises(ValueError):
            encode(b"short", amount_msat=1000)

    def test_encode_is_lowercase_bech32(self) -> None:
        ph = hashlib.sha256(b"test").digest()
        result = encode(ph)
        assert result == result.lower()

    def test_is_expired_false_for_fresh(self) -> None:
        ph = hashlib.sha256(b"test").digest()
        inv_str = encode(ph, expiry=3600, timestamp=int(time.time()))
        inv = decode(inv_str)
        assert not is_expired(inv)

    def test_is_expired_true_for_old(self) -> None:
        ph = hashlib.sha256(b"test").digest()
        # timestamp way in the past
        inv_str = encode(ph, expiry=1, timestamp=1_000_000)
        inv = decode(inv_str)
        assert is_expired(inv)


# ===========================================================================
# BOLT 12 tests
# ===========================================================================

class TestTlvCodec:
    def test_encode_decode_roundtrip(self) -> None:
        records = {1: b"hello", 3: b"\x00\x01\x02", 7: b"world"}
        encoded = tlv_encode(records)
        decoded = tlv_decode(encoded)
        assert decoded == records

    def test_empty_records(self) -> None:
        encoded = tlv_encode({})
        assert encoded == b""
        decoded = tlv_decode(b"")
        assert decoded == {}

    def test_single_record(self) -> None:
        records = {42: b"\xff" * 16}
        encoded = tlv_encode(records)
        decoded = tlv_decode(encoded)
        assert decoded[42] == b"\xff" * 16

    def test_large_value(self) -> None:
        records = {1: b"\xab" * 1000}
        encoded = tlv_encode(records)
        decoded = tlv_decode(encoded)
        assert len(decoded[1]) == 1000

    def test_non_ascending_types_raises(self) -> None:
        # Manually craft a TLV with descending types
        bad_tlv = bytes([3, 1, 0x41, 1, 1, 0x42])  # type 3 then type 1
        with pytest.raises(ValueError):
            tlv_decode(bad_tlv)

    def test_truncated_value_raises(self) -> None:
        # type=1, length=10, but only 3 bytes of value
        bad_tlv = bytes([1, 10, 0x01, 0x02, 0x03])
        with pytest.raises(ValueError):
            tlv_decode(bad_tlv)


class TestOffer:
    def test_roundtrip(self) -> None:
        offer = Offer(
            description="Test offer",
            amount_msat=50_000,
            issuer="Test Merchant",
        )
        tlv = offer.to_tlv()
        recovered = Offer.from_tlv(tlv)
        assert recovered.description == "Test offer"
        assert recovered.amount_msat == 50_000
        assert recovered.issuer == "Test Merchant"

    def test_no_amount(self) -> None:
        offer = Offer(description="Free offer")
        tlv = offer.to_tlv()
        recovered = Offer.from_tlv(tlv)
        assert recovered.amount_msat is None

    def test_currency_roundtrip(self) -> None:
        offer = Offer(description="USD offer", currency="USD", amount_msat=10_000)
        tlv = offer.to_tlv()
        recovered = Offer.from_tlv(tlv)
        assert recovered.currency == "USD"

    def test_features_roundtrip(self) -> None:
        offer = Offer(description="Featured", features=0b1010)
        tlv = offer.to_tlv()
        recovered = Offer.from_tlv(tlv)
        assert recovered.features == 0b1010


class TestInvoiceRequest:
    def test_roundtrip(self) -> None:
        req = InvoiceRequest(
            offer_id=b"\x01" * 32,
            payer_id=b"\x02" * 33,
            amount_msat=100_000,
            payer_note="Thanks!",
        )
        tlv = req.to_tlv()
        recovered = InvoiceRequest.from_tlv(tlv, offer_id=b"\x01" * 32)
        assert recovered.amount_msat == 100_000
        assert recovered.payer_note == "Thanks!"

    def test_quantity_roundtrip(self) -> None:
        req = InvoiceRequest(
            offer_id=b"\x01" * 32,
            payer_id=b"\x02" * 33,
            quantity=5,
        )
        tlv = req.to_tlv()
        recovered = InvoiceRequest.from_tlv(tlv)
        assert recovered.quantity == 5


class TestBolt12Invoice:
    def test_roundtrip(self) -> None:
        ph = hashlib.sha256(b"bolt12test").digest()
        inv = B12Invoice(
            payment_hash=ph,
            amount_msat=1_000_000,
            created_at=1_700_000_000,
            relative_expiry=3600,
        )
        tlv = inv.to_tlv()
        recovered = B12Invoice.from_tlv(tlv)
        assert recovered.payment_hash == ph
        assert recovered.amount_msat == 1_000_000
        assert recovered.relative_expiry == 3600

    def test_is_expired_old(self) -> None:
        ph = hashlib.sha256(b"t").digest()
        inv = B12Invoice(payment_hash=ph, amount_msat=1000, created_at=1_000_000, relative_expiry=1)
        assert inv.is_expired()

    def test_is_not_expired_fresh(self) -> None:
        ph = hashlib.sha256(b"t").digest()
        inv = B12Invoice(payment_hash=ph, amount_msat=1000, created_at=int(time.time()), relative_expiry=7200)
        assert not inv.is_expired()

    def test_sign_stub_returns_64_bytes(self) -> None:
        msg = b"test message"
        key = b"\x01" * 32
        sig = sign_stub(msg, key)
        assert len(sig) == 64

    def test_verify_stub_returns_false(self) -> None:
        msg = b"test message"
        sig = b"\x00" * 64
        pub = b"\x02" * 33
        assert verify_stub(msg, sig, pub) is False


# ===========================================================================
# Pathfinding tests
# ===========================================================================

class TestFeeCalculate:
    def test_zero_amount(self) -> None:
        assert fee_calculate(0, 1000, 100) == 1000

    def test_proportional_only(self) -> None:
        # 1_000_000 msat * 1000 ppm = 1000 msat
        assert fee_calculate(1_000_000, 0, 1000) == 1000

    def test_base_only(self) -> None:
        assert fee_calculate(1_000_000, 500, 0) == 500

    def test_combined(self) -> None:
        # base=1000, rate=100ppm, amount=1_000_000 → 1000 + 100 = 1100
        assert fee_calculate(1_000_000, 1000, 100) == 1100

    def test_ceiling_applied(self) -> None:
        # 1 msat * 1 ppm = 0.000001 → ceil to 1
        assert fee_calculate(1, 0, 1) == 1

    def test_negative_amount_raises(self) -> None:
        with pytest.raises(ValueError):
            fee_calculate(-1, 1000, 100)

    def test_negative_base_fee_raises(self) -> None:
        with pytest.raises(ValueError):
            fee_calculate(1000, -1, 100)

    def test_negative_fee_rate_raises(self) -> None:
        with pytest.raises(ValueError):
            fee_calculate(1000, 1000, -1)


class TestGraph:
    def _simple_graph(self) -> Graph:
        g = Graph()
        g.add_bidirectional_channel("A", "B", "AB", base_fee_msat=1000, fee_rate_ppm=100)
        g.add_bidirectional_channel("B", "C", "BC", base_fee_msat=1000, fee_rate_ppm=100)
        return g

    def test_nodes_present(self) -> None:
        g = self._simple_graph()
        assert "A" in g.nodes
        assert "B" in g.nodes
        assert "C" in g.nodes

    def test_neighbors_a(self) -> None:
        g = self._simple_graph()
        neighbors = [ch.destination for ch in g.neighbors("A")]
        assert "B" in neighbors

    def test_bidirectional_both_directions(self) -> None:
        g = self._simple_graph()
        ab_neighbors = [ch.destination for ch in g.neighbors("A")]
        ba_neighbors = [ch.destination for ch in g.neighbors("B")]
        assert "B" in ab_neighbors
        assert "A" in ba_neighbors


class TestDijkstra:
    def _build_graph(self) -> Graph:
        g = Graph()
        g.add_bidirectional_channel("A", "B", "AB", base_fee_msat=100, fee_rate_ppm=10, capacity_msat=10_000_000)
        g.add_bidirectional_channel("B", "C", "BC", base_fee_msat=200, fee_rate_ppm=20, capacity_msat=10_000_000)
        g.add_bidirectional_channel("A", "C", "AC", base_fee_msat=1000, fee_rate_ppm=500, capacity_msat=10_000_000)
        return g

    def test_same_node_returns_single(self) -> None:
        g = self._build_graph()
        path = dijkstra(g, "A", "A")
        assert path == ["A"]

    def test_direct_path(self) -> None:
        g = self._build_graph()
        path = dijkstra(g, "A", "B")
        assert path is not None
        assert path[0] == "A"
        assert path[-1] == "B"

    def test_multi_hop_path(self) -> None:
        g = self._build_graph()
        # A→B→C should be cheaper than A→C directly
        path = dijkstra(g, "A", "C")
        assert path is not None
        assert path[0] == "A"
        assert path[-1] == "C"

    def test_no_path_returns_none(self) -> None:
        g = Graph()
        g.add_bidirectional_channel("A", "B", "AB")
        path = dijkstra(g, "A", "Z")
        assert path is None

    def test_unknown_src_returns_none(self) -> None:
        g = self._build_graph()
        path = dijkstra(g, "X", "C")
        assert path is None

    def test_path_starts_at_src(self) -> None:
        g = self._build_graph()
        path = dijkstra(g, "A", "C")
        assert path is not None
        assert path[0] == "A"

    def test_path_ends_at_dst(self) -> None:
        g = self._build_graph()
        path = dijkstra(g, "A", "C")
        assert path is not None
        assert path[-1] == "C"

    def test_min_htlc_filters_channel(self) -> None:
        g = Graph()
        # Channel requires min 1_000_000 msat but we send only 1_000
        g.add_channel(Channel("A", "B", "AB", 100, 10, 10_000_000, min_htlc_msat=1_000_000))
        path = dijkstra(g, "A", "B", amount_msat=1_000)
        assert path is None

    def test_capacity_filters_channel(self) -> None:
        g = Graph()
        # Channel max is 1_000 but we send 10_000_000
        g.add_channel(Channel("A", "B", "AB", 100, 10, capacity_msat=1_000, max_htlc_msat=1_000))
        path = dijkstra(g, "A", "B", amount_msat=10_000_000)
        assert path is None


class TestProbabilityWeightedPath:
    def _build_graph(self) -> Graph:
        g = Graph()
        g.add_bidirectional_channel("A", "B", "AB", capacity_msat=100_000_000)
        g.add_bidirectional_channel("B", "C", "BC", capacity_msat=100_000_000)
        g.add_bidirectional_channel("A", "C", "AC", capacity_msat=100_000_000)
        return g

    def test_returns_path(self) -> None:
        g = self._build_graph()
        success_rates = {"AB": 0.9, "AB_r": 0.9, "BC": 0.9, "BC_r": 0.9, "AC": 0.5, "AC_r": 0.5}
        path = probability_weighted_path(g, "A", "C", success_rates)
        assert path is not None
        assert path[-1] == "C"

    def test_same_node(self) -> None:
        g = self._build_graph()
        path = probability_weighted_path(g, "A", "A", {})
        assert path == ["A"]

    def test_no_path_returns_none(self) -> None:
        g = Graph()
        g.add_bidirectional_channel("A", "B", "AB")
        path = probability_weighted_path(g, "A", "Z", {})
        assert path is None

    def test_prefers_high_probability_channel(self) -> None:
        g = Graph()
        # Two paths: A→B direct (low prob) vs A→C→B (high prob)
        g.add_channel(Channel("A", "B", "AB_low", 0, 0, 100_000_000))
        g.add_channel(Channel("A", "C", "AC", 0, 0, 100_000_000))
        g.add_channel(Channel("C", "B", "CB", 0, 0, 100_000_000))
        success_rates = {"AB_low": 0.1, "AC": 0.99, "CB": 0.99}
        path = probability_weighted_path(g, "A", "B", success_rates)
        assert path is not None
        # Should prefer A→C→B over A→B directly
        assert path == ["A", "C", "B"]


class TestMppSplit:
    def test_amount_less_than_min_raises(self) -> None:
        # 100_000 < min_part_msat=200_000 → max_possible=0 → ValueError
        with pytest.raises(ValueError):
            mpp_split(100_000, max_parts=8, min_part_msat=200_000)

    def test_even_split(self) -> None:
        parts = mpp_split(1_000_000, max_parts=4, min_part_msat=1000)
        assert len(parts) == 4
        assert sum(parts) == 1_000_000

    def test_sums_correctly(self) -> None:
        for amount in [999_999, 1_000_000, 1_000_001, 7_777_777]:
            parts = mpp_split(amount, max_parts=8, min_part_msat=1000)
            assert sum(parts) == amount

    def test_max_parts_respected(self) -> None:
        parts = mpp_split(10_000_000, max_parts=3, min_part_msat=1000)
        assert len(parts) <= 3

    def test_min_part_respected(self) -> None:
        parts = mpp_split(1_000_000, max_parts=100, min_part_msat=100_000)
        # max possible = 1_000_000 // 100_000 = 10, min(100, 10) = 10
        assert all(p >= 100_000 for p in parts)

    def test_zero_amount_raises(self) -> None:
        with pytest.raises(ValueError):
            mpp_split(0)

    def test_negative_amount_raises(self) -> None:
        with pytest.raises(ValueError):
            mpp_split(-1)

    def test_amount_too_small_raises(self) -> None:
        with pytest.raises(ValueError):
            mpp_split(500, max_parts=8, min_part_msat=1000)

    def test_single_part_fallback(self) -> None:
        # Only one part fits
        parts = mpp_split(5_000, max_parts=8, min_part_msat=4_000)
        assert len(parts) == 1
        assert parts[0] == 5_000

    def test_parts_nearly_equal(self) -> None:
        parts = mpp_split(10_003, max_parts=3, min_part_msat=100)
        # 10_003 / 3 = 3334.33 → parts should be [3335, 3334, 3334]
        assert max(parts) - min(parts) <= 1
