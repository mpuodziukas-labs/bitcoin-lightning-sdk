# bitcoin-lightning-sdk

Pure Python Lightning Network SDK implementing BOLT 11, BOLT 12, and pathfinding algorithms.

## Modules

- `lightning/bolt11.py` — BOLT 11 invoice decode/encode with known test vectors
- `lightning/bolt12.py` — BOLT 12 Offer/InvoiceRequest/Invoice with TLV codec
- `lightning/pathfinding.py` — Dijkstra, probability-weighted path, MPP split, fee calculation

## Quick Start

```python
from lightning.bolt11 import decode, encode
from lightning.pathfinding import Graph, dijkstra, fee_calculate, mpp_split
import hashlib

# Decode a BOLT 11 invoice
invoice = decode("lnbc2500u1pvjluez...")
print(f"Amount: {invoice.amount_msat} msat")

# Encode a new invoice
payment_hash = hashlib.sha256(b"my payment").digest()
inv_str = encode(payment_hash, amount_msat=100_000, description="Test")

# Find a path
g = Graph()
g.add_bidirectional_channel("Alice", "Bob", "AB")
g.add_bidirectional_channel("Bob", "Carol", "BC")
path = dijkstra(g, "Alice", "Carol")

# Split a payment (MPP)
parts = mpp_split(1_000_000, max_parts=4)
print(f"Split into {len(parts)} parts: {parts}")
```

## Zero External Dependencies

Pure Python standard library only. No numpy, no cryptography libraries required.

## Installation

```bash
pip install pytest
pytest tests/
```
