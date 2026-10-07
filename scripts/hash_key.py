"""Hash a team API key for config/gateway.yaml.

    python scripts/hash_key.py my-secret-key   # hash an existing key
    python scripts/hash_key.py --new           # generate a random key and its hash
"""

import hashlib
import secrets
import sys

if len(sys.argv) != 2:
    sys.exit(__doc__)
key = "gw-" + secrets.token_urlsafe(24) if sys.argv[1] == "--new" else sys.argv[1]
if sys.argv[1] == "--new":
    print(f"key:            {key}   (give this to the team; it is not stored anywhere)")
print(f"api_key_sha256: {hashlib.sha256(key.encode()).hexdigest()}")
