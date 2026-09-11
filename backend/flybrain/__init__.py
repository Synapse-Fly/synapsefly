"""FlyBrain / SynapseFly backend package (SPEC v1.0).

Import root is ``backend/``; ``flybrain`` is a top-level package. Runtime hard
dependencies are numpy, fastapi, uvicorn, websockets, pydantic and httpx; every
optional dependency (torch, anthropic, tweepy, pyarrow, pandas, neuprint) is
imported inside functions only (SPEC section 0.1).
"""

__version__ = "0.1.0"

__all__ = ["__version__"]
