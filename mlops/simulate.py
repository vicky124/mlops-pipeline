"""Send production-like traffic to a running model server, optionally drifted.

    python -m mlops.simulate --rows 2000                          # traffic like training data
    python -m mlops.simulate --rows 2000 --price-increase 25 --m2m-share 0.75 --ticket-rate 2.5
"""

from __future__ import annotations

import argparse

import httpx

from mlops.data import FEATURES, DriftSpec, generate


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--url", default="http://localhost:8080")
    p.add_argument("--rows", type=int, default=2_000)
    p.add_argument("--batch", type=int, default=100)
    p.add_argument("--price-increase", type=float, default=0.0)
    p.add_argument("--m2m-share", type=float, default=0.55)
    p.add_argument("--ticket-rate", type=float, default=1.2)
    p.add_argument("--seed", type=int, default=99)
    args = p.parse_args()

    drift = DriftSpec(charges_shift=args.price_increase, m2m_share=args.m2m_share,
                      ticket_rate=args.ticket_rate)
    df = generate(args.rows, seed=args.seed, drift=drift)[FEATURES]
    churn = 0
    with httpx.Client(base_url=args.url, timeout=30) as client:
        for start in range(0, len(df), args.batch):
            records = df.iloc[start:start + args.batch].to_dict("records")
            res = client.post("/predict", json={"records": records})
            res.raise_for_status()
            churn += sum(p["churn"] for p in res.json()["predictions"])
    print(f"scored {len(df)} rows; flagged {churn} ({churn / len(df):.1%}) as likely to churn")


if __name__ == "__main__":
    main()
