"""Interactive command-line interface for chatting with the MLOps Agent."""

from __future__ import annotations

import json

import httpx

AGENT_URL = "http://localhost:9000"


def main():
    print("=" * 60)
    print(" MLOps AI Agent Terminal Client")
    print(f" Target Server: {AGENT_URL}")
    print(" Commands: 'exit' to quit, 'confirm <id>' to confirm an action")
    print("=" * 60)

    with httpx.Client(base_url=AGENT_URL, timeout=120.0) as client:
        # Check health
        try:
            h = client.get("/health")
            if h.status_code != 200:
                print(f"[Warning] Agent health check failed: {h.text}")
        except Exception as e:
            print(f"[Warning] Could not connect to agent at {AGENT_URL}: {e}")

        while True:
            try:
                user_input = input("\n[You] > ").strip()
                if not user_input:
                    continue
                if user_input.lower() in ("exit", "quit"):
                    print("Exiting. Goodbye!")
                    break

                if user_input.startswith("confirm "):
                    cid = user_input.split(" ", 1)[1].strip()
                    print(f"Confirming action {cid}...")
                    resp = client.post(f"/confirm/{cid}")
                    print(json.dumps(resp.json(), indent=2))
                    continue

                resp = client.post("/chat", json={"message": user_input})
                if resp.status_code != 200:
                    print(f"[Error {resp.status_code}] {resp.text}")
                    continue

                data = resp.json()
                answer = data.get("answer", {})
                print("\n[Agent Summary]")
                print(answer.get("summary", ""))

                facts = answer.get("known_facts", [])
                if facts:
                    print("\n[Known Facts]")
                    for f in facts:
                        eids = ", ".join(f.get("evidence_ids", []))
                        print(f"  • {f.get('statement')} (Evidence: {eids})")

                uncertainties = answer.get("uncertainties", [])
                if uncertainties:
                    print("\n[Uncertainties / Limitations]")
                    for u in uncertainties:
                        print(f"  ? {u}")

                actions = answer.get("recommended_actions", [])
                if actions:
                    print("\n[Recommended Actions]")
                    for a in actions:
                        print(f"  → {a}")

                pending = answer.get("pending_confirmations", [])
                if pending:
                    print("\n[Pending Confirmations Required]")
                    for p in pending:
                        print(f"  ! ID: {p} (Run: confirm {p})")

            except (KeyboardInterrupt, EOFError):
                print("\nSession terminated.")
                break
            except Exception as e:
                print(f"[Error] {e}")


if __name__ == "__main__":
    main()
