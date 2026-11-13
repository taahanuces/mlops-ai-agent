"""Run all 5 demo scenarios against the agent API and save transcripts with raw tool evidence."""

from __future__ import annotations

import json
from pathlib import Path

import httpx

AGENT_URL = "http://localhost:9000"
REPORTS_DIR = Path(__file__).resolve().parent.parent / "reports" / "conversations"

SCENARIOS = [
    {
        "id": 1,
        "title": "Scenario 1 — Dataset Provenance for Specific Experiment",
        "prompt": "Which dataset version was used for experiment 7? Provide Git commit and reproduction commands.",
    },
    {
        "id": 2,
        "title": "Scenario 2 — Experiment Comparison and Best Model Recommendation",
        "prompt": "Compare the last five training runs and recommend the best candidate by F1 score. Explain any caveats or exclusions.",
    },
    {
        "id": 3,
        "title": "Scenario 3 — Authorized Pipeline Trigger with Confirmation Flow",
        "prompt": "Trigger the model training pipeline and report whether it succeeded.",
        "requires_confirm": True,
    },
    {
        "id": 4,
        "title": "Scenario 4 — Kubernetes Incident Diagnosis and Root Cause Analysis",
        "prompt": "Why is my prediction API unavailable in Kubernetes?",
    },
    {
        "id": 5,
        "title": "Scenario 5 — Mandatory Multi-System Operational Correlated Scenario",
        "prompt": "Find the best recent model, identify the dataset version used to train it, check the deployment status, and explain whether retraining is necessary.",
    },
]


def run_scenario(client: httpx.Client, scenario: dict) -> None:
    sid = scenario["id"]
    title = scenario["title"]
    prompt = scenario["prompt"]
    print("\n========================================================")
    print(f" Executing {title}")
    print(f" Prompt: '{prompt}'")
    print("========================================================")

    resp = client.post("/chat", json={"message": prompt})
    if resp.status_code != 200:
        print(f"Error {resp.status_code}: {resp.text}")
        return

    data = resp.json()
    answer = data.get("answer", {})
    evidence = data.get("evidence", [])
    pending = answer.get("pending_confirmations", [])

    confirm_result = None
    if scenario.get("requires_confirm") and pending:
        cid = pending[0]
        print(f"Detected pending action: {cid}. Simulating explicit user confirmation...")
        c_resp = client.post(f"/confirm/{cid}")
        confirm_result = c_resp.json()
        print(f"Confirmation response: {confirm_result.get('status')}")

    # Save JSON output
    json_path = REPORTS_DIR / f"scenario_{sid}.json"
    full_output = {
        "scenario": scenario,
        "chat_response": data,
        "confirmation_execution": confirm_result,
    }
    with open(json_path, "w") as f:
        json.dump(full_output, f, indent=2)

    # Save Markdown transcript
    md_path = REPORTS_DIR / f"scenario_{sid}.md"
    with open(md_path, "w") as f:
        f.write(f"# {title}\n\n")
        f.write(f"**User Request:**\n> {prompt}\n\n")
        f.write(f"## Agent Summary\n{answer.get('summary', '')}\n\n")

        f.write("## Known Facts (Tool-Grounded)\n")
        for fact in answer.get("known_facts", []):
            eids = ", ".join(fact.get("evidence_ids", []))
            f.write(f"- {fact.get('statement')} `[{eids}]`\n")
        f.write("\n")

        f.write("## Uncertainties & Service Failures\n")
        uncertainties = answer.get("uncertainties", [])
        if uncertainties:
            for u in uncertainties:
                f.write(f"- {u}\n")
        else:
            f.write("- None identified.\n")
        f.write("\n")

        f.write("## Recommended Actions\n")
        actions = answer.get("recommended_actions", [])
        if actions:
            for a in actions:
                f.write(f"- {a}\n")
        else:
            f.write("- None.\n")
        f.write("\n")

        if confirm_result:
            f.write("## Confirmation Action Execution\n")
            f.write(f"```json\n{json.dumps(confirm_result, indent=2)}\n```\n\n")

        f.write("## Supporting Evidence Log\n")
        f.write("| Evidence ID | Tool | Source | Status | Duration (ms) |\n")
        f.write("|---|---|---|---|---|\n")
        for ev in evidence:
            f.write(
                f"| `{ev.get('evidence_id')}` | `{ev.get('tool')}` | `{ev.get('source')}` | `{ev.get('status')}` | {ev.get('duration_ms')} |\n"
            )
        f.write("\n")

    print(f"✓ Saved scenario {sid} transcript to {md_path} and {json_path}")


def main():
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    with httpx.Client(base_url=AGENT_URL, timeout=120.0) as client:
        for scen in SCENARIOS:
            run_scenario(client, scen)


if __name__ == "__main__":
    main()
