"""One-time migration: add split, difficulty, and chunk_relevance to the eval dataset."""
import json
import pathlib

DATASET_PATH = pathlib.Path("data/qa/retrieval_recall_dataset.json")

TEST_IDS = {"R6", "R11", "R15", "R19", "R23", "R30", "R36", "R39", "R42", "R49"}

_DIFFICULTY_MAP: dict[str, set[str]] = {
    "easy": {
        "R3", "R5", "R8", "R9", "R11", "R15", "R16", "R22", "R25",
        "R33", "R35", "R38", "R39", "R43", "R45", "R46", "R47", "R48", "R49", "R50",
    },
    "medium": {
        "R2", "R4", "R6", "R7", "R10", "R13", "R17", "R18", "R20", "R23",
        "R24", "R26", "R27", "R28", "R29", "R31", "R34", "R36", "R37",
        "R40", "R41", "R42", "R44",
    },
    "hard": {"R1", "R12", "R14", "R19", "R21", "R30", "R32"},
}
ID_TO_DIFF = {cid: diff for diff, ids in _DIFFICULTY_MAP.items() for cid in ids}


def main() -> None:
    data = json.loads(DATASET_PATH.read_text(encoding="utf-8"))
    for case in data["cases"]:
        cid = case["case_id"]
        case["split"] = "test" if cid in TEST_IDS else "dev"
        case["difficulty"] = ID_TO_DIFF.get(cid, "medium")
        # Seed chunk_relevance: all existing expected IDs start at 2 (fully relevant)
        existing_relevance = case.get("chunk_relevance", {})
        for chunk_id in case.get("expected_chunk_ids", []):
            if chunk_id and chunk_id not in existing_relevance:
                existing_relevance[chunk_id] = 2
        case["chunk_relevance"] = existing_relevance

    DATASET_PATH.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Migrated {len(data['cases'])} cases in {DATASET_PATH}")

    # Verify
    by_split: dict[str, int] = {}
    by_diff: dict[str, int] = {}
    for case in data["cases"]:
        by_split[case["split"]] = by_split.get(case["split"], 0) + 1
        by_diff[case["difficulty"]] = by_diff.get(case["difficulty"], 0) + 1
    print(f"  Split distribution: {dict(sorted(by_split.items()))}")
    print(f"  Difficulty distribution: {dict(sorted(by_diff.items()))}")


if __name__ == "__main__":
    main()
