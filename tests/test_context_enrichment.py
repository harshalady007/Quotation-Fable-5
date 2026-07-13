"""V3 context-enrichment review and audit tests."""

from __future__ import annotations

import json

import pandas as pd
import pytest

from context_enrichment import (ContextEnrichmentError,
                                build_context_review_queue,
                                merge_context_reviews, write_context_review,
                                write_correction_manifest)
from data_corrections import (DataCorrectionError,
                              validate_correction_manifest)
from pricing_dataset import load_pricing_dataset


def _workbook(tmp_path):
    path = tmp_path / "quotation_items.xlsx"
    products = pd.DataFrame([
        {
            "Source File": f"Q-{index}.pdf",
            "Quotation Date": f"2026-0{index}-15",
            "Scope of Work": "Supply Only",
            "Item Name": "Bench",
            "Description": (
                f"Mild steel backless bench length {1200 + index * 300}mm"
            ),
            "Unit": "Nos",
            "Unit Price": 900 + index * 100,
        }
        for index in range(1, 4)
    ])
    products.to_excel(path, sheet_name="Products", index=False)
    return path


def _empty_corrections(tmp_path):
    path = tmp_path / "pricing_corrections.json"
    path.write_text(
        json.dumps({"schema_version": 2, "records": []}), encoding="utf-8"
    )
    return path


def _review_file(tmp_path, data_path, corrections_path):
    rows, summary = build_context_review_queue(
        str(data_path), str(corrections_path), family="bench"
    )
    path = tmp_path / "context-review.csv"
    write_context_review(path, rows, summary)
    return path, rows, summary


def test_context_queue_uses_stable_ids_and_profiles_only_missing_fields(tmp_path):
    data_path = _workbook(tmp_path)
    corrections_path = _empty_corrections(tmp_path)
    _, rows, summary = _review_file(tmp_path, data_path, corrections_path)

    assert len(rows) == 3
    assert len({row["record_id"] for row in rows}) == 3
    assert summary["scope_rows"] == 3
    assert summary["fields"]["quantity"]["coverage"] == 0.0
    assert summary["fields"]["quotation_date"]["coverage"] == 1.0
    assert rows[0]["fields_needing_review"] == [
        "quantity", "supplier", "location"
    ]
    assert all(row["dataset_fingerprint"] == summary["dataset_fingerprint"]
               for row in rows)


def test_approved_context_review_merges_with_field_evidence_and_applies(tmp_path):
    data_path = _workbook(tmp_path)
    corrections_path = _empty_corrections(tmp_path)
    review_path, _, _ = _review_file(tmp_path, data_path, corrections_path)
    review = pd.read_csv(review_path, dtype=object, keep_default_na=False)
    record_id = review.loc[0, "record_id"]
    review.loc[0, [
        "review_status", "reviewed_quantity", "reviewed_supplier",
        "reviewed_location", "evidence_reference", "review_reason",
        "reviewed_by", "reviewed_at",
    ]] = [
        "approved", "12", " ACME Industries ", " Abu Dhabi ",
        "Q-1.pdf page 2", "Confirmed against the signed quotation line.",
        "estimator@example.com", "2026-07-13",
    ]
    review.to_csv(review_path, index=False)

    manifest, summary = merge_context_reviews(
        review_path, str(data_path), str(corrections_path)
    )
    correction = manifest["records"][0]
    assert manifest["schema_version"] == 2
    assert correction["set"] == {
        "quantity": 12,
        "supplier": "acme industries",
        "location": "abu dhabi",
    }
    assert set(correction["field_reviews"]) == {
        "quantity", "supplier", "location"
    }
    assert summary["approved_fields"] == {
        "location": 1, "quantity": 1, "supplier": 1
    }

    write_correction_manifest(corrections_path, manifest)
    enriched = load_pricing_dataset(str(data_path), str(corrections_path))
    row = enriched.loc[enriched["record_id"] == record_id].iloc[0]
    assert row["quantity"] == 12
    assert row["supplier"] == "acme industries"
    assert row["location"] == "abu dhabi"
    assert row["record_id"] == record_id
    assert "abu dhabi" not in row["search_text"]


def test_stale_review_is_rejected_before_any_manifest_change(tmp_path):
    data_path = _workbook(tmp_path)
    corrections_path = _empty_corrections(tmp_path)
    review_path, _, _ = _review_file(tmp_path, data_path, corrections_path)
    review = pd.read_csv(review_path, dtype=object, keep_default_na=False)
    review["dataset_fingerprint"] = "stale-fingerprint"
    review.to_csv(review_path, index=False)

    with pytest.raises(ContextEnrichmentError, match="stale"):
        merge_context_reviews(review_path, str(data_path), str(corrections_path))
    assert json.loads(corrections_path.read_text(encoding="utf-8"))["records"] == []


def test_existing_context_requires_explicit_override_and_conflicts_never_win(tmp_path):
    data_path = _workbook(tmp_path)
    corrections_path = _empty_corrections(tmp_path)
    review_path, _, _ = _review_file(tmp_path, data_path, corrections_path)
    review = pd.read_csv(review_path, dtype=object, keep_default_na=False)
    review.loc[0, [
        "review_status", "reviewed_quotation_date", "evidence_reference",
        "review_reason", "reviewed_by", "reviewed_at",
    ]] = [
        "approved", "2026-06-01", "Q-1.pdf cover",
        "Corrected from the signed quotation cover.",
        "estimator@example.com", "2026-07-13",
    ]
    review.to_csv(review_path, index=False)

    with pytest.raises(ContextEnrichmentError, match="allow_override"):
        merge_context_reviews(review_path, str(data_path), str(corrections_path))


def test_schema_two_rejects_approved_context_without_field_audit():
    manifest = {
        "schema_version": 2,
        "records": [{
            "record_id": "stable-record",
            "status": "approved",
            "reason": "Estimator supplied quantity.",
            "reviewed_by": "estimator@example.com",
            "set": {"quantity": 10},
        }],
    }
    with pytest.raises(DataCorrectionError, match="field-level evidence"):
        validate_correction_manifest(manifest)


def test_import_will_not_activate_an_unrelated_proposed_correction(tmp_path):
    data_path = _workbook(tmp_path)
    corrections_path = _empty_corrections(tmp_path)
    review_path, rows, _ = _review_file(tmp_path, data_path, corrections_path)
    corrections_path.write_text(json.dumps({
        "schema_version": 2,
        "records": [{
            "record_id": rows[0]["record_id"],
            "status": "proposed",
            "reason": "Unit still needs a separate review.",
            "set": {"unit": "set"},
        }],
    }), encoding="utf-8")

    # The correction changed the active fingerprint, so first regenerate the
    # template against that exact proposed manifest.
    review_path, _, _ = _review_file(tmp_path, data_path, corrections_path)
    review = pd.read_csv(review_path, dtype=object, keep_default_na=False)
    target = review.index[review["record_id"] == rows[0]["record_id"]][0]
    review.loc[target, [
        "review_status", "reviewed_quantity", "evidence_reference",
        "review_reason", "reviewed_by", "reviewed_at",
    ]] = [
        "approved", "8", "Q-1.pdf page 2",
        "Confirmed against the quotation.",
        "estimator@example.com", "2026-07-13",
    ]
    review.to_csv(review_path, index=False)

    with pytest.raises(ContextEnrichmentError, match="non-approved correction"):
        merge_context_reviews(review_path, str(data_path), str(corrections_path))
