from typing import Any

import pandas as pd
import streamlit as st

from frontend.pdf_export import build_compare_pdf, build_summary_pdf


def render_summary_response(response: dict[str, Any], heading: str = "Result") -> None:
    extraction = response.get("extraction", {})
    verification = response.get("verification", {})
    warnings = response.get("warnings", [])

    heading_col, dl_col = st.columns([5, 1])
    with heading_col:
        st.subheader(heading)
    with dl_col:
        pdf_bytes = build_summary_pdf(response, title=heading)
        st.download_button(
            label="Download PDF",
            data=pdf_bytes,
            file_name="lease_summary.pdf",
            mime="application/pdf",
            use_container_width=True,
        )

    render_guardrail_checks(verification, warnings)
    render_extraction_overview(extraction)

    summary = extraction.get("plain_english_summary")
    if summary:
        st.markdown("#### Plain-English Summary")
        st.write(summary)

    render_list_section("Tenant Obligations", extraction.get("tenant_obligations"))
    render_list_section("Landlord Obligations", extraction.get("landlord_obligations"))
    render_list_section("Unusual Clauses", extraction.get("unusual_clauses"))

    with st.expander("Raw JSON"):
        st.json(response)


def render_extraction_overview(extraction: dict[str, Any]) -> None:
    rows = [
        ("Tenant", extraction.get("tenant_name")),
        ("Landlord", extraction.get("landlord_name")),
        ("Property", extraction.get("property_address")),
        ("Lease start", extraction.get("lease_start_date")),
        ("Lease end", extraction.get("lease_end_date")),
        ("Monthly rent", extraction.get("monthly_rent_amount")),
        ("Rent due", extraction.get("rent_payment_due_date")),
        ("Security deposit", extraction.get("security_deposit_amount")),
        ("Notice to vacate", extraction.get("notice_period_to_vacate")),
    ]
    st.dataframe(
        pd.DataFrame(
            [{"field": field, "value": display_value(value)} for field, value in rows]
        ),
        hide_index=True,
        use_container_width=True,
    )


def render_list_section(title: str, values: list[str] | None) -> None:
    if not values:
        return

    st.markdown(f"#### {title}")
    for value in values:
        st.markdown(f"- {value}")


def render_guardrail_checks(
    verification: dict[str, Any],
    warnings: list[str] | None = None,
) -> None:
    st.markdown("#### Grounding Check")
    overall_supported = verification.get("overall_supported")

    if warnings:
        st.warning("\n".join(f"- {warning}" for warning in warnings))

    if overall_supported is True:
        st.success("All extracted values were marked as supported.")
    elif overall_supported is False:
        if not warnings:
            st.warning("Some extracted values need review.")
    else:
        st.info("Grounding status was not returned.")

    checks = verification.get("checks") or []
    if not checks:
        return

    rows = [
        {
            "field": check.get("field_name"),
            "status": check.get("status"),
            "evidence": check.get("evidence"),
            "explanation": check.get("explanation"),
        }
        for check in checks
    ]
    st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)


def render_compare_response(response: dict[str, Any]) -> None:
    comparison = response.get("comparison", {})
    differences = comparison.get("differences") or []

    render_compare_guardrail_summary(response)

    heading_col, dl_col = st.columns([5, 1])
    with heading_col:
        st.subheader("Comparison")
    with dl_col:
        pdf_bytes = build_compare_pdf(response)
        st.download_button(
            label="Download PDF",
            data=pdf_bytes,
            file_name="lease_comparison.pdf",
            mime="application/pdf",
            use_container_width=True,
        )

    summary = comparison.get("summary")
    if summary:
        st.write(summary)

    if differences:
        rows = [
            {
                "field": item.get("field_name"),
                "lease_a": stringify_value(item.get("lease_a_value")),
                "lease_b": stringify_value(item.get("lease_b_value")),
                "difference": item.get("difference"),
                "practical_impact": item.get("practical_impact"),
            }
            for item in differences
        ]
        st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
    else:
        st.info("No material differences were returned.")

    lease_a_tab, lease_b_tab, raw_tab = st.tabs(["Lease A Result", "Lease B Result", "Raw JSON"])
    with lease_a_tab:
        render_summary_response(response.get("lease_a", {}), heading="Lease A")
    with lease_b_tab:
        render_summary_response(response.get("lease_b", {}), heading="Lease B")
    with raw_tab:
        st.json(response)


def render_compare_guardrail_summary(response: dict[str, Any]) -> None:
    lease_a = response.get("lease_a", {})
    lease_b = response.get("lease_b", {})
    rows = [
        build_guardrail_summary_row("Lease A", lease_a),
        build_guardrail_summary_row("Lease B", lease_b),
    ]

    st.subheader("Grounding Check")
    needs_review = any(
        result.get("verification", {}).get("overall_supported") is not True
        or bool(result.get("warnings"))
        for result in [lease_a, lease_b]
    )
    if needs_review:
        st.warning("Review grounding warnings before using the comparison.")
    else:
        st.success("Both leases completed grounding checks without unsupported values.")

    st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)


def build_guardrail_summary_row(label: str, result: dict[str, Any]) -> dict[str, str]:
    verification = result.get("verification", {})
    warnings = result.get("warnings") or []
    checks = verification.get("checks") or []
    unsupported_count = sum(1 for check in checks if check.get("status") == "unsupported")

    return {
        "lease": label,
        "overall_supported": stringify_value(verification.get("overall_supported")),
        "unsupported_fields": str(unsupported_count),
        "warnings": "; ".join(warnings) if warnings else "None",
    }


def display_value(value: Any) -> str:
    if value in (None, "", []):
        return "Not found"
    if isinstance(value, list):
        return ", ".join(str(item) for item in value)
    return str(value)


def stringify_value(value: Any) -> str:
    if value is None:
        return "Not found"
    if isinstance(value, list):
        return "; ".join(str(item) for item in value)
    return str(value)
