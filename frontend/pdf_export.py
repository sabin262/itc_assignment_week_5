from datetime import date
from typing import Any

from fpdf import FPDF


_UNICODE_MAP = str.maketrans({
    "—": "--",   # em dash
    "–": "-",    # en dash
    "‘": "'",    # left single quote
    "’": "'",    # right single quote
    "“": '"',    # left double quote
    "”": '"',    # right double quote
    "•": "-",    # bullet
    " ": " ",    # non-breaking space
    "…": "...",  # ellipsis
})


def _safe(text: str) -> str:
    """Return latin-1-safe string for use with fpdf2 core fonts."""
    text = text.translate(_UNICODE_MAP)
    return text.encode("latin-1", errors="replace").decode("latin-1")


def _clean(value: Any) -> str:
    if value in (None, "", []):
        return "Not found"
    if isinstance(value, list):
        return "; ".join(str(item) for item in value)
    return str(value)


class _PDF(FPDF):
    _LINE_H = 5.0

    def header(self) -> None:
        self.set_font("Helvetica", "B", 13)
        self.cell(0, 9, _safe(self.title), align="C", new_x="LMARGIN", new_y="NEXT")
        self.set_font("Helvetica", "", 8)
        self.cell(0, 5, f"Generated: {date.today()}", align="C", new_x="LMARGIN", new_y="NEXT")
        self.ln(3)

    def footer(self) -> None:
        self.set_y(-11)
        self.set_font("Helvetica", "I", 7)
        self.cell(0, 8, f"Page {self.page_no()}", align="C")

    def h2(self, text: str) -> None:
        self.set_font("Helvetica", "B", 10)
        self.set_fill_color(220, 220, 220)
        self.set_x(self.l_margin)
        self.cell(self.epw, 7, _safe(text), fill=True, new_x="LMARGIN", new_y="NEXT")
        self.set_fill_color(255, 255, 255)
        self.ln(1)

    def kv(self, key: str, value: str) -> None:
        y0 = self.get_y()
        self.set_font("Helvetica", "B", 9)
        self.set_x(self.l_margin)
        self.cell(52, self._LINE_H, _safe(key + ":"))
        self.set_font("Helvetica", "", 9)
        self.set_xy(self.l_margin + 52, y0)
        self.multi_cell(self.epw - 52, self._LINE_H, _safe(value))

    def bullets(self, items: list[str]) -> None:
        self.set_font("Helvetica", "", 9)
        for item in items:
            self.set_x(self.l_margin)
            self.multi_cell(self.epw, self._LINE_H, _safe(f"-  {item}"))
        self.ln(1)

    def body(self, text: str) -> None:
        self.set_font("Helvetica", "", 9)
        self.set_x(self.l_margin)
        self.multi_cell(self.epw, self._LINE_H, _safe(text))
        self.ln(2)

    def tbl_head(self, cols: list[tuple[str, float]]) -> None:
        self.set_font("Helvetica", "B", 8)
        self.set_fill_color(200, 200, 200)
        self.set_x(self.l_margin)
        for label, w in cols:
            self.cell(w, 6, _safe(label), border=1, fill=True)
        self.ln()
        self.set_fill_color(255, 255, 255)

    def tbl_row(self, cells: list[tuple[str, float]]) -> None:
        line_h = 4.0
        x0 = self.l_margin
        y0 = self.get_y()

        safe_cells = [(_safe(text), w) for text, w in cells]

        # Compute row height from content
        self.set_font("Helvetica", "", 7.5)
        max_lines = 1
        for text, w in safe_cells:
            usable = w - 2
            line_w = 0.0
            lines = 1
            for word in text.split():
                word_w = self.get_string_width(word + " ")
                if line_w + word_w > usable:
                    lines += 1
                    line_w = word_w
                else:
                    line_w += word_w
            max_lines = max(max_lines, lines)
        row_h = max_lines * line_h + 2

        if y0 + row_h > self.h - self.b_margin - 5:
            self.add_page()
            y0 = self.get_y()

        x = x0
        for text, w in safe_cells:
            self.rect(x, y0, w, row_h)
            self.set_xy(x + 1, y0 + 1)
            self.multi_cell(w - 2, line_h, text, border=0)
            x += w

        self.set_xy(x0, y0 + row_h)


def _extraction_section(pdf: _PDF, result: dict[str, Any]) -> None:
    extraction = result.get("extraction", {})
    for key, field in [
        ("Tenant", "tenant_name"),
        ("Landlord", "landlord_name"),
        ("Property", "property_address"),
        ("Lease Start", "lease_start_date"),
        ("Lease End", "lease_end_date"),
        ("Monthly Rent", "monthly_rent_amount"),
        ("Rent Due", "rent_payment_due_date"),
        ("Security Deposit", "security_deposit_amount"),
        ("Notice to Vacate", "notice_period_to_vacate"),
    ]:
        pdf.kv(key, _clean(extraction.get(field)))
    pdf.ln(2)

    summary = extraction.get("plain_english_summary")
    if summary:
        pdf.h2("Plain-English Summary")
        pdf.body(str(summary))

    for title, field in [
        ("Tenant Obligations", "tenant_obligations"),
        ("Landlord Obligations", "landlord_obligations"),
        ("Unusual Clauses", "unusual_clauses"),
    ]:
        items = extraction.get(field) or []
        if items:
            pdf.h2(title)
            pdf.bullets(items)


def _grounding_section(pdf: _PDF, result: dict[str, Any], label: str = "") -> None:
    verification = result.get("verification", {})
    warnings = result.get("warnings") or []
    prefix = f"{label} - " if label else ""

    overall = verification.get("overall_supported")
    if overall is True:
        status = "All extracted values supported"
    elif overall is False:
        status = "Some extracted values need review"
    else:
        status = "Status not available"
    pdf.kv(f"{prefix}Result", status)

    if warnings:
        pdf.kv(f"{prefix}Warnings", "; ".join(warnings))

    checks = verification.get("checks") or []
    unsupported = sum(1 for c in checks if c.get("status") == "unsupported")
    if unsupported:
        pdf.kv(f"{prefix}Unsupported Fields", str(unsupported))


def build_summary_pdf(response: dict[str, Any], title: str = "Lease Summary Report") -> bytes:
    pdf = _PDF()
    pdf.title = title
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()

    pdf.h2("Grounding Check")
    _grounding_section(pdf, response)
    pdf.ln(2)

    pdf.h2("Key Details")
    _extraction_section(pdf, response)

    verification = response.get("verification", {})
    checks = verification.get("checks") or []
    if checks:
        pdf.h2("Grounding Check Details")
        cols: list[tuple[str, float]] = [
            ("Field", 45.0), ("Status", 25.0), ("Evidence", 60.0), ("Explanation", 60.0),
        ]
        pdf.tbl_head(cols)
        for check in checks:
            pdf.tbl_row([
                (_clean(check.get("field_name")), 45.0),
                (_clean(check.get("status")), 25.0),
                (_clean(check.get("evidence")), 60.0),
                (_clean(check.get("explanation")), 60.0),
            ])

    return bytes(pdf.output())


def build_compare_pdf(response: dict[str, Any]) -> bytes:
    comparison = response.get("comparison", {})
    differences = comparison.get("differences") or []
    lease_a = response.get("lease_a", {})
    lease_b = response.get("lease_b", {})

    pdf = _PDF()
    pdf.title = "Lease Comparison Report"
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()

    pdf.h2("Grounding Check Summary")
    _grounding_section(pdf, lease_a, "Lease A")
    _grounding_section(pdf, lease_b, "Lease B")
    pdf.ln(2)

    summary = comparison.get("summary")
    if summary:
        pdf.h2("Comparison Summary")
        pdf.body(str(summary))

    pdf.h2("Differences")
    if differences:
        cols_diff: list[tuple[str, float]] = [
            ("Field", 35.0), ("Lease A", 35.0), ("Lease B", 35.0),
            ("Difference", 50.0), ("Practical Impact", 35.0),
        ]
        pdf.tbl_head(cols_diff)
        for item in differences:
            pdf.tbl_row([
                (_clean(item.get("field_name")), 35.0),
                (_clean(item.get("lease_a_value")), 35.0),
                (_clean(item.get("lease_b_value")), 35.0),
                (_clean(item.get("difference")), 50.0),
                (_clean(item.get("practical_impact")), 35.0),
            ])
    else:
        pdf.body("No material differences were returned.")

    pdf.add_page()
    pdf.h2("Lease A - Full Summary")
    _grounding_section(pdf, lease_a)
    pdf.ln(2)
    _extraction_section(pdf, lease_a)

    pdf.add_page()
    pdf.h2("Lease B - Full Summary")
    _grounding_section(pdf, lease_b)
    pdf.ln(2)
    _extraction_section(pdf, lease_b)

    return bytes(pdf.output())
