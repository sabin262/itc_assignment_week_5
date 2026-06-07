from dataclasses import dataclass


@dataclass
class TestResult:
    nodeid: str
    status: str
    duration: float


_RESULTS_BY_NODEID: dict[str, TestResult] = {}
_RESULT_ORDER: list[str] = []


def pytest_sessionstart(session) -> None:
    _RESULTS_BY_NODEID.clear()
    _RESULT_ORDER.clear()


def pytest_runtest_logreport(report) -> None:
    if report.when == "call":
        _record_result(report)
    elif report.when in {"setup", "teardown"} and (report.failed or report.skipped):
        _record_result(report)


def pytest_terminal_summary(terminalreporter, exitstatus, config) -> None:
    results = [_RESULTS_BY_NODEID[nodeid] for nodeid in _RESULT_ORDER]
    if not results:
        return

    passed = sum(1 for result in results if result.status == "PASS")
    total = len(results)
    pass_percentage = (passed / total) * 100

    terminalreporter.write_sep("=", "Per-test Results")
    for result in results:
        terminalreporter.write_line(
            f"{result.status:<4} {result.nodeid} ({result.duration:.2f}s)"
        )

    terminalreporter.write_sep("=", "Pass Percentage")
    terminalreporter.write_line(f"{passed}/{total} tests passed ({pass_percentage:.1f}%).")


def _record_result(report) -> None:
    nodeid = report.nodeid
    if nodeid not in _RESULTS_BY_NODEID:
        _RESULT_ORDER.append(nodeid)

    status = _status_for_report(report)
    existing = _RESULTS_BY_NODEID.get(nodeid)

    if existing is None or status == "FAIL" or existing.status != "FAIL":
        _RESULTS_BY_NODEID[nodeid] = TestResult(
            nodeid=nodeid,
            status=status,
            duration=report.duration,
        )


def _status_for_report(report) -> str:
    if report.failed:
        return "FAIL"
    if report.passed:
        return "PASS"
    if report.skipped:
        return "SKIP"
    return report.outcome.upper()
