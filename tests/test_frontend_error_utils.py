import httpx

from frontend.error_utils import extract_error_message


def test_extract_error_message_from_string_detail():
    response = httpx.Response(
        422,
        json={"detail": "Lease text must contain at least 100 words."},
    )

    assert extract_error_message(response) == "Lease text must contain at least 100 words."


def test_extract_error_message_from_fastapi_validation_detail_list():
    response = httpx.Response(
        422,
        json={
            "detail": [
                {
                    "loc": ["body", "file"],
                    "msg": "Field required",
                    "type": "missing",
                }
            ]
        },
    )

    assert extract_error_message(response) == "body.file: Field required"
