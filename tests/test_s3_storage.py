from datetime import UTC, datetime

from app.s3_storage import S3LeaseStorage


class FakePaginator:
    def paginate(self, Bucket: str, Prefix: str):
        assert Bucket == "test-bucket"
        assert Prefix == "sample_leases/"
        return [
            {
                "Contents": [
                    {
                        "Key": "sample_leases/valid_lease_a.txt",
                        "Size": 100,
                        "LastModified": datetime(2026, 1, 1, tzinfo=UTC),
                    },
                    {
                        "Key": "sample_leases/valid_lease_b.pdf",
                        "Size": 200,
                        "LastModified": datetime(2026, 1, 2, tzinfo=UTC),
                    },
                    {
                        "Key": "sample_leases/valid_lease_c.docx",
                        "Size": 300,
                        "LastModified": datetime(2026, 1, 3, tzinfo=UTC),
                    },
                    {
                        "Key": "sample_leases/notes.rtf",
                        "Size": 400,
                        "LastModified": datetime(2026, 1, 4, tzinfo=UTC),
                    },
                    {
                        "Key": "sample_leases/folder/",
                        "Size": 0,
                    },
                ]
            }
        ]


class FakeS3Client:
    def get_paginator(self, name: str):
        assert name == "list_objects_v2"
        return FakePaginator()


def test_s3_storage_lists_supported_lease_files_only():
    storage = S3LeaseStorage(
        bucket="test-bucket",
        prefix="sample_leases",
        client=FakeS3Client(),
    )

    leases = storage.list_lease_files()

    assert [lease.key for lease in leases] == [
        "sample_leases/valid_lease_a.txt",
        "sample_leases/valid_lease_b.pdf",
        "sample_leases/valid_lease_c.docx",
    ]
    assert [lease.filename for lease in leases] == [
        "valid_lease_a.txt",
        "valid_lease_b.pdf",
        "valid_lease_c.docx",
    ]
