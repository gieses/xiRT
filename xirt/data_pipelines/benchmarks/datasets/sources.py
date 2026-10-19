"""Inventory and selectively retrieve the study's linked Zenodo archives.

HTTP ranges avoid downloading gigabytes of spectra/results merely to identify
model-ready peptide/RT tables. Archive member CRCs are checked by ZipFile; the
published whole-archive MD5 is recorded but cannot verify partial downloads.

Example:
    >>> reader = RangeReader("https://example.org/data.zip", 1000)  # doctest: +SKIP
"""

import argparse
import hashlib
import io
import json
import shutil
import urllib.request
import zipfile
from pathlib import Path, PurePosixPath
from typing import Any


class RangeReader(io.RawIOBase):
    """Provide a seekable read-only HTTP-range stream to Python's ZIP reader."""

    def __init__(self, url: str, size: int) -> None:
        super().__init__()
        self.url, self.size, self.position = url, size, 0
        self.buffer_start, self.buffer = 0, b""

    def seekable(self) -> bool:
        """Report that archive offsets can be read in arbitrary order."""
        return True

    def readable(self) -> bool:
        """Report that the remote archive is readable, never writable."""
        return True

    def tell(self) -> int:
        """Return the current virtual archive offset."""
        return self.position

    def seek(self, offset: int, whence: int = io.SEEK_SET) -> int:
        """Set the virtual offset with standard seek semantics."""
        self.position = (
            offset + {io.SEEK_SET: 0, io.SEEK_CUR: self.position, io.SEEK_END: self.size}[whence]
        )
        if self.position < 0:
            raise ValueError("Cannot seek before the archive.")
        return self.position

    def read(self, size: int = -1) -> bytes:
        """Retrieve exactly the requested range and reject servers ignoring Range."""
        length = self.size - self.position if size < 0 else min(size, self.size - self.position)
        if length <= 0:
            return b""
        requested = self.position
        if self.buffer_start <= requested and requested + length <= self.buffer_start + len(
            self.buffer
        ):
            self.position += length
            return self.buffer[
                requested - self.buffer_start : requested - self.buffer_start + length
            ]
        start = min(requested, max(0, self.size - 4 * 1024 * 1024))
        count = min(max(length, 4 * 1024 * 1024), self.size - start)
        end = start + count - 1
        request = urllib.request.Request(
            self.url,
            headers={
                "Range": f"bytes={start}-{end}",
                "Accept-Encoding": "identity",
            },
        )
        with urllib.request.urlopen(request, timeout=120) as response:
            if response.status != 206 or response.headers.get("Content-Range") != (
                f"bytes {start}-{end}/{self.size}"
            ):
                raise OSError("Server did not honor the requested archive range.")
            data = response.read(count + 1)
        if len(data) != count:
            raise OSError("Incomplete archive range response.")
        self.buffer_start, self.buffer = start, data
        self.position += length
        return data[requested - start : requested - start + length]


def metadata(record: int) -> dict[str, Any]:
    """Fetch the published record and preserve its canonical version ID."""
    with urllib.request.urlopen(f"https://zenodo.org/api/records/{record}", timeout=60) as stream:
        return json.load(stream)


def main() -> None:
    """List linked archive members or extract one explicitly named, safe member."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--record", type=int, default=4720760)
    parser.add_argument(
        "--output", type=Path, default=Path("benchmarks/results/sources-inventory.json")
    )
    parser.add_argument("--cache", type=Path, default=Path("benchmarks/.cache/sources"))
    parser.add_argument("--extract", nargs=2, metavar=("ARCHIVE", "MEMBER"))
    args = parser.parse_args()
    record = metadata(args.record)
    files = record["files"]
    if args.extract:
        archive, member = args.extract
        item = next(file for file in files if file["key"] == archive)
        name = PurePosixPath(member)
        if name.is_absolute() or ".." in name.parts:
            raise ValueError("Unsafe archive member path.")
        destination = args.cache / Path(archive).stem / str(name)
        destination.parent.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(RangeReader(item["links"]["self"], item["size"])) as stream:
            with stream.open(member) as source, destination.open("wb") as target:
                shutil.copyfileobj(source, target)
        digest = hashlib.sha256(destination.read_bytes()).hexdigest()
        print(
            json.dumps(
                {
                    "path": str(destination),
                    "sha256": digest,
                    "record": record["id"],
                    "archive": archive,
                    "member": member,
                }
            )
        )
        return
    result = {"record": record["id"], "doi": record.get("doi"), "files": []}
    for item in files:
        entry = {
            "name": item["key"],
            "size": item["size"],
            "checksum": item["checksum"],
            "url": item["links"]["self"],
            "members": [],
        }
        if item["key"].endswith(".zip"):
            with zipfile.ZipFile(RangeReader(item["links"]["self"], item["size"])) as stream:
                entry["members"] = [
                    {
                        "name": info.filename,
                        "size": info.file_size,
                        "compressed_size": info.compress_size,
                        "crc32": info.CRC,
                    }
                    for info in stream.infolist()
                ]
            print(item["key"], len(entry["members"]), "members", flush=True)
        result["files"].append(entry)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
