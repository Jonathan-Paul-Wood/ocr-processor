"""Bounded, content-validated CSV input (filenames are never trusted)."""
import csv
from io import StringIO


def csv_rows(payload):
    try:
        encoding = 'utf-16' if payload.startswith((b'\xff\xfe', b'\xfe\xff')) else 'utf-8-sig'
        text = payload.decode(encoding)
        if any(ord(char) < 32 and char not in '\r\n\t' for char in text):
            return None
        dialect = csv.Sniffer().sniff(text[:8192], delimiters=',;\t')
        rows = [row for row in csv.reader(StringIO(text), dialect, strict=True) if any(cell.strip() for cell in row)]
        if len(rows) < 2 or len(rows[0]) < 2 or len(set(rows[0])) != len(rows[0]):
            return None
        if any(len(row) != len(rows[0]) for row in rows):
            return None
        return rows
    except (UnicodeError, csv.Error):
        return None
