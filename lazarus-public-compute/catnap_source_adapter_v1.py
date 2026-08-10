from __future__ import annotations

import csv
import hashlib
import json
import math
import re
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
from scipy import sparse
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import HashingVectorizer

TRAINING_RELEASE = "2026-07-01"
TARGET_RELEASE = "2026-09-01"
EXPECTED_TRAINING_HASHES = {
    "assay_2026-07-01.txt": "e6b575e0785b43c5dbd84b7ef3eb08e2d072fa32117d1ad0bdd31eb5e243bdd2",
    "abs_2026-07-01.txt": "fe9c01fe781e479a429cb27dfab68466fec0645e400032defdf8db33eafe772c",
    "heavy_seqs_aa_2026-07-01.fasta": "f768e68f792ad9760041a07838e521284790a323adb181a76ecbc7098a485863",
    "light_seqs_aa_2026-07-01.fasta": "964b5c8299f51bfdfe53814122692cf6eb88b000af0cdfec809ce6d60873fcc6",
    "virseqs_aa_O_2026-07-01.fasta": "9c83b868d7159ad6148f628878027ccbaf2e07131baf4efd3221f1aa8ec568cf",
}
EXPECTED_FAMILY_SHA256 = "6e7080e2d22e66322ea0884515be9ae5c6762eeb24acc16005ca45d68ecba788"
EXPECTED_TRAINING_PAIRS = 63112
EXPECTED_TRAINING_ANTIBODIES = 575
EXPECTED_TRAINING_VIRUSES = 1507
EXPECTED_TRAINING_FAMILIES = 228
EXPECTED_RESOLVED_ANTIBODIES = 576
RAW_COMPONENTS = 24
SELECTED_COMPONENTS = 48
HASH_FEATURES = 4096
POSITION_HASH_FEATURES = 16384
SVD_RANDOM_STATE = 626
RAW_WIDTH = 624
SELECTED_WIDTH = 2400

CENSORED = re.compile(r"^\s*(?P<op>>=|<=|>|<|GT|LT)?\s*(?P<value>(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)\s*$", re.I)
MISSING = {"", "NA", "N/A", "NT", "ND", "NONE", "NULL", "-"}
ACCESSION_BOUNDARY = re.compile(r"^(.+?)_[A-Z]{1,5}\d{5,}(?:\.\d+)?_")
IMMDB = re.compile(r"ImmDBID\s*([0-9]+)", re.I)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def clean(value: str | None) -> str:
    return " ".join(unicodedata.normalize("NFKC", value or "").replace("\ufeff", "").strip().split())


def norm(value: str) -> str:
    return "".join(ch for ch in clean(value).casefold() if ch.isalnum())


def read_table(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    text = path.read_text(encoding="utf-8-sig", errors="replace")
    if "<html" in text[:500].casefold() or "<!doctype" in text[:500].casefold():
        raise ValueError(f"HTML payload: {path}")
    lines = [line for line in text.splitlines() if line.strip()]
    delimiter = "\t" if lines and "\t" in lines[0] else ","
    reader = csv.DictReader(lines, delimiter=delimiter)
    fields = [clean(field) for field in (reader.fieldnames or [])]
    rows = [{clean(key): clean(value) for key, value in row.items() if key is not None} for row in reader]
    return fields, rows


def find_field(fields: Sequence[str], exact: Sequence[str], contains: Sequence[str] = ()) -> str | None:
    by_norm = {norm(field): field for field in fields}
    for candidate in exact:
        if norm(candidate) in by_norm:
            return by_norm[norm(candidate)]
    for field in fields:
        key = norm(field)
        if any(norm(candidate) in key for candidate in contains):
            return field
    return None


def parse_val(raw: str | None) -> float | None:
    text = "" if raw is None else str(raw).strip()
    if text.upper() in MISSING:
        return None
    match = CENSORED.match(text)
    if not match:
        return None
    value = float(match.group("value"))
    if value <= 0:
        return None
    operator = (match.group("op") or "").upper()
    multiplier = 0.5 if operator in ("<", "<=", "LT") else 2.0 if operator in (">", ">=", "GT") else 1.0
    return -math.log(value * multiplier)


def split_aliases(value: str) -> list[str]:
    return [clean(piece) for piece in re.split(r"[|;,]", clean(value)) if clean(piece)]


@dataclass(frozen=True)
class Meta:
    name: str
    aliases: tuple[str, ...]
    ids: tuple[str, ...]


def load_meta(path: Path) -> dict[str, list[Meta]]:
    fields, rows = read_table(path)
    name_field = find_field(fields, ("Name", "Antibody", "Antibody name"))
    alias_field = find_field(fields, ("Alias", "Aliases", "Synonym"), ("alias", "synonym"))
    immdb_field = find_field(fields, ("Immuno DB ID", "ImmDB ID"), ("immunodbid", "immdbid"))
    if name_field is None:
        raise ValueError("CATNAP antibody metadata name field missing")
    index: defaultdict[str, list[Meta]] = defaultdict(list)
    for row in rows:
        name = clean(row.get(name_field, ""))
        if not name:
            continue
        aliases = tuple(dict.fromkeys(split_aliases(row.get(alias_field, "")) if alias_field else []))
        ids = tuple(dict.fromkeys("ImmDBID" + item for item in re.findall(r"\d+", row.get(immdb_field, "")))) if immdb_field else ()
        record = Meta(name, aliases, ids)
        for identifier in (name, *aliases):
            key = norm(identifier)
            if key:
                index[key].append(record)
    return dict(index)


@dataclass(frozen=True)
class FastaRecord:
    record_id: str
    header: str
    sequence: str
    identifiers: tuple[str, ...]


def header_identifiers(header: str) -> tuple[str, ...]:
    header = clean(header)
    identifiers: list[str] = []
    if "__" in header:
        identifiers.append(header.split("__", 1)[0])
    accession = ACCESSION_BOUNDARY.match(header)
    if accession:
        identifiers.append(accession.group(1))
    if "_ImmDBID" in header:
        identifiers.append(header.split("_ImmDBID", 1)[0].split("__", 1)[0])
    identifiers.extend("ImmDBID" + item for item in IMMDB.findall(header))
    if "_" in header and any(marker in header.casefold() for marker in ("heavy_chain", "light_chain", "chain_heavy", "chain_light")):
        identifiers.append(header.split("_", 1)[0])
    output: list[str] = []
    seen: set[str] = set()
    for identifier in identifiers:
        key = norm(identifier)
        if len(key) >= 3 and key not in seen:
            seen.add(key)
            output.append(clean(identifier))
    return tuple(output)


def fasta_records(path: Path) -> list[FastaRecord]:
    records: list[FastaRecord] = []
    header: str | None = None
    parts: list[str] = []

    def finish() -> None:
        nonlocal header, parts
        if header is None:
            return
        sequence = "".join(parts).replace(" ", "").upper()
        if not sequence:
            raise ValueError(f"empty FASTA sequence: {header}")
        records.append(FastaRecord(f"r{len(records)+1}", header, sequence, header_identifiers(header)))

    for raw in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw.strip()
        if not line:
            continue
        if line.startswith(">"):
            finish()
            header = line[1:].strip()
            parts = []
        else:
            if header is None:
                raise ValueError("FASTA sequence before header")
            parts.append(line)
    finish()
    if not records:
        raise ValueError(f"no FASTA records: {path}")
    return records


class Resolver:
    def __init__(self, records: Sequence[FastaRecord]) -> None:
        self.records = {record.record_id: record for record in records}
        index: defaultdict[str, set[str]] = defaultdict(set)
        for record in records:
            for identifier in record.identifiers:
                index[norm(identifier)].add(record.record_id)
        self.index = dict(index)

    def resolve(self, name: str, metadata: Sequence[Meta]) -> FastaRecord | None:
        tiers: list[list[str]] = []
        immdb = [identifier for record in metadata for identifier in record.ids]
        if immdb:
            tiers.append(immdb)
        declared = [name]
        for record in metadata:
            declared.extend((record.name, *record.aliases))
        tiers.append(declared)
        for identifiers in tiers:
            found: set[str] = set()
            for identifier in identifiers:
                found.update(self.index.get(norm(identifier), ()))
            if len(found) == 1:
                return self.records[next(iter(found))]
            if found:
                return None
        return None


def load_family_map(path: Path) -> tuple[dict[str, str], dict[str, list[str]]]:
    if sha256_file(path) != EXPECTED_FAMILY_SHA256:
        raise ValueError("primary family graph SHA-256 drifted")
    by_antibody: dict[str, str] = {}
    families: dict[str, list[str]] = {}
    with path.open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            family = row["family_id"].strip()
            members = [item.strip() for item in row["members"].split("|") if item.strip()]
            if len(members) != int(row["size"]):
                raise ValueError("family size drift")
            families[family] = members
            for antibody in members:
                if antibody in by_antibody:
                    raise ValueError("antibody occurs in multiple families")
                by_antibody[antibody] = family
    if len(families) != 229 or len(by_antibody) != 576:
        raise ValueError("frozen primary family graph dimensions drifted")
    return by_antibody, families


def resolve_antibodies(training_data: Path, family_by: dict[str, str]) -> dict[str, str]:
    metadata = load_meta(training_data / "abs_2026-07-01.txt")
    heavy = Resolver(fasta_records(training_data / "heavy_seqs_aa_2026-07-01.fasta"))
    light = Resolver(fasta_records(training_data / "light_seqs_aa_2026-07-01.fasta"))
    output: dict[str, str] = {}
    for antibody in sorted(family_by):
        records = metadata.get(norm(antibody), [])
        h = heavy.resolve(antibody, records)
        l = light.resolve(antibody, records)
        if h is None or l is None:
            raise ValueError(f"unresolved frozen antibody: {antibody}")
        output[antibody] = h.sequence + "|" + l.sequence
    if len(output) != EXPECTED_RESOLVED_ANTIBODIES:
        raise ValueError("resolved antibody count drifted")
    return output


def virus_sequences(path: Path) -> tuple[dict[str, str], set[str]]:
    output: dict[str, str] = {}
    valid: set[str] = set()
    header: str | None = None
    parts: list[str] = []

    def finish() -> None:
        nonlocal header, parts
        if header is None:
            return
        fields = header.split(".")
        key = fields[-2].strip() if len(fields) >= 2 else ""
        if key:
            if key in output:
                raise ValueError(f"duplicate virus FASTA key: {key}")
            sequence = "".join(parts).replace(" ", "").upper()
            if not sequence:
                raise ValueError(f"empty virus sequence: {key}")
            output[key] = sequence
            if "HXB2" not in header.upper():
                valid.add(key)

    for raw in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw.strip()
        if not line:
            continue
        if line.startswith(">"):
            finish()
            header = line[1:].strip()
            parts = []
        else:
            if header is None:
                raise ValueError("virus sequence before header")
            parts.append(line)
    finish()
    if not output:
        raise ValueError("empty virus FASTA")
    return output, valid


def generic_matrix(assay: Path, fasta: Path, *, minimum_virus_edges: int) -> tuple[dict[tuple[str, str], tuple[float, int]], dict[str, str]]:
    if minimum_virus_edges not in (1, 10):
        raise ValueError("unexpected virus-edge minimum")
    sequences, valid = virus_sequences(fasta)
    fields, rows = read_table(assay)
    antibody_field = find_field(fields, ("Antibody",))
    virus_field = find_field(fields, ("Virus",))
    ic50_field = find_field(fields, ("IC50",))
    if not antibody_field or not virus_field or not ic50_field:
        raise ValueError("required CATNAP fields missing")
    antibody_order: list[str] = []
    virus_order: list[str] = []
    seen_antibodies: set[str] = set()
    seen_viruses: set[str] = set()
    values: defaultdict[tuple[str, str], list[float]] = defaultdict(list)
    for row in rows:
        antibody = clean(row.get(antibody_field, ""))
        virus = clean(row.get(virus_field, ""))
        if antibody not in seen_antibodies:
            seen_antibodies.add(antibody)
            antibody_order.append(antibody)
        if virus not in seen_viruses:
            seen_viruses.add(virus)
            virus_order.append(virus)
        if "polyclonal" in antibody.casefold() or "/" in antibody or "+" in antibody or virus not in valid:
            continue
        target = parse_val(row.get(ic50_field, ""))
        if target is not None:
            values[(antibody, virus)].append(float(target))
    eligible_antibodies = [a for a in antibody_order if "polyclonal" not in a.casefold() and "/" not in a and "+" not in a]
    eligible_viruses = [v for v in virus_order if v in valid]
    for _ in range(20):
        antibody_set = set(eligible_antibodies)
        virus_set = set(eligible_viruses)
        antibody_counts: Counter[str] = Counter()
        virus_counts: Counter[str] = Counter()
        for antibody, virus in values:
            if antibody in antibody_set and virus in virus_set:
                antibody_counts[antibody] += 1
                virus_counts[virus] += 1
        next_antibodies = [a for a in eligible_antibodies if antibody_counts[a] >= 1]
        next_viruses = [v for v in eligible_viruses if virus_counts[v] >= minimum_virus_edges]
        if next_antibodies == eligible_antibodies and next_viruses == eligible_viruses:
            break
        eligible_antibodies, eligible_viruses = next_antibodies, next_viruses
    antibody_set = set(eligible_antibodies)
    virus_set = set(eligible_viruses)
    matrix = {
        (antibody, virus): (float(sum(items) / len(items)), len(items))
        for (antibody, virus), items in values.items()
        if antibody in antibody_set and virus in virus_set
    }
    return matrix, sequences


@dataclass(frozen=True)
class Pair:
    antibody: str
    family_id: str
    virus: str
    target: float


@dataclass(frozen=True)
class AdapterDesignSet:
    row_id: np.ndarray
    antibody_id: np.ndarray
    antibody_family: np.ndarray
    virus_id: np.ndarray
    virus_cluster: np.ndarray
    truth: np.ndarray | None
    x_raw: np.ndarray
    x_residual: np.ndarray
    x_router: np.ndarray
    antibody_embedding: np.ndarray
    virus_embedding: np.ndarray

    def validate(self, require_truth: bool) -> None:
        arrays = (self.row_id, self.antibody_id, self.antibody_family, self.virus_id, self.virus_cluster,
                  self.x_raw, self.x_residual, self.x_router, self.antibody_embedding, self.virus_embedding)
        n = len(self.row_id)
        if n == 0 or any(len(value) != n for value in arrays):
            raise ValueError("design arrays must be nonempty and row-aligned")
        if require_truth and self.truth is None:
            raise ValueError("training truth is required")
        if self.truth is not None and len(self.truth) != n:
            raise ValueError("truth is not row-aligned")
        if len(set(map(str, self.row_id))) != n:
            raise ValueError("row IDs must be unique")
        numeric = [self.x_raw, self.x_residual, self.x_router, self.antibody_embedding, self.virus_embedding]
        if self.truth is not None:
            numeric.append(self.truth)
        if not all(np.all(np.isfinite(np.asarray(value, dtype=float))) for value in numeric):
            raise ValueError("non-finite design input")
        if len(np.unique(self.virus_cluster.astype(str))) < 5:
            raise ValueError("fewer than five virus clusters")


class SequenceRepresentation:
    def __init__(self, *, kind: str, components: int, selected: bool) -> None:
        if kind not in ("antibody", "virus"):
            raise ValueError("invalid sequence kind")
        self.kind = kind
        self.components = components
        self.selected = selected
        self.svd: TruncatedSVD | None = None

    def _char_matrix(self, sequences: dict[str, str], ids: list[str]):
        vectorizer = HashingVectorizer(analyzer="char", ngram_range=(2, 3), n_features=HASH_FEATURES,
                                       alternate_sign=False, norm="l2", lowercase=False)
        return vectorizer.transform([sequences[item] for item in ids]).tocsr()

    def _position_matrix(self, sequences: dict[str, str], ids: list[str]):
        if self.kind == "antibody":
            texts = [" ".join([*(f"H{i:04d}_{c}" for i, c in enumerate(sequences[item].split("|")[0])),
                               *(f"L{i:04d}_{c}" for i, c in enumerate(sequences[item].split("|")[1]))]) for item in ids]
        else:
            texts = [" ".join(f"p{i:04d}_{c}" for i, c in enumerate(sequences[item])) for item in ids]
        vectorizer = HashingVectorizer(analyzer="word", tokenizer=str.split, preprocessor=None, token_pattern=None,
                                       n_features=POSITION_HASH_FEATURES, alternate_sign=False, norm="l2", lowercase=False)
        return vectorizer.transform(texts).tocsr()

    def _matrix(self, sequences: dict[str, str], ids: list[str]):
        char = self._char_matrix(sequences, ids)
        if not self.selected:
            return char
        return sparse.hstack((char, self._position_matrix(sequences, ids)), format="csr")

    def fit_transform(self, sequences: dict[str, str], ids: list[str]) -> np.ndarray:
        if len(ids) < self.components + 1:
            raise ValueError("too few training entities for frozen SVD width")
        matrix = self._matrix(sequences, ids)
        components = self.components if self.selected else min(self.components, matrix.shape[0] - 1, matrix.shape[1] - 1)
        self.svd = TruncatedSVD(n_components=components, random_state=SVD_RANDOM_STATE)
        values = np.asarray(self.svd.fit_transform(matrix))
        return self._normalize(values)

    def transform(self, sequences: dict[str, str], ids: list[str]) -> np.ndarray:
        if self.svd is None:
            raise RuntimeError("sequence representation has not been fit")
        if not ids:
            return np.empty((0, self.components), dtype=np.float32)
        values = np.asarray(self.svd.transform(self._matrix(sequences, ids)))
        return self._normalize(values)

    def _normalize(self, values: np.ndarray) -> np.ndarray:
        if not self.selected:
            return values.astype(np.float32, copy=False)
        norms = np.linalg.norm(values, axis=1, keepdims=True)
        return (values / np.where(norms == 0, 1.0, norms)).astype(np.float32, copy=False)


def build_joint(rows: list[Pair], antibody: dict[str, np.ndarray], virus: dict[str, np.ndarray]) -> np.ndarray:
    a = np.asarray([antibody[row.antibody] for row in rows], dtype=np.float32)
    v = np.asarray([virus[row.virus] for row in rows], dtype=np.float32)
    a = a / np.linalg.norm(a, axis=1, keepdims=True)
    v = v / np.linalg.norm(v, axis=1, keepdims=True)
    output = np.column_stack((a, v, np.einsum("ij,ik->ijk", a, v, optimize=True).reshape(len(rows), a.shape[1] * v.shape[1]))).astype(np.float32)
    return output


def build_raw(rows: list[Pair], antibody: dict[str, np.ndarray], virus: dict[str, np.ndarray]) -> np.ndarray:
    a = np.asarray([antibody[row.antibody] for row in rows], dtype=np.float32)
    v = np.asarray([virus[row.virus] for row in rows], dtype=np.float32)
    return np.column_stack((a, v, np.einsum("ij,ik->ijk", a, v, optimize=True).reshape(len(rows), a.shape[1] * v.shape[1]))).astype(np.float32)


def row_ids(rows: list[Pair]) -> np.ndarray:
    return np.asarray([json.dumps([row.antibody, row.virus], ensure_ascii=False, separators=(",", ":")) for row in rows], dtype=object)


def virus_cluster_id(sequence: str) -> str:
    # Frozen implementation hardening for the contract phrase "virus sequence cluster":
    # exact normalized Env-sequence identity, never labels or target statistics.
    return "exact-env-sha256:" + hashlib.sha256(sequence.upper().encode("ascii", errors="strict")).hexdigest()


class CatnapSourceAdapterV1:
    def __init__(self) -> None:
        self.training_keys: set[tuple[str, str]] | None = None
        self.training_antibody_sequences: dict[str, str] | None = None
        self.training_virus_sequences: dict[str, str] | None = None
        self.family_by: dict[str, str] | None = None
        self.raw_antibody = SequenceRepresentation(kind="antibody", components=RAW_COMPONENTS, selected=False)
        self.raw_virus = SequenceRepresentation(kind="virus", components=RAW_COMPONENTS, selected=False)
        self.selected_antibody = SequenceRepresentation(kind="antibody", components=SELECTED_COMPONENTS, selected=True)
        self.selected_virus = SequenceRepresentation(kind="virus", components=SELECTED_COMPONENTS, selected=True)

    @property
    def adapter_sha256(self) -> str:
        return sha256_file(Path(__file__))

    def _verify_training_files(self, training_data: Path) -> None:
        for name, expected in EXPECTED_TRAINING_HASHES.items():
            path = training_data / name
            if not path.is_file():
                raise FileNotFoundError(path)
            actual = sha256_file(path)
            if actual != expected:
                raise ValueError(f"training hash mismatch {name}: {actual}")

    def _design(self, rows: list[Pair], antibody_sequences: dict[str, str], virus_sequences_map: dict[str, str], *, fit: bool) -> AdapterDesignSet:
        antibodies = sorted({row.antibody for row in rows})
        viruses = sorted({row.virus for row in rows})
        if fit:
            raw_a_values = self.raw_antibody.fit_transform(antibody_sequences, antibodies)
            raw_v_values = self.raw_virus.fit_transform(virus_sequences_map, viruses)
            selected_a_values = self.selected_antibody.fit_transform(antibody_sequences, antibodies)
            selected_v_values = self.selected_virus.fit_transform(virus_sequences_map, viruses)
        else:
            raw_a_values = self.raw_antibody.transform(antibody_sequences, antibodies)
            raw_v_values = self.raw_virus.transform(virus_sequences_map, viruses)
            selected_a_values = self.selected_antibody.transform(antibody_sequences, antibodies)
            selected_v_values = self.selected_virus.transform(virus_sequences_map, viruses)
        raw_a = {name: raw_a_values[index] for index, name in enumerate(antibodies)}
        raw_v = {name: raw_v_values[index] for index, name in enumerate(viruses)}
        selected_a = {name: selected_a_values[index] for index, name in enumerate(antibodies)}
        selected_v = {name: selected_v_values[index] for index, name in enumerate(viruses)}
        x_raw = build_raw(rows, raw_a, raw_v)
        x_residual = build_joint(rows, selected_a, selected_v)
        if x_raw.shape[1] != RAW_WIDTH or x_residual.shape[1] != SELECTED_WIDTH:
            raise RuntimeError(f"feature width drift: raw={x_raw.shape}, residual={x_residual.shape}")
        antibody_embedding = np.asarray([selected_a[row.antibody] for row in rows], dtype=np.float32)
        virus_embedding = np.asarray([selected_v[row.virus] for row in rows], dtype=np.float32)
        clusters = np.asarray([virus_cluster_id(virus_sequences_map[row.virus]) for row in rows], dtype=object)
        design = AdapterDesignSet(
            row_id=row_ids(rows),
            antibody_id=np.asarray([row.antibody for row in rows], dtype=object),
            antibody_family=np.asarray([row.family_id for row in rows], dtype=object),
            virus_id=np.asarray([row.virus for row in rows], dtype=object),
            virus_cluster=clusters,
            truth=np.asarray([row.target for row in rows], dtype=np.float64),
            x_raw=x_raw,
            x_residual=x_residual,
            x_router=x_raw,
            antibody_embedding=antibody_embedding,
            virus_embedding=virus_embedding,
        )
        design.validate(True)
        return design

    def build_training_design(self, training_data: Path, families: Path) -> AdapterDesignSet:
        self._verify_training_files(training_data)
        family_by, _ = load_family_map(families)
        antibody_sequences = resolve_antibodies(training_data, family_by)
        matrix, virus_map = generic_matrix(training_data / "assay_2026-07-01.txt", training_data / "virseqs_aa_O_2026-07-01.fasta", minimum_virus_edges=10)
        rows = [Pair(antibody, family_by[antibody], virus, target)
                for (antibody, virus), (target, _) in sorted(matrix.items())
                if antibody in family_by and antibody in antibody_sequences and virus in virus_map]
        observed = (len(rows), len({row.antibody for row in rows}), len({row.virus for row in rows}), len({row.family_id for row in rows}))
        expected = (EXPECTED_TRAINING_PAIRS, EXPECTED_TRAINING_ANTIBODIES, EXPECTED_TRAINING_VIRUSES, EXPECTED_TRAINING_FAMILIES)
        if observed != expected:
            raise ValueError(f"July training universe drift: observed={observed} expected={expected}")
        self.training_keys = {(row.antibody, row.virus) for row in rows}
        self.training_antibody_sequences = antibody_sequences
        self.training_virus_sequences = virus_map
        self.family_by = family_by
        return self._design(rows, antibody_sequences, virus_map, fit=True)

    def build_target_design(self, target_data: Path, families: Path, training_design: AdapterDesignSet) -> AdapterDesignSet:
        if self.training_keys is None or self.training_antibody_sequences is None or self.training_virus_sequences is None or self.family_by is None:
            raise RuntimeError("training design must be built before target design")
        family_by, _ = load_family_map(families)
        if family_by != self.family_by:
            raise ValueError("family graph changed between training and target")
        assay = target_data / "assay_2026-09-01.txt"
        fasta = target_data / "virseqs_aa_O_2026-09-01.fasta"
        if not assay.is_file() or not fasta.is_file():
            raise FileNotFoundError("exact September target members are required")
        matrix, target_viruses = generic_matrix(assay, fasta, minimum_virus_edges=1)
        combined_viruses = dict(target_viruses)
        combined_viruses.update(self.training_virus_sequences)
        rows = [Pair(antibody, family_by[antibody], virus, target)
                for (antibody, virus), (target, _) in sorted(matrix.items())
                if antibody in family_by
                and antibody in self.training_antibody_sequences
                and virus in combined_viruses
                and (antibody, virus) not in self.training_keys]
        if not rows:
            raise ValueError("empty new-pair target after frozen eligibility rules")
        if any((row.antibody, row.virus) in self.training_keys for row in rows):
            raise AssertionError("target newness rule failed")
        design = self._design(rows, self.training_antibody_sequences, combined_viruses, fit=False)
        training_keys_from_design = set(zip(training_design.antibody_id.astype(str), training_design.virus_id.astype(str), strict=True))
        target_keys = set(zip(design.antibody_id.astype(str), design.virus_id.astype(str), strict=True))
        if training_keys_from_design & target_keys:
            raise AssertionError("training-target pair overlap")
        return design


ADAPTER = CatnapSourceAdapterV1()
