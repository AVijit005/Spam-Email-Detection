from __future__ import annotations

import pickle
import tempfile
import unittest
from pathlib import Path

from app.ml.registry import (
    ModelIntegrityError,
    _compute_hash,
    _hash_path,
    load_model,
    save_model,
)


class DummyModel:
    pass


class DummyVectorizer:
    pass


class TestRegistry(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.tmp_path = Path(self.tmp.name)
        self.model_path = self.tmp_path / "model.pkl"
        self.vec_path = self.tmp_path / "vectorizer.pkl"
        self.meta_path = self.tmp_path / "metadata.json"

    def tearDown(self):
        self.tmp.cleanup()

    def _write_artifact(self):
        save_model(
            DummyModel(), DummyVectorizer(), {"version": "1.0"},
            self.model_path, self.vec_path, self.meta_path,
        )

    def test_load_model_verifies_sha256_and_rejects_tampered_file(self):
        self._write_artifact()

        model_hash_path = _hash_path(self.model_path)
        original_hash = model_hash_path.read_text().strip()
        model_hash_path.write_text("0" * 64)

        with self.assertRaises(ModelIntegrityError) as ctx:
            load_model(self.model_path, self.vec_path, self.meta_path)
        self.assertIn("integrity check failed", str(ctx.exception))

    def test_load_model_returns_none_when_model_file_missing(self):
        result = load_model(self.model_path, self.vec_path, self.meta_path)
        self.assertIsNone(result)

    def test_load_model_returns_none_when_vectorizer_file_missing(self):
        self.model_path.write_bytes(b"\x80\x04\x95\x06\x00\x00\x00\x00\x00\x00\x00.")
        result = load_model(self.model_path, self.vec_path, self.meta_path)
        self.assertIsNone(result)

    def test_save_model_creates_sha256_sidecar_files(self):
        self._write_artifact()

        model_hash_path = _hash_path(self.model_path)
        vec_hash_path = _hash_path(self.vec_path)
        self.assertTrue(model_hash_path.exists(), "model .sha256 missing")
        self.assertTrue(vec_hash_path.exists(), "vectorizer .sha256 missing")

        expected_model = _compute_hash(self.model_path)
        expected_vec = _compute_hash(self.vec_path)
        self.assertEqual(model_hash_path.read_text().strip(), expected_model)
        self.assertEqual(vec_hash_path.read_text().strip(), expected_vec)

    def test_load_model_succeeds_with_matching_hash(self):
        self._write_artifact()

        artifact = load_model(self.model_path, self.vec_path, self.meta_path)
        self.assertIsNotNone(artifact)
        self.assertIsInstance(artifact.model, DummyModel)
        self.assertIsInstance(artifact.vectorizer, DummyVectorizer)
        self.assertEqual(artifact.metadata, {"version": "1.0"})

    def test_load_model_succeeds_when_sidecar_absent(self):
        save_model(
            DummyModel(), DummyVectorizer(), {"version": "1.0"},
            self.model_path, self.vec_path, self.meta_path,
        )
        _hash_path(self.model_path).unlink()
        _hash_path(self.vec_path).unlink()

        artifact = load_model(self.model_path, self.vec_path, self.meta_path)
        self.assertIsNotNone(artifact)

    def test_save_model_creates_parent_directory(self):
        deep_path = self.tmp_path / "nested" / "dir"
        model_p = deep_path / "model.pkl"
        vec_p = deep_path / "vectorizer.pkl"
        meta_p = deep_path / "metadata.json"

        save_model(DummyModel(), DummyVectorizer(), {}, model_p, vec_p, meta_p)
        self.assertTrue(model_p.exists())
        self.assertTrue(vec_p.exists())
