import tempfile
import unittest
from pathlib import Path

from PIL import Image

from diffsynth_engine.generate_kwargs import resolve_generate_kwargs


class TestResolveGenerateKwargs(unittest.TestCase):
    def test_resolve_image_names_to_single_image(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "a.png"
            Image.new("RGB", (4, 4), color="red").save(path)

            resolved = resolve_generate_kwargs(
                {
                    "dataset_dir": tmp,
                    "image_names": ["a.png"],
                    "prompt": "test",
                }
            )

            self.assertNotIn("image_names", resolved)
            self.assertNotIn("dataset_dir", resolved)
            self.assertIsInstance(resolved["image"], Image.Image)
            self.assertEqual(resolved["prompt"], "test")

    def test_resolve_image_names_to_list(self):
        with tempfile.TemporaryDirectory() as tmp:
            for name in ("a.png", "b.png"):
                Image.new("RGB", (4, 4)).save(Path(tmp) / name)

            resolved = resolve_generate_kwargs(
                {
                    "dataset_dir": tmp,
                    "image_names": ["a.png", "b.png"],
                    "prompt": "test",
                }
            )

            self.assertIsInstance(resolved["image"], list)
            self.assertEqual(len(resolved["image"]), 2)

    def test_passthrough_without_image_names(self):
        kwargs = {"prompt": "txt2img"}
        self.assertIs(resolve_generate_kwargs(kwargs), kwargs)

    def test_missing_dataset_dir_raises(self):
        with self.assertRaises(ValueError):
            resolve_generate_kwargs({"image_names": ["a.png"], "prompt": "test"})


if __name__ == "__main__":
    unittest.main()
