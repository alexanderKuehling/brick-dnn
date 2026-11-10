"""Segmentation datasets: the layouts and mask formats that are read, what is refused, the
ZIP extraction and the built-in Shapes dataset (needs Pillow and NumPy, not PyTorch)."""

from __future__ import annotations

import json
import zipfile

import pytest

np = pytest.importorskip("numpy")
pytest.importorskip("PIL")

from brickdnn.training import (  # noqa: E402
    BUILTIN_DATASETS,
    DatasetError,
    DatasetInfo,
    builtin_info,
    extract_image_zip,
    inspect_image_folder,
    inspect_segmentation_folder,
    is_segmentation_folder,
)
from brickdnn.training.datasets import SHAPES_MEAN, SHAPES_PIXEL_SHARE, SHAPES_STD  # noqa: E402
from brickdnn.training.segmentation import (  # noqa: E402
    CLASS_COLORS,
    IGNORE_INDEX,
    MAX_SEGMENTATION_CLASSES,
    MaskDecoder,
    class_colors,
    describe,
    find_pairs,
)
from brickdnn.training.shapes import SHAPE_CLASSES, shapes_arrays, shapes_sample  # noqa: E402

from .segmentation_helpers import (  # noqa: E402
    SIZE,
    dataset_files,
    encode_mask,
    png,
    sample,
    write_files,
    zip_bytes,
)

# --------------------------------------------------------------------------- #
# Layouts
# --------------------------------------------------------------------------- #


def test_images_and_masks_folders(tmp_path):
    write_files(tmp_path, dataset_files(8))
    assert is_segmentation_folder(tmp_path)
    info = inspect_segmentation_folder(tmp_path, name="Squares")
    assert (info.kind, info.task, info.name) == ("segmentation", "segmentation", "Squares")
    assert info.num_samples == 8
    assert info.sample_shape == [3, SIZE, SIZE]
    assert info.class_names == ["background", "class 1", "class 2"]
    assert info.num_classes == 3
    assert info.predefined_split is None
    details = info.segmentation
    assert details["mask_format"] == "indices"
    assert details["ignore_index"] is None
    assert details["image_sizes"] == {
        "most_common": [SIZE, SIZE],
        "min": [SIZE, SIZE],
        "max": [SIZE, SIZE],
        "different": 1,
    }
    assert (details["unmatched_images"], details["unmatched_masks"]) == (0, 0)
    assert details["mask_size_mismatches"] == 0
    # The pixel shares are those of the masks, counted here by hand.
    counts = np.zeros(3)
    for index in range(8):
        counts += np.bincount(sample(index)[1].ravel(), minlength=3)
    assert details["class_pixel_share"] == pytest.approx(counts / counts.sum(), abs=1e-6)
    assert sum(details["class_pixel_share"]) == pytest.approx(1.0, abs=1e-5)
    # It survives the trip through JSON, like every dataset info.
    assert DatasetInfo.from_dict(json.loads(json.dumps(info.to_dict()))) == info


def test_wrapper_folders_other_names_and_mask_suffixes(tmp_path):
    files = dataset_files(
        6, prefix="my data/", images="img", mask_dir="labels", mask_suffix="_mask"
    )
    write_files(tmp_path, files)
    info = inspect_segmentation_folder(tmp_path)
    assert info.num_samples == 6
    pairs = find_pairs(tmp_path).pairs
    assert [pair.key for pair in pairs][:2] == ["my data/000", "my data/001"]
    assert pairs[0].image.name == "000.png" and pairs[0].mask.name == "000_mask.png"


@pytest.mark.parametrize("layout", ["split", "split_inside"])
def test_split_folders_are_the_split(tmp_path, layout):
    write_files(tmp_path, dataset_files(12, layout=layout))
    info = inspect_segmentation_folder(tmp_path)
    assert info.num_samples == 12
    assert info.predefined_split == [6, 3, 3]
    assert {pair.split for pair in find_pairs(tmp_path).pairs} == {0, 1, 2}


def test_split_needs_training_and_validation_folders(tmp_path):
    files = {
        name.replace("val/", "extra/").replace("test/", "extra/"): content
        for name, content in dataset_files(12, layout="split").items()
    }
    write_files(tmp_path, files)
    info = inspect_segmentation_folder(tmp_path)
    assert info.num_samples == 12
    assert info.predefined_split is None  # split at random, like a dataset without folders


def test_unmatched_files_and_other_sizes_are_reported(tmp_path):
    files = dataset_files(8)
    del files["masks/007.png"]
    del files["images/006.png"]
    picture, mask = sample(0, size=8)
    files["masks/005.png"] = encode_mask(mask)  # half the size of its image
    write_files(tmp_path, files)
    info = inspect_segmentation_folder(tmp_path)
    assert info.num_samples == 6
    details = info.segmentation
    assert (details["unmatched_images"], details["unmatched_masks"]) == (1, 1)
    assert details["unmatched_examples"] == ["images/007.png", "masks/006.png"]
    assert details["mask_size_mismatches"] == 1
    assert details["mask_size_examples"] == ["masks/005.png"]
    notes = describe(info)
    assert notes[0].startswith("Not used: 1 image without a mask and 1 mask without an image")
    assert "another size than the image" in notes[1]


def test_images_of_different_sizes_and_grayscale(tmp_path):
    files = dataset_files(6)
    for index in (0, 1):
        picture, mask = sample(index, size=24)
        files[f"images/{index:03d}.png"] = png(picture, "L")
        files[f"masks/{index:03d}.png"] = encode_mask(mask)
    write_files(tmp_path, files)
    info = inspect_segmentation_folder(tmp_path)
    assert info.sample_shape == [3, SIZE, SIZE]  # some images are RGB; the most common size
    assert info.segmentation["image_sizes"]["max"] == [24, 24]
    assert info.segmentation["image_sizes"]["different"] == 2

    gray = tmp_path / "gray"
    write_files(
        gray,
        {
            n: png(sample(i)[0], "L") if n.startswith("images") else c
            for i, (n, c) in enumerate(sorted(dataset_files(4).items()))
        },
    )
    assert inspect_segmentation_folder(gray).sample_shape[0] == 1


def test_hidden_files_are_skipped(tmp_path):
    files = dataset_files(4)
    files["__MACOSX/images/000.png"] = b"junk"
    files["images/.hidden.png"] = b"junk"
    files[".git/masks/000.png"] = b"junk"
    write_files(tmp_path, files)
    assert inspect_segmentation_folder(tmp_path).num_samples == 4


# --------------------------------------------------------------------------- #
# Mask formats
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("masks", "classes", "expected_format", "names", "values"),
    [
        ("indices", 3, "indices", ["background", "class 1", "class 2"], None),
        ("palette", 3, "indices", ["background", "class 1", "class 2"], None),
        ("binary", 2, "binary", ["background", "foreground"], None),
        ("levels", 3, "levels", ["background", "class 1", "class 2"], [0, 127, 254]),
        (
            "colors",
            3,
            "colors",
            ["background", "class 1", "class 2"],
            [[0, 0, 0], [0, 0, 255], [255, 0, 0]],
        ),
    ],
)
def test_mask_formats(tmp_path, masks, classes, expected_format, names, values):
    write_files(tmp_path, dataset_files(6, masks=masks, classes=classes))
    info = inspect_segmentation_folder(tmp_path)
    assert info.segmentation["mask_format"] == expected_format
    assert info.segmentation["mask_values"] == values
    assert info.class_names == names
    # Decoded, every mask is the class map it was made from (colours: sorted, black first).
    decoder = MaskDecoder(info.segmentation, info.num_classes, tmp_path)
    for index, pair in enumerate(find_pairs(tmp_path).pairs):
        expected = sample(index, classes=classes)[1]
        if masks == "colors":
            expected = np.array([0, 2, 1], dtype=np.uint8)[expected]  # blue sorts before red
        decoded = decoder.decode(pair.mask)
        assert decoded.dtype == np.uint8
        assert (decoded == expected).all()


def test_ignore_index_255(tmp_path):
    files = dataset_files(6)
    for index in range(6):
        mask = sample(index)[1].copy()
        mask[0, :] = IGNORE_INDEX  # a border that is not labelled
        files[f"masks/{index:03d}.png"] = encode_mask(mask)
    write_files(tmp_path, files)
    info = inspect_segmentation_folder(tmp_path)
    details = info.segmentation
    assert details["mask_format"] == "indices"
    assert details["ignore_index"] == IGNORE_INDEX
    assert details["ignored_share"] == pytest.approx(1 / SIZE)
    assert info.num_classes == 3
    assert sum(details["class_pixel_share"]) == pytest.approx(1.0, abs=1e-5)
    decoded = MaskDecoder(details, 3, tmp_path).decode(find_pairs(tmp_path).pairs[0].mask)
    assert (decoded[0] == IGNORE_INDEX).all() and decoded[1:].max() <= 2


def test_sixteen_bit_and_rgb_grey_masks_are_indices(tmp_path):
    from PIL import Image

    files = dataset_files(4)
    write_files(tmp_path, files)
    for index in range(4):
        mask = sample(index)[1]
        path = tmp_path / "masks" / f"{index:03d}.png"
        if index % 2:
            Image.fromarray(mask.astype(np.uint16)).save(path)  # 16-bit grey
        else:
            Image.fromarray(np.stack([mask] * 3, axis=-1)).save(path)  # grey stored as RGB
    info = inspect_segmentation_folder(tmp_path)
    assert info.segmentation["mask_format"] == "indices"
    assert info.num_classes == 3


def test_class_names_from_a_text_file(tmp_path):
    files = dataset_files(4)
    files["classes.txt"] = b"sky\n\n road \ncar\n"
    write_files(tmp_path, files)
    assert inspect_segmentation_folder(tmp_path).class_names == ["sky", "road", "car"]


def test_class_file_can_list_classes_no_mask_shows(tmp_path):
    files = dataset_files(4)
    files["classes.json"] = json.dumps(["sky", "road", "car", "tree"]).encode()
    write_files(tmp_path, files)
    info = inspect_segmentation_folder(tmp_path)
    assert info.class_names == ["sky", "road", "car", "tree"]
    assert info.segmentation["class_pixel_share"][3] == 0


def test_class_file_gives_the_order_of_colours(tmp_path):
    files = dataset_files(4, masks="colors")
    files["classes.json"] = json.dumps(
        {
            "classes": [
                {"name": "nothing", "color": [0, 0, 0]},
                {"name": "red", "color": [255, 0, 0]},
                {"name": "blue", "color": [0, 0, 255]},
            ]
        }
    ).encode()
    write_files(tmp_path, files)
    info = inspect_segmentation_folder(tmp_path)
    assert info.class_names == ["nothing", "red", "blue"]
    assert info.segmentation["mask_values"] == [[0, 0, 0], [255, 0, 0], [0, 0, 255]]
    decoder = MaskDecoder(info.segmentation, 3, tmp_path)
    assert (decoder.decode(find_pairs(tmp_path).pairs[1].mask) == sample(1)[1]).all()


# --------------------------------------------------------------------------- #
# What is refused
# --------------------------------------------------------------------------- #


def refusal(tmp_path, files) -> str:
    write_files(tmp_path, files)
    with pytest.raises(DatasetError) as error:
        inspect_segmentation_folder(tmp_path)
    return str(error.value)


def test_no_folders(tmp_path):
    files = {f"cats/{i}.png": png(sample(i)[0]) for i in range(4)}
    assert not is_segmentation_folder(write_files(tmp_path, files))
    message = refusal(tmp_path, {})
    assert message.startswith("No folders with images and masks were found.")
    assert "images/001.jpg and masks/001.png" in message


def test_images_without_masks_and_the_other_way_round(tmp_path):
    images = {n: c for n, c in dataset_files(4).items() if n.startswith("images/")}
    assert "has images but no masks" in refusal(tmp_path / "a", images)
    masks = {n: c for n, c in dataset_files(4).items() if n.startswith("masks/")}
    assert "has masks but no images" in refusal(tmp_path / "b", masks)


def test_names_that_do_not_match(tmp_path):
    files = {
        n.replace("masks/0", "masks/x") if n.startswith("masks/") else n: c
        for n, c in dataset_files(4).items()
    }
    message = refusal(tmp_path, files)
    assert "No image has a mask with the same file name" in message
    assert "'images/000.png' and 'masks/x00.png'" in message


def test_too_few_pairs(tmp_path):
    assert "at least 4 images with a mask" in refusal(tmp_path, dataset_files(3))


def test_two_files_with_the_same_name(tmp_path):
    files = dataset_files(4)
    files["masks/000.bmp"] = files["masks/000.png"]
    assert "has the same name as another file" in refusal(tmp_path, files)


def test_jpeg_masks_are_refused_with_the_reason(tmp_path):
    message = refusal(tmp_path, dataset_files(4, masks="jpeg"))
    assert "different grey values" in message
    assert "JPEG compression changes the pixel values" in message
    assert "Save the masks as PNG" in message


def test_smoothed_masks_are_refused(tmp_path):
    files = dataset_files(4)
    for index in range(4):
        ramp = np.tile(np.arange(SIZE, dtype=np.uint8) * 9, (SIZE, 1))  # 16 grey values
        ramp[0, :] = np.arange(100, 100 + SIZE)  # 15 more (108 is there already)
        ramp[1, :3] = (200, 201, 202)  # 34 values: too many to be classes
        files[f"masks/{index:03d}.png"] = png(ramp)
    message = refusal(tmp_path, files)
    assert "The masks hold 34 different grey values (0, 9, 18" in message
    assert "anti-aliased" in message and "JPEG" not in message


def test_too_many_classes(tmp_path):
    files = dataset_files(4)
    values = np.arange(SIZE * SIZE, dtype=np.uint8).reshape(SIZE, SIZE) % 40  # indices 0 … 39
    for index in range(4):
        files[f"masks/{index:03d}.png"] = png(values)
    message = refusal(tmp_path, files)
    assert f"at most {MAX_SEGMENTATION_CLASSES} classes" in message


def test_too_many_colours(tmp_path):
    files = dataset_files(4)
    rng = np.random.default_rng(0)
    for index in range(4):
        files[f"masks/{index:03d}.png"] = png(
            rng.integers(0, 255, size=(SIZE, SIZE, 3), dtype=np.uint8)
        )
    message = refusal(tmp_path, files)
    assert "different colours" in message and "one colour per class" in message


def test_masks_that_are_all_background(tmp_path):
    files = dataset_files(4)
    for index in range(4):
        files[f"masks/{index:03d}.png"] = png(np.zeros((SIZE, SIZE), dtype=np.uint8))
    assert "The masks are all background" in refusal(tmp_path, files)


def test_colours_the_class_file_does_not_list(tmp_path):
    files = dataset_files(4, masks="colors")
    files["classes.json"] = json.dumps(
        [{"name": "nothing", "color": [0, 0, 0]}, {"name": "red", "color": [255, 0, 0]}]
    ).encode()
    message = refusal(tmp_path, files)
    assert "colours that classes.json does not list, e.g. (0, 0, 255)" in message


@pytest.mark.parametrize(
    ("content", "message"),
    [
        (b"{not json", "must be a list of class names"),
        (b"[1, 2]", "must be a list of class names"),
        (json.dumps([f"c{i}" for i in range(40)]).encode(), "at most 32"),
        (json.dumps(["a"]).encode(), None),  # fewer names than classes: the rest get a name
    ],
)
def test_class_file_problems(tmp_path, content, message):
    files = dataset_files(4)
    files["classes.json"] = content
    if message is None:
        write_files(tmp_path, files)
        assert inspect_segmentation_folder(tmp_path).class_names == ["a", "class 1", "class 2"]
    else:
        assert message in refusal(tmp_path, files)


def test_more_class_names_than_values(tmp_path):
    files = dataset_files(4, masks="binary", classes=2)
    files["classes.txt"] = b"a\nb\nc\n"
    assert "lists 3 classes, but the masks only hold 2" in refusal(tmp_path, files)


def test_broken_and_huge_files(tmp_path, monkeypatch):
    files = dataset_files(4)
    files["masks/002.png"] = b"\x89PNG not really an image"
    message = refusal(tmp_path / "broken", files)
    assert message == "The mask 'masks/002.png' cannot be read. Remove it or save it again."

    from brickdnn.training import segmentation

    monkeypatch.setattr(segmentation, "MAX_SIDE", 12)
    message = refusal(tmp_path / "huge", dataset_files(4))
    assert "has 16×16 pixels; at most 12×12 are supported" in message
    monkeypatch.setattr(segmentation, "MAX_SIDE", 4096)
    monkeypatch.setattr(segmentation, "MAX_PAIRS", 5)
    assert "at most 5 are" in refusal(tmp_path / "many", dataset_files(6))


def test_a_dataset_that_changed_is_noticed_when_it_is_read(tmp_path):
    write_files(tmp_path, dataset_files(4))
    info = inspect_segmentation_folder(tmp_path)
    decoder = MaskDecoder(info.segmentation, info.num_classes, tmp_path)
    (tmp_path / "masks" / "000.png").write_bytes(png(np.full((SIZE, SIZE), 9, dtype=np.uint8)))
    with pytest.raises(DatasetError, match="values that were not there when the dataset"):
        decoder.decode(tmp_path / "masks" / "000.png")


# --------------------------------------------------------------------------- #
# ZIP files
# --------------------------------------------------------------------------- #


def test_zip_keeps_class_files_and_only_counts_images(tmp_path):
    files = dataset_files(4)
    files["classes.txt"] = b"sky\nroad\ncar\n"
    files["readme.md"] = b"not extracted"
    files["notes/todo.txt"] = b"not extracted"
    archive = tmp_path / "data.zip"
    archive.write_bytes(zip_bytes(files))
    assert extract_image_zip(archive, tmp_path / "out") == 8
    extracted = sorted(
        p.relative_to(tmp_path / "out").as_posix()
        for p in (tmp_path / "out").rglob("*")
        if p.is_file()
    )
    assert "classes.txt" in extracted and len(extracted) == 9
    assert inspect_segmentation_folder(tmp_path / "out").class_names == ["sky", "road", "car"]


def test_zip_safety_limits_also_hold_for_segmentation_zips(tmp_path):
    files = dataset_files(4)
    archive = tmp_path / "slip.zip"
    archive.write_bytes(zip_bytes({**files, "../masks/evil.png": b"x"}))
    with pytest.raises(DatasetError, match="unsafe path"):
        extract_image_zip(archive, tmp_path / "a")
    assert not (tmp_path / "a").exists() and not (tmp_path / "masks").exists()

    archive.write_bytes(zip_bytes({**files, "/abs/classes.txt": b"x"}))
    with pytest.raises(DatasetError, match="unsafe path"):
        extract_image_zip(archive, tmp_path / "b")

    # Class files count as files (many tiny ones are refused) and as bytes (a huge one too).
    many = {f"d{i}/classes.txt": b"x" for i in range(20)}
    archive.write_bytes(zip_bytes({**files, **many}))
    with pytest.raises(DatasetError, match="more than 10 images"):
        extract_image_zip(archive, tmp_path / "c", max_files=10)
    archive.write_bytes(zip_bytes({**files, "classes.txt": b"x" * 200_000}))
    with pytest.raises(DatasetError, match="larger than"):
        extract_image_zip(archive, tmp_path / "d", max_bytes=100_000)
    assert not (tmp_path / "d").exists()

    with zipfile.ZipFile(archive, "w") as bomb:  # compresses to almost nothing
        bomb.writestr("masks/000.png", b"\0" * 5_000_000, compress_type=zipfile.ZIP_DEFLATED)
    assert archive.stat().st_size < 50_000
    with pytest.raises(DatasetError, match="larger than"):
        extract_image_zip(archive, tmp_path / "e", max_bytes=1_000_000)


def test_class_folders_are_still_class_folders(tmp_path):
    files = {f"{label}/{i}.png": png(sample(i)[0]) for label in ("cats", "dogs") for i in range(3)}
    write_files(tmp_path, files)
    assert not is_segmentation_folder(tmp_path)
    assert inspect_image_folder(tmp_path).kind == "images"


# --------------------------------------------------------------------------- #
# The built-in Shapes dataset
# --------------------------------------------------------------------------- #


def test_shapes_is_a_built_in_segmentation_dataset():
    spec = BUILTIN_DATASETS["shapes"]
    assert (spec.task, spec.generated, spec.download_mb) == ("segmentation", True, 0)
    assert spec.class_names == SHAPE_CLASSES
    info = builtin_info("shapes")
    assert (info.kind, info.task, info.builtin) == ("builtin", "segmentation", "shapes")
    assert info.sample_shape == [3, 64, 64]
    assert (info.train_size, info.test_size, info.num_samples) == (400, 100, 500)
    assert info.segmentation["mask_format"] == "indices"
    assert spec.to_dict()["task"] == "segmentation"
    assert BUILTIN_DATASETS["mnist"].to_dict()["task"] == "classification"
    assert builtin_info("mnist").task is None


def test_shapes_images_are_deterministic_and_look_like_described():
    picture, mask = shapes_sample(7)
    again, mask_again = shapes_sample(7)
    assert (picture == again).all() and (mask == mask_again).all()
    assert picture.shape == (64, 64, 3) and picture.dtype == np.uint8
    assert mask.shape == (64, 64) and mask.dtype == np.uint8
    other, _ = shapes_sample(8)
    assert (picture != other).any()

    images, masks = shapes_arrays(0, 400)
    assert images.shape == (400, 3, 64, 64) and masks.shape == (400, 64, 64)
    assert (images[7].transpose(1, 2, 0) == picture).all()
    assert set(np.unique(masks)) == {0, 1, 2, 3}
    # Every image shows a shape, and every class occurs in many images.
    assert (masks.reshape(400, -1).max(axis=1) > 0).all()
    for label in (1, 2, 3):
        assert ((masks == label).reshape(400, -1).any(axis=1)).sum() > 100
    # The statistics the training normalises with, and the shares shown to the user.
    scaled = images / 255.0
    assert scaled.mean(axis=(0, 2, 3)) == pytest.approx(SHAPES_MEAN, abs=2e-4)
    assert scaled.std(axis=(0, 2, 3)) == pytest.approx(SHAPES_STD, abs=2e-4)
    shares = [float((masks == label).mean()) for label in range(4)]
    assert shares == pytest.approx(SHAPES_PIXEL_SHARE, abs=2e-4)


def test_shapes_start_and_size():
    images, masks = shapes_arrays(398, 4)
    assert (images[0].transpose(1, 2, 0) == shapes_sample(398)[0]).all()
    small, small_mask = shapes_sample(3, size=32)
    assert small.shape == (32, 32, 3) and small_mask.shape == (32, 32)


def test_class_colours():
    assert len(CLASS_COLORS) == MAX_SEGMENTATION_CLASSES == len(set(CLASS_COLORS))
    assert class_colors(3) == list(CLASS_COLORS[:3])
    assert len(class_colors(40)) == 40
