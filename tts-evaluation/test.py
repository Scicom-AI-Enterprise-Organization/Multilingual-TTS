from evaluate import Dataset, repeat_interleave, summarize
import os
import pytest 
import multiprocessing as mp
import random 
import time

@pytest.fixture()
def setup_dummy_files():
    ds = Dataset(
        dataset_name="Scicom-intl/Evaluation-Multilingual-VC",
        length=3,
    )
    for sample in ds: 
        lang = sample["language"]
        id = sample["id"]
        for i in range(3):
            with open(f"output_{lang}_{id}_{i}.wav", "w") as f:
                f.write(f"dummy_content")
    
    yield
    
    for sample in ds:
        lang = sample["language"]
        id = sample["id"]
        for i in range(3):
            os.remove(f"output_{lang}_{id}_{i}.wav")

def test_dataset(setup_dummy_files):
    ds = Dataset(
        dataset_name="Scicom-intl/Evaluation-Multilingual-VC",
        length=13,
    )
    assert len(ds) == 13
    
    split_ds = ds.split(
        split_num=2,
        prefix="output", 
        output_dir="./", 
        sampling_size=3,
    )
    assert len(split_ds) == 2
    
    for s_ds in split_ds:
        assert len(s_ds) == 5
    
    split_ds = ds.split(
        split_num=20, 
        prefix="output",
        output_dir="./",
        sampling_size=3,
    )
    assert len(split_ds) == 10
    
    split_ds = ds.split(
        split_num=20, 
        prefix="output",
        output_dir="./",
        sampling_size=4,
    )
    assert len(split_ds) == 13

def test_repeat_interleave():
    lst = ["a", "b", "c"]
    times = 3
    expected_output = ["a", "a", "a", "b", "b", "b", "c", "c", "c"]
    assert repeat_interleave(lst, times) == expected_output

def test_summarize():
    ds = Dataset(
        dataset_name="Scicom-intl/Evaluation-Multilingual-VC",
        length=None,
    )
    summarize(
        dataset=ds, 
        output_dir="./", 
        sample_size=3, 
        skip_mos=True
    )
