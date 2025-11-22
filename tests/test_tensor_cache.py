# tests/test_tensor_cache.py

import torch

from envs.pokemon.tensor_cache import OneHotCache, create_new_one_hot


def test_one_hot_cache_correctness():
    """Tests that the cache returns the correct one-hot tensor."""
    cache = OneHotCache()
    length = 10
    index = 3

    expected_tensor = torch.zeros(length)
    expected_tensor[index] = 1

    # First call, should create and cache
    cached_tensor = cache.get_one_hot(length, index)
    assert torch.equal(cached_tensor, expected_tensor)

    # Second call, should return from cache
    cached_tensor_2 = cache.get_one_hot(length, index)
    assert torch.equal(cached_tensor_2, expected_tensor)

    # Check that it's the same object in memory
    assert id(cached_tensor) == id(cached_tensor_2)


def test_one_hot_cache_multiple_entries():
    """Tests the cache with multiple different entries."""
    cache = OneHotCache()

    tensor1 = cache.get_one_hot(10, 1)
    tensor2 = cache.get_one_hot(10, 2)
    tensor3 = cache.get_one_hot(20, 5)

    expected1 = torch.zeros(10)
    expected1[1] = 1
    expected2 = torch.zeros(10)
    expected2[2] = 1
    expected3 = torch.zeros(20)
    expected3[5] = 1

    assert torch.equal(tensor1, expected1)
    assert torch.equal(tensor2, expected2)
    assert torch.equal(tensor3, expected3)
    assert not torch.equal(tensor1, tensor2)


def test_performance_with_one_hot_cache(benchmark):
    """Compares the performance of getting a one-hot tensor with and without the cache."""
    length = 1000
    index = 500
    cache = OneHotCache()

    # Benchmark the caching method
    # The first call will be slower due to cache miss, but subsequent calls are faster.
    # We call it once to populate the cache before benchmarking.
    cache.get_one_hot(length, index)
    benchmark(cache.get_one_hot, length, index)


def test_performance_without_one_hot_cache(benchmark):
    """Compares the performance of getting a one-hot tensor with and without the cache."""
    length = 1000
    index = 500

    # Benchmark the non-caching method
    benchmark(create_new_one_hot, length, index)
