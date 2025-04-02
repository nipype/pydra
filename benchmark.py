import asyncio
import time


def sync_function(x):
    return x * 2


async def async_function(x):
    return x * 2


def benchmark_sync():
    start_time = time.time()
    for _ in range(1000000):
        sync_function(10)
    end_time = time.time()
    return end_time - start_time


async def benchmark_async():
    start_time = time.time()
    for _ in range(1000000):
        await async_function(10)
    end_time = time.time()
    return end_time - start_time


def main():
    sync_time = benchmark_sync()
    print(f"Sync function time: {sync_time:.6f} seconds")

    async_time = asyncio.run(benchmark_async())
    print(f"Async function time: {async_time:.6f} seconds")


if __name__ == "__main__":
    main()
