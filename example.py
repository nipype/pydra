import asyncio


def is_coroutine_function(func):
    return asyncio.iscoroutinefunction(func)


async def call_function(func, *args, **kwargs):
    if is_coroutine_function(func):
        return await func(*args, **kwargs)
    else:
        return func(*args, **kwargs)


# Example usage
async def async_function(x):
    await asyncio.sleep(1)
    return x * 2


def sync_function(x):
    return x * 2


async def main():
    result1 = await call_function(async_function, 10)
    result2 = await call_function(sync_function, 10)
    print(result1)  # Output: 20
    print(result2)  # Output: 20


# To run the example
if __name__ == "__main__":
    asyncio.run(main())
