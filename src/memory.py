import torch

_cache_ = {
    "memory_allocated": 0,
    "max_memory_allocated": 0,
    "memory_reserved": 0,
    "max_memory_reserved": 0,
}


def _get_memory_info(info_name: str, unit: str) -> str:
    if info_name == "memory_allocated":
        current_value = torch.cuda.memory.memory_allocated()
    elif info_name == "max_memory_allocated":
        current_value = torch.cuda.memory.max_memory_allocated()
    elif info_name == "memory_reserved":
        current_value = torch.cuda.memory.memory_reserved()
    elif info_name == "max_memory_reserved":
        current_value = torch.cuda.memory.max_memory_reserved()
    else:
        raise ValueError("Unrecognized `info_name` argument.")

    divisor: int = 1
    if unit.lower() == "kb":
        divisor = 1024
    elif unit.lower() == "mb":
        divisor = 1024 * 1024
    elif unit.lower() == "gb":
        divisor = 1024 * 1024 * 1024
    else:
        raise ValueError("Unrecognized `unit` argument.")

    diff_value = current_value - _cache_[info_name]
    _cache_[info_name] = current_value

    return (
        f"{info_name}: \t {current_value} ({current_value/divisor:.3f} {unit.upper()})"
        f"\t diff_{info_name}: {diff_value} ({diff_value/divisor:.3f} {unit.upper()})"
    )


def print_memory_info(unit: str = "kb") -> None:
    print(_get_memory_info("memory_allocated", unit))
    print(_get_memory_info("max_memory_allocated", unit))
    print(_get_memory_info("memory_reserved", unit))
    print(_get_memory_info("max_memory_reserved", unit))
    print("")
