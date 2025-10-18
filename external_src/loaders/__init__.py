from .ham_loader import get_ham_loaders
from .oct_loader import get_oct_loaders

def get_loader(name, batch_size, **kwargs):
    if name.lower() == "ham10000":
        return get_ham_loaders(batch_size=batch_size, **kwargs)
    elif name.lower() == "oct2017":
        return get_oct_loaders(batch_size=batch_size, **kwargs)
    else:
        raise ValueError(f"Unknown dataset {name}")
