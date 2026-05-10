import string

from dataloader.lowercase_letters_dataloader import (
    S_LIST,
    get_letters_dataloader,
)


C_LIST = list(string.ascii_uppercase)


def get_dataloader(
    data_dir,
    batch_size,
    n_fragments=26,
    fragment_len=32,
    num_workers=0,
    portion=1,
    shuffle=True,
    distributed=False,
):
    return get_letters_dataloader(
        data_dir=data_dir,
        batch_size=batch_size,
        n_fragments=n_fragments,
        fragment_len=fragment_len,
        num_workers=num_workers,
        portion=portion,
        shuffle=shuffle,
        distributed=distributed,
        c_list=C_LIST,
        s_list=S_LIST,
    )
