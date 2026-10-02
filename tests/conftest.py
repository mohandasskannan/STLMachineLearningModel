import torch


def pytest_sessionstart(session):
    # Tiny CPU fixtures are faster and more stable without large thread pools.
    torch.set_num_threads(2)
