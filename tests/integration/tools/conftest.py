import os

import pytest


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    if os.environ.get("ATM_ENABLE_DOCKER_TESTS") != "1":
        skip_docker = pytest.mark.skip(
            reason="Docker tests disabled (set ATM_ENABLE_DOCKER_TESTS=1)"
        )
        for item in items:
            if "docker" in item.keywords:
                item.add_marker(skip_docker)
    if os.environ.get("ATM_ENABLE_NETWORK_TESTS") != "1":
        skip_network = pytest.mark.skip(
            reason="Network tests disabled (set ATM_ENABLE_NETWORK_TESTS=1)"
        )
        for item in items:
            if "network" in item.keywords:
                item.add_marker(skip_network)
