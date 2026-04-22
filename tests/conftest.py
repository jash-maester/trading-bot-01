import pytest


@pytest.fixture(scope="session")
def db_url() -> str:
    import os

    return os.getenv(
        "TEST_DATABASE_URL",
        "postgresql+psycopg://trader:trader@localhost:5432/trader",
    )
