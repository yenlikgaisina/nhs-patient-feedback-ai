"""Shared pytest fixtures and configuration."""
from pathlib import Path

import pytest

from predict import MODEL_PATH


def pytest_configure(config):
    """Fail fast with a clear message if the model artefact is missing."""
    if not Path(MODEL_PATH).exists():
        pytest.exit(
            f"\nModel artefact missing at {MODEL_PATH}.\n"
            "Run `make data && make train` (or "
            "`python data/make_dataset.py && python train_model.py`) "
            "before running tests.\n",
            returncode=2,
        )


SAMPLE_COMMENTS = {
    "appointment_access": (
        "I tried for three weeks to book an appointment and could not get through."
    ),
    "communication": (
        "Nobody explained what the procedure would involve or what to expect."
    ),
    "staff_attitude": (
        "The receptionist was rude and unhelpful when I asked for help."
    ),
    "waiting_time": (
        "I had to wait hours in A&E with no information about how long it would take."
    ),
    "treatment_quality": (
        "The clinical care was excellent and the doctor was thorough."
    ),
    "administration": (
        "My prescription was lost and my referral was sent to the wrong department."
    ),
    "facilities": (
        "The waiting room was dirty and the parking was impossible."
    ),
}


@pytest.fixture(scope="session")
def sample_comments():
    return SAMPLE_COMMENTS


@pytest.fixture(scope="session")
def sample_text():
    return "I waited far too long for an appointment and nobody explained the delay."
