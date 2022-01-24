def pytest_addoption(parser):
    parser.addoption(
        "--regenerate-screenshots",
        action="store_true",
        dest="regenerate_screenshots",
        default=False,
    )
