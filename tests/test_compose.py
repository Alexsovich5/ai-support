"""Static checks on docker-compose.yml and the tracked files.

Published ports must listen on loopback only, every service runs with the
least privilege it needs, credentials reach containers only as files, and
no committed file holds a real-looking secret.
"""

import ipaddress
import json
import os
import re
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parent.parent
LANE_SUBNET = ipaddress.ip_network("172.77.0.0/24")
SECRET_NAME = re.compile(r"KEY|TOKEN|PASSWORD|SECRET", re.IGNORECASE)
SECRET_WORDS = re.compile(r"api[_-]?key|token|password|secret|sk-", re.IGNORECASE)


def _sources():
    """The committed file, plus the ``docker compose config`` output when make test provides it."""
    sources = [("file", yaml.safe_load((ROOT / "docker-compose.yml").read_text(encoding="utf-8")))]
    rendered = os.environ.get("COMPOSE_RENDERED")
    if rendered:
        sources.append(("rendered", json.loads(Path(rendered).read_text(encoding="utf-8"))))
    return sources


SOURCES = _sources()
SOURCE_IDS = [name for name, _ in SOURCES]
SERVICE_CASES = [
    pytest.param(compose, name, id=f"{label}-{name}")
    for label, compose in SOURCES
    for name in sorted(compose["services"])
]


@pytest.fixture(params=[compose for _, compose in SOURCES], ids=SOURCE_IDS)
def compose(request):
    return request.param


def test_make_test_checks_the_rendered_config_too():
    makefile = (ROOT / "Makefile").read_text(encoding="utf-8")
    assert "docker compose --profile test config --format json" in makefile
    assert "COMPOSE_RENDERED=" in makefile


def _env_items(service):
    env = service.get("environment") or {}
    if isinstance(env, dict):
        return [(name, "" if value is None else str(value)) for name, value in env.items()]
    items = []
    for entry in env:
        name, _, value = str(entry).partition("=")
        items.append((name, value))
    return items


def _volumes(service):
    """(source, read_only) for each mount, short or long syntax."""
    mounts = []
    for volume in service.get("volumes") or []:
        if isinstance(volume, dict):
            mounts.append((str(volume.get("source", "")), bool(volume.get("read_only"))))
        else:
            parts = str(volume).split(":")
            mounts.append((parts[0], len(parts) > 2 and "ro" in parts[2].split(",")))
    return mounts


def _ports(service):
    """(host_ip, published) for each published port, short or long syntax."""
    ports = []
    for port in service.get("ports") or []:
        if isinstance(port, dict):
            ports.append((port.get("host_ip"), int(port["published"])))
        else:
            parts = str(port).split(":")
            if len(parts) == 3:
                ports.append((parts[0], int(parts[1])))
            else:
                ports.append((None, int(parts[0]) if len(parts) == 2 else 0))
    return ports


def _as_text(value) -> str:
    if value is None:
        return ""
    if isinstance(value, (list, tuple)):
        return " ".join(str(part) for part in value)
    return str(value)


def test_every_published_port_is_bound_to_loopback(compose):
    published = [p for service in compose["services"].values() for p in _ports(service)]
    assert published
    for host_ip, _ in published:
        assert host_ip == "127.0.0.1"


def test_published_host_ports_stay_in_the_lane(compose):
    for service in compose["services"].values():
        for _, host_port in _ports(service):
            assert 22700 <= host_port <= 22799


@pytest.mark.parametrize("compose, name", SERVICE_CASES)
def test_service_runs_with_least_privilege(compose, name):
    service = compose["services"][name]
    assert not service.get("privileged")
    assert service.get("network_mode") != "host"
    assert service.get("pid") != "host"
    assert service.get("ipc") != "host"
    assert not any("docker.sock" in source for source, _ in _volumes(service))
    assert "no-new-privileges:true" in (service.get("security_opt") or [])
    assert service.get("cap_drop") == ["ALL"]
    assert not service.get("cap_add")
    assert str(service.get("user", "root")).split(":")[0] not in ("root", "0")


@pytest.mark.parametrize("compose, name", SERVICE_CASES)
def test_no_credentials_in_environment_or_command_lines(compose, name):
    service = compose["services"][name]
    for key, value in _env_items(service):
        if SECRET_NAME.search(key):
            assert key.endswith("_FILE"), f"{name}: {key} must point at a secret file"
            assert value.startswith("/"), f"{name}: {key} must be an absolute file path"
    for field in ("command", "entrypoint"):
        assert not SECRET_WORDS.search(_as_text(service.get(field))), (name, field)
    healthcheck = (service.get("healthcheck") or {}).get("test")
    assert not SECRET_WORDS.search(_as_text(healthcheck)), (name, "healthcheck")


def test_only_the_api_and_tester_read_the_secrets_and_only_init_writes_them(compose):
    readers = {"api", "tester"}
    for name, service in compose["services"].items():
        mounts = [ro for source, ro in _volumes(service) if source == "secrets"]
        if name == "secrets-init":
            assert mounts and not any(mounts)
        elif name in readers:
            assert mounts and all(mounts), name
        else:
            assert not mounts, f"{name} must not see the secrets volume"


def test_stub_sits_on_an_internal_network_without_published_ports(compose):
    assert compose["networks"]["backend"].get("internal") is True
    stub = compose["services"]["openai-stub"]
    assert list(stub["networks"]) == ["backend"]
    assert not stub.get("ports")


def test_networks_use_only_the_lane_subnet(compose):
    for network in compose["networks"].values():
        for pool in network["ipam"]["config"]:
            assert ipaddress.ip_network(pool["subnet"]).subnet_of(LANE_SUBNET)


@pytest.mark.parametrize("name", ["api", "openai-stub", "secrets-init"])
def test_runtime_services_have_read_only_root_filesystem(compose, name):
    assert compose["services"][name].get("read_only") is True


REAL_LOOKING_SECRETS = [
    re.compile(r"sk-(?:proj-)?[A-Za-z0-9]{20,}"),
    re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----"),
    re.compile(r"AKIA[0-9A-Z]{16}"),
    re.compile(r"gh[pousr]_[A-Za-z0-9]{30,}"),
    re.compile(r"xox[abprs]-[A-Za-z0-9-]{10,}"),
]
SKIP_DIRS = {".git", "__pycache__", ".pytest_cache", "chroma", "secrets", "htmlcov", ".venv"}


def _repo_text_files():
    for path in ROOT.rglob("*"):
        if not path.is_file() or SKIP_DIRS.intersection(path.relative_to(ROOT).parts):
            continue
        if path.stat().st_size > 1_000_000:
            continue
        try:
            yield path, path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue


def test_repository_files_hold_no_real_looking_secrets():
    checked = 0
    for path, text in _repo_text_files():
        checked += 1
        for pattern in REAL_LOOKING_SECRETS:
            assert not pattern.search(text), f"{path.relative_to(ROOT)} matches {pattern.pattern}"
    assert checked > 20


def test_generated_secrets_are_git_ignored():
    ignored = (ROOT / ".gitignore").read_text(encoding="utf-8").split()
    assert "secrets/" in ignored
    assert ".env" in ignored
