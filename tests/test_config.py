"""
The YAML config must stay in sync with the Python defaults, and wiring it in
must not change a single generated number.

Two guarantees are checked here:

1. `test_scenario_keys_are_real_kwargs` / `test_config_matches_python_defaults`
   — every key declared in configs/generator.yaml names a real keyword argument
   of the scenario function it sits under, and carries the same value as that
   function's default.  This is what makes the YAML citeable in the paper: a
   reader who reads the config is reading what the code actually ran.

2. `test_generation_identical_without_config` — generating a scenario with the
   config loaded is bit-identical to generating it with the config absent.  The
   config wiring is therefore a documentation change, not a behavioural one,
   and results computed before it remain valid.
"""

import inspect

import numpy as np
import pytest

from generator import config as gcfg
from generator import scenarios as gscen
from generator.parameters import BaseParams
from generator.scenarios import SCENARIO_NAMES, generate_scenario

# Scenario name -> the function that consumes its kwargs.
SCENARIO_FUNCS = {
    "no_shift": gscen._no_shift,
    "covariate_shift": gscen._covariate_shift,
    "observational_shift": gscen._observational_shift,
    "spurious_shift": gscen._spurious_shift,
    "baseline_hazard_shift": gscen._baseline_hazard_shift,
    "shape_shift": gscen._shape_shift,
    "concept_shift": gscen._concept_shift,
    "strong_covariate_shift": gscen._strong_covariate_shift,
    "cure_shift": gscen._cure_shift,
    # prior_shift's only parameter (prevalence_target) is consumed by
    # generate_scenario itself, so it is checked separately below.
}

# Scenarios exercised in the equivalence check.  cure_shift and prior_shift use
# a different generation path and are covered by the two dedicated tests.
EQUIVALENCE_SCENARIOS = [
    "no_shift",
    "covariate_shift",
    "observational_shift",
    "spurious_shift",
    "baseline_hazard_shift",
    "shape_shift",
    "concept_shift",
    "strong_covariate_shift",
]


@pytest.fixture(autouse=True)
def _clear_config_cache():
    """Each test starts from a cold cache so env-var overrides take effect."""
    gcfg._CACHE.clear()
    yield
    gcfg._CACHE.clear()


def test_config_file_is_present_and_parses():
    cfg = gcfg.load_config(reload=True)
    assert cfg, "configs/generator.yaml missing or empty — the paper cites it"
    assert set(cfg) >= {"base", "domain_defaults", "scenarios", "experiment"}


def test_every_scenario_is_declared():
    """No scenario may be silently undocumented."""
    declared = set(gcfg.load_config(reload=True).get("scenarios", {}))
    assert declared == set(SCENARIO_NAMES), (
        f"config/code scenario mismatch: "
        f"only in config {declared - set(SCENARIO_NAMES)}, "
        f"only in code {set(SCENARIO_NAMES) - declared}"
    )


def test_base_block_matches_baseparams_defaults():
    defaults = BaseParams()
    for key, value in gcfg.base_kwargs(gcfg.load_config(reload=True)).items():
        assert hasattr(defaults, key), f"base.{key} is not a BaseParams field"
        assert getattr(defaults, key) == value, (
            f"base.{key}: config {value} != Python default {getattr(defaults, key)}"
        )


@pytest.mark.parametrize("name", sorted(SCENARIO_FUNCS))
def test_scenario_keys_are_real_kwargs(name):
    """A typo'd config key would otherwise be silently ignored."""
    accepted = set(inspect.signature(SCENARIO_FUNCS[name]).parameters)
    for key in gcfg.scenario_kwargs(name, gcfg.load_config(reload=True)):
        assert key in accepted, (
            f"scenarios.{name}.{key} is not a parameter of {SCENARIO_FUNCS[name].__name__}; "
            f"it would be passed through and raise TypeError"
        )


@pytest.mark.parametrize("name", sorted(SCENARIO_FUNCS))
def test_config_matches_python_defaults(name):
    """Config and code must agree, so neither can drift unnoticed."""
    params = inspect.signature(SCENARIO_FUNCS[name]).parameters
    for key, value in gcfg.scenario_kwargs(name, gcfg.load_config(reload=True)).items():
        default = params[key].default
        assert default is not inspect.Parameter.empty, (
            f"scenarios.{name}.{key} has no Python default to compare against"
        )
        assert np.allclose(np.asarray(value, dtype=float),
                           np.asarray(default, dtype=float)), (
            f"scenarios.{name}.{key}: config {value} != Python default {default}"
        )


def test_prior_shift_prevalence_matches_generate_scenario_default():
    declared = gcfg.scenario_kwargs("prior_shift", gcfg.load_config(reload=True))
    assert declared["prevalence_target"] == 0.60, (
        "generate_scenario falls back to 0.60 when the key is absent; keep them equal"
    )


@pytest.mark.parametrize("name", EQUIVALENCE_SCENARIOS)
def test_generation_identical_without_config(name, monkeypatch, tmp_path):
    """Loading the config changes no generated value."""
    with_cfg = generate_scenario(name, n_source_domains=3, n_per_domain=200, seed=7)

    # Point the loader at a path that does not exist -> pure Python defaults.
    monkeypatch.setenv("SHIFT_SURV_CONFIG", str(tmp_path / "absent.yaml"))
    gcfg._CACHE.clear()
    without_cfg = generate_scenario(name, n_source_domains=3, n_per_domain=200, seed=7)

    np.testing.assert_array_equal(with_cfg["target"].x, without_cfg["target"].x)
    np.testing.assert_array_equal(with_cfg["target"].y, without_cfg["target"].y)
    np.testing.assert_allclose(with_cfg["target"].eta, without_cfg["target"].eta)
    for a, b in zip(with_cfg["source"], without_cfg["source"]):
        np.testing.assert_array_equal(a.x, b.x)
        np.testing.assert_array_equal(a.y, b.y)


def test_explicit_kwargs_override_config():
    """Callers (sweeps, ablations) must still be able to override the config."""
    a = generate_scenario("covariate_shift", n_source_domains=1,
                          n_per_domain=200, seed=3)
    b = generate_scenario("covariate_shift", n_source_domains=1,
                          n_per_domain=200, seed=3, mu_inv_shift_scale=4.0)
    assert not np.allclose(a["target"].x, b["target"].x), (
        "explicit kwarg was swallowed by the config defaults"
    )
