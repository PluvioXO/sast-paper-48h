import copy
import json
from pathlib import Path
from types import SimpleNamespace
import time

import numpy as np
import pytest
import torch
from torch.nn import functional as F

from src import paper_suite as suite
from src import paper_suite_models as models
from src.paper_suite_analysis import linear_cka, ridge_probe, run_toy
from src.paper_suite_data import (legacy_engine, load_data, prepare_case_data, shd_frames,
                                  speaker_split, write_json)


@pytest.fixture(autouse=True, scope="module")
def cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def small_model(vision=False):
    return models.SuiteSNN({"architecture": "cnn2" if vision else "mlp2", "widths": [2, 3],
        "classes": 3, "spatial_size": 8 if vision else None, "input_channels": 4,
        "leak": .9, "slope": 5., "threshold": 1.}).double()


def batches(vision=False):
    torch.manual_seed(17)
    shape = (5, 4, 2, 8, 8) if vision else (5, 4, 4)
    return [(torch.rand(shape, dtype=torch.float64), torch.tensor([0, 1, 2, 1, 0])) for _ in range(2)]


SETTINGS = {"rho": .05, "beta": .3, "input_penalty": .01, "margin_target": .25}


def test_configuration_defaults_and_reject_bad_protocol():
    c = suite.read_config()
    assert c["budget"]["hours"] == 48
    assert (c["leak"], c["slope"], c["threshold"]) == (.5, 1., 1.)
    assert suite.plan(c)["jobs"]["ablate"] == 0
    assert suite.plan(c)["jobs"]["confirm"] == 140
    for method in c["methods"]:
        selected = suite.candidates(c, method)
        assert len({s["lr"] for s in selected}) == 3
        if method in ("sast", "sam", "asam", "sast_margin"):
            assert len({s["rho"] for s in selected}) == 3
    c["seeds"][0] = c["pilot_seeds"][0]
    with pytest.raises(ValueError, match="disjoint"):
        suite.validate_config(c)


def test_new_cnn_matches_existing_audited_recurrence():
    engine = legacy_engine()
    cfg = engine.ReplicationConfig(channels=(2, 3), slope=5., leak=.9)
    old = engine.ReplicationSNN(cfg, "nmnist").double()
    spec = {"architecture": "cnn2", "widths": [2, 3], "classes": 10,
            "spatial_size": 34, "input_channels": 2, "leak": .9, "slope": 5., "threshold": cfg.threshold}
    new = models.SuiteSNN(spec).double()
    new.load_state_dict({key.replace("convs.", "blocks."): value for key, value in old.state_dict().items()})
    x = torch.rand(2, 4, 2, 34, 34, dtype=torch.float64)
    for mode in ("smooth", "hard", "ste"):
        for scales in ((1., 1.), (.5, 2.)):
            torch.testing.assert_close(new(x, mode=mode, scales=scales),
                old(x, mode=mode, threshold_scales=scales), rtol=1e-12, atol=1e-12)


def test_recurrent_smooth_gradient_matches_finite_difference():
    model = small_model()
    x, y = batches()[0]
    loss = F.cross_entropy(model(x), y)
    loss.backward()
    original = models.parameter_vector(model)
    analytic = models.gradient_vector(model).clone()
    for index in (0, 4, 10, len(original)-1):
        values = []
        for sign in (1, -1):
            changed = original.clone()
            changed[index] += sign*1e-5
            models.assign_vector(model, changed)
            values.append(float(F.cross_entropy(model(x), y).detach()))
        assert (values[0]-values[1])/2e-5 == pytest.approx(float(analytic[index]), rel=1e-5, abs=1e-9)
    models.assign_vector(model, original)


@pytest.mark.parametrize("method", ["sast", "sam", "asam", "random", "rho0", "avg2", "margin", "input_gradient"])
def test_short_microbatch_matches_effective_batch(method):
    model = small_model()
    other = copy.deepcopy(model)
    a, b = batches()
    for current, microbatch in ((model, 2), (other, 5)):
        optimizer = torch.optim.SGD(current.parameters(), lr=.01)
        models.train_step(current, optimizer, b, a, method, SETTINGS, microbatch,
                          torch.Generator().manual_seed(9))
    torch.testing.assert_close(models.parameter_vector(model), models.parameter_vector(other), rtol=1e-10, atol=1e-12)


def test_rho_zero_is_ordinary_descent_and_avg2_is_average():
    a, b = batches()
    initial = small_model()
    for method in ("rho0", "avg2"):
        delivered, reference = copy.deepcopy(initial), copy.deepcopy(initial)
        optimizer = torch.optim.SGD(delivered.parameters(), lr=.02)
        measured = models.train_step(delivered, optimizer, b, a, method, SETTINGS, 2, torch.Generator().manual_seed(1))
        expected = F.cross_entropy(reference(b[0]), b[1])
        if method == "avg2":
            expected = (expected+F.cross_entropy(reference(a[0]), a[1]))/2
        expected.backward()
        torch.optim.SGD(reference.parameters(), lr=.02).step()
        torch.testing.assert_close(models.parameter_vector(delivered), models.parameter_vector(reference), rtol=1e-12, atol=1e-12)
        assert measured["pairs"] == 2 and measured["forward_backward_examples"] == 10


@pytest.mark.parametrize("method", ["sast", "sam", "asam", "asam_independent"])
def test_two_pass_update_against_explicit_reference(method):
    reference = small_model()
    actual = copy.deepcopy(reference)
    a, b = batches()
    base = models.parameter_vector(reference)
    first = b if method in ("sam", "asam") else a
    F.cross_entropy(reference(first[0]), first[1]).backward()
    direction = models.gradient_vector(reference)
    scale = torch.cat([(p.detach().abs()+.01 if name.endswith("weight") else torch.ones_like(p)).flatten()
        for name, p in reference.named_parameters()]) if method.startswith("asam") else torch.ones_like(base)
    epsilon = SETTINGS["rho"]*scale.square()*direction/((scale*direction).norm()+1e-12)
    models.assign_vector(reference, base+epsilon)
    reference.zero_grad(set_to_none=True)
    F.cross_entropy(reference(b[0]), b[1]).backward()
    expected = base-.02*models.gradient_vector(reference)
    models.train_step(actual, torch.optim.SGD(actual.parameters(), lr=.02), b, a, method, SETTINGS, 2,
                      torch.Generator().manual_seed(1))
    torch.testing.assert_close(models.parameter_vector(actual), expected, rtol=1e-11, atol=1e-12)


def test_perturbation_restored_when_second_pass_fails(monkeypatch):
    model = small_model()
    before = models.parameter_vector(model).clone()
    actual, calls = models.backward_mean, []
    def fail_second(*args, **kwargs):
        calls.append(1)
        if len(calls) == 2:
            assert not torch.equal(before, models.parameter_vector(model))
            raise RuntimeError("injected interruption")
        return actual(*args, **kwargs)
    monkeypatch.setattr(models, "backward_mean", fail_second)
    a, b = batches()
    with pytest.raises(RuntimeError, match="interruption"):
        models.train_step(model, torch.optim.SGD(model.parameters(), lr=.1), b, a, "sast", SETTINGS, 2,
                          torch.Generator().manual_seed(1))
    torch.testing.assert_close(before, models.parameter_vector(model), rtol=0, atol=0)


@pytest.mark.parametrize("vision", [False, True])
def test_transfer_bound_encloses_actual_errors(vision):
    for seed in range(3):
        torch.manual_seed(seed)
        model = small_model(vision)
        x = batches(vision)[0][0]
        for scales in (None, [.25, 4.], [2., .5]):
            result = models.transfer_error_bound(model, x, scales)
            assert torch.all(result["observed_logit_error"] <= result["logit_bound"]+1e-10)
            assert torch.all(result["observed_feature_error"] <= result["feature_error_bound"]+1e-10)
            assert not torch.any(result["agreement_condition"]&~result["observed_agreement"])


def test_speaker_disjoint_split_and_last_timestamp():
    labels, speakers = np.tile([0, 1, 2], 4), np.repeat(np.arange(4), 3)
    parts = speaker_split(labels, speakers, seed=2, count=1)
    assert not set(speakers[parts["train"]])&set(speakers[parts["validation"]])
    assert set(labels[parts["validation"]]) == {0, 1, 2}
    events = np.array([(0, 1), (10, 2)], dtype=[("t", "i8"), ("x", "i8")])
    frames = shd_frames(events, 2)
    assert frames[0, 1] == frames[1, 2] == 1


def test_zero_feature_probe_and_undefined_cka():
    x, labels = np.zeros((6, 10)), np.array([0, 1, 2, 0, 1, 2])
    assert linear_cka(x, x) is None
    metadata, predictions, _ = ridge_probe(x, labels, x, labels, {"hard": x}, [.001, .1], 3)
    assert np.isfinite(predictions["hard"]).all()
    assert metadata["selected_lambda"] in (.001, .1)
    nonzero = np.random.default_rng(1).normal(size=(6, 10))
    assert linear_cka(nonzero, nonzero*2) == pytest.approx(1.)


def test_epoch_resume_replays_same_training(tmp_path, monkeypatch):
    c = suite.smoke_config()
    c["data_root"] = str(tmp_path/"data")
    c["epochs"]["shd"] = 2
    descriptor = prepare_case_data(c, tmp_path, "shd", 2, 2026)
    data, _ = load_data(tmp_path, "shd_t2_split2026")
    case = next(case for case in suite.cases(c) if case["dataset"] == "shd")
    settings = suite.candidates(c, "sast")[0]
    full = tmp_path/"full"
    suite.fit(c, case, "sast", 1000, settings, full, data, descriptor, torch.device("cpu"), 2)
    original = suite.atomic_torch_save
    def interrupted(path, value):
        original(path, value)
        if Path(path).name == "last.pt" and value["epoch"] == 1:
            raise RuntimeError("injected stop after checkpoint")
    monkeypatch.setattr(suite, "atomic_torch_save", interrupted)
    resumed = tmp_path/"resumed"
    with pytest.raises(RuntimeError, match="injected"):
        suite.fit(c, case, "sast", 1000, settings, resumed, data, descriptor, torch.device("cpu"), 2)
    monkeypatch.setattr(suite, "atomic_torch_save", original)
    suite.fit(c, case, "sast", 1000, settings, resumed, data, descriptor, torch.device("cpu"), 2)
    a = torch.load(full/"last.pt", weights_only=True)
    b = torch.load(resumed/"last.pt", weights_only=True)
    for key in a["model"]:
        torch.testing.assert_close(a["model"][key], b["model"][key], rtol=0, atol=0)
    assert a["counters"]["pairs"] == b["counters"]["pairs"] == 4


def test_artifact_integrity_and_protocol_freeze(tmp_path):
    c = suite.read_config()
    suite.freeze_protocol(c, tmp_path)
    changed = copy.deepcopy(c)
    changed["slope"] = c["slope"]*2
    with pytest.raises(RuntimeError, match="changed"):
        suite.freeze_protocol(changed, tmp_path)
    write_json(tmp_path/"result.json", {"accuracy": .5})
    suite.mark_completed(tmp_path, "test")
    assert suite.completed(tmp_path, "test")
    write_json(tmp_path/"result.json", {"accuracy": 1.})
    with pytest.raises(RuntimeError, match="changed"):
        suite.completed(tmp_path, "test")


def test_toy_exhibits_small_smooth_loss_and_bad_transfer(tmp_path):
    c = suite.smoke_config()
    result = run_toy(c, tmp_path)
    row = next(r for r in result["rows"] if r["slope"] == 1 and r["time_bins"] == 16)
    assert row["smooth_loss"] < 1e-8 and row["smooth_accuracy"] == 1
    assert row["hard_accuracy"] == .5 and row["hard_constant_throughout_box"]
    assert row["box_loss_upper_numerical"] < 1e-4


def test_timing_plan_preserves_protocol_and_ignores_accuracy(tmp_path):
    c = suite.read_config()
    class Pool:
        def submit(self, _, case):
            timing = {"case": case["id"], "pair_example_seconds": .00001,
                      "forward_example_seconds": .000005, "counts": {"train": 100, "validation": 30, "test": 30}}
            return SimpleNamespace(result=lambda: timing)
    chosen = suite.budget_plan(c, Pool(), ["cuda:0"]*6, tmp_path, time.time()+48*3600)
    assert chosen == c
    assert json.loads((tmp_path/"budget_plan.json").read_text())["seeds"] == c["seeds"]


def test_partial_report_keeps_completed_causal_arms(tmp_path):
    from src.paper_suite_analysis import write_report
    c = suite.read_config()
    task = suite.stage_jobs(c, "causal")[0]
    arm = tmp_path/"jobs"/task["id"]/"erm2x"
    write_json(arm/"result.json", {"method": "erm2x", "initial_checkpoint_sha256": "shared",
        "costs": {"pairs": 2}, "test": {mode: {"accuracy": .5} for mode in ("smooth", "hard", "calibrated")}})
    suite.mark_completed(arm, "test")
    summary = write_report(c, tmp_path)
    assert not summary["complete"]
    assert "shared" in (tmp_path/"reports"/"causal.csv").read_text()


def test_independent_worker_processes_and_cached_rerun(tmp_path):
    c = suite.smoke_config()
    c["datasets"], c["methods"] = ["shd"], ["erm2x", "sast"]
    c["causal_datasets"], c["ablation"]["datasets"] = [], []
    c["ablation"]["methods"] = c["methods"][:]
    c["data_root"] = str(tmp_path/"data")
    suite.validate_config(c)
    stages = ["tune", "confirm"]
    suite.run(c, tmp_path/"run", ["cpu", "cpu"], microbatch=2, threads=1, stages=stages)
    files = sorted((tmp_path/"run"/"jobs").glob("*/runtime.json"))
    assert len({json.loads(p.read_text())["pid"] for p in files}) == 2
    timestamps = {p: p.stat().st_mtime_ns for p in files}
    suite.run(c, tmp_path/"run", ["cpu", "cpu"], microbatch=2, threads=1, stages=stages)
    assert {p: p.stat().st_mtime_ns for p in files} == timestamps


def test_frame_cache_resumes_completed_chunks(tmp_path):
    from src.paper_suite_data import _cache_split
    ids = np.arange(129)
    calls = []
    def interrupted(index):
        if index == 128:
            raise RuntimeError("interrupted cache")
        return np.full((1, 2), index, dtype=np.float32)
    with pytest.raises(RuntimeError, match="cache"):
        _cache_split(tmp_path, "train", ids, ids % 3, (1, 2), interrupted)
    def resumed(index):
        calls.append(index)
        return np.full((1, 2), index, dtype=np.float32)
    info = _cache_split(tmp_path, "train", ids, ids % 3, (1, 2), resumed)
    assert calls == [128]
    frames = np.load(info["frames"])
    assert frames[127, 0, 0] == 127 and frames[128, 0, 0] == 128


def test_dvs_archive_verification_and_extraction_repair(tmp_path, monkeypatch):
    import tonic
    from src import paper_suite_data as cache
    downloads = []
    class FakeGesture:
        train_url, test_url = "https://figshare.com/ndownloader/files/1", "https://figshare.com/ndownloader/files/2"
        train_filename, test_filename = "train.zip", "test.zip"
        train_md5, test_md5 = "verified_train", "verified_test"
        repaired = False
        def __init__(self, save_to, train):
            self.train = train
            count = (1077 if self.repaired else 1000) if train else 264
            paths = [tmp_path/f"label{i}.npy" for i in range(11)]
            for path in paths:
                path.write_bytes(b"fixture")
            self.targets = (np.arange(count) % 11).tolist()
            self.data = [str(paths[label]) for label in self.targets]
        def __len__(self):
            return len(self.data)
        def download(self):
            type(self).repaired = True
        def __getitem__(self, index):
            events = np.array([(0, 1, 1, 0), (10, 2, 2, 1)],
                dtype=[("t", "i8"), ("x", "i8"), ("y", "i8"), ("p", "i8")])
            return events, self.targets[index]
    def cache_fixture(directory, split, ids, labels, shape, get_frame, speakers=None):
        assert get_frame(int(ids[0])).shape == (2, 2, 64, 64)
        return {"count": len(ids)}
    monkeypatch.setattr(tonic.datasets, "DVSGesture", FakeGesture)
    monkeypatch.setattr(tonic.download_utils, "download_url", lambda *args: downloads.append(args))
    monkeypatch.setattr(cache, "_cache_split", cache_fixture)
    c = suite.smoke_config()
    c["data_root"] = str(tmp_path/"data")
    parameters = {"dataset": "dvsgesture", "time_bins": 2, "split_seed": 2026}
    result = cache._real_data(c, parameters, tmp_path/"frames")
    assert FakeGesture.repaired
    assert [args[3] for args in downloads] == ["verified_train", "verified_test"]
    assert all("api.figshare.com/v2/file/download" in args[0] for args in downloads)
    assert result["splits"]["train"]["count"]+result["splits"]["validation"]["count"] == 1077
    assert result["splits"]["test"]["count"] == 264
