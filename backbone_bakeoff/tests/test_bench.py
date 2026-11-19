import pandas as pd
import torch

from bakeoff.bench import imgs_per_s, main

TINY = dict(bs=2, img=16, warmup=1, iters=2, device="cpu")


def tiny_model(in_chans):
    return torch.nn.Sequential(torch.nn.Conv2d(in_chans, 4, 3), torch.nn.AdaptiveAvgPool2d(1),
                               torch.nn.Flatten(), torch.nn.Linear(4, 10))


def test_inference_throughput_positive():
    ips, _ = imgs_per_s(tiny_model(13), in_chans=13, train=False, **TINY)
    assert ips > 0


def test_train_step_throughput_positive():
    ips, _ = imgs_per_s(tiny_model(3), in_chans=3, train=True, **TINY)
    assert ips > 0


def test_cli_writes_bench_csv_without_gpu_name(tmp_path):
    out = tmp_path / "bench.csv"
    main(["--out", str(out), "--backbones", "resnet50", "--bs", "2", "--img", "32",
          "--warmup", "1", "--iters", "1", "--device", "cpu"])
    df = pd.read_csv(out)
    assert list(df.columns) == ["backbone", "in_chans", "params_M", "infer_ips", "infer_mem_MB",
                                "train_ips", "train_mem_MB"]
    assert df[["backbone", "in_chans"]].values.tolist() == [["resnet50", 3], ["resnet50", 13]]
    assert (df.infer_ips > 0).all() and (df.train_ips > 0).all()
