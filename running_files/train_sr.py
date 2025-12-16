import torch
from torch.utils.data import DataLoader

from betty.engine import Engine
from betty.configs import Config, EngineConfig

from datasets.paired_hr_dataset import HRDataset
from datasets.real_lr_dataset import RealLRDataset
from datasets.val_sr_dataset import ValSRDataset

from models.degradation.prior import DegradationPriorNet
from models.degradation.pipeline import ConditionedDegradation
from models.sr.network import SimpleSR

from models.problems.degradation_problem import DegradationProblem
from models.problems.sr_problem import SRProblem
from models.problems.arch_problem import ArchProblem


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # =========================
    # Datasets
    # =========================
    # 请按需修改下面这几个路径
    hr_root = "data/hr"
    lr_root = "data/lr"
    val_lr_root = "data/val_lr"
    val_hr_root = "data/val_hr"

    hr_loader = DataLoader(HRDataset(hr_root), batch_size=4, shuffle=True, num_workers=4)
    lr_loader = DataLoader(RealLRDataset(lr_root), batch_size=4, shuffle=True, num_workers=4)
    val_loader = DataLoader(
        ValSRDataset(val_lr_root, val_hr_root),
        batch_size=2,
        shuffle=False,
        num_workers=4,
    )

    # =========================
    # Models
    # =========================
    deg_prior = DegradationPriorNet().to(device)
    deg_pipeline = ConditionedDegradation(scale_factor=4)  # 可改成你需要的放大倍数
    sr_net = SimpleSR(in_channels=3, out_channels=3, scale_factor=4).to(device)

    # =========================
    # Problems（三层依赖）
    # =========================
    # 第一级：退化先验 + 退化模拟，逼近真实 LR
    deg_problem = DegradationProblem(
        name="degradation",
        prior=deg_prior,
        pipeline=deg_pipeline,
        optimizer=torch.optim.Adam(deg_prior.parameters(), lr=1e-4),
        train_data_loader=zip(hr_loader, lr_loader),
        config=Config(type="darts", unroll_steps=1),
        device=device,
    )

    # 第二级：用合成 LR 训练 SR
    sr_problem = SRProblem(
        name="sr",
        sr_net=sr_net,
        deg_prior=deg_prior,
        deg_pipeline=deg_pipeline,
        optimizer=torch.optim.Adam(sr_net.parameters(), lr=1e-4),
        train_data_loader=hr_loader,
        config=Config(),
        device=device,
    )

    # 第三级：用验证集的 SR 表现更新第一级退化先验的 NAS 架构参数（mixerop 的 alpha）
    arch_problem = ArchProblem(
        name="arch",
        sr_net=sr_net,
        deg_prior=deg_prior,
        optimizer=torch.optim.Adam(deg_prior.arch_parameters(), lr=1e-4),
        train_data_loader=val_loader,
        config=Config(retain_graph=True),
        device=device,
    )

    # =========================
    # Engine（多级优化）
    # =========================
    engine_config = EngineConfig(
        train_iters=1000,
        valid_step=50,
        roll_back=True,
    )

    problems = [deg_problem, sr_problem, arch_problem]
    # low-to-upper: 第 1 级 → 第 2 级 → 第 3 级
    l2u = {deg_problem: [sr_problem], sr_problem: [arch_problem]}
    # upper-to-low: 第 3 级 → 第 1 级
    u2l = {arch_problem: [deg_problem]}
    dependencies = {"l2u": l2u, "u2l": u2l}

    engine = Engine(
        config=engine_config,
        problems=problems,
        dependencies=dependencies,
    )

    engine.run()


if __name__ == "__main__":
    main()