import torch


def test_step_lr_step_called():
    model = torch.nn.Linear(2, 2)
    opt = torch.optim.SGD(model.parameters(), lr=0.1)
    sched = torch.optim.lr_scheduler.StepLR(opt, step_size=1, gamma=0.1)
    initial_lr = opt.param_groups[0]['lr']
    opt.step()
    sched.step()
    assert opt.param_groups[0]['lr'] != initial_lr