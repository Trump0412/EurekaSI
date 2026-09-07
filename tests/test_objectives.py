import pytest
import torch

from spatial_intelligence.objectives import group_advantages, kl_loss, policy_loss, sft_loss


@pytest.mark.parametrize("direction", ["forward", "reverse"])
def test_kl_zero_and_teacher_no_gradient(direction):
    student = torch.randn(1,4,7,requires_grad=True)
    teacher = student.detach().clone().requires_grad_(True)
    loss = kl_loss(student, teacher, direction=direction)
    assert abs(loss.item()) < 1e-6
    loss.backward()
    assert teacher.grad is None


@pytest.mark.parametrize("direction", ["forward", "reverse"])
def test_kl_optimization_moves_toward_teacher(direction):
    student = torch.tensor([[[0.,0.,0.]]], requires_grad=True)
    teacher = torch.tensor([[[4.,-2.,-2.]]], requires_grad=True)
    optimizer = torch.optim.SGD([student],lr=.3)
    initial = kl_loss(student, teacher, direction=direction).item()
    for _ in range(10):
        optimizer.zero_grad(); loss=kl_loss(student,teacher,direction=direction);loss.backward();optimizer.step()
    assert kl_loss(student,teacher,direction=direction).item() < initial
    assert teacher.grad is None


def test_response_sft_gradient():
    logits = torch.zeros(1,2,4,requires_grad=True)
    loss = sft_loss(logits,torch.tensor([[2,3]]))
    loss.backward()
    assert logits.grad[0,0,2] < 0 and logits.grad[0,1,3] < 0


def test_constant_reward_has_no_advantage():
    assert torch.equal(group_advantages([1.,1.,1.]),torch.zeros(3))


@pytest.mark.parametrize("algorithm", ["grpo", "gspo"])
def test_policy_improves_positive_advantage(algorithm):
    new = torch.tensor([[-1.,-2.]],requires_grad=True)
    old = new.detach().clone()
    loss = policy_loss(new,old,old,torch.tensor(1.),algorithm=algorithm,clip=.2,beta=0.)
    loss.backward()
    assert (new.grad < 0).all()


def test_gspo_sequence_ratio_differs_from_grpo():
    new=torch.tensor([[-.5,-2.5]],requires_grad=True)
    old=torch.tensor([[-1.,-2.]])
    a=policy_loss(new,old,old,1.,algorithm="grpo",clip=.2,beta=0.)
    b=policy_loss(new,old,old,1.,algorithm="gspo",clip=.2,beta=0.)
    assert not torch.allclose(a,b)
