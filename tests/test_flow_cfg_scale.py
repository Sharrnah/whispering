import torch


def test_causal_conditional_cfm_accepts_flow_cfg_scale_kwarg():
    """Regression test for: TypeError: ...forward() got an unexpected keyword argument 'flow_cfg_scale'."""

    from Models.TTS.chatterbox.models.s3gen.flow_matching import CausalConditionalCFM

    class DummyEstimator(torch.nn.Module):
        @property
        def dtype(self):
            return torch.float32

        def forward(self, x, mask=None, mu=None, t=None, spks=None, cond=None, r=None):
            # Return a tensor with the same shape as x, as expected by the Euler solver.
            return torch.zeros_like(x)

    model = CausalConditionalCFM(
        in_channels=80,
        n_spks=1,
        spk_emb_dim=80,
        estimator=DummyEstimator(),
    )

    B, C, T = 1, 80, 4
    mu = torch.zeros(B, C, T)
    mask = torch.ones(B, 1, T)
    spks = torch.zeros(B, 80)
    cond = torch.zeros(B, 80, T)

    # Should not raise.
    out, _ = model.forward(
        mu=mu,
        mask=mask,
        n_timesteps=1,
        spks=spks,
        cond=cond,
        meanflow=False,
        flow_cfg_scale=0.7,
    )
    assert out.shape == (B, C, T)

