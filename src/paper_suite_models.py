"""Smooth/hard LIF models and compute-accounted updates for the paper experiments."""
import math

import torch
from torch import nn
from torch.nn import functional as F


METHODS = {
    "erm": (1, 1), "erm2x": (1, 2), "sast": (2, 1), "sam": (2, 1),
    "asam": (2, 1), "asam_independent": (2, 1), "hard_sg": (1, 2), "margin": (1, 2),
    "sast_margin": (2, 1), "avg2": (2, 1), "random": (2, 1),
    "rho0": (2, 1), "input_gradient": (1, 1),
    "sgd2x": (1, 2), "sast_sgd": (2, 1), "sam_sgd": (2, 1),
}


def smooth(x, slope):
    return .5 + torch.atan(slope*x)/math.pi


class HardSurrogate(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, slope):
        ctx.save_for_backward(x)
        ctx.slope = slope
        return (x >= 0).to(x.dtype)

    @staticmethod
    def backward(ctx, gradient):
        (x,) = ctx.saved_tensors
        k = ctx.slope
        return gradient*k/(math.pi*(1+(k*x).square())), None


class SuiteSNN(nn.Module):
    """Zero initial state on every call; no detached reset, BatchNorm, or dropout."""
    def __init__(self, spec):
        super().__init__()
        self.spec = dict(spec)
        self.leak, self.slope = spec["leak"], spec["slope"]
        self.register_buffer("threshold", torch.tensor(float(spec["threshold"])))
        self.vision = spec["architecture"].startswith("cnn")
        widths = spec["widths"]
        if self.vision:
            previous, size = 2, spec["spatial_size"]
            self.blocks = nn.ModuleList()
            for width in widths:
                self.blocks.append(nn.Conv2d(previous, width, 3, padding=1))
                previous, size = width, size//2
            if size < 1:
                raise ValueError("Too many pooling layers for the input size.")
            features = previous*size*size
        else:
            dims = [spec["input_channels"]]+widths
            self.blocks = nn.ModuleList(nn.Linear(a, b) for a, b in zip(dims, dims[1:]))
            features = widths[-1]
        self.readout = nn.Linear(features, spec["classes"])

    def forward(self, x, mode="smooth", scales=None, return_info=False,
                penalty=False, margin_target=.25, trace=False,
                output_hardness=None, reset_hardness=None):
        if mode not in ("smooth", "hard", "ste"):
            raise ValueError(mode)
        scales = [1.]*len(self.blocks) if scales is None else scales
        if len(scales) != len(self.blocks) or any(s <= 0 for s in scales):
            raise ValueError("One positive threshold scale is required per layer.")
        batch, bins = x.shape[:2]
        z, features, traces = x, [], []
        state_penalty = torch.zeros(batch, device=x.device, dtype=x.dtype)
        min_margin = torch.full((batch,), float("inf"), device=x.device, dtype=x.dtype)
        margin_sum = torch.zeros_like(state_penalty)
        near_sum, activity_sum = torch.zeros_like(state_penalty), torch.zeros_like(state_penalty)
        temporal_gain = torch.zeros_like(state_penalty)
        for layer, block in enumerate(self.blocks):
            drive = block(z.flatten(0, 1))
            if self.vision:
                drive = F.avg_pool2d(drive, 2)
            drive = drive.reshape(batch, bins, *drive.shape[1:])
            theta = self.threshold*scales[layer]
            state = previous_reset = None
            outputs, membranes = [], []
            for t in range(bins):
                u = drive[:, t] if state is None else self.leak*state+drive[:, t]-theta*previous_reset
                distance = u-theta
                soft = smooth(distance, self.slope) if mode == "smooth" or output_hardness is not None else None
                if output_hardness is not None:
                    hard = (distance >= 0).to(x.dtype)
                    spike = (1-output_hardness)*soft+output_hardness*hard
                    tau = output_hardness if reset_hardness is None else reset_hardness
                    reset = (1-tau)*soft+tau*hard
                else:
                    spike = soft if mode == "smooth" else (
                        (distance >= 0).to(x.dtype) if mode == "hard" else
                        HardSurrogate.apply(distance, self.slope))
                    reset = spike
                state = u
                previous_reset = reset.detach() if self.spec.get("detach_reset", False) else reset
                outputs.append(spike)
                if penalty:
                    state_penalty = state_penalty+F.relu(margin_target-distance.abs()).square().flatten(1).mean(1)
                if return_info:
                    flat = distance.detach().abs().flatten(1)
                    min_margin = torch.minimum(min_margin, flat.amin(1))
                    margin_sum += flat.mean(1)
                    near_sum += (flat < .1).to(x.dtype).mean(1)
                    activity_sum += spike.detach().flatten(1).mean(1)
                    derivative = self.slope/(math.pi*(1+(self.slope*distance.detach()).square()))
                    gain = (self.leak-theta*derivative).abs().flatten(1).amax(1)
                    temporal_gain = torch.maximum(temporal_gain, gain)
                if trace:
                    membranes.append(u)
            z = torch.stack(outputs, 1)
            if return_info:
                features.append(z.mean(1).flatten(1))
            if trace:
                traces.append({"u": torch.stack(membranes, 1), "s": z, "threshold": theta})
        representation = z.sum(1)/bins
        logits = self.readout(representation.flatten(1))
        if not (return_info or penalty or trace):
            return logits
        info = {"penalty": state_penalty/(bins*len(self.blocks)), "features": features,
                "min_margin": min_margin, "mean_margin": margin_sum/(bins*len(self.blocks)),
                "near_threshold": near_sum/(bins*len(self.blocks)),
                "activity": activity_sum/(bins*len(self.blocks)), "temporal_gain": temporal_gain,
                "traces": traces}
        return logits, info


def backward_mean(model, batch, microbatch, *, mode="smooth", beta=0.,
                  margin_target=.25, coefficient=1., input_penalty=0.):
    """Accumulate a full effective-batch mean, including a shorter final microbatch."""
    x, y = batch
    total, calls = 0., 0
    for start in range(0, len(y), microbatch):
        xx, yy = x[start:start+microbatch], y[start:start+microbatch]
        if input_penalty:
            xx = xx.detach().requires_grad_(True)
        output = model(xx, mode=mode, penalty=bool(beta), margin_target=margin_target)
        logits, info = output if beta else (output, None)
        loss = F.cross_entropy(logits, yy)
        if beta:
            loss = loss+beta*info["penalty"].mean()
        if input_penalty:
            per_example = torch.autograd.grad(loss*len(yy), xx, create_graph=True)[0]
            loss = loss+input_penalty*per_example.flatten(1).square().sum(1).mean()
        if not torch.isfinite(loss):
            raise FloatingPointError("Non-finite training objective.")
        weight = len(yy)/len(y)
        (loss*(coefficient*weight)).backward()
        total += float(loss.detach())*weight
        calls += 1
    return total, calls


def grad_norm(model):
    values = [p.grad.norm() for p in model.parameters() if p.grad is not None]
    norm = torch.linalg.vector_norm(torch.stack(values))
    if not torch.isfinite(norm):
        raise FloatingPointError("Non-finite parameter gradient.")
    return norm


def perturb(model, method, rho, generator, delta=1e-12):
    named = list(model.named_parameters())
    scales = [p.detach().abs()+.01 if method == "asam" and n.endswith("weight")
              else torch.ones_like(p) for n, p in named]
    directions = [torch.randn(p.shape, generator=generator, dtype=p.dtype).to(p.device)
                  if method == "random" else (p.grad.detach() if p.grad is not None else torch.zeros_like(p))
                  for _, p in named]
    norm = torch.linalg.vector_norm(torch.stack([(s*d).norm() for s, d in zip(scales, directions)]))
    if not torch.isfinite(norm):
        raise FloatingPointError("Non-finite ascent direction.")
    radius = 0. if method == "rho0" else rho
    with torch.no_grad():
        for (_, p), scale, direction in zip(named, scales, directions):
            p.add_(radius*scale.square()*direction/(norm+delta))


def train_step(model, optimizer, descent, ascent, method, settings, microbatch, generator):
    if method not in METHODS:
        raise ValueError(method)
    passes, _ = METHODS[method]
    method = {"sgd2x": "erm2x", "sast_sgd": "sast", "sam_sgd": "sam"}.get(method, method)
    mode = "ste" if method == "hard_sg" else "smooth"
    beta = settings.get("beta", 0.) if method in ("margin", "sast_margin") else 0.
    options = dict(mode=mode, beta=beta, margin_target=settings["margin_target"],
                   input_penalty=settings.get("input_penalty", 0.) if method == "input_gradient" else 0.)
    model.train()
    optimizer.zero_grad(set_to_none=True)
    originals = [p.detach().clone() for p in model.parameters()] if passes == 2 and method != "avg2" else None
    calls = 0
    try:
        if passes == 2:
            first = descent if method in ("sam", "asam") else ascent
            _, n = backward_mean(model, first, microbatch, coefficient=.5 if method == "avg2" else 1., **options)
            calls += n
            grad_norm(model)
            if method != "avg2":
                perturb(model, "asam" if method == "asam_independent" else method, settings["rho"], generator)
                optimizer.zero_grad(set_to_none=True)
        loss, n = backward_mean(model, descent, microbatch,
                                coefficient=.5 if method == "avg2" else 1., **options)
        calls += n
        norm = float(grad_norm(model))
    finally:
        if originals is not None:
            with torch.no_grad():
                for p, saved in zip(model.parameters(), originals):
                    p.copy_(saved)
    optimizer.step()
    return {"loss": loss, "gradient_norm": norm, "pairs": passes,
            "micro_backward_calls": calls, "forward_backward_examples": passes*len(descent[1]),
            "input_vjps": n if method == "input_gradient" else 0}


def parameter_vector(model):
    return torch.cat([p.detach().flatten() for p in model.parameters()])


def gradient_vector(model):
    return torch.cat([(p.grad.detach() if p.grad is not None else torch.zeros_like(p)).flatten()
                      for p in model.parameters()])


def assign_vector(model, vector):
    offset = 0
    with torch.no_grad():
        for p in model.parameters():
            p.copy_(vector[offset:offset+p.numel()].view_as(p))
            offset += p.numel()
    if offset != vector.numel():
        raise ValueError("Parameter-vector size mismatch.")


@torch.no_grad()
def transfer_error_bound(model, x, scales=None):
    """A posteriori componentwise bound in real arithmetic; floating evaluation is not formal certification.

    Uses hard trajectories, a signed reset-Jacobian bound, and class-specific readout rows.
    Different hard thresholds/reset magnitudes are covered by the explicit reset residual.
    """
    if x.dtype != torch.float64:
        raise ValueError("Evaluate transfer bounds in float64.")
    soft_logits = model(x)
    hard_logits, hard = model(x, mode="hard", scales=scales, trace=True, return_info=True)
    errors = torch.zeros_like(x)
    for layer, (block, saved) in enumerate(zip(model.blocks, hard["traces"])):
        weights = block.weight.abs()
        incoming = errors.flatten(0, 1)
        drive = F.conv2d(incoming, weights, padding=1) if model.vision else F.linear(incoming, weights)
        if model.vision:
            drive = F.avg_pool2d(drive, 2)
        drive = drive.reshape(x.shape[0], x.shape[1], *drive.shape[1:])
        theta_s, theta_h = model.threshold, saved["threshold"]
        previous_error = previous_u = previous_h = None
        output_errors = []
        for t in range(x.shape[1]):
            u_h, spike_h = saved["u"][:, t], saved["s"][:, t]
            e = drive[:, t]
            if previous_error is not None:
                lo, hi = previous_u-previous_error-theta_s, previous_u+previous_error-theta_s
                d_lo = model.slope/(math.pi*(1+(model.slope*lo).square()))
                d_hi = model.slope/(math.pi*(1+(model.slope*hi).square()))
                dmin, dmax = torch.minimum(d_lo, d_hi), torch.maximum(d_lo, d_hi)
                dmax = torch.where((lo <= 0)&(hi >= 0), model.slope/math.pi, dmax)
                q = torch.maximum((model.leak-theta_s*dmin).abs(), (model.leak-theta_s*dmax).abs())
                reset_residual = (theta_h*previous_h-theta_s*smooth(previous_u-theta_s, model.slope)).abs()
                e = e+q*previous_error+reset_residual
            a = (smooth(u_h-e-theta_s, model.slope)-spike_h).abs()
            b = (smooth(u_h+e-theta_s, model.slope)-spike_h).abs()
            output_errors.append(torch.maximum(a, b))
            previous_error, previous_u, previous_h = e, u_h, spike_h
        errors = torch.stack(output_errors, 1)
    feature_error = errors.mean(1).flatten(1)
    winner = soft_logits.argmax(1)
    rows = model.readout.weight[winner, None, :]-model.readout.weight[None, :, :]
    difference_bounds = (rows.abs()*feature_error[:, None, :]).sum(2)
    margins = soft_logits.gather(1, winner[:, None])-soft_logits
    mask = F.one_hot(winner, soft_logits.shape[1]).bool()
    safe = ((margins > difference_bounds)|mask).all(1)
    actual_error = (soft_logits-hard_logits).abs().amax(1)
    logit_bound = F.linear(feature_error, model.readout.weight.abs()).amax(1)
    return {"agreement_condition": safe, "observed_agreement": winner == hard_logits.argmax(1),
            "logit_bound": logit_bound, "observed_logit_error": actual_error,
            "feature_error_bound": feature_error,
            "observed_feature_error": (model(x, return_info=True)[1]["features"][-1]-hard["features"][-1]).abs()
            if hard["features"] else None}
