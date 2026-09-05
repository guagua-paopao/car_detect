from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TruckHierarchy:
    generic: int
    light: int
    heavy: int

    @property
    def family(self) -> tuple[int, int, int]:
        return self.generic, self.light, self.heavy


def truck_hierarchy(labels: list[str]) -> TruckHierarchy:
    required = ("truck", "light_truck", "heavy_truck")
    missing = [label for label in required if label not in labels]
    if missing:
        raise ValueError(f"truck hierarchy labels are missing: {missing}")
    return TruckHierarchy(*(labels.index(label) for label in required))


def car_family_indices(labels: list[str]) -> tuple[int, ...]:
    """Return deployable fine classes compatible with an official coarse car box."""

    required = ("sedan", "suv", "mpv", "other")
    missing = [label for label in required if label not in labels]
    if missing:
        raise ValueError(f"car family labels are missing: {missing}")
    return tuple(labels.index(label) for label in required)


def coarse_truck_family_indices(labels: list[str]) -> tuple[int, ...]:
    """Return available truck-family outputs without requiring a generic class.

    Production taxonomy v1 has only light/heavy truck outputs, while the
    offline taxonomy v2 also has a generic truck fallback.  Official coarse
    truck truth may supervise the summed probability mass in either taxonomy.
    """

    required_fine = ("light_truck", "heavy_truck")
    missing = [label for label in required_fine if label not in labels]
    if missing:
        raise ValueError(f"coarse truck family labels are missing: {missing}")
    members = [label for label in ("truck", *required_fine) if label in labels]
    return tuple(labels.index(label) for label in members)


def coarse_color_group_indices(labels: list[str]) -> dict[int, tuple[int, ...]]:
    """Map legacy merged CCTV color truth onto compatible v2 outputs.

    A legacy ``silver_gray`` label cannot safely be fabricated as either gray
    or silver.  It therefore supervises only their summed probability mass.
    The v2 deployment taxonomy has no orange or beige output, so the audited
    merged yellow/orange and brown/beige groups map to the corresponding
    deployable yellow and brown classes.
    """

    specifications = {
        1: ("gray", "silver"),
        2: ("yellow",),
        3: ("brown",),
    }
    missing = sorted(
        {
            label
            for members in specifications.values()
            for label in members
            if label not in labels
        }
    )
    if missing:
        raise ValueError(f"coarse color group labels are missing: {missing}")
    return {
        code: tuple(labels.index(label) for label in members)
        for code, members in specifications.items()
    }


def partial_coarse_family_loss(
    logits,
    family_indices: tuple[int, ...],
    sample_weights=None,
    gamma: float = 0.0,
):
    """Supervise probability mass for a coarse family without fabricating a class."""

    import torch

    if logits.ndim != 2:
        raise ValueError("logits must have shape [batch, classes]")
    if not family_indices:
        raise ValueError("family_indices must not be empty")
    if min(family_indices) < 0 or max(family_indices) >= logits.shape[1]:
        raise ValueError("family index is outside the classifier output")
    index = torch.tensor(family_indices, device=logits.device, dtype=torch.long)
    log_probabilities = torch.log_softmax(logits, dim=1)
    family_log_probability = torch.logsumexp(
        log_probabilities.index_select(1, index), dim=1
    )
    values = -family_log_probability
    if gamma > 0:
        values = values * (1.0 - family_log_probability.exp()).clamp_min(1e-6).pow(gamma)
    if sample_weights is not None:
        weighted = values * sample_weights
        return weighted.sum() / sample_weights.sum().clamp_min(1e-6)
    return values.mean()


def partial_label_group_loss(
    logits,
    group_targets,
    groups: dict[int, tuple[int, ...]],
    sample_weights=None,
    gamma: float = 0.0,
):
    """Supervise per-row probability mass for one of several partial groups."""

    import torch

    if logits.ndim != 2:
        raise ValueError("logits must have shape [batch, classes]")
    if group_targets.ndim != 1 or group_targets.shape[0] != logits.shape[0]:
        raise ValueError("group targets must have one value per logit row")
    if not groups:
        raise ValueError("groups must not be empty")
    if sample_weights is not None and sample_weights.shape != group_targets.shape:
        raise ValueError("sample weights must match group targets")
    loss_dtype = torch.promote_types(logits.dtype, torch.float32)
    values = torch.empty(logits.shape[0], device=logits.device, dtype=loss_dtype)
    assigned = torch.zeros(logits.shape[0], device=logits.device, dtype=torch.bool)
    log_probabilities = torch.log_softmax(logits.to(loss_dtype), dim=1)
    for code, member_indices in groups.items():
        mask = group_targets == code
        if not mask.any():
            continue
        if not member_indices:
            raise ValueError(f"partial group {code} has no members")
        if min(member_indices) < 0 or max(member_indices) >= logits.shape[1]:
            raise ValueError(f"partial group {code} index is outside the classifier output")
        index = torch.tensor(member_indices, device=logits.device, dtype=torch.long)
        group_log_probability = torch.logsumexp(
            log_probabilities[mask].index_select(1, index), dim=1
        )
        group_values = -group_log_probability
        if gamma > 0:
            group_values = group_values * (
                1.0 - group_log_probability.exp()
            ).clamp_min(1e-6).pow(gamma)
        values[mask] = group_values
        assigned[mask] = True
    if not assigned.all():
        unknown = sorted(set(group_targets[~assigned].detach().cpu().tolist()))
        raise ValueError(f"unknown partial group targets: {unknown}")
    if sample_weights is not None:
        weights = sample_weights.to(device=logits.device, dtype=loss_dtype)
        weighted = values * weights
        return weighted.sum() / weights.sum().clamp_min(1e-6)
    return values.mean()


def partial_truck_family_loss(
    logits,
    targets,
    hierarchy: TruckHierarchy,
    ordinary_loss_values,
    sample_weights=None,
    coarse_class_weight=None,
    gamma: float = 0.0,
):
    """Replace generic-truck CE with a coarse truck-family likelihood.

    A row whose official truth is only ``truck`` must not be fabricated as a
    visual class disjoint from light/heavy truck.  It supervises the summed
    probability of the whole truck family.  Fine rows keep their ordinary
    exact-class loss.
    """

    import torch

    if ordinary_loss_values.ndim != 1 or targets.ndim != 1:
        raise ValueError("loss values and targets must be one-dimensional")
    if ordinary_loss_values.shape[0] != targets.shape[0]:
        raise ValueError("loss values and targets must have equal length")
    values = ordinary_loss_values.clone()
    generic_mask = targets == hierarchy.generic
    if generic_mask.any():
        log_probabilities = torch.log_softmax(logits[generic_mask], dim=1)
        family_index = torch.tensor(
            hierarchy.family, device=logits.device, dtype=torch.long
        )
        family_log_probability = torch.logsumexp(
            log_probabilities.index_select(1, family_index), dim=1
        )
        coarse_values = -family_log_probability
        if gamma > 0:
            coarse_values = coarse_values * (
                1.0 - family_log_probability.exp()
            ).clamp_min(1e-6).pow(gamma)
        if coarse_class_weight is not None:
            coarse_values = coarse_values * coarse_class_weight
        values[generic_mask] = coarse_values
    if sample_weights is not None:
        weighted = values * sample_weights
        return weighted.sum() / sample_weights.sum().clamp_min(1e-6)
    return values.mean()


def decode_truck_family(logits, hierarchy: TruckHierarchy, subtype_threshold: float = 0.65):
    """Decode a hierarchical truck family while preserving fail-closed fallback.

    If truck-family probability beats every outside class but neither fine
    subtype is sufficiently dominant inside the family, the result is the
    coarse ``truck`` label.  Confidence is the family mass for a coarse result
    and the exact class probability for a fine result.
    """

    import torch

    if logits.ndim != 2:
        raise ValueError("logits must have shape [batch, classes]")
    if not 0.0 < subtype_threshold <= 1.0:
        raise ValueError("subtype_threshold must be in (0, 1]")
    probabilities = torch.softmax(logits, dim=1)
    predictions = probabilities.argmax(dim=1)
    confidences = probabilities.amax(dim=1)
    family_index = torch.tensor(hierarchy.family, device=logits.device, dtype=torch.long)
    outside_mask = torch.ones(logits.shape[1], device=logits.device, dtype=torch.bool)
    outside_mask[family_index] = False
    family_probabilities = probabilities.index_select(1, family_index)
    family_mass = family_probabilities.sum(dim=1)
    outside_best = (
        probabilities[:, outside_mask].amax(dim=1)
        if outside_mask.any()
        else torch.zeros_like(family_mass)
    )
    choose_family = family_mass > outside_best
    subtype_index = torch.tensor(
        (hierarchy.light, hierarchy.heavy), device=logits.device, dtype=torch.long
    )
    subtype_probabilities = probabilities.index_select(1, subtype_index)
    best_subtype_probability, best_subtype_offset = subtype_probabilities.max(dim=1)
    conditional_subtype = best_subtype_probability / family_mass.clamp_min(1e-12)
    choose_subtype = choose_family & (conditional_subtype >= subtype_threshold)
    choose_generic = choose_family & ~choose_subtype
    predictions = predictions.clone()
    confidences = confidences.clone()
    predictions[choose_subtype] = subtype_index[best_subtype_offset[choose_subtype]]
    confidences[choose_subtype] = best_subtype_probability[choose_subtype]
    predictions[choose_generic] = hierarchy.generic
    confidences[choose_generic] = family_mass[choose_generic]
    return predictions, confidences


def truck_family_match(prediction: int, target: int, hierarchy: TruckHierarchy) -> bool:
    """Return semantic correctness when either side has only coarse truth."""

    if prediction == target:
        return True
    family = set(hierarchy.family)
    return prediction in family and target in family and (
        prediction == hierarchy.generic or target == hierarchy.generic
    )
