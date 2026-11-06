use std::cmp::Ordering;
use std::time::{Duration, Instant};

use crate::constraints::{NativeConstraints, WholeSlatePredicate};

// Bound native final-materialization work between cooperative deadline checks.
// Python callbacks may still exceed the budget because they cannot be safely
// interrupted while executing.
const PREDICATE_BATCH_CHUNK: usize = 16;

#[derive(Debug, Clone)]
pub struct OracleConfig {
    pub max_items: usize,
    pub structural_upper_bound: usize,
    pub structural_upper_source: String,
    pub beam_width: usize,
    pub max_expansions: usize,
    pub timeout: Duration,
    pub confidence_radius: f64,
    pub safety_threshold: f64,
}

#[derive(Debug, Clone)]
pub struct SlateInputs {
    pub rewards: Vec<f64>,
    pub safety_means: Vec<f64>,
    pub safety_variances: Vec<f64>,
    pub covariance: Option<Vec<Vec<f64>>>,
    pub linear_features: Option<Vec<Vec<f64>>>,
    pub linear_transformed_features: Option<Vec<Vec<f64>>>,
}

#[derive(Debug, Clone)]
pub struct OracleOutcome {
    pub selected_indices: Vec<usize>,
    pub k_lower: usize,
    pub k_upper: usize,
    pub cardinality_gap: usize,
    pub first_unresolved_depth: Option<usize>,
    pub upper_bound_source: String,
    pub reward: f64,
    pub safety_mean: f64,
    pub safety_variance: f64,
    pub safety_lcb: f64,
    pub status: String,
    pub stop_reason: String,
    pub exact: bool,
    pub rechecked: bool,
    pub deadline_reached: bool,
    pub expansions: usize,
    pub deepest_completed_depth: usize,
    pub incumbent_updates: usize,
    pub predicate_calls: usize,
    pub predicate_batches: usize,
    pub predicate_errors: usize,
    pub elapsed_seconds: f64,
}

#[derive(Debug, Clone)]
struct State {
    selected: Vec<usize>,
    reward: f64,
    safety_mean: f64,
    safety_variance: f64,
    safety_lcb: f64,
}

impl State {
    fn baseline() -> Self {
        Self {
            selected: Vec::new(),
            reward: 0.0,
            safety_mean: 0.0,
            safety_variance: 0.0,
            safety_lcb: 0.0,
        }
    }
}

#[derive(Debug, Default)]
struct SearchStats {
    expansions: usize,
    deepest_completed_depth: usize,
    incumbent_updates: usize,
    predicate_calls: usize,
    predicate_batches: usize,
    predicate_errors: usize,
    deadline_reached: bool,
    stop_reason: String,
    complete: bool,
}

pub fn optimize(
    inputs: &SlateInputs,
    config: &OracleConfig,
    constraints: &NativeConstraints,
    predicate: &mut dyn WholeSlatePredicate,
) -> Result<OracleOutcome, String> {
    validate_inputs(inputs, config)?;
    let item_count = inputs.rewards.len();
    let max_depth = config
        .max_items
        .min(config.structural_upper_bound)
        .min(item_count);
    let started = Instant::now();
    let deadline = started + config.timeout;
    let mut stats = SearchStats {
        complete: true,
        stop_reason: "search_exhausted".to_owned(),
        ..SearchStats::default()
    };
    let mut incumbent = State::baseline();

    greedy_floor(
        inputs,
        config,
        constraints,
        predicate,
        deadline,
        &mut incumbent,
        &mut stats,
    );

    if stats.expansions < config.max_expansions && Instant::now() < deadline {
        beam_search(
            inputs,
            config,
            constraints,
            predicate,
            deadline,
            &mut incumbent,
            &mut stats,
        );
    } else {
        stats.complete = false;
        if Instant::now() >= deadline {
            stats.deadline_reached = true;
            stats.stop_reason = "deadline".to_owned();
        } else {
            stats.stop_reason = "expansion_budget".to_owned();
        }
    }

    if Instant::now() >= deadline {
        stats.deadline_reached = true;
    }

    // Final checks are mandatory and are not skipped when the search budget is
    // exhausted. Predicate exceptions fail closed to the registered baseline.
    let mut rechecked =
        constraints.accepts(&incumbent.selected) && incumbent.safety_lcb >= config.safety_threshold;
    if rechecked && !incumbent.selected.is_empty() {
        stats.predicate_calls += 1;
        stats.predicate_batches += 1;
        match predicate.accepts(&incumbent.selected) {
            Ok(accepted) => rechecked = accepted,
            Err(_) => {
                record_predicate_error(&mut stats);
                rechecked = false;
            }
        }
    }
    if !rechecked {
        stats.complete = false;
        if stats.stop_reason == "search_exhausted" {
            stats.stop_reason = "final_recheck_failed".to_owned();
        }
        incumbent = State::baseline();
        // The empty slate is the registered unchanged baseline. The public
        // API requires a relative threshold <= 0, so this fallback is safe.
        rechecked = true;
    }

    let exact = stats.complete && rechecked;
    let k_lower = incumbent.selected.len();
    let k_upper = if exact { k_lower } else { max_depth };
    let cardinality_gap = k_upper.saturating_sub(k_lower);
    let first_unresolved_depth = (!exact && k_lower < k_upper).then_some(k_lower + 1);
    let upper_bound_source = if exact {
        "exact".to_owned()
    } else {
        config.structural_upper_source.clone()
    };
    let status = if k_lower == 0 {
        "baseline"
    } else if exact {
        "exact"
    } else {
        "bounded"
    };

    Ok(OracleOutcome {
        selected_indices: incumbent.selected,
        k_lower,
        k_upper,
        cardinality_gap,
        first_unresolved_depth,
        upper_bound_source,
        reward: incumbent.reward,
        safety_mean: incumbent.safety_mean,
        safety_variance: incumbent.safety_variance,
        safety_lcb: incumbent.safety_lcb,
        status: status.to_owned(),
        stop_reason: stats.stop_reason,
        exact,
        rechecked,
        deadline_reached: stats.deadline_reached,
        expansions: stats.expansions,
        deepest_completed_depth: stats.deepest_completed_depth,
        incumbent_updates: stats.incumbent_updates,
        predicate_calls: stats.predicate_calls,
        predicate_batches: stats.predicate_batches,
        predicate_errors: stats.predicate_errors,
        elapsed_seconds: started.elapsed().as_secs_f64(),
    })
}

fn greedy_floor(
    inputs: &SlateInputs,
    config: &OracleConfig,
    constraints: &NativeConstraints,
    predicate: &mut dyn WholeSlatePredicate,
    deadline: Instant,
    incumbent: &mut State,
    stats: &mut SearchStats,
) {
    let mut order: Vec<usize> = (0..inputs.rewards.len()).collect();
    order.sort_by(|&left, &right| {
        inputs.rewards[right]
            .total_cmp(&inputs.rewards[left])
            .then_with(|| left.cmp(&right))
    });
    let mut selected = Vec::new();

    for item in order {
        if selected.len() >= config.max_items
            || stats.expansions >= config.max_expansions
            || Instant::now() >= deadline
        {
            break;
        }
        let mut candidate = selected.clone();
        candidate.push(item);
        candidate.sort_unstable();
        stats.expansions += 1;
        if !constraints.accepts(&candidate) {
            continue;
        }
        let state = evaluate(inputs, config, candidate);
        if state.safety_lcb < config.safety_threshold {
            continue;
        }
        stats.predicate_calls += 1;
        stats.predicate_batches += 1;
        match predicate.accepts(&state.selected) {
            Ok(true) => {
                selected = state.selected.clone();
                if better(&state, incumbent) {
                    *incumbent = state;
                    stats.incumbent_updates += 1;
                }
            }
            Ok(false) => {}
            Err(_) => record_predicate_error(stats),
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn beam_search(
    inputs: &SlateInputs,
    config: &OracleConfig,
    constraints: &NativeConstraints,
    predicate: &mut dyn WholeSlatePredicate,
    deadline: Instant,
    incumbent: &mut State,
    stats: &mut SearchStats,
) {
    let item_count = inputs.rewards.len();
    let max_depth = config
        .max_items
        .min(config.structural_upper_bound)
        .min(item_count);
    let mut beam = vec![State::baseline()];

    for depth in 1..=max_depth {
        let mut next = Vec::new();
        let mut safe_candidates = Vec::new();
        let mut interrupted = false;
        for state in &beam {
            let start_item = state.selected.last().map_or(0, |item| item + 1);
            for item in start_item..item_count {
                if stats.expansions >= config.max_expansions {
                    stats.complete = false;
                    stats.stop_reason = "expansion_budget".to_owned();
                    interrupted = true;
                    break;
                }
                if Instant::now() >= deadline {
                    stats.complete = false;
                    stats.deadline_reached = true;
                    stats.stop_reason = "deadline".to_owned();
                    interrupted = true;
                    break;
                }

                let mut selected = state.selected.clone();
                selected.push(item);
                stats.expansions += 1;
                if !constraints.accepts(&selected) {
                    continue;
                }
                let candidate = evaluate(inputs, config, selected);
                if candidate.safety_lcb >= config.safety_threshold {
                    safe_candidates.push(candidate.clone());
                }
                // A black-box predicate and the safety score can be
                // non-monotone, so neither rejection prunes this descendant.
                next.push(candidate);
            }
            if interrupted {
                break;
            }
        }
        if !safe_candidates.is_empty() {
            for candidates in safe_candidates.chunks(PREDICATE_BATCH_CHUNK) {
                if Instant::now() >= deadline {
                    stats.complete = false;
                    stats.deadline_reached = true;
                    stats.stop_reason = "deadline".to_owned();
                    interrupted = true;
                    break;
                }

                let selected: Vec<Vec<usize>> = candidates
                    .iter()
                    .map(|candidate| candidate.selected.clone())
                    .collect();
                stats.predicate_calls += selected.len();
                stats.predicate_batches += 1;
                let decisions = predicate.accepts_many(&selected);
                if decisions.len() != candidates.len() {
                    for _ in 0..candidates.len() {
                        record_predicate_error(stats);
                    }
                } else {
                    for (candidate, decision) in candidates.iter().zip(decisions) {
                        match decision {
                            Ok(true) if better(candidate, incumbent) => {
                                *incumbent = candidate.clone();
                                stats.incumbent_updates += 1;
                            }
                            Ok(_) => {}
                            Err(_) => record_predicate_error(stats),
                        }
                    }
                }
            }
        }
        if interrupted {
            break;
        }
        stats.deepest_completed_depth = depth;
        if next.is_empty() {
            break;
        }
        next.sort_by(state_search_order);
        if depth < max_depth && next.len() > config.beam_width {
            next.truncate(config.beam_width);
            stats.complete = false;
        }
        beam = next;
    }

    if !stats.complete && stats.stop_reason == "search_exhausted" {
        stats.stop_reason = "beam_truncated".to_owned();
    }
}

fn record_predicate_error(stats: &mut SearchStats) {
    stats.predicate_errors += 1;
    stats.complete = false;
    if stats.stop_reason == "search_exhausted" {
        stats.stop_reason = "predicate_error".to_owned();
    }
}

fn state_search_order(left: &State, right: &State) -> Ordering {
    right
        .safety_lcb
        .total_cmp(&left.safety_lcb)
        .then_with(|| right.reward.total_cmp(&left.reward))
        .then_with(|| left.selected.cmp(&right.selected))
}

fn better(candidate: &State, incumbent: &State) -> bool {
    candidate.selected.len() > incumbent.selected.len()
        || (candidate.selected.len() == incumbent.selected.len()
            && candidate.reward.total_cmp(&incumbent.reward) == Ordering::Greater)
}

fn evaluate(inputs: &SlateInputs, config: &OracleConfig, selected: Vec<usize>) -> State {
    let reward = selected.iter().map(|&item| inputs.rewards[item]).sum();
    let safety_mean = selected.iter().map(|&item| inputs.safety_means[item]).sum();
    let safety_variance = if let (Some(features), Some(transformed_features)) =
        (&inputs.linear_features, &inputs.linear_transformed_features)
    {
        let feature_dimension = features.first().map_or(0, Vec::len);
        let mut feature_sum = vec![0.0; feature_dimension];
        let mut transformed_sum = vec![0.0; feature_dimension];
        for &item in &selected {
            for dimension in 0..feature_dimension {
                feature_sum[dimension] += features[item][dimension];
                transformed_sum[dimension] += transformed_features[item][dimension];
            }
        }
        feature_sum
            .iter()
            .zip(&transformed_sum)
            .map(|(feature, transformed)| feature * transformed)
            .sum::<f64>()
            .max(0.0)
    } else if let Some(covariance) = &inputs.covariance {
        selected
            .iter()
            .flat_map(|&left| selected.iter().map(move |&right| covariance[left][right]))
            .sum::<f64>()
            .max(0.0)
    } else {
        selected
            .iter()
            .map(|&item| inputs.safety_variances[item])
            .sum()
    };
    let safety_lcb = safety_mean - config.confidence_radius * safety_variance.sqrt();
    State {
        selected,
        reward,
        safety_mean,
        safety_variance,
        safety_lcb,
    }
}

fn validate_inputs(inputs: &SlateInputs, config: &OracleConfig) -> Result<(), String> {
    let item_count = inputs.rewards.len();
    if inputs.safety_means.len() != item_count || inputs.safety_variances.len() != item_count {
        return Err(
            "rewards, safety_means, and safety_variances must have equal length".to_owned(),
        );
    }
    if config.beam_width == 0 || config.max_expansions == 0 {
        return Err("beam_width and max_expansions must be positive".to_owned());
    }
    if config.structural_upper_bound > item_count {
        return Err("structural_upper_bound cannot exceed the operation count".to_owned());
    }
    if !config.confidence_radius.is_finite() || config.confidence_radius < 0.0 {
        return Err("confidence_radius must be finite and nonnegative".to_owned());
    }
    if !config.safety_threshold.is_finite() || config.safety_threshold > 0.0 {
        return Err(
            "safety_threshold must be finite and <= 0 because the unchanged baseline has contrast 0"
                .to_owned(),
        );
    }
    for (name, values) in [
        ("rewards", &inputs.rewards),
        ("safety_means", &inputs.safety_means),
        ("safety_variances", &inputs.safety_variances),
    ] {
        if values.iter().any(|value| !value.is_finite()) {
            return Err(format!("{name} contains NaN or infinity"));
        }
    }
    if inputs
        .safety_variances
        .iter()
        .any(|variance| *variance < 0.0)
    {
        return Err("safety_variances must be nonnegative".to_owned());
    }
    if let Some(covariance) = &inputs.covariance {
        validate_covariance(covariance, &inputs.safety_variances)?;
    }
    validate_linear_factor(inputs)?;
    Ok(())
}

fn validate_linear_factor(inputs: &SlateInputs) -> Result<(), String> {
    match (&inputs.linear_features, &inputs.linear_transformed_features) {
        (None, None) => Ok(()),
        (Some(_), None) | (None, Some(_)) => Err(
            "linear_features and linear_transformed_features must be supplied together".to_owned(),
        ),
        (Some(features), Some(transformed)) => {
            if inputs.covariance.is_some() {
                return Err(
                    "full covariance and factorized linear uncertainty are mutually exclusive"
                        .to_owned(),
                );
            }
            let operation_count = inputs.rewards.len();
            if features.len() != operation_count || transformed.len() != operation_count {
                return Err(format!(
                    "linear uncertainty must have {operation_count} operation rows"
                ));
            }
            let feature_dimension = features.first().map_or(0, Vec::len);
            if features.iter().any(|row| row.len() != feature_dimension)
                || transformed.iter().any(|row| row.len() != feature_dimension)
            {
                return Err("linear uncertainty rows have inconsistent dimensions".to_owned());
            }
            if features
                .iter()
                .chain(transformed.iter())
                .flat_map(|row| row.iter())
                .any(|value| !value.is_finite())
            {
                return Err("linear uncertainty contains NaN or infinity".to_owned());
            }
            let variance_scale = inputs
                .safety_variances
                .iter()
                .copied()
                .fold(1.0_f64, f64::max);
            let tolerance = 1e-10 * variance_scale;
            for operation in 0..operation_count {
                let variance = features[operation]
                    .iter()
                    .zip(&transformed[operation])
                    .map(|(feature, transformed_feature)| feature * transformed_feature)
                    .sum::<f64>()
                    .max(0.0);
                if (variance - inputs.safety_variances[operation]).abs() > tolerance {
                    return Err(
                        "factorized linear uncertainty diagonal must match safety_variances"
                            .to_owned(),
                    );
                }
            }
            Ok(())
        }
    }
}

fn validate_covariance(covariance: &[Vec<f64>], variances: &[f64]) -> Result<(), String> {
    let size = variances.len();
    if covariance.len() != size || covariance.iter().any(|row| row.len() != size) {
        return Err(format!("covariance must have shape {size} x {size}"));
    }
    let scale = variances.iter().copied().fold(1.0_f64, f64::max);
    let tolerance = 1e-10 * scale;
    for row in covariance {
        if row.iter().any(|value| !value.is_finite()) {
            return Err("covariance contains NaN or infinity".to_owned());
        }
    }
    for (index, (row, variance)) in covariance.iter().zip(variances).enumerate() {
        if row[index] < -tolerance {
            return Err("covariance has a negative diagonal entry".to_owned());
        }
        if (row[index] - variance).abs() > tolerance {
            return Err("covariance diagonal must match safety_variances".to_owned());
        }
        for (other, other_row) in covariance.iter().enumerate().skip(index + 1) {
            if (row[other] - other_row[index]).abs() > tolerance {
                return Err("covariance must be symmetric".to_owned());
            }
        }
    }

    // Pivoted-free LDL^T check for positive semidefiniteness. Zero pivots may
    // only have zero residual cross terms.
    let mut lower = vec![vec![0.0; size]; size];
    let mut diagonal = vec![0.0; size];
    for i in 0..size {
        for j in 0..i {
            let residual = covariance[i][j]
                - (0..j)
                    .map(|k| lower[i][k] * diagonal[k] * lower[j][k])
                    .sum::<f64>();
            if diagonal[j].abs() <= tolerance {
                if residual.abs() > tolerance {
                    return Err("covariance is not positive semidefinite".to_owned());
                }
                lower[i][j] = 0.0;
            } else {
                lower[i][j] = residual / diagonal[j];
            }
        }
        let pivot = covariance[i][i]
            - (0..i)
                .map(|k| lower[i][k] * lower[i][k] * diagonal[k])
                .sum::<f64>();
        if pivot < -tolerance {
            return Err("covariance is not positive semidefinite".to_owned());
        }
        diagonal[i] = pivot.max(0.0);
        lower[i][i] = 1.0;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::constraints::AcceptAll;
    use std::collections::HashMap;

    #[test]
    fn exact_search_is_cardinality_first() {
        let inputs = SlateInputs {
            rewards: vec![100.0, -1_000.0, 2.0],
            safety_means: vec![1.0, 1.0, 1.0],
            safety_variances: vec![0.0; 3],
            covariance: None,
            linear_features: None,
            linear_transformed_features: None,
        };
        let config = OracleConfig {
            max_items: 2,
            structural_upper_bound: 3,
            structural_upper_source: "test_bound".to_owned(),
            beam_width: 16,
            max_expansions: 1_000,
            timeout: Duration::from_secs(1),
            confidence_radius: 1.96,
            safety_threshold: 0.0,
        };
        let constraints = NativeConstraints::new(3, None, HashMap::new(), vec![]).unwrap();
        let mut predicate = AcceptAll;
        let result = optimize(&inputs, &config, &constraints, &mut predicate).unwrap();
        assert!(result.exact);
        assert_eq!(result.selected_indices, vec![0, 2]);
        assert_eq!(result.k_lower, 2);
        assert_eq!(result.k_upper, 2);
    }

    #[test]
    fn covariance_validation_rejects_indefinite_matrix() {
        let covariance = vec![vec![1.0, 2.0], vec![2.0, 1.0]];
        assert!(validate_covariance(&covariance, &[1.0, 1.0]).is_err());
    }
}
