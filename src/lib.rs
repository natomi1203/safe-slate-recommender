mod constraints;
mod oracle;
mod uncertainty;

use std::collections::{HashMap, HashSet};
use std::time::Duration;

use constraints::{AcceptAll, NativeConstraints, PartitionConstraint, WholeSlatePredicate};
use oracle::{OracleConfig, OracleOutcome, SlateInputs};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyDict, PyList, PyModule};
use sha2::{Digest, Sha256};
use uncertainty::{prepare_linear_contrasts, project_linear_contrasts};

type PreparedFactorizedInput = (Option<Vec<Vec<f64>>>, Option<Vec<Vec<f64>>>);

#[pyclass(frozen, module = "safe_slate._core")]
#[derive(Clone)]
pub struct ProjectedContrastUncertainty {
    #[pyo3(get)]
    covariance: Vec<Vec<f64>>,
    #[pyo3(get)]
    safety_variances: Vec<f64>,
    #[pyo3(get)]
    individual_lcbs: Vec<f64>,
    #[pyo3(get)]
    feature_dimension: usize,
}

#[pyclass(frozen, module = "safe_slate._core")]
#[derive(Clone)]
pub struct PreparedLinearContrastMarginals {
    #[pyo3(get)]
    safety_variances: Vec<f64>,
    #[pyo3(get)]
    feature_dimension: usize,
    #[pyo3(get)]
    contrast_features_sha256: String,
    #[pyo3(get)]
    parameter_covariance_sha256: String,
    #[pyo3(get)]
    matrix_digest_encoding: String,
}

#[pymethods]
impl PreparedLinearContrastMarginals {
    fn __repr__(&self) -> String {
        format!(
            "PreparedLinearContrastMarginals(operation_count={}, feature_dimension={}, matrix_digest_encoding='{}')",
            self.safety_variances.len(),
            self.feature_dimension,
            self.matrix_digest_encoding
        )
    }
}

#[pymethods]
impl ProjectedContrastUncertainty {
    fn __repr__(&self) -> String {
        format!(
            "ProjectedContrastUncertainty(operation_count={}, feature_dimension={})",
            self.safety_variances.len(),
            self.feature_dimension
        )
    }
}

/// Materialize the full operation covariance for small audits and debugging.
/// Production linear re-ranking keeps the same uncertainty factorized.
#[pyfunction]
#[pyo3(signature = (
    safety_means,
    contrast_features,
    parameter_covariance,
    *,
    confidence_radius=1.96
))]
fn project_linear_contrast_uncertainty(
    safety_means: Vec<f64>,
    contrast_features: Vec<Vec<f64>>,
    parameter_covariance: Vec<Vec<f64>>,
    confidence_radius: f64,
) -> PyResult<ProjectedContrastUncertainty> {
    let projection = project_linear_contrasts(
        &safety_means,
        &contrast_features,
        &parameter_covariance,
        confidence_radius,
    )
    .map_err(PyValueError::new_err)?;
    Ok(ProjectedContrastUncertainty {
        covariance: projection.covariance,
        safety_variances: projection.variances,
        individual_lcbs: projection.individual_lcbs,
        feature_dimension: projection.feature_dimension,
    })
}

/// Return only operation-level marginal variances without an N-by-N matrix.
#[pyfunction]
fn project_linear_contrast_marginals(
    contrast_features: Vec<Vec<f64>>,
    parameter_covariance: Vec<Vec<f64>>,
) -> PyResult<Vec<f64>> {
    Ok(
        prepare_linear_contrasts(&contrast_features, &parameter_covariance)
            .map_err(PyValueError::new_err)?
            .variances,
    )
}

fn matrix_f64le_sha256(matrix: &[Vec<f64>], name: &str) -> Result<String, String> {
    let row_count = matrix.len();
    let column_count = matrix.first().map_or(0, Vec::len);
    if matrix.iter().any(|row| row.len() != column_count) {
        return Err(format!("{name} rows must have equal length"));
    }
    if matrix
        .iter()
        .flat_map(|row| row.iter())
        .any(|value| !value.is_finite())
    {
        return Err(format!("{name} contains NaN or infinity"));
    }
    let rows = u64::try_from(row_count).map_err(|_| format!("{name} has too many rows"))?;
    let columns =
        u64::try_from(column_count).map_err(|_| format!("{name} has too many columns"))?;
    let mut hasher = Sha256::new();
    hasher.update(b"safe-slate-matrix-f64le-v1\0");
    hasher.update(rows.to_le_bytes());
    hasher.update(columns.to_le_bytes());
    for value in matrix.iter().flat_map(|row| row.iter()) {
        hasher.update(value.to_le_bytes());
    }
    Ok(format!("{:x}", hasher.finalize()))
}

/// Project operation-level marginals and compute both immutable matrix
/// identities in one Rust crossing for latency-sensitive serving paths.
#[pyfunction]
fn prepare_linear_contrast_marginals_f64le(
    contrast_features: Vec<Vec<f64>>,
    parameter_covariance: Vec<Vec<f64>>,
) -> PyResult<PreparedLinearContrastMarginals> {
    let prepared = prepare_linear_contrasts(&contrast_features, &parameter_covariance)
        .map_err(PyValueError::new_err)?;
    let contrast_features_sha256 = matrix_f64le_sha256(&contrast_features, "contrast_features")
        .map_err(PyValueError::new_err)?;
    let parameter_covariance_sha256 =
        matrix_f64le_sha256(&parameter_covariance, "parameter_covariance")
            .map_err(PyValueError::new_err)?;
    Ok(PreparedLinearContrastMarginals {
        safety_variances: prepared.variances,
        feature_dimension: prepared.feature_dimension,
        contrast_features_sha256,
        parameter_covariance_sha256,
        matrix_digest_encoding: "f64le-v1".to_owned(),
    })
}

fn prepare_factorized_uncertainty(
    safety_variances: &[f64],
    covariance_present: bool,
    contrast_features: Option<Vec<Vec<f64>>>,
    parameter_covariance: Option<Vec<Vec<f64>>>,
) -> PyResult<PreparedFactorizedInput> {
    match (contrast_features, parameter_covariance) {
        (None, None) => Ok((None, None)),
        (Some(_), None) | (None, Some(_)) => Err(PyValueError::new_err(
            "contrast_features and parameter_covariance must be supplied together",
        )),
        (Some(features), Some(parameter_covariance)) => {
            if covariance_present {
                return Err(PyValueError::new_err(
                    "covariance and factorized linear uncertainty are mutually exclusive",
                ));
            }
            let prepared = prepare_linear_contrasts(&features, &parameter_covariance)
                .map_err(PyValueError::new_err)?;
            if prepared.variances.len() != safety_variances.len() {
                return Err(PyValueError::new_err(format!(
                    "contrast_features has {} rows, expected {}",
                    prepared.variances.len(),
                    safety_variances.len()
                )));
            }
            let scale = safety_variances
                .iter()
                .chain(prepared.variances.iter())
                .copied()
                .fold(1.0_f64, f64::max);
            let tolerance = 1e-10 * scale;
            if safety_variances
                .iter()
                .zip(&prepared.variances)
                .any(|(declared, projected)| (declared - projected).abs() > tolerance)
            {
                return Err(PyValueError::new_err(
                    "factorized linear uncertainty diagonal must match safety_variances",
                ));
            }
            Ok((Some(features), Some(prepared.transformed_features)))
        }
    }
}

struct PythonPredicate {
    callback: Option<Py<PyAny>>,
    batch_callback: Option<Py<PyAny>>,
}

struct ReplacementPredicate {
    callback: Option<Py<PyAny>>,
    batch_callback: Option<Py<PyAny>>,
    remove_positions: Vec<usize>,
    add_item_ids: Vec<usize>,
    baseline_item_ids: Option<Vec<usize>>,
}

impl WholeSlatePredicate for ReplacementPredicate {
    fn accepts(&mut self, selected: &[usize]) -> Result<bool, String> {
        let materialized = materialize_slate(
            self.baseline_item_ids.as_deref(),
            &self.remove_positions,
            &self.add_item_ids,
            selected,
        );
        if self.baseline_item_ids.is_some() && materialized.is_none() {
            return Ok(false);
        }
        if let Some(callback) = &self.callback {
            return Python::attach(|py| {
                let plan = replacement_plan(
                    py,
                    selected,
                    &self.remove_positions,
                    &self.add_item_ids,
                    materialized,
                )?;
                callback
                    .bind(py)
                    .call1((plan,))
                    .and_then(|result| result.extract::<bool>())
                    .map_err(|error| error.to_string())
            });
        }
        if self.batch_callback.is_some() {
            return self
                .accepts_many(&[selected.to_vec()])
                .into_iter()
                .next()
                .unwrap_or_else(|| Err("batch predicate returned no decision".to_owned()));
        }
        Ok(true)
    }

    fn accepts_many(&mut self, candidates: &[Vec<usize>]) -> Vec<Result<bool, String>> {
        let Some(callback) = &self.batch_callback else {
            return candidates
                .iter()
                .map(|candidate| self.accepts(candidate))
                .collect();
        };
        Python::attach(|py| {
            let plans = PyList::empty(py);
            let mut valid_positions = Vec::new();
            let mut decisions = vec![Ok(false); candidates.len()];
            for (position, selected) in candidates.iter().enumerate() {
                let materialized = materialize_slate(
                    self.baseline_item_ids.as_deref(),
                    &self.remove_positions,
                    &self.add_item_ids,
                    selected,
                );
                if self.baseline_item_ids.is_some() && materialized.is_none() {
                    continue;
                }
                match replacement_plan(
                    py,
                    selected,
                    &self.remove_positions,
                    &self.add_item_ids,
                    materialized,
                ) {
                    Ok(plan) => match plans.append(plan) {
                        Ok(()) => valid_positions.push(position),
                        Err(error) => decisions[position] = Err(error.to_string()),
                    },
                    Err(error) => decisions[position] = Err(error),
                }
            }
            if valid_positions.is_empty() {
                return decisions;
            }
            match callback
                .bind(py)
                .call1((plans,))
                .and_then(|result| result.extract::<Vec<bool>>())
            {
                Ok(batch) if batch.len() == valid_positions.len() => {
                    for (position, accepted) in valid_positions.into_iter().zip(batch) {
                        decisions[position] = Ok(accepted);
                    }
                }
                Ok(batch) => {
                    let message = format!(
                        "batch predicate returned {} decisions for {} candidates",
                        batch.len(),
                        valid_positions.len()
                    );
                    for position in valid_positions {
                        decisions[position] = Err(message.clone());
                    }
                }
                Err(error) => {
                    let message = error.to_string();
                    for position in valid_positions {
                        decisions[position] = Err(message.clone());
                    }
                }
            }
            decisions
        })
    }
}

fn replacement_plan<'py>(
    py: Python<'py>,
    selected: &[usize],
    remove_positions: &[usize],
    add_item_ids: &[usize],
    materialized: Option<Vec<usize>>,
) -> Result<Bound<'py, PyDict>, String> {
    let plan = PyDict::new(py);
    let removed: Vec<usize> = selected
        .iter()
        .map(|&operation| remove_positions[operation])
        .collect();
    let added: Vec<usize> = selected
        .iter()
        .map(|&operation| add_item_ids[operation])
        .collect();
    plan.set_item("operation_indices", selected.to_vec())
        .map_err(|error| error.to_string())?;
    plan.set_item("remove_positions", removed)
        .map_err(|error| error.to_string())?;
    plan.set_item("add_item_ids", added)
        .map_err(|error| error.to_string())?;
    plan.set_item("materialized_slate", materialized)
        .map_err(|error| error.to_string())?;
    Ok(plan)
}

fn batch_error_results(count: usize, message: String) -> Vec<Result<bool, String>> {
    (0..count).map(|_| Err(message.clone())).collect()
}

impl WholeSlatePredicate for PythonPredicate {
    fn accepts(&mut self, selected: &[usize]) -> Result<bool, String> {
        if let Some(callback) = &self.callback {
            return Python::attach(|py| {
                callback
                    .bind(py)
                    .call1((selected.to_vec(),))
                    .and_then(|result| result.extract::<bool>())
                    .map_err(|error| error.to_string())
            });
        }
        let decisions = self.accepts_many(&[selected.to_vec()]);
        decisions
            .into_iter()
            .next()
            .unwrap_or_else(|| Err("batch predicate returned no decision".to_owned()))
    }

    fn accepts_many(&mut self, candidates: &[Vec<usize>]) -> Vec<Result<bool, String>> {
        let Some(callback) = &self.batch_callback else {
            return candidates
                .iter()
                .map(|candidate| self.accepts(candidate))
                .collect();
        };
        Python::attach(|py| {
            match callback
                .bind(py)
                .call1((candidates.to_vec(),))
                .and_then(|result| result.extract::<Vec<bool>>())
            {
                Ok(decisions) if decisions.len() == candidates.len() => {
                    decisions.into_iter().map(Ok).collect()
                }
                Ok(decisions) => batch_error_results(
                    candidates.len(),
                    format!(
                        "batch predicate returned {} decisions for {} candidates",
                        decisions.len(),
                        candidates.len()
                    ),
                ),
                Err(error) => batch_error_results(candidates.len(), error.to_string()),
            }
        })
    }
}

#[pyclass(frozen, module = "safe_slate._core")]
#[derive(Clone)]
pub struct CertifiedOracleResult {
    #[pyo3(get)]
    selected_indices: Vec<usize>,
    #[pyo3(get)]
    k_lower: usize,
    #[pyo3(get)]
    k_upper: usize,
    #[pyo3(get)]
    cardinality_gap: usize,
    #[pyo3(get)]
    first_unresolved_depth: Option<usize>,
    #[pyo3(get)]
    upper_bound_source: String,
    #[pyo3(get)]
    reward: f64,
    #[pyo3(get)]
    safety_mean: f64,
    #[pyo3(get)]
    safety_variance: f64,
    #[pyo3(get)]
    safety_lcb: f64,
    #[pyo3(get)]
    status: String,
    #[pyo3(get)]
    stop_reason: String,
    #[pyo3(get)]
    exact: bool,
    #[pyo3(get)]
    rechecked: bool,
    #[pyo3(get)]
    deadline_reached: bool,
    #[pyo3(get)]
    expansions: usize,
    #[pyo3(get)]
    deepest_completed_depth: usize,
    #[pyo3(get)]
    incumbent_updates: usize,
    #[pyo3(get)]
    predicate_calls: usize,
    #[pyo3(get)]
    predicate_batches: usize,
    #[pyo3(get)]
    predicate_errors: usize,
    #[pyo3(get)]
    elapsed_seconds: f64,
}

#[pymethods]
impl CertifiedOracleResult {
    fn __repr__(&self) -> String {
        format!(
            "CertifiedOracleResult(status='{}', k_lower={}, k_upper={}, reward={:.6}, rechecked={})",
            self.status, self.k_lower, self.k_upper, self.reward, self.rechecked
        )
    }
}

impl From<OracleOutcome> for CertifiedOracleResult {
    fn from(value: OracleOutcome) -> Self {
        Self {
            selected_indices: value.selected_indices,
            k_lower: value.k_lower,
            k_upper: value.k_upper,
            cardinality_gap: value.cardinality_gap,
            first_unresolved_depth: value.first_unresolved_depth,
            upper_bound_source: value.upper_bound_source,
            reward: value.reward,
            safety_mean: value.safety_mean,
            safety_variance: value.safety_variance,
            safety_lcb: value.safety_lcb,
            status: value.status,
            stop_reason: value.stop_reason,
            exact: value.exact,
            rechecked: value.rechecked,
            deadline_reached: value.deadline_reached,
            expansions: value.expansions,
            deepest_completed_depth: value.deepest_completed_depth,
            incumbent_updates: value.incumbent_updates,
            predicate_calls: value.predicate_calls,
            predicate_batches: value.predicate_batches,
            predicate_errors: value.predicate_errors,
            elapsed_seconds: value.elapsed_seconds,
        }
    }
}

#[pyclass(frozen, module = "safe_slate._core")]
#[derive(Clone)]
pub struct CertifiedReplacementResult {
    #[pyo3(get)]
    selected_operation_indices: Vec<usize>,
    #[pyo3(get)]
    remove_positions: Vec<usize>,
    #[pyo3(get)]
    add_item_ids: Vec<usize>,
    #[pyo3(get)]
    materialized_slate: Option<Vec<usize>>,
    #[pyo3(get)]
    k_lower: usize,
    #[pyo3(get)]
    k_upper: usize,
    #[pyo3(get)]
    cardinality_gap: usize,
    #[pyo3(get)]
    first_unresolved_depth: Option<usize>,
    #[pyo3(get)]
    upper_bound_source: String,
    #[pyo3(get)]
    reward: f64,
    #[pyo3(get)]
    safety_mean: f64,
    #[pyo3(get)]
    safety_variance: f64,
    #[pyo3(get)]
    safety_lcb: f64,
    #[pyo3(get)]
    status: String,
    #[pyo3(get)]
    stop_reason: String,
    #[pyo3(get)]
    exact: bool,
    #[pyo3(get)]
    rechecked: bool,
    #[pyo3(get)]
    deadline_reached: bool,
    #[pyo3(get)]
    expansions: usize,
    #[pyo3(get)]
    deepest_completed_depth: usize,
    #[pyo3(get)]
    incumbent_updates: usize,
    #[pyo3(get)]
    predicate_calls: usize,
    #[pyo3(get)]
    predicate_batches: usize,
    #[pyo3(get)]
    predicate_errors: usize,
    #[pyo3(get)]
    elapsed_seconds: f64,
}

#[pymethods]
impl CertifiedReplacementResult {
    fn __repr__(&self) -> String {
        format!(
            "CertifiedReplacementResult(status='{}', k_lower={}, k_upper={}, reward={:.6}, rechecked={})",
            self.status, self.k_lower, self.k_upper, self.reward, self.rechecked
        )
    }
}

impl CertifiedReplacementResult {
    fn from_outcome(
        value: OracleOutcome,
        remove_positions: &[usize],
        add_item_ids: &[usize],
        baseline_item_ids: Option<&[usize]>,
    ) -> Self {
        let removed = value
            .selected_indices
            .iter()
            .map(|&operation| remove_positions[operation])
            .collect();
        let added = value
            .selected_indices
            .iter()
            .map(|&operation| add_item_ids[operation])
            .collect();
        let materialized_slate = materialize_slate(
            baseline_item_ids,
            remove_positions,
            add_item_ids,
            &value.selected_indices,
        );
        Self {
            selected_operation_indices: value.selected_indices,
            remove_positions: removed,
            add_item_ids: added,
            materialized_slate,
            k_lower: value.k_lower,
            k_upper: value.k_upper,
            cardinality_gap: value.cardinality_gap,
            first_unresolved_depth: value.first_unresolved_depth,
            upper_bound_source: value.upper_bound_source,
            reward: value.reward,
            safety_mean: value.safety_mean,
            safety_variance: value.safety_variance,
            safety_lcb: value.safety_lcb,
            status: value.status,
            stop_reason: value.stop_reason,
            exact: value.exact,
            rechecked: value.rechecked,
            deadline_reached: value.deadline_reached,
            expansions: value.expansions,
            deepest_completed_depth: value.deepest_completed_depth,
            incumbent_updates: value.incumbent_updates,
            predicate_calls: value.predicate_calls,
            predicate_batches: value.predicate_batches,
            predicate_errors: value.predicate_errors,
            elapsed_seconds: value.elapsed_seconds,
        }
    }
}

#[pyfunction]
#[pyo3(signature = (
    rewards,
    safety_means,
    safety_variances,
    max_items,
    *,
    safety_threshold=0.0,
    confidence_radius=1.96,
    covariance=None,
    contrast_features=None,
    parameter_covariance=None,
    group_ids=None,
    group_caps=None,
    conflicts=None,
    beam_width=128,
    max_expansions=100_000,
    timeout_ms=1_000,
    custom_predicate=None,
    custom_predicate_batch=None
))]
#[allow(clippy::too_many_arguments)]
fn optimize_slate(
    rewards: Vec<f64>,
    safety_means: Vec<f64>,
    safety_variances: Vec<f64>,
    max_items: usize,
    safety_threshold: f64,
    confidence_radius: f64,
    covariance: Option<Vec<Vec<f64>>>,
    contrast_features: Option<Vec<Vec<f64>>>,
    parameter_covariance: Option<Vec<Vec<f64>>>,
    group_ids: Option<Vec<i64>>,
    group_caps: Option<HashMap<i64, usize>>,
    conflicts: Option<Vec<(usize, usize)>>,
    beam_width: usize,
    max_expansions: usize,
    timeout_ms: u64,
    custom_predicate: Option<Py<PyAny>>,
    custom_predicate_batch: Option<Py<PyAny>>,
) -> PyResult<CertifiedOracleResult> {
    let item_count = rewards.len();
    let (linear_features, linear_transformed_features) = prepare_factorized_uncertainty(
        &safety_variances,
        covariance.is_some(),
        contrast_features,
        parameter_covariance,
    )?;
    let constraints = NativeConstraints::new(
        item_count,
        group_ids,
        group_caps.unwrap_or_default(),
        conflicts.unwrap_or_default(),
    )
    .map_err(PyValueError::new_err)?;
    let inputs = SlateInputs {
        rewards,
        safety_means,
        safety_variances,
        covariance,
        linear_features,
        linear_transformed_features,
    };
    let config = OracleConfig {
        max_items,
        structural_upper_bound: item_count,
        structural_upper_source: "declared_cap".to_owned(),
        beam_width,
        max_expansions,
        timeout: Duration::from_millis(timeout_ms),
        confidence_radius,
        safety_threshold,
    };

    let outcome = if custom_predicate.is_some() || custom_predicate_batch.is_some() {
        let mut predicate = PythonPredicate {
            callback: custom_predicate,
            batch_callback: custom_predicate_batch,
        };
        oracle::optimize(&inputs, &config, &constraints, &mut predicate)
    } else {
        let mut predicate = AcceptAll;
        oracle::optimize(&inputs, &config, &constraints, &mut predicate)
    }
    .map_err(PyValueError::new_err)?;

    Ok(outcome.into())
}

#[pyfunction]
#[pyo3(signature = (
    remove_positions,
    add_item_ids,
    reward_deltas,
    safety_means,
    safety_variances,
    baseline_size,
    protected_count,
    max_replacements,
    *,
    safety_threshold=0.0,
    confidence_radius=1.96,
    covariance=None,
    contrast_features=None,
    parameter_covariance=None,
    baseline_item_ids=None,
    remove_group_ids=None,
    remove_group_caps=None,
    add_group_ids=None,
    add_group_caps=None,
    operation_conflicts=None,
    beam_width=128,
    max_expansions=100_000,
    timeout_ms=1_000,
    custom_predicate=None,
    custom_predicate_batch=None
))]
#[allow(clippy::too_many_arguments)]
fn optimize_replacements(
    remove_positions: Vec<usize>,
    add_item_ids: Vec<usize>,
    reward_deltas: Vec<f64>,
    safety_means: Vec<f64>,
    safety_variances: Vec<f64>,
    baseline_size: usize,
    protected_count: usize,
    max_replacements: usize,
    safety_threshold: f64,
    confidence_radius: f64,
    covariance: Option<Vec<Vec<f64>>>,
    contrast_features: Option<Vec<Vec<f64>>>,
    parameter_covariance: Option<Vec<Vec<f64>>>,
    baseline_item_ids: Option<Vec<usize>>,
    remove_group_ids: Option<Vec<i64>>,
    remove_group_caps: Option<HashMap<i64, usize>>,
    add_group_ids: Option<Vec<i64>>,
    add_group_caps: Option<HashMap<i64, usize>>,
    operation_conflicts: Option<Vec<(usize, usize)>>,
    beam_width: usize,
    max_expansions: usize,
    timeout_ms: u64,
    custom_predicate: Option<Py<PyAny>>,
    custom_predicate_batch: Option<Py<PyAny>>,
) -> PyResult<CertifiedReplacementResult> {
    let operation_count = reward_deltas.len();
    for (name, length) in [
        ("remove_positions", remove_positions.len()),
        ("add_item_ids", add_item_ids.len()),
        ("safety_means", safety_means.len()),
        ("safety_variances", safety_variances.len()),
    ] {
        if length != operation_count {
            return Err(PyValueError::new_err(format!(
                "{name} has length {length}, expected {operation_count}"
            )));
        }
    }
    if protected_count > baseline_size {
        return Err(PyValueError::new_err(
            "protected_count cannot exceed baseline_size",
        ));
    }
    if let Some(baseline) = &baseline_item_ids {
        if baseline.len() != baseline_size {
            return Err(PyValueError::new_err(format!(
                "baseline_item_ids has length {}, expected {baseline_size}",
                baseline.len()
            )));
        }
        if baseline.iter().copied().collect::<HashSet<_>>().len() != baseline.len() {
            return Err(PyValueError::new_err(
                "baseline_item_ids must be unique before replacement",
            ));
        }
    }
    if let Some(position) = remove_positions
        .iter()
        .find(|&&position| position < protected_count || position >= baseline_size)
    {
        return Err(PyValueError::new_err(format!(
            "remove position {position} is protected or outside the baseline"
        )));
    }

    let remove_caps = unit_caps(&remove_positions)?;
    let add_caps = unit_caps(&add_item_ids)?;
    let mut partitions = vec![
        PartitionConstraint {
            ids: to_partition_ids(&remove_positions)?,
            caps: remove_caps,
        },
        PartitionConstraint {
            ids: to_partition_ids(&add_item_ids)?,
            caps: add_caps,
        },
    ];
    let remove_group_upper = optional_partition_upper(
        "remove",
        operation_count,
        remove_group_ids.as_deref(),
        remove_group_caps.as_ref(),
    )?;
    let add_group_upper = optional_partition_upper(
        "add",
        operation_count,
        add_group_ids.as_deref(),
        add_group_caps.as_ref(),
    )?;
    append_optional_partition(remove_group_ids, remove_group_caps, &mut partitions);
    append_optional_partition(add_group_ids, add_group_caps, &mut partitions);

    let explicit_conflicts = operation_conflicts.unwrap_or_default();
    let constraints =
        NativeConstraints::with_partitions(operation_count, partitions, explicit_conflicts.clone())
            .map_err(PyValueError::new_err)?;
    let mut structural_upper_bound = replacement_structural_upper(
        &remove_positions,
        &add_item_ids,
        baseline_size - protected_count,
        max_replacements,
    );
    if let Some(group_upper) = remove_group_upper {
        structural_upper_bound = structural_upper_bound.min(group_upper);
    }
    if let Some(group_upper) = add_group_upper {
        structural_upper_bound = structural_upper_bound.min(group_upper);
    }
    structural_upper_bound = structural_upper_bound.min(conflict_clique_cover_upper(
        &remove_positions,
        &add_item_ids,
        &explicit_conflicts,
    ));
    let (linear_features, linear_transformed_features) = prepare_factorized_uncertainty(
        &safety_variances,
        covariance.is_some(),
        contrast_features,
        parameter_covariance,
    )?;
    let inputs = SlateInputs {
        rewards: reward_deltas,
        safety_means,
        safety_variances,
        covariance,
        linear_features,
        linear_transformed_features,
    };
    let config = OracleConfig {
        max_items: max_replacements,
        structural_upper_bound,
        structural_upper_source: "remove_add_group_clique_cover".to_owned(),
        beam_width,
        max_expansions,
        timeout: Duration::from_millis(timeout_ms),
        confidence_radius,
        safety_threshold,
    };

    let mut predicate = ReplacementPredicate {
        callback: custom_predicate,
        batch_callback: custom_predicate_batch,
        remove_positions: remove_positions.clone(),
        add_item_ids: add_item_ids.clone(),
        baseline_item_ids: baseline_item_ids.clone(),
    };
    let outcome = oracle::optimize(&inputs, &config, &constraints, &mut predicate)
        .map_err(PyValueError::new_err)?;

    Ok(CertifiedReplacementResult::from_outcome(
        outcome,
        &remove_positions,
        &add_item_ids,
        baseline_item_ids.as_deref(),
    ))
}

fn materialize_slate(
    baseline_item_ids: Option<&[usize]>,
    remove_positions: &[usize],
    add_item_ids: &[usize],
    selected: &[usize],
) -> Option<Vec<usize>> {
    let baseline = baseline_item_ids?;
    let mut materialized = baseline.to_vec();
    for &operation in selected {
        materialized[remove_positions[operation]] = add_item_ids[operation];
    }
    (materialized.iter().copied().collect::<HashSet<_>>().len() == materialized.len())
        .then_some(materialized)
}

fn to_partition_ids(values: &[usize]) -> PyResult<Vec<i64>> {
    values
        .iter()
        .map(|&value| {
            i64::try_from(value)
                .map_err(|_| PyValueError::new_err("partition identifier exceeds i64"))
        })
        .collect()
}

fn unit_caps(values: &[usize]) -> PyResult<HashMap<i64, usize>> {
    Ok(to_partition_ids(values)?
        .into_iter()
        .map(|value| (value, 1))
        .collect())
}

fn append_optional_partition(
    ids: Option<Vec<i64>>,
    caps: Option<HashMap<i64, usize>>,
    partitions: &mut Vec<PartitionConstraint>,
) {
    // `optional_partition_upper` has already validated the pair and its length.
    // Keeping this helper infallible avoids duplicating unreachable error edges.
    if let (Some(ids), Some(caps)) = (ids, caps) {
        partitions.push(PartitionConstraint { ids, caps });
    }
}

fn optional_partition_upper(
    side: &str,
    operation_count: usize,
    ids: Option<&[i64]>,
    caps: Option<&HashMap<i64, usize>>,
) -> PyResult<Option<usize>> {
    match (ids, caps) {
        (None, None) => Ok(None),
        (Some(ids), Some(caps)) => {
            if ids.len() != operation_count {
                return Err(PyValueError::new_err(format!(
                    "{side}_group_ids has length {}, expected {operation_count}",
                    ids.len()
                )));
            }
            let mut counts: HashMap<i64, usize> = HashMap::new();
            for &group in ids {
                *counts.entry(group).or_default() += 1;
            }
            Ok(Some(
                counts
                    .into_iter()
                    .map(|(group, count)| caps.get(&group).copied().unwrap_or(count).min(count))
                    .sum(),
            ))
        }
        _ => Err(PyValueError::new_err(format!(
            "{side}_group_ids and {side}_group_caps must be supplied together"
        ))),
    }
}

fn replacement_structural_upper(
    remove_positions: &[usize],
    add_item_ids: &[usize],
    mutable_count: usize,
    max_replacements: usize,
) -> usize {
    let unique_removes = remove_positions
        .iter()
        .copied()
        .collect::<HashSet<_>>()
        .len();
    let unique_adds = add_item_ids.iter().copied().collect::<HashSet<_>>().len();
    max_replacements
        .min(remove_positions.len())
        .min(unique_removes)
        .min(unique_adds)
        .min(mutable_count)
}

/// A clique cover of the operation-conflict graph is a certified upper bound:
/// an independent feasible operation set can take at most one vertex from
/// each clique. This greedy cover need not be minimum, but it is always sound.
fn conflict_clique_cover_upper(
    remove_positions: &[usize],
    add_item_ids: &[usize],
    explicit_conflicts: &[(usize, usize)],
) -> usize {
    let operation_count = remove_positions.len();
    if operation_count == 0 {
        return 0;
    }
    let mut conflicts = HashSet::new();
    for &(left, right) in explicit_conflicts {
        conflicts.insert(normalized_pair(left, right));
    }
    for left in 0..operation_count {
        for right in (left + 1)..operation_count {
            if remove_positions[left] == remove_positions[right]
                || add_item_ids[left] == add_item_ids[right]
            {
                conflicts.insert((left, right));
            }
        }
    }

    let mut degrees = vec![0_usize; operation_count];
    for &(left, right) in &conflicts {
        if left < operation_count && right < operation_count {
            degrees[left] += 1;
            degrees[right] += 1;
        }
    }
    let mut vertices: Vec<usize> = (0..operation_count).collect();
    vertices.sort_by(|&left, &right| {
        degrees[right]
            .cmp(&degrees[left])
            .then_with(|| left.cmp(&right))
    });

    let mut cliques: Vec<Vec<usize>> = Vec::new();
    for vertex in vertices {
        if let Some(clique) = cliques.iter_mut().find(|clique| {
            clique
                .iter()
                .all(|&member| conflicts.contains(&normalized_pair(vertex, member)))
        }) {
            clique.push(vertex);
        } else {
            cliques.push(vec![vertex]);
        }
    }
    cliques.len()
}

fn normalized_pair(left: usize, right: usize) -> (usize, usize) {
    if left < right {
        (left, right)
    } else {
        (right, left)
    }
}

// The Python module initializer belongs only to the extension library. Emitting
// a second `PyInit__core` in the Rust unit-test binary gives LLVM two coverage
// maps for the same exported symbol and makes otherwise compatible profiles
// appear mismatched when Python and Rust coverage are merged.
#[cfg_attr(not(test), pymodule)]
fn _core(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<CertifiedOracleResult>()?;
    module.add_class::<CertifiedReplacementResult>()?;
    module.add_class::<ProjectedContrastUncertainty>()?;
    module.add_class::<PreparedLinearContrastMarginals>()?;
    module.add_function(wrap_pyfunction!(optimize_slate, module)?)?;
    module.add_function(wrap_pyfunction!(optimize_replacements, module)?)?;
    module.add_function(wrap_pyfunction!(
        project_linear_contrast_uncertainty,
        module
    )?)?;
    module.add_function(wrap_pyfunction!(project_linear_contrast_marginals, module)?)?;
    module.add_function(wrap_pyfunction!(
        prepare_linear_contrast_marginals_f64le,
        module
    )?)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn python_module_registration_succeeds() {
        Python::initialize();
        Python::attach(|py| {
            let module = PyModule::new(py, "_core").unwrap();
            _core(&module).unwrap();
            assert!(module.getattr("optimize_slate").is_ok());
            assert!(module.getattr("CertifiedOracleResult").is_ok());
        });
    }

    #[test]
    fn clique_cover_is_an_upper_bound_for_every_five_vertex_graph() {
        let vertex_count = 5;
        let pairs: Vec<(usize, usize)> = (0..vertex_count)
            .flat_map(|left| ((left + 1)..vertex_count).map(move |right| (left, right)))
            .collect();
        let remove_positions: Vec<usize> = (0..vertex_count).collect();
        let add_item_ids: Vec<usize> = (100..(100 + vertex_count)).collect();

        for graph_mask in 0..(1_usize << pairs.len()) {
            let conflicts: Vec<(usize, usize)> = pairs
                .iter()
                .enumerate()
                .filter_map(|(edge, &pair)| (graph_mask & (1 << edge) != 0).then_some(pair))
                .collect();
            let upper = conflict_clique_cover_upper(&remove_positions, &add_item_ids, &conflicts);
            let exact = (0..(1_usize << vertex_count))
                .filter(|subset| {
                    conflicts.iter().all(|&(left, right)| {
                        subset & (1 << left) == 0 || subset & (1 << right) == 0
                    })
                })
                .map(|subset| subset.count_ones() as usize)
                .max()
                .unwrap_or(0);
            assert!(exact <= upper, "graph mask {graph_mask}: {exact} > {upper}");
        }
    }

    #[test]
    fn private_conversion_defenses_fail_closed() {
        assert!(matrix_f64le_sha256(&[vec![1.0], vec![1.0, 2.0]], "matrix").is_err());
        assert!(matrix_f64le_sha256(&[vec![f64::NAN]], "matrix").is_err());
        assert!(to_partition_ids(&[usize::MAX]).is_err());

        assert!(materialize_slate(Some(&[10, 11]), &[1], &[10], &[0]).is_none());
    }
}
