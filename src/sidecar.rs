use std::collections::HashSet;
use std::sync::Arc;
use std::time::{Duration, Instant};

use serde::Deserialize;
use sha2::{Digest, Sha256};
use tonic::{Request, Response, Status};

use crate::constraints::{NativeConstraints, PartitionConstraint, WholeSlatePredicate};
use crate::oracle::{self, OracleConfig, SlateInputs};
use crate::uncertainty::prepare_linear_contrasts;

// Generated tonic clients return tonic::Status by value. Newer clippy versions
// flag that generated signature, which this crate cannot change safely.
#[allow(clippy::result_large_err)]
pub mod proto {
    tonic::include_proto!("safe_slate.v1");
}

use proto::safe_slate_reranker_server::SafeSlateReranker;
use proto::{ProposalStatus, ProposeSlateRequest, ProposeSlateResponse, ReplacementOperation};

const MODEL_SCHEMA_VERSION: &str = "safe-slate-model/v1";
const MAX_CANDIDATES: usize = 100;
const REQUIRED_PROTECTED_COUNT: usize = 2;
const REQUIRED_MAX_REPLACEMENTS: usize = 2;
const MAX_DEADLINE_MILLIS: u64 = 10;

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
struct ModelBundle {
    schema_version: String,
    model_id: String,
    feature_names: Vec<String>,
    reward_intercept: f64,
    reward_coefficients: Vec<f64>,
    safety_intercept: f64,
    safety_coefficients: Vec<f64>,
    safety_parameter_covariance: Vec<Vec<f64>>,
    confidence_radius: f64,
    safety_relative_noninferiority: f64,
    beam_width: usize,
    max_expansions: usize,
    training_snapshot_sha256: String,
    feature_dictionary_sha256: String,
    dependency_lock_sha256: String,
}

impl ModelBundle {
    fn parse(bytes: &[u8]) -> Result<Self, String> {
        let model: Self = serde_json::from_slice(bytes).map_err(|error| error.to_string())?;
        model.validate()?;
        Ok(model)
    }

    fn validate(&self) -> Result<(), String> {
        let dimension = self.feature_names.len();
        if self.schema_version != MODEL_SCHEMA_VERSION || self.model_id.is_empty() || dimension == 0
        {
            return Err("invalid model identity".to_owned());
        }
        if self.reward_coefficients.len() != dimension
            || self.safety_coefficients.len() != dimension
            || self.safety_parameter_covariance.len() != dimension
            || self
                .safety_parameter_covariance
                .iter()
                .any(|row| row.len() != dimension)
        {
            return Err("model dimensions do not match feature_names".to_owned());
        }
        if self.feature_names.iter().any(|name| name.is_empty())
            || self.feature_names.iter().collect::<HashSet<_>>().len() != dimension
        {
            return Err("feature_names must be nonempty and unique".to_owned());
        }
        if !self.reward_intercept.is_finite()
            || !self.safety_intercept.is_finite()
            || !self.confidence_radius.is_finite()
            || self.confidence_radius < 0.0
            || self.safety_relative_noninferiority != 0.01
            || self.beam_width == 0
            || self.max_expansions == 0
        {
            return Err("invalid model parameters".to_owned());
        }
        if self
            .reward_coefficients
            .iter()
            .chain(&self.safety_coefficients)
            .any(|value| !value.is_finite())
        {
            return Err("model coefficients contain NaN or infinity".to_owned());
        }
        prepare_linear_contrasts(&[], &self.safety_parameter_covariance)?;
        for digest in [
            &self.training_snapshot_sha256,
            &self.feature_dictionary_sha256,
            &self.dependency_lock_sha256,
        ] {
            if !is_sha256(digest) {
                return Err("model provenance digest is invalid".to_owned());
            }
        }
        Ok(())
    }

    fn reward(&self, features: &[f64]) -> f64 {
        linear(self.reward_intercept, &self.reward_coefficients, features).max(0.0)
    }

    fn safety_probability(&self, features: &[f64]) -> f64 {
        logistic(linear(
            self.safety_intercept,
            &self.safety_coefficients,
            features,
        ))
    }

    fn safety_gradient(&self, features: &[f64]) -> Vec<f64> {
        let probability = self.safety_probability(features);
        let scale = probability * (1.0 - probability);
        features.iter().map(|feature| scale * feature).collect()
    }
}

#[derive(Clone)]
pub struct RerankerService {
    expected_contract_digest: Arc<str>,
    model_digest: Arc<str>,
    model: Arc<ModelBundle>,
}

impl RerankerService {
    pub fn from_model_bytes(
        expected_contract_digest: String,
        model_bytes: &[u8],
    ) -> Result<Self, String> {
        if !is_sha256(&expected_contract_digest) {
            return Err("contract digest must be lowercase SHA-256".to_owned());
        }
        let model = ModelBundle::parse(model_bytes)?;
        let model_digest = sha256_hex(model_bytes);
        Ok(Self {
            expected_contract_digest: expected_contract_digest.into(),
            model_digest: model_digest.into(),
            model: Arc::new(model),
        })
    }

    pub fn model_digest(&self) -> &str {
        &self.model_digest
    }

    fn propose(&self, request: &ProposeSlateRequest) -> ProposeSlateResponse {
        let started = Instant::now();
        let input_digest = digest_request(request);
        let baseline = baseline_slate(request);

        if request.contract_digest != self.expected_contract_digest.as_ref()
            || request.model_digest != self.model_digest.as_ref()
        {
            return fallback_response(
                ProposalStatus::FallbackIdentityMismatch,
                baseline,
                self.model_digest.as_ref(),
                input_digest,
                "identity_mismatch",
                started,
            );
        }
        if let Err(reason) = validate_request(request, self.model.feature_names.len()) {
            return fallback_response(
                ProposalStatus::FallbackInvalidInput,
                baseline,
                self.model_digest.as_ref(),
                input_digest,
                reason,
                started,
            );
        }

        match coverage_string_result!(SidecarOptimize, self.optimize(request)) {
            Ok(mut response) => {
                response.input_digest = input_digest;
                response.model_digest = self.model_digest.to_string();
                response.rust_duration_micros = elapsed_micros(started);
                response
            }
            Err(reason) => fallback_response(
                ProposalStatus::FallbackInternal,
                baseline,
                self.model_digest.as_ref(),
                input_digest,
                &reason,
                started,
            ),
        }
    }

    fn optimize(&self, request: &ProposeSlateRequest) -> Result<ProposeSlateResponse, String> {
        let baseline_size = request.baseline_size as usize;
        let protected_count = request.protected_count as usize;
        let max_replacements = request.max_replacements as usize;
        let baseline: Vec<String> = request.candidates[..baseline_size]
            .iter()
            .map(|candidate| candidate.item_id.clone())
            .collect();

        let rewards: Vec<f64> = request
            .candidates
            .iter()
            .map(|candidate| self.model.reward(&candidate.features))
            .collect();
        let safety_probabilities: Vec<f64> = request
            .candidates
            .iter()
            .map(|candidate| self.model.safety_probability(&candidate.features))
            .collect();
        let safety_gradients: Vec<Vec<f64>> = request
            .candidates
            .iter()
            .map(|candidate| self.model.safety_gradient(&candidate.features))
            .collect();

        let mut remove_positions = Vec::new();
        let mut add_candidate_indices = Vec::new();
        let mut reward_deltas = Vec::new();
        let mut safety_means = Vec::new();
        let mut contrast_features = Vec::new();
        let noninferiority_multiplier = 1.0 - self.model.safety_relative_noninferiority;
        for remove_position in protected_count..baseline_size {
            for add_candidate_index in baseline_size..request.candidates.len() {
                remove_positions.push(remove_position);
                add_candidate_indices.push(add_candidate_index);
                reward_deltas.push(rewards[add_candidate_index] - rewards[remove_position]);
                safety_means.push(
                    safety_probabilities[add_candidate_index]
                        - noninferiority_multiplier * safety_probabilities[remove_position],
                );
                contrast_features.push(
                    safety_gradients[add_candidate_index]
                        .iter()
                        .zip(&safety_gradients[remove_position])
                        .map(|(add, remove)| add - noninferiority_multiplier * remove)
                        .collect::<Vec<_>>(),
                );
            }
        }
        if reward_deltas.is_empty() {
            return Ok(fallback_response(
                ProposalStatus::FallbackNoSafeChange,
                baseline,
                self.model_digest.as_ref(),
                String::new(),
                "no_insertion_candidates",
                Instant::now(),
            ));
        }

        let prepared = coverage_string_result!(
            SidecarProjection,
            prepare_linear_contrasts(&contrast_features, &self.model.safety_parameter_covariance)
        )?;
        // Request validation caps both indices below 100, so these conversions
        // are exact on every supported target.
        let remove_ids = remove_positions
            .iter()
            .map(|position| *position as i64)
            .collect::<Vec<_>>();
        let add_ids = add_candidate_indices
            .iter()
            .map(|index| *index as i64)
            .collect::<Vec<_>>();
        let remove_partition = PartitionConstraint {
            caps: remove_ids.iter().copied().map(|id| (id, 1)).collect(),
            ids: remove_ids,
        };
        let add_partition = PartitionConstraint {
            caps: add_ids.iter().copied().map(|id| (id, 1)).collect(),
            ids: add_ids,
        };
        let constraints = coverage_string_result!(
            SidecarConstraints,
            NativeConstraints::with_partitions(
                reward_deltas.len(),
                vec![remove_partition, add_partition],
                Vec::new(),
            )
        )?;
        let inputs = SlateInputs {
            rewards: reward_deltas,
            safety_means,
            safety_variances: prepared.variances,
            covariance: None,
            linear_features: Some(contrast_features),
            linear_transformed_features: Some(prepared.transformed_features),
        };
        let config = OracleConfig {
            max_items: max_replacements,
            structural_upper_bound: max_replacements
                .min(baseline_size - protected_count)
                .min(request.candidates.len() - baseline_size),
            structural_upper_source: "replacement_cap".to_owned(),
            beam_width: self.model.beam_width,
            max_expansions: self.model.max_expansions,
            timeout: Duration::from_millis(u64::from(request.deadline_millis)),
            confidence_radius: self.model.confidence_radius,
            safety_threshold: 0.0,
        };
        let mut predicate = MaterializedSlatePredicate {
            baseline: baseline.clone(),
            candidates: request
                .candidates
                .iter()
                .map(|candidate| candidate.item_id.clone())
                .collect(),
            remove_positions: remove_positions.clone(),
            add_candidate_indices: add_candidate_indices.clone(),
            protected_count,
            max_replacements,
        };
        let outcome = coverage_string_result!(
            SidecarOracle,
            oracle::optimize(&inputs, &config, &constraints, &mut predicate)
        )?;
        if outcome.selected_indices.is_empty() {
            return Ok(ProposeSlateResponse {
                status: ProposalStatus::FallbackNoSafeChange.into(),
                proposed_slate: baseline,
                operations: Vec::new(),
                k_lower: outcome.k_lower as u32,
                k_upper: outcome.k_upper as u32,
                final_recheck_passed: outcome.rechecked,
                fallback: true,
                model_digest: String::new(),
                input_digest: String::new(),
                rust_duration_micros: 0,
                stop_reason: outcome.stop_reason,
                deadline_reached: outcome.deadline_reached,
            });
        }

        let mut proposed = baseline.clone();
        let mut operations = Vec::with_capacity(outcome.selected_indices.len());
        for operation in &outcome.selected_indices {
            let remove_position = remove_positions[*operation];
            let add_candidate_index = add_candidate_indices[*operation];
            let add_item_id = request.candidates[add_candidate_index].item_id.clone();
            proposed[remove_position] = add_item_id.clone();
            operations.push(ReplacementOperation {
                remove_position: remove_position as u32,
                add_item_id,
            });
        }
        let final_recheck_passed = outcome.rechecked
            && final_recheck(
                &baseline,
                &proposed,
                &operations,
                protected_count,
                max_replacements,
            );
        #[cfg(coverage)]
        let final_recheck_passed = if crate::coverage_failpoints::take(
            crate::coverage_failpoints::CoverageFailpoint::SidecarFinalRecheck,
        ) {
            false
        } else {
            final_recheck_passed
        };
        if !final_recheck_passed {
            return Ok(ProposeSlateResponse {
                status: ProposalStatus::FallbackFinalRecheck.into(),
                proposed_slate: baseline,
                operations: Vec::new(),
                k_lower: 0,
                k_upper: outcome.k_upper as u32,
                final_recheck_passed: false,
                fallback: true,
                model_digest: String::new(),
                input_digest: String::new(),
                rust_duration_micros: 0,
                stop_reason: "sidecar_final_recheck_failed".to_owned(),
                deadline_reached: outcome.deadline_reached,
            });
        }

        Ok(ProposeSlateResponse {
            status: ProposalStatus::Ok.into(),
            proposed_slate: proposed,
            operations,
            k_lower: outcome.k_lower as u32,
            k_upper: outcome.k_upper as u32,
            final_recheck_passed,
            fallback: false,
            model_digest: String::new(),
            input_digest: String::new(),
            rust_duration_micros: 0,
            stop_reason: outcome.stop_reason,
            deadline_reached: outcome.deadline_reached,
        })
    }
}

#[tonic::async_trait]
impl SafeSlateReranker for RerankerService {
    async fn propose_slate(
        &self,
        request: Request<ProposeSlateRequest>,
    ) -> Result<Response<ProposeSlateResponse>, Status> {
        Ok(Response::new(self.propose(request.get_ref())))
    }
}

struct MaterializedSlatePredicate {
    baseline: Vec<String>,
    candidates: Vec<String>,
    remove_positions: Vec<usize>,
    add_candidate_indices: Vec<usize>,
    protected_count: usize,
    max_replacements: usize,
}

impl WholeSlatePredicate for MaterializedSlatePredicate {
    fn accepts(&mut self, selected: &[usize]) -> Result<bool, String> {
        if selected.len() > self.max_replacements {
            return Ok(false);
        }
        let mut proposed = self.baseline.clone();
        let mut operations = Vec::with_capacity(selected.len());
        for operation in selected {
            let Some(remove_position) = self.remove_positions.get(*operation).copied() else {
                return Ok(false);
            };
            let Some(add_candidate_index) = self.add_candidate_indices.get(*operation).copied()
            else {
                return Ok(false);
            };
            let Some(add_item_id) = self.candidates.get(add_candidate_index).cloned() else {
                return Ok(false);
            };
            proposed[remove_position] = add_item_id.clone();
            operations.push(ReplacementOperation {
                remove_position: remove_position as u32,
                add_item_id,
            });
        }
        Ok(final_recheck(
            &self.baseline,
            &proposed,
            &operations,
            self.protected_count,
            self.max_replacements,
        ))
    }
}

fn validate_request(
    request: &ProposeSlateRequest,
    feature_dimension: usize,
) -> Result<(), &'static str> {
    let baseline_size = request.baseline_size as usize;
    if request.request_id.is_empty() || request.trace_id.is_empty() {
        return Err("missing_request_identity");
    }
    if request.candidates.is_empty()
        || request.candidates.len() > MAX_CANDIDATES
        || baseline_size < REQUIRED_PROTECTED_COUNT
        || baseline_size > request.candidates.len()
    {
        return Err("invalid_candidate_or_baseline_size");
    }
    if request.protected_count as usize != REQUIRED_PROTECTED_COUNT
        || request.max_replacements as usize != REQUIRED_MAX_REPLACEMENTS
        || request.deadline_millis == 0
        || u64::from(request.deadline_millis) > MAX_DEADLINE_MILLIS
    {
        return Err("invalid_guardrails");
    }
    let mut item_ids = HashSet::with_capacity(request.candidates.len());
    for (index, candidate) in request.candidates.iter().enumerate() {
        if candidate.item_id.is_empty()
            || candidate.input_rank as usize != index + 1
            || candidate.features.len() != feature_dimension
            || candidate.features.iter().any(|value| !value.is_finite())
            || !item_ids.insert(&candidate.item_id)
        {
            return Err("invalid_candidate");
        }
    }
    Ok(())
}

fn final_recheck(
    baseline: &[String],
    proposed: &[String],
    operations: &[ReplacementOperation],
    protected_count: usize,
    max_replacements: usize,
) -> bool {
    if baseline.len() != proposed.len()
        || operations.len() > max_replacements
        || baseline[..protected_count] != proposed[..protected_count]
        || proposed.iter().collect::<HashSet<_>>().len() != proposed.len()
    {
        return false;
    }
    let changed_positions: HashSet<usize> = baseline
        .iter()
        .zip(proposed)
        .enumerate()
        .filter_map(|(position, (before, after))| (before != after).then_some(position))
        .collect();
    let operation_positions: HashSet<usize> = operations
        .iter()
        .map(|operation| operation.remove_position as usize)
        .collect();
    changed_positions == operation_positions && changed_positions.len() == operations.len()
}

fn baseline_slate(request: &ProposeSlateRequest) -> Vec<String> {
    request
        .candidates
        .iter()
        .take(request.baseline_size as usize)
        .map(|candidate| candidate.item_id.clone())
        .collect()
}

fn fallback_response(
    status: ProposalStatus,
    baseline: Vec<String>,
    model_digest: &str,
    input_digest: String,
    stop_reason: &str,
    started: Instant,
) -> ProposeSlateResponse {
    ProposeSlateResponse {
        status: status.into(),
        proposed_slate: baseline,
        operations: Vec::new(),
        k_lower: 0,
        k_upper: 0,
        final_recheck_passed: false,
        fallback: true,
        model_digest: model_digest.to_owned(),
        input_digest,
        rust_duration_micros: elapsed_micros(started),
        stop_reason: stop_reason.to_owned(),
        deadline_reached: false,
    }
}

fn digest_request(request: &ProposeSlateRequest) -> String {
    let mut hasher = Sha256::new();
    hasher.update(b"safe-slate-propose-input-v1\0");
    for value in [
        &request.contract_digest,
        &request.model_digest,
        &request.request_id,
        &request.trace_id,
    ] {
        hasher.update((value.len() as u64).to_le_bytes());
        hasher.update(value.as_bytes());
    }
    for value in [
        request.baseline_size,
        request.protected_count,
        request.max_replacements,
        request.deadline_millis,
    ] {
        hasher.update(value.to_le_bytes());
    }
    hasher.update((request.candidates.len() as u64).to_le_bytes());
    for candidate in &request.candidates {
        hasher.update((candidate.item_id.len() as u64).to_le_bytes());
        hasher.update(candidate.item_id.as_bytes());
        hasher.update(candidate.input_rank.to_le_bytes());
        hasher.update((candidate.features.len() as u64).to_le_bytes());
        for feature in &candidate.features {
            hasher.update(feature.to_le_bytes());
        }
    }
    format!("{:x}", hasher.finalize())
}

fn linear(intercept: f64, coefficients: &[f64], features: &[f64]) -> f64 {
    intercept
        + coefficients
            .iter()
            .zip(features)
            .map(|(coefficient, feature)| coefficient * feature)
            .sum::<f64>()
}

fn logistic(value: f64) -> f64 {
    if value >= 0.0 {
        1.0 / (1.0 + (-value).exp())
    } else {
        let exp = value.exp();
        exp / (1.0 + exp)
    }
}

fn sha256_hex(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

fn is_sha256(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

fn elapsed_micros(started: Instant) -> u64 {
    u64::try_from(started.elapsed().as_micros()).unwrap_or(u64::MAX)
}

#[cfg(any(test, coverage))]
pub(crate) mod tests {
    use super::*;
    use proto::Candidate;
    use proto::safe_slate_reranker_client::SafeSlateRerankerClient;
    use proto::safe_slate_reranker_server::SafeSlateReranker;
    use proto::safe_slate_reranker_server::SafeSlateRerankerServer;
    use tokio::net::TcpListener;
    use tokio_stream::wrappers::TcpListenerStream;
    use tonic::transport::Server;

    const CONTRACT_DIGEST: &str =
        "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";

    fn model_bytes() -> Vec<u8> {
        br#"{
  "schema_version": "safe-slate-model/v1",
  "model_id": "test-model",
  "feature_names": ["quality"],
  "reward_intercept": 0.0,
  "reward_coefficients": [1.0],
  "safety_intercept": 0.0,
  "safety_coefficients": [0.0],
  "safety_parameter_covariance": [[0.0]],
  "confidence_radius": 1.96,
  "safety_relative_noninferiority": 0.01,
  "beam_width": 128,
  "max_expansions": 100000,
  "training_snapshot_sha256": "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",
  "feature_dictionary_sha256": "cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc",
  "dependency_lock_sha256": "dddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddd"
}"#
        .to_vec()
    }

    fn parsed_model() -> ModelBundle {
        ModelBundle::parse(&model_bytes()).unwrap()
    }

    fn assert_model_invalid(mutator: impl FnOnce(&mut ModelBundle), message: &str) {
        let mut model = parsed_model();
        mutator(&mut model);
        assert_eq!(model.validate(), Err(message.to_owned()));
    }

    fn service() -> RerankerService {
        RerankerService::from_model_bytes(CONTRACT_DIGEST.to_owned(), &model_bytes()).unwrap()
    }

    fn request(service: &RerankerService) -> ProposeSlateRequest {
        ProposeSlateRequest {
            contract_digest: CONTRACT_DIGEST.to_owned(),
            model_digest: service.model_digest().to_owned(),
            request_id: "request-1".to_owned(),
            trace_id: "trace-1".to_owned(),
            candidates: vec![
                candidate("1", 1, 3.0),
                candidate("2", 2, 2.0),
                candidate("3", 3, 1.0),
                candidate("4", 4, 4.0),
            ],
            baseline_size: 3,
            protected_count: 2,
            max_replacements: 2,
            deadline_millis: 10,
        }
    }

    fn candidate(item_id: &str, input_rank: u32, quality: f64) -> Candidate {
        Candidate {
            item_id: item_id.to_owned(),
            input_rank,
            features: vec![quality],
        }
    }

    coverage_async_tests! {
    run_async_coverage_tests;
    async fn proposes_safe_replacement() {
        let service = service();
        let response = service
            .propose_slate(Request::new(request(&service)))
            .await
            .unwrap()
            .into_inner();
        assert_eq!(response.status, ProposalStatus::Ok as i32);
        assert_eq!(response.proposed_slate, vec!["1", "2", "4"]);
        assert_eq!(response.operations.len(), 1);
        assert!(response.final_recheck_passed);
        assert!(!response.fallback);
        assert_eq!(response.model_digest, service.model_digest());
        assert!(is_sha256(&response.input_digest));
    }

    async fn serves_proposal_over_localhost_grpc() {
        let service = service();
        let input = request(&service);
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        let server = tokio::spawn(
            Server::builder()
                .add_service(SafeSlateRerankerServer::new(service))
                .serve_with_incoming(TcpListenerStream::new(listener)),
        );
        let mut client = SafeSlateRerankerClient::connect(format!("http://{address}"))
            .await
            .unwrap();
        let response = client.propose_slate(input).await.unwrap().into_inner();
        server.abort();

        assert_eq!(response.status, ProposalStatus::Ok as i32);
        assert_eq!(response.proposed_slate, vec!["1", "2", "4"]);
        assert!(response.final_recheck_passed);
    }

    async fn is_deterministic() {
        let service = service();
        let first = service
            .propose_slate(Request::new(request(&service)))
            .await
            .unwrap()
            .into_inner();
        let second = service
            .propose_slate(Request::new(request(&service)))
            .await
            .unwrap()
            .into_inner();
        assert_eq!(first.proposed_slate, second.proposed_slate);
        assert_eq!(first.operations, second.operations);
        assert_eq!(first.input_digest, second.input_digest);
    }

    async fn falls_back_on_model_digest_mismatch() {
        let service = service();
        let mut input = request(&service);
        input.model_digest =
            "eeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee".to_owned();
        let response = service
            .propose_slate(Request::new(input))
            .await
            .unwrap()
            .into_inner();
        assert_eq!(
            response.status,
            ProposalStatus::FallbackIdentityMismatch as i32
        );
        assert_eq!(response.proposed_slate, vec!["1", "2", "3"]);
        assert!(response.fallback);
    }

    async fn falls_back_on_contract_digest_mismatch() {
        let service = service();
        let mut input = request(&service);
        input.contract_digest =
            "eeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee".to_owned();
        let response = service
            .propose_slate(Request::new(input))
            .await
            .unwrap()
            .into_inner();
        assert_eq!(
            response.status,
            ProposalStatus::FallbackIdentityMismatch as i32
        );
        assert!(response.fallback);
    }

    async fn falls_back_on_duplicate_candidate() {
        let service = service();
        let mut input = request(&service);
        input.candidates[3].item_id = "3".to_owned();
        let response = service
            .propose_slate(Request::new(input))
            .await
            .unwrap()
            .into_inner();
        assert_eq!(response.status, ProposalStatus::FallbackInvalidInput as i32);
        assert!(response.fallback);
    }

    async fn falls_back_on_malformed_features() {
        let service = service();
        let mut input = request(&service);
        input.candidates[3].features.clear();
        let response = service
            .propose_slate(Request::new(input))
            .await
            .unwrap()
            .into_inner();
        assert_eq!(response.status, ProposalStatus::FallbackInvalidInput as i32);
        assert!(response.fallback);
    }

    async fn rejects_deadline_above_ten_millis() {
        let service = service();
        let mut input = request(&service);
        input.deadline_millis = 11;
        let response = service
            .propose_slate(Request::new(input))
            .await
            .unwrap()
            .into_inner();
        assert_eq!(response.status, ProposalStatus::FallbackInvalidInput as i32);
        assert!(response.fallback);
    }

    async fn falls_back_when_purchase_certificate_rejects_reward_gain() {
        let mut value: serde_json::Value = serde_json::from_slice(&model_bytes()).unwrap();
        value["reward_coefficients"] = serde_json::json!([-1.0]);
        value["safety_coefficients"] = serde_json::json!([10.0]);
        let bytes = serde_json::to_vec(&value).unwrap();
        let service =
            RerankerService::from_model_bytes(CONTRACT_DIGEST.to_owned(), &bytes).unwrap();
        let mut input = request(&service);
        input.candidates[3].features = vec![-4.0];
        let response = service
            .propose_slate(Request::new(input))
            .await
            .unwrap()
            .into_inner();
        assert_eq!(response.status, ProposalStatus::FallbackNoSafeChange as i32);
        assert_eq!(response.proposed_slate, vec!["1", "2", "3"]);
        assert!(response.fallback);
    }

    async fn rejects_changed_guardrails() {
        let service = service();
        let mut input = request(&service);
        input.protected_count = 1;
        let response = service
            .propose_slate(Request::new(input))
            .await
            .unwrap()
            .into_inner();
        assert_eq!(response.status, ProposalStatus::FallbackInvalidInput as i32);
        assert_eq!(response.proposed_slate, vec!["1", "2", "3"]);
    }

    async fn falls_back_without_insertion_candidates() {
        let service = service();
        let mut input = request(&service);
        input.candidates.truncate(3);
        let response = service
            .propose_slate(Request::new(input))
            .await
            .unwrap()
            .into_inner();
        assert_eq!(response.status, ProposalStatus::FallbackNoSafeChange as i32);
        assert_eq!(response.stop_reason, "no_insertion_candidates");
        assert!(response.fallback);
    }
    }

    coverage_tests! {
    run_sync_coverage_tests;
    fn rejects_invalid_model_covariance() {
        let invalid = model_bytes().into_iter().collect::<Vec<_>>();
        let mut value: serde_json::Value = serde_json::from_slice(&invalid).unwrap();
        value["safety_parameter_covariance"] = serde_json::json!([[-1.0]]);
        let bytes = serde_json::to_vec(&value).unwrap();
        assert!(RerankerService::from_model_bytes(CONTRACT_DIGEST.to_owned(), &bytes).is_err());
    }

    fn rejects_every_invalid_model_contract() {
        assert!(ModelBundle::parse(b"{").is_err());
        let mut value: serde_json::Value = serde_json::from_slice(&model_bytes()).unwrap();
        value["unexpected"] = serde_json::json!(true);
        assert!(ModelBundle::parse(&serde_json::to_vec(&value).unwrap()).is_err());

        for invalid_digest in ["", "A", "0".repeat(63).as_str()] {
            assert!(RerankerService::from_model_bytes(
                invalid_digest.to_owned(),
                &model_bytes()
            )
            .is_err());
        }

        assert_model_invalid(
            |model| model.schema_version = "safe-slate-model/v2".to_owned(),
            "invalid model identity",
        );
        assert_model_invalid(
            |model| model.model_id.clear(),
            "invalid model identity",
        );
        assert_model_invalid(
            |model| model.feature_names.clear(),
            "invalid model identity",
        );
        assert_model_invalid(
            |model| model.reward_coefficients.clear(),
            "model dimensions do not match feature_names",
        );
        assert_model_invalid(
            |model| model.safety_coefficients.clear(),
            "model dimensions do not match feature_names",
        );
        assert_model_invalid(
            |model| model.safety_parameter_covariance.clear(),
            "model dimensions do not match feature_names",
        );
        assert_model_invalid(
            |model| model.safety_parameter_covariance[0].clear(),
            "model dimensions do not match feature_names",
        );
        assert_model_invalid(
            |model| model.feature_names[0].clear(),
            "feature_names must be nonempty and unique",
        );
        assert_model_invalid(
            |model| {
                model.feature_names.push("quality".to_owned());
                model.reward_coefficients.push(1.0);
                model.safety_coefficients.push(0.0);
                model.safety_parameter_covariance = vec![vec![0.0; 2]; 2];
            },
            "feature_names must be nonempty and unique",
        );
        assert_model_invalid(
            |model| model.reward_intercept = f64::NAN,
            "invalid model parameters",
        );
        assert_model_invalid(
            |model| model.safety_intercept = f64::INFINITY,
            "invalid model parameters",
        );
        assert_model_invalid(
            |model| model.confidence_radius = f64::NAN,
            "invalid model parameters",
        );
        assert_model_invalid(
            |model| model.confidence_radius = -1.0,
            "invalid model parameters",
        );
        assert_model_invalid(
            |model| model.safety_relative_noninferiority = 0.02,
            "invalid model parameters",
        );
        assert_model_invalid(
            |model| model.beam_width = 0,
            "invalid model parameters",
        );
        assert_model_invalid(
            |model| model.max_expansions = 0,
            "invalid model parameters",
        );
        assert_model_invalid(
            |model| model.reward_coefficients[0] = f64::NAN,
            "model coefficients contain NaN or infinity",
        );
        assert_model_invalid(
            |model| model.safety_coefficients[0] = f64::INFINITY,
            "model coefficients contain NaN or infinity",
        );
        assert_model_invalid(
            |model| model.training_snapshot_sha256.clear(),
            "model provenance digest is invalid",
        );
        assert_model_invalid(
            |model| model.feature_dictionary_sha256 = "A".repeat(64),
            "model provenance digest is invalid",
        );
        assert_model_invalid(
            |model| model.dependency_lock_sha256 = "0".repeat(63),
            "model provenance digest is invalid",
        );

        assert!(is_sha256(&"0".repeat(64)));
        assert!(is_sha256(&"a".repeat(64)));
        assert!(!is_sha256(&"A".repeat(64)));
        assert!(!is_sha256(&"a".repeat(63)));
    }

    fn rejects_every_invalid_request_contract() {
        let service = service();
        let valid = request(&service);
        assert_eq!(validate_request(&valid, 1), Ok(()));

        let mut input = valid.clone();
        input.request_id.clear();
        assert_eq!(validate_request(&input, 1), Err("missing_request_identity"));
        let mut input = valid.clone();
        input.trace_id.clear();
        assert_eq!(validate_request(&input, 1), Err("missing_request_identity"));
        let mut input = valid.clone();
        input.candidates.clear();
        assert_eq!(
            validate_request(&input, 1),
            Err("invalid_candidate_or_baseline_size")
        );
        let mut input = valid.clone();
        input.candidates = (0..=MAX_CANDIDATES)
            .map(|index| candidate(&index.to_string(), index as u32 + 1, 1.0))
            .collect();
        assert_eq!(
            validate_request(&input, 1),
            Err("invalid_candidate_or_baseline_size")
        );
        let mut input = valid.clone();
        input.baseline_size = 1;
        assert_eq!(
            validate_request(&input, 1),
            Err("invalid_candidate_or_baseline_size")
        );
        let mut input = valid.clone();
        input.baseline_size = 5;
        assert_eq!(
            validate_request(&input, 1),
            Err("invalid_candidate_or_baseline_size")
        );

        for mutator in [
            |input: &mut ProposeSlateRequest| input.protected_count = 1,
            |input: &mut ProposeSlateRequest| input.max_replacements = 1,
            |input: &mut ProposeSlateRequest| input.deadline_millis = 0,
            |input: &mut ProposeSlateRequest| input.deadline_millis = 11,
        ] {
            let mut input = valid.clone();
            mutator(&mut input);
            assert_eq!(validate_request(&input, 1), Err("invalid_guardrails"));
        }

        let mut input = valid.clone();
        input.candidates[0].item_id.clear();
        assert_eq!(validate_request(&input, 1), Err("invalid_candidate"));
        let mut input = valid.clone();
        input.candidates[0].input_rank = 2;
        assert_eq!(validate_request(&input, 1), Err("invalid_candidate"));
        let mut input = valid.clone();
        input.candidates[0].features.clear();
        assert_eq!(validate_request(&input, 1), Err("invalid_candidate"));
        let mut input = valid.clone();
        input.candidates[0].features[0] = f64::NAN;
        assert_eq!(validate_request(&input, 1), Err("invalid_candidate"));
        let mut input = valid;
        input.candidates[1].item_id = input.candidates[0].item_id.clone();
        assert_eq!(validate_request(&input, 1), Err("invalid_candidate"));
    }

    fn predicate_and_final_recheck_defenses_fail_closed() {
        let baseline = vec!["1".to_owned(), "2".to_owned(), "3".to_owned()];
        let candidates = vec![
            "1".to_owned(),
            "2".to_owned(),
            "3".to_owned(),
            "4".to_owned(),
        ];
        let mut predicate = MaterializedSlatePredicate {
            baseline: baseline.clone(),
            candidates: candidates.clone(),
            remove_positions: vec![2],
            add_candidate_indices: vec![3],
            protected_count: 2,
            max_replacements: 1,
        };
        assert_eq!(predicate.accepts(&[0]), Ok(true));
        assert_eq!(predicate.accepts(&[0, 0]), Ok(false));
        assert_eq!(predicate.accepts(&[1]), Ok(false));

        predicate.remove_positions.push(2);
        predicate.add_candidate_indices.clear();
        assert_eq!(predicate.accepts(&[0]), Ok(false));
        predicate.add_candidate_indices.push(99);
        assert_eq!(predicate.accepts(&[0]), Ok(false));

        let operation = ReplacementOperation {
            remove_position: 2,
            add_item_id: "4".to_owned(),
        };
        assert!(final_recheck(
            &baseline,
            &["1".to_owned(), "2".to_owned(), "4".to_owned()],
            std::slice::from_ref(&operation),
            2,
            1,
        ));
        assert!(!final_recheck(&baseline, &baseline[..2], &[], 2, 1));
        assert!(!final_recheck(
            &baseline,
            &baseline,
            &[operation.clone(), operation.clone()],
            2,
            1,
        ));
        assert!(!final_recheck(
            &baseline,
            &["9".to_owned(), "2".to_owned(), "3".to_owned()],
            &[],
            2,
            1,
        ));
        assert!(!final_recheck(
            &baseline,
            &["1".to_owned(), "2".to_owned(), "2".to_owned()],
            std::slice::from_ref(&operation),
            2,
            1,
        ));
        assert!(!final_recheck(
            &baseline,
            &["1".to_owned(), "2".to_owned(), "4".to_owned()],
            &[],
            2,
            1,
        ));
    }
    }

    #[cfg(coverage)]
    pub(crate) fn run_failpoint_coverage_tests() {
        use crate::coverage_failpoints::{self, CoverageFailpoint};

        for point in [
            CoverageFailpoint::SidecarOptimize,
            CoverageFailpoint::SidecarProjection,
            CoverageFailpoint::SidecarConstraints,
            CoverageFailpoint::SidecarOracle,
        ] {
            let service = service();
            let input = request(&service);
            coverage_failpoints::set(point);
            let response = service.propose(&input);
            assert_eq!(response.status, ProposalStatus::FallbackInternal as i32);
            assert!(response.fallback);
            assert!(coverage_failpoints::is_clear());
        }

        let service = service();
        let input = request(&service);
        coverage_failpoints::set(CoverageFailpoint::SidecarFinalRecheck);
        let response = service.propose(&input);
        assert_eq!(response.status, ProposalStatus::FallbackFinalRecheck as i32);
        assert!(response.fallback);
        assert!(coverage_failpoints::is_clear());
    }
}
