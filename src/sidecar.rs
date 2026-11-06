use std::collections::HashSet;
use std::sync::Arc;
use std::time::{Duration, Instant};

use serde::Deserialize;
use sha2::{Digest, Sha256};
use tonic::{Request, Response, Status};

use crate::constraints::{NativeConstraints, PartitionConstraint, WholeSlatePredicate};
use crate::oracle::{self, OracleConfig, SlateInputs};
use crate::uncertainty::prepare_linear_contrasts;

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

        match self.optimize(request) {
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

        let prepared =
            prepare_linear_contrasts(&contrast_features, &self.model.safety_parameter_covariance)?;
        let remove_ids = remove_positions
            .iter()
            .map(|position| {
                i64::try_from(*position).map_err(|_| "remove position overflow".to_owned())
            })
            .collect::<Result<Vec<_>, _>>()?;
        let add_ids = add_candidate_indices
            .iter()
            .map(|index| {
                i64::try_from(*index).map_err(|_| "add candidate index overflow".to_owned())
            })
            .collect::<Result<Vec<_>, _>>()?;
        let remove_partition = PartitionConstraint {
            caps: remove_ids.iter().copied().map(|id| (id, 1)).collect(),
            ids: remove_ids,
        };
        let add_partition = PartitionConstraint {
            caps: add_ids.iter().copied().map(|id| (id, 1)).collect(),
            ids: add_ids,
        };
        let constraints = NativeConstraints::with_partitions(
            reward_deltas.len(),
            vec![remove_partition, add_partition],
            Vec::new(),
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
        let outcome = oracle::optimize(&inputs, &config, &constraints, &mut predicate)?;
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
    changed_positions == operation_positions
        && changed_positions.len() == operations.len()
        && changed_positions
            .iter()
            .all(|position| *position >= protected_count)
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

#[cfg(test)]
mod tests {
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

    #[tokio::test]
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

    #[tokio::test]
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

    #[tokio::test]
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

    #[tokio::test]
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

    #[tokio::test]
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

    #[tokio::test]
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

    #[tokio::test]
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

    #[tokio::test]
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

    #[tokio::test]
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

    #[test]
    fn rejects_invalid_model_covariance() {
        let invalid = model_bytes().into_iter().collect::<Vec<_>>();
        let mut value: serde_json::Value = serde_json::from_slice(&invalid).unwrap();
        value["safety_parameter_covariance"] = serde_json::json!([[-1.0]]);
        let bytes = serde_json::to_vec(&value).unwrap();
        assert!(RerankerService::from_model_bytes(CONTRACT_DIGEST.to_owned(), &bytes).is_err());
    }
}
