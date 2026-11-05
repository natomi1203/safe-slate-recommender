#[derive(Debug, Clone)]
pub struct LinearContrastProjection {
    pub covariance: Vec<Vec<f64>>,
    pub variances: Vec<f64>,
    pub individual_lcbs: Vec<f64>,
    pub feature_dimension: usize,
}

#[derive(Debug, Clone)]
pub struct PreparedLinearContrasts {
    pub transformed_features: Vec<Vec<f64>>,
    pub variances: Vec<f64>,
    pub feature_dimension: usize,
}

pub fn prepare_linear_contrasts(
    contrast_features: &[Vec<f64>],
    parameter_covariance: &[Vec<f64>],
) -> Result<PreparedLinearContrasts, String> {
    let feature_dimension = parameter_covariance.len();
    if parameter_covariance
        .iter()
        .any(|row| row.len() != feature_dimension)
    {
        return Err(format!(
            "parameter_covariance must have shape {feature_dimension} x {feature_dimension}"
        ));
    }
    if !contrast_features.is_empty() && feature_dimension == 0 {
        return Err("parameter_covariance must be nonempty when contrasts are present".to_owned());
    }
    for (row, features) in contrast_features.iter().enumerate() {
        if features.len() != feature_dimension {
            return Err(format!(
                "contrast_features row {row} has length {}, expected {feature_dimension}",
                features.len()
            ));
        }
        if features.iter().any(|value| !value.is_finite()) {
            return Err("contrast_features contains NaN or infinity".to_owned());
        }
    }
    validate_symmetric_psd(parameter_covariance, "parameter_covariance")?;

    let operation_count = contrast_features.len();
    let mut transformed_features = vec![vec![0.0; feature_dimension]; operation_count];
    let mut variances = Vec::with_capacity(operation_count);
    for (operation, features) in contrast_features.iter().enumerate() {
        for (row, covariance_row) in parameter_covariance.iter().enumerate() {
            transformed_features[operation][row] = covariance_row
                .iter()
                .zip(features)
                .map(|(covariance, feature)| covariance * feature)
                .sum();
        }
        let variance = features
            .iter()
            .zip(&transformed_features[operation])
            .map(|(feature, transformed_feature)| feature * transformed_feature)
            .sum::<f64>();
        let scale = features
            .iter()
            .zip(&transformed_features[operation])
            .map(|(feature, transformed_feature)| (feature * transformed_feature).abs())
            .sum::<f64>()
            .max(1.0);
        if variance < -1e-10 * scale {
            return Err("projected covariance has a negative diagonal entry".to_owned());
        }
        variances.push(variance.max(0.0));
    }
    Ok(PreparedLinearContrasts {
        transformed_features,
        variances,
        feature_dimension,
    })
}

pub fn project_linear_contrasts(
    safety_means: &[f64],
    contrast_features: &[Vec<f64>],
    parameter_covariance: &[Vec<f64>],
    confidence_radius: f64,
) -> Result<LinearContrastProjection, String> {
    if !confidence_radius.is_finite() || confidence_radius < 0.0 {
        return Err("confidence_radius must be finite and nonnegative".to_owned());
    }
    if safety_means.len() != contrast_features.len() {
        return Err(format!(
            "contrast_features has {} rows, expected {}",
            contrast_features.len(),
            safety_means.len()
        ));
    }
    if safety_means.iter().any(|value| !value.is_finite()) {
        return Err("safety_means contains NaN or infinity".to_owned());
    }

    let prepared = prepare_linear_contrasts(contrast_features, parameter_covariance)?;
    let feature_dimension = prepared.feature_dimension;
    let operation_count = contrast_features.len();
    let mut covariance = vec![vec![0.0; operation_count]; operation_count];
    for left in 0..operation_count {
        for right in left..operation_count {
            let value = contrast_features[left]
                .iter()
                .zip(&prepared.transformed_features[right])
                .map(|(feature, transformed_feature)| feature * transformed_feature)
                .sum::<f64>();
            covariance[left][right] = value;
            covariance[right][left] = value;
        }
    }

    let covariance_scale = covariance
        .iter()
        .flat_map(|row| row.iter())
        .map(|value| value.abs())
        .fold(1.0_f64, f64::max);
    let tolerance = 1e-10 * covariance_scale;
    for (index, row) in covariance.iter_mut().enumerate() {
        if row[index] < -tolerance {
            return Err("projected covariance has a negative diagonal entry".to_owned());
        }
        row[index] = prepared.variances[index];
    }
    validate_symmetric_psd(&covariance, "projected covariance")?;

    let individual_lcbs = safety_means
        .iter()
        .zip(&prepared.variances)
        .map(|(mean, variance)| mean - confidence_radius * variance.sqrt())
        .collect();
    Ok(LinearContrastProjection {
        covariance,
        variances: prepared.variances,
        individual_lcbs,
        feature_dimension,
    })
}

pub(crate) fn validate_symmetric_psd(matrix: &[Vec<f64>], name: &str) -> Result<(), String> {
    let size = matrix.len();
    if matrix.iter().any(|row| row.len() != size) {
        return Err(format!("{name} must have shape {size} x {size}"));
    }
    if matrix
        .iter()
        .flat_map(|row| row.iter())
        .any(|value| !value.is_finite())
    {
        return Err(format!("{name} contains NaN or infinity"));
    }
    let scale = matrix
        .iter()
        .flat_map(|row| row.iter())
        .map(|value| value.abs())
        .fold(1.0_f64, f64::max);
    let tolerance = 1e-10 * scale;
    for index in 0..size {
        if matrix[index][index] < -tolerance {
            return Err(format!("{name} has a negative diagonal entry"));
        }
        for other in (index + 1)..size {
            if (matrix[index][other] - matrix[other][index]).abs() > tolerance {
                return Err(format!("{name} must be symmetric"));
            }
        }
    }

    // Pivot-free LDL^T is sufficient here because a zero pivot may only have
    // zero residual cross terms in a positive-semidefinite matrix.
    let mut lower = vec![vec![0.0; size]; size];
    let mut diagonal = vec![0.0; size];
    for i in 0..size {
        for j in 0..i {
            let residual = matrix[i][j]
                - (0..j)
                    .map(|k| lower[i][k] * diagonal[k] * lower[j][k])
                    .sum::<f64>();
            if diagonal[j].abs() <= tolerance {
                if residual.abs() > tolerance {
                    return Err(format!("{name} is not positive semidefinite"));
                }
                lower[i][j] = 0.0;
            } else {
                lower[i][j] = residual / diagonal[j];
            }
        }
        let pivot = matrix[i][i]
            - (0..i)
                .map(|k| lower[i][k] * lower[i][k] * diagonal[k])
                .sum::<f64>();
        if pivot < -tolerance {
            return Err(format!("{name} is not positive semidefinite"));
        }
        diagonal[i] = pivot.max(0.0);
        lower[i][i] = 1.0;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn projects_shared_parameter_uncertainty() {
        let projection = project_linear_contrasts(
            &[0.3, 0.2],
            &[vec![1.0, 1.0], vec![1.0, -1.0]],
            &[vec![2.0, 0.5], vec![0.5, 1.0]],
            2.0,
        )
        .unwrap();

        assert_eq!(projection.feature_dimension, 2);
        assert_eq!(projection.variances, vec![4.0, 2.0]);
        assert_eq!(projection.covariance, vec![vec![4.0, 1.0], vec![1.0, 2.0]]);
        assert!((projection.individual_lcbs[0] + 3.7).abs() < 1e-12);
    }

    #[test]
    fn rejects_indefinite_parameter_covariance() {
        let result = project_linear_contrasts(
            &[0.1],
            &[vec![1.0, 0.0]],
            &[vec![1.0, 2.0], vec![2.0, 1.0]],
            1.96,
        );
        assert!(result.is_err());
    }
}
