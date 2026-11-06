use std::collections::{HashMap, HashSet};

/// A whole-slate predicate may be non-monotone. Search code must never use a
/// rejection from this interface to prune all descendants of a partial slate.
pub trait WholeSlatePredicate {
    fn accepts(&mut self, selected: &[usize]) -> Result<bool, String>;

    fn accepts_many(&mut self, candidates: &[Vec<usize>]) -> Vec<Result<bool, String>> {
        candidates
            .iter()
            .map(|candidate| self.accepts(candidate))
            .collect()
    }
}

#[derive(Default)]
pub struct AcceptAll;

impl WholeSlatePredicate for AcceptAll {
    fn accepts(&mut self, _selected: &[usize]) -> Result<bool, String> {
        Ok(true)
    }
}

#[derive(Debug, Clone)]
pub struct NativeConstraints {
    item_count: usize,
    partitions: Vec<PartitionConstraint>,
    conflicts: HashSet<(usize, usize)>,
}

#[derive(Debug, Clone)]
pub struct PartitionConstraint {
    pub ids: Vec<i64>,
    pub caps: HashMap<i64, usize>,
}

impl NativeConstraints {
    pub fn new(
        item_count: usize,
        group_ids: Option<Vec<i64>>,
        group_caps: HashMap<i64, usize>,
        conflicts: Vec<(usize, usize)>,
    ) -> Result<Self, String> {
        if let Some(ids) = &group_ids {
            if ids.len() != item_count {
                return Err(format!(
                    "group_ids has length {}, expected {item_count}",
                    ids.len()
                ));
            }
        } else if !group_caps.is_empty() {
            return Err("group_caps requires group_ids".to_owned());
        }

        let mut normalized = HashSet::with_capacity(conflicts.len());
        for (left, right) in conflicts {
            if left >= item_count || right >= item_count {
                return Err(format!(
                    "conflict ({left}, {right}) contains an out-of-range item"
                ));
            }
            if left == right {
                return Err(format!("self-conflict for item {left} is invalid"));
            }
            normalized.insert(if left < right {
                (left, right)
            } else {
                (right, left)
            });
        }

        let partitions = group_ids
            .map(|ids| {
                vec![PartitionConstraint {
                    ids,
                    caps: group_caps,
                }]
            })
            .unwrap_or_default();
        Self::with_partitions_normalized(item_count, partitions, normalized)
    }

    pub fn with_partitions(
        item_count: usize,
        partitions: Vec<PartitionConstraint>,
        conflicts: Vec<(usize, usize)>,
    ) -> Result<Self, String> {
        let mut normalized = HashSet::with_capacity(conflicts.len());
        for (left, right) in conflicts {
            if left >= item_count || right >= item_count {
                return Err(format!(
                    "conflict ({left}, {right}) contains an out-of-range operation"
                ));
            }
            if left == right {
                return Err(format!("self-conflict for operation {left} is invalid"));
            }
            normalized.insert(if left < right {
                (left, right)
            } else {
                (right, left)
            });
        }
        Self::with_partitions_normalized(item_count, partitions, normalized)
    }

    fn with_partitions_normalized(
        item_count: usize,
        partitions: Vec<PartitionConstraint>,
        conflicts: HashSet<(usize, usize)>,
    ) -> Result<Self, String> {
        for (index, partition) in partitions.iter().enumerate() {
            if partition.ids.len() != item_count {
                return Err(format!(
                    "partition {index} has length {}, expected {item_count}",
                    partition.ids.len()
                ));
            }
        }
        Ok(Self {
            item_count,
            partitions,
            conflicts,
        })
    }

    pub fn accepts(&self, selected: &[usize]) -> bool {
        let mut seen = HashSet::with_capacity(selected.len());
        let mut partition_counts: Vec<HashMap<i64, usize>> =
            (0..self.partitions.len()).map(|_| HashMap::new()).collect();

        for &item in selected {
            if item >= self.item_count || !seen.insert(item) {
                return false;
            }
            for (partition, counts) in self.partitions.iter().zip(&mut partition_counts) {
                let group = partition.ids[item];
                let count = counts.entry(group).or_default();
                *count += 1;
                if partition.caps.get(&group).is_some_and(|cap| *count > *cap) {
                    return false;
                }
            }
        }

        for left_pos in 0..selected.len() {
            for right_pos in (left_pos + 1)..selected.len() {
                let left = selected[left_pos];
                let right = selected[right_pos];
                let pair = if left < right {
                    (left, right)
                } else {
                    (right, left)
                };
                if self.conflicts.contains(&pair) {
                    return false;
                }
            }
        }
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn native_constraints_reject_conflicts_and_group_overflow() {
        let constraints = NativeConstraints::new(
            4,
            Some(vec![0, 0, 1, 1]),
            HashMap::from([(0, 1), (1, 2)]),
            vec![(1, 2)],
        )
        .unwrap();

        assert!(constraints.accepts(&[0, 2, 3]));
        assert!(!constraints.accepts(&[0, 1]));
        assert!(!constraints.accepts(&[1, 2]));
        assert!(!constraints.accepts(&[0, 0]));
    }

    #[test]
    fn multiple_partitions_enforce_remove_and_add_uniqueness() {
        let constraints = NativeConstraints::with_partitions(
            4,
            vec![
                PartitionConstraint {
                    ids: vec![10, 10, 11, 12],
                    caps: HashMap::from([(10, 1), (11, 1), (12, 1)]),
                },
                PartitionConstraint {
                    ids: vec![20, 21, 21, 22],
                    caps: HashMap::from([(20, 1), (21, 1), (22, 1)]),
                },
            ],
            vec![],
        )
        .unwrap();

        assert!(constraints.accepts(&[0, 2]));
        assert!(!constraints.accepts(&[0, 1]));
        assert!(!constraints.accepts(&[1, 2]));
    }

    #[test]
    fn malformed_constraints_fail_closed_and_conflicts_are_normalized() {
        assert!(
            NativeConstraints::new(2, Some(vec![0]), HashMap::new(), vec![]).is_err()
        );
        assert!(
            NativeConstraints::new(2, None, HashMap::from([(0, 1)]), vec![]).is_err()
        );
        assert!(NativeConstraints::new(2, None, HashMap::new(), vec![(0, 2)]).is_err());
        assert!(NativeConstraints::new(2, None, HashMap::new(), vec![(1, 1)]).is_err());
        assert!(
            NativeConstraints::with_partitions(2, vec![], vec![(0, 2)]).is_err()
        );
        assert!(
            NativeConstraints::with_partitions(2, vec![], vec![(0, 0)]).is_err()
        );
        assert!(NativeConstraints::with_partitions(
            2,
            vec![PartitionConstraint {
                ids: vec![0],
                caps: HashMap::new(),
            }],
            vec![],
        )
        .is_err());

        let constraints =
            NativeConstraints::with_partitions(2, vec![], vec![(1, 0)]).unwrap();
        assert!(!constraints.accepts(&[1, 0]));
        assert!(!constraints.accepts(&[2]));
    }
}
