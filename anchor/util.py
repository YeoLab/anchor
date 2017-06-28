from .visualize import MODALITY_ORDER

n_events = 'Number of alternative events'


def tidify_modalities(modality_assignments, name='event_id'):
    modalities_tidy = modality_assignments.stack().reset_index()
    modalities_tidy = modalities_tidy.rename(
        columns={'level_1': name, 0: "modality"})
    return modalities_tidy


def count_modalities(tidy_modalities, name='event_id', group_name='phenotype'):
    modalities_counts = tidy_modalities.groupby(
        [group_name, 'modality']).count().reset_index()
    modalities_counts = modalities_counts.rename(
        columns={name: n_events})
    n_events_grouped = modalities_counts.groupby('phenotype')[n_events]
    modalities_counts['percentage'] = 100*n_events_grouped.apply(
        lambda x: x/x.sum())
    return modalities_counts


def twodee_counts(modality_counts, group_name='phenotype', group_order=None):
    modalities_counts_2d = modality_counts.pivot(
        index=group_name, columns='modality', values=n_events)
    modalities_counts_2d = modalities_counts_2d.reindex(
        columns=MODALITY_ORDER, index=group_order)
    modalities_counts_2d = modalities_counts_2d.T
    return modalities_counts_2d
