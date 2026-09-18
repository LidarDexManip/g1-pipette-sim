"""Select explicitly configured recording-session overrides without changing actions."""
import copy


def for_episode(config, episode):
    profile = config.get('dataset_profile', {})
    if not profile.get('sessions'):
        return config
    if 'episodes_46_193' not in episode.provenance:
        raise ValueError('The HIL194 session profile requires the HIL194 dataset; use its matching dataset or a different config')
    selected = next((s for s in profile['sessions'] if s['start'] <= episode.index < s['end']), None)
    if selected is None:
        raise ValueError(f'No recording-session profile for episode {episode.index}')
    if config.get('selected_session') == selected['name']:
        return config
    result = copy.deepcopy(config)
    for key, value in selected['overrides'].items():
        result[key] = copy.deepcopy(value)
    result['selected_session'] = selected['name']
    return result
