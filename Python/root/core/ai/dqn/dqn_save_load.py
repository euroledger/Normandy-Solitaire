"""DQN-only snapshots of a NormandyEnv at a decision boundary.

This module does not modify or import the interactive save/load implementation.
Save/load files are trusted local JSON snapshots. Call capture_state BEFORE the
source game advances; snapshots contain no live mutable references.
"""
from __future__ import annotations

import base64
import copy
import json
import pickle
import random
from enum import Enum
from pathlib import Path

from cards import decks
from core.global_game_state import GlobalGameState
from core.ai.headless_action_masks import HeadlessDecisionState
from core import german_units, allied_armies
from core.map.map_model import (
    transport_track, supply_track, hitler_approval_track,
    in_transit_box, strategic_reserve_box, eliminated_units_box,
)
from core.map.map_spaces_us_1 import us_1_track
from core.map.map_spaces_brit_2 import brit_2_track
from core.map.map_spaces_can_1 import can_1_track
from core.map.map_spaces_us_3 import us_viii_track, us_xv_track
from core.models import GermanUnit, AlliedArmy
from core.tables.weather import WeatherResult, WeatherType
from core.models import Strategy
from core.enums import SideType

VERSION = 1
DEFAULT_DIR = Path(__file__).resolve().parent / 'saved_games'
CARDS = {c.card_id: c for c in decks.draw_deck + decks.mid_deck + decks.late_deck}


def _spaces():
    result = {}
    for track in (us_1_track, brit_2_track, can_1_track, us_viii_track, us_xv_track):
        for space in track:
            if space.name in result and result[space.name] is not space:
                raise ValueError(f'Duplicate map-space name: {space.name}')
            result[space.name] = space
    return result


def _named_units():
    # Collect actual module-level singletons, including commanders.
    result = {}
    for module in (german_units, allied_armies):
        for key, obj in vars(module).items():
            if isinstance(obj, (GermanUnit, AlliedArmy)):
                result[id(obj)] = (f'{module.__name__}.{key}', obj)
    return {name: obj for name, obj in result.values()}


def _pack_rng():
    return base64.b64encode(pickle.dumps(random.getstate(), protocol=4)).decode('ascii')


def _unpack_rng(value):
    return pickle.loads(base64.b64decode(value.encode('ascii')))


def capture_state(env, *, metadata=None):
    """Return an independent JSON-compatible snapshot of the active DQN game."""
    spaces = _spaces()
    boxes = {'in_transit': in_transit_box, 'strategic_reserve': strategic_reserve_box,
             'eliminated_units': eliminated_units_box}
    named = _named_units()
    objects = {}
    id_to_key = {}
    for name, obj in named.items():
        id_to_key[id(obj)] = name
    next_dynamic = 0

    def unit_key(unit):
        nonlocal next_dynamic
        identity = id(unit)
        if identity not in id_to_key:
            id_to_key[identity] = f'dynamic:{next_dynamic}'
            next_dynamic += 1
        key = id_to_key[identity]
        if key not in objects:
            if isinstance(unit, GermanUnit):
                objects[key] = {'kind': 'german', 'name': unit.name,
                                'type': unit.type.name, 'combat_value': unit.combat_value}
            elif isinstance(unit, AlliedArmy):
                objects[key] = {'kind': 'allied', 'name': unit.name,
                                'strength': unit._strength, 'reverse_strength': unit.reverse_strength,
                                'flipped': unit.flipped, 'merged': unit.merged,
                                'location': unit.location.name if unit.location else None}
            else:
                raise TypeError(f'Unexpected unit: {unit!r}')
        return key

    def encode(value):
        if value is None or isinstance(value, (bool, int, float, str)):
            return value
        if isinstance(value, Enum):
            return {'$enum': [type(value).__module__, type(value).__name__, value.name]}
        if isinstance(value, (GermanUnit, AlliedArmy)):
            return {'$unit': unit_key(value)}
        if hasattr(value, 'card_id'):
            return {'$card': value.card_id}
        if isinstance(value, WeatherResult):
            return {'$weather': {'weather_type': value.weather_type.name,
                                 'available_jabos': value.available_jabos,
                                 'resource_drm': value.resource_drm,
                                 'carpet_bombing_drm': value.carpet_bombing_drm}}
        if hasattr(value, 'name') and value.name in spaces and spaces[value.name] is value:
            return {'$space': value.name}
        if isinstance(value, (list, tuple)):
            return [encode(x) for x in value]
        if isinstance(value, set):
            return {'$set': [encode(x) for x in value]}
        if isinstance(value, dict):
            return {'$dict': [[encode(k), encode(v)] for k, v in value.items()]}
        raise TypeError(f'Cannot serialize {type(value).__name__}: {value!r}')

    global_values = {k: encode(v) for k, v in vars(GlobalGameState).items()
                     if not k.startswith('_') and not isinstance(v, (classmethod, staticmethod, property)) and not callable(v)}
    # Class methods are descriptors, not callable() at the class-dict level.
    # Persist only the three actual mutable decision-state fields.
    decision_values = {
        key: encode(getattr(HeadlessDecisionState, key))
        for key in (
            'hitler_intervention_stage',
            'hitler_intervention_target',
            'hitler_intervention_redeployments',
        )
    }
    space_values = {name: {'controlling_player': space.controlling_player.name,
                           'fortified_village_modifier': space.fortified_village_modifier,
                           'fortified': space.fortified,
                           'model_modifier': space.model_modifier,
                           'under_siege': space.under_siege,
                           'units': [unit_key(u) for u in space.units]}
                    for name, space in spaces.items()}
    box_values = {name: [unit_key(u) for u in box.units] for name, box in boxes.items()}
    # Register every named unit, including those temporarily absent from the board.
    for unit in named.values():
        unit_key(unit)
    result = {
        'version': VERSION,
        'metadata': copy.deepcopy(metadata or {}),
        'global_game_state': global_values,
        'decision_state': decision_values,
        'current_decision_type': env.current_decision_type,
        'decks': {'draw': [c.card_id for c in env.draw_deck],
                  'mid': [c.card_id for c in env.mid_deck],
                  'late': [c.card_id for c in env.late_deck]},
        'resources': {'transport': transport_track.value, 'supply': supply_track.value,
                      'hitler_approval': hitler_approval_track.value},
        'spaces': space_values,
        'boxes': box_values,
        'units': objects,
        'python_rng': _pack_rng(),
    }
    # Fail at capture time if the snapshot is not JSON-serializable.
    json.dumps(result)
    return result


def restore_state(env, snapshot):
    """Restore in-place into an environment that has already been reset()."""
    if snapshot.get('version') != VERSION:
        raise ValueError(f'Unsupported DQN snapshot version: {snapshot.get("version")}')
    spaces = _spaces()
    named = _named_units()
    units = {}
    for key, data in snapshot['units'].items():
        if key in named:
            unit = named[key]
        elif key.startswith('dynamic:') and data['kind'] == 'german':
            from core.enums import ReinforcementType
            unit = GermanUnit(type=ReinforcementType[data['type']], name=data['name'],
                              combat_value=data['combat_value'])
        else:
            raise ValueError(f'Unknown saved unit identifier: {key}')
        units[key] = unit

    def decode(value):
        if isinstance(value, list):
            return [decode(x) for x in value]
        if not isinstance(value, dict):
            return value
        if '$unit' in value:
            return units[value['$unit']]
        if '$card' in value:
            return CARDS[value['$card']]
        if '$space' in value:
            return spaces[value['$space']]
        if '$set' in value:
            return set(decode(x) for x in value['$set'])
        if '$dict' in value:
            return {decode(k): decode(v) for k, v in value['$dict']}
        if '$weather' in value:
            v = value['$weather']
            return WeatherResult(weather_type=WeatherType[v['weather_type']],
                                 available_jabos=v['available_jabos'],
                                 resource_drm=v['resource_drm'],
                                 carpet_bombing_drm=v['carpet_bombing_drm'])
        if '$enum' in value:
            import importlib
            module, cls, member = value['$enum']
            if not module.startswith('core.'):
                raise ValueError(f'Unexpected enum module: {module}')
            return getattr(importlib.import_module(module), cls)[member]
        return {k: decode(v) for k, v in value.items()}

    for name, data in snapshot['units'].items():
        unit = units[name]
        if data['kind'] == 'german':
            from core.enums import ReinforcementType
            unit.type = ReinforcementType[data['type']]
            unit.name = data['name']
            unit.combat_value = data['combat_value']
        else:
            unit._strength = data['strength']
            unit.reverse_strength = data['reverse_strength']
            unit.flipped = data['flipped']
            unit.merged = data['merged']
            unit.location = spaces[data['location']] if data['location'] else None

    for name, value in snapshot['global_game_state'].items():
        setattr(GlobalGameState, name, decode(value))
    for name, value in snapshot['decision_state'].items():
        setattr(HeadlessDecisionState, name, decode(value))
    for key, track in (('transport', transport_track), ('supply', supply_track),
                       ('hitler_approval', hitler_approval_track)):
        track.value = snapshot['resources'][key]

    for name, data in snapshot['spaces'].items():
        space = spaces[name]
        space.controlling_player = SideType[data['controlling_player']]
        space.fortified_village_modifier = data['fortified_village_modifier']
        space.fortified = data['fortified']
        space.model_modifier = data['model_modifier']
        space.under_siege = data['under_siege']
        space.units[:] = [units[key] for key in data['units']]
    for name, box in (('in_transit', in_transit_box), ('strategic_reserve', strategic_reserve_box),
                      ('eliminated_units', eliminated_units_box)):
        box.units[:] = [units[key] for key in snapshot['boxes'][name]]

    env.draw_deck = [CARDS[c] for c in snapshot['decks']['draw']]
    env.mid_deck = [CARDS[c] for c in snapshot['decks']['mid']]
    env.late_deck = [CARDS[c] for c in snapshot['decks']['late']]
    env.current_decision_type = snapshot['current_decision_type']
    random.setstate(_unpack_rng(snapshot['python_rng']))
    return snapshot.get('metadata', {})


def save_game(env, path, *, metadata=None):
    """Capture and save immediately. For a later final outcome, capture_state first."""
    snapshot = capture_state(env, metadata=metadata)
    save_snapshot(snapshot, path)
    return Path(path)


def save_snapshot(snapshot, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8') as f:
        json.dump(snapshot, f, separators=(',', ':'))
    return path


def load_snapshot(path):
    with Path(path).open('r', encoding='utf-8') as f:
        snapshot = json.load(f)
    if snapshot.get('version') != VERSION:
        raise ValueError('Incompatible snapshot format')
    return snapshot


def load_game(env, path):
    return restore_state(env, load_snapshot(path))


# ---------------------------------------------------------------------------
# Mid-game state library: collection and random selection
# ---------------------------------------------------------------------------

def _midgame_config(config=None):
    """Use the project's existing configuration unless one is supplied."""
    if config is None:
        from core.ai.dqn.midgame_config import MIDGAME_CONFIG
        config = MIDGAME_CONFIG
    return config


def _midgame_library(config):
    return Path(config['library_path'])


class MidgameCollector:
    """Collect first-decision snapshots during ONE source episode.

    Each environment worker should own its own collector. Call consider()
    before each decision and finish() once the source game terminates.
    """

    def __init__(self, config=None):
        self.config = _midgame_config(config)
        self.pending = {}
        self.source_game_id = None

    def start_game(self, source_game_id):
        """Start a fresh source episode; discard unsaved pending snapshots."""
        self.source_game_id = str(source_game_id)
        self.pending.clear()

    def consider(self, env):
        """Capture only the first decision at each configured turn."""
        if not self.config.get('collect_states', False):
            return False
        if self.source_game_id is None:
            raise RuntimeError('Call start_game(source_game_id) first')
        turn = int(GlobalGameState.cards_drawn)
        if turn not in self.config['start_turns'] or turn in self.pending:
            return False
        if env.current_decision_type not in ('action_phase', 'hitler_intervention'):
            return False
        self.pending[turn] = capture_state(env, metadata={
            'source_game_id': self.source_game_id,
            'start_turn': turn,
        })
        return True

    def finish(self, final_turn, result, *, source_checkpoint=None, original_epsilon=None):
        """Persist captured positions with the original episode's outcome.

        Returns saved paths. Call before resetting the source environment.
        """
        if self.source_game_id is None:
            raise RuntimeError('Call start_game(source_game_id) first')
        library = _midgame_library(self.config)
        saved = []
        for turn, snapshot in sorted(self.pending.items()):
            meta = snapshot['metadata']
            meta['original_final_turn'] = int(final_turn)
            meta['original_result'] = str(result)
            if source_checkpoint is not None:
                meta['source_checkpoint'] = str(source_checkpoint)
            if original_epsilon is not None:
                meta['original_epsilon'] = float(original_epsilon)
            # A source_game_id must be unique across workers and runs.
            safe_id = ''.join(c if c.isalnum() or c in '-_' else '_'
                              for c in self.source_game_id)
            # path = library / f'turn_{turn:02d}' / f'{safe_id}.json'
            
            path = (
                library / f'turn_{turn:02d}'
                / f'game_start{turn:02d}_end{int(final_turn):02d}_{result}_{safe_id[:8]}.json'
            )
            if path.exists():
                raise FileExistsError(f'Midgame snapshot already exists: {path}')
            saved.append(save_snapshot(snapshot, path))
        self.pending.clear()
        self.source_game_id = None
        return saved


def choose_midgame_snapshot(config=None, *, rng=None):
    """Weighted turn choice, then uniform random file choice within that turn.

    Returns (snapshot, path). Empty turns are excluded and weights normalized.
    """
    config = _midgame_config(config)
    rng = rng or random
    turns = config['start_turns']
    weights = config['turn_weights']
    if len(turns) != len(weights) or not turns:
        raise ValueError('start_turns and turn_weights must be nonempty and aligned')
    if config.get('selection', 'random') != 'random':
        raise ValueError('Only random midgame selection is supported')
    if any(float(w) < 0 for w in weights):
        raise ValueError('turn_weights cannot be negative')
    candidates = []
    for turn, weight in zip(turns, weights):
        paths = sorted((_midgame_library(config) / f'turn_{int(turn):02d}').glob('*.json'))
        if paths and weight > 0:
            candidates.append((int(turn), float(weight), paths))
    if not candidates:
        raise FileNotFoundError('No eligible midgame snapshots in the configured library')
    selected = rng.choices(candidates, weights=[x[1] for x in candidates], k=1)[0]
    path = rng.choice(selected[2])
    snapshot = load_snapshot(path)
    if int(snapshot['metadata']['start_turn']) != selected[0]:
        raise ValueError(f'Snapshot turn metadata does not match folder: {path}')
    return snapshot, path


def restore_random_midgame(env, config=None, *, rng=None):
    """Restore one random saved state into an ALREADY RESET environment.

    Returns (metadata, path). Caller must rebuild the current decision's
    observation/action mask after restoring; env.reset() normally starts at
    turn 1 and must not be called again after this operation.
    """
    snapshot, path = choose_midgame_snapshot(config, rng=rng)
    metadata = restore_state(env, snapshot)
    return metadata, path
