"""Stores globals related to the current environment."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import gin


_observation_spec = None
_action_spec = None


@gin.configurable
def observation_spec():
  if _observation_spec is None:
    raise RuntimeError('observation_spec has not been set.')
  return _observation_spec


@gin.configurable
def action_spec():
  if _action_spec is None:
    raise RuntimeError('action_spec has not been set.')
  return _action_spec


def set_observation_spec(spec):
  global _observation_spec
  _observation_spec = spec


def set_action_spec(spec):
  global _action_spec
  _action_spec = spec