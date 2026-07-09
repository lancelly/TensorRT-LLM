# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Pure unit tests for BlockRadixTree.match_block_keys (hash-free probing)."""

import unittest
from collections.abc import Iterator
from importlib.util import find_spec
from typing import TYPE_CHECKING, cast

if not TYPE_CHECKING and find_spec("kv_cache_manager_v2") is not None:
    from kv_cache_manager_v2 import ReuseScope, TokenId
    from kv_cache_manager_v2._block_radix_tree import (
        Block,
        BlockRadixTree,
        remove_subtree,
        sequence_to_blockchain_keys,
    )
    from kv_cache_manager_v2._life_cycle_registry import LifeCycleRegistry
else:
    from tensorrt_llm.runtime.kv_cache_manager_v2 import ReuseScope, TokenId
    from tensorrt_llm.runtime.kv_cache_manager_v2._block_radix_tree import (
        Block,
        BlockRadixTree,
        remove_subtree,
        sequence_to_blockchain_keys,
    )
    from tensorrt_llm.runtime.kv_cache_manager_v2._life_cycle_registry import LifeCycleRegistry


class _EmptyLifeCycles:
    size = 0

    @property
    def ssm_life_cycle_id(self) -> None:
        return None

    def attention_life_cycles(self) -> Iterator[tuple[object, object]]:
        return iter(())


TPB = 2


class TestMatchBlockKeys(unittest.TestCase):
    def setUp(self) -> None:
        self.tree = BlockRadixTree(cast(LifeCycleRegistry, _EmptyLifeCycles()), tokens_per_block=TPB)
        self.scope = ReuseScope(lora_id=7, salt=11)
        self.tokens = [TokenId(t) for t in (1, 2, 3, 4, 5, 6)]
        root = self.tree.add_or_get_existing(self.scope)
        self.blocks: list[Block] = []
        prev: object = root
        for beg in range(0, len(self.tokens), TPB):
            block = Block(self.tokens[beg : beg + TPB], prev)
            self.blocks.append(block)
            prev = block

    def _committed_keys(self) -> list[bytes]:
        return [block.key for block in self.blocks]

    def test_matches_full_chain_like_match(self) -> None:
        expected = self.tree.match(self.scope, self.tokens)
        result = self.tree.match_block_keys(self.scope, self._committed_keys())
        self.assertEqual(result.blocks, expected.blocks)
        self.assertEqual(result.num_tokens, expected.num_tokens)
        self.assertEqual(result.num_tokens, len(self.tokens))

    def test_key_prefix_matches_shorter_chain(self) -> None:
        result = self.tree.match_block_keys(self.scope, self._committed_keys()[:2])
        self.assertEqual(result.blocks, self.blocks[:2])
        self.assertEqual(result.num_tokens, 2 * TPB)

    def test_wrong_scope_matches_nothing(self) -> None:
        other_scope = ReuseScope(lora_id=7, salt=12)
        result = self.tree.match_block_keys(other_scope, self._committed_keys())
        self.assertEqual(result.blocks, [])
        self.assertEqual(result.num_tokens, 0)

    def test_unknown_key_stops_the_walk(self) -> None:
        keys = self._committed_keys()
        keys[1] = b"\x00" * len(keys[1])
        result = self.tree.match_block_keys(self.scope, keys)
        self.assertEqual(result.blocks, self.blocks[:1])
        self.assertEqual(result.num_tokens, TPB)

    def test_removed_subtree_degrades_to_deepest_survivor(self) -> None:
        # Simulate eviction dropping block 1 (and thus its descendants): the
        # walk must stop at the deepest surviving block instead of reporting
        # a match for detached nodes.
        remove_subtree(self.blocks[1])
        result = self.tree.match_block_keys(self.scope, self._committed_keys())
        self.assertEqual(result.blocks, self.blocks[:1])
        self.assertEqual(result.num_tokens, TPB)

    def test_partial_tail_block_counts_partial_tokens(self) -> None:
        partial_tokens = [TokenId(9)]
        partial = Block(partial_tokens, self.blocks[-1])
        keys = self._committed_keys() + [partial.key]
        result = self.tree.match_block_keys(self.scope, keys)
        self.assertEqual(result.blocks, self.blocks + [partial])
        self.assertEqual(result.num_tokens, len(self.tokens) + len(partial_tokens))

    def test_keys_agree_with_sequence_to_blockchain_keys(self) -> None:
        # committed_block_keys() feeds keys produced during commit; verify the
        # tree-node keys equal what hashing the token stream would produce, so
        # the by-keys walk is exactly the hash walk minus the hashing.
        hashed = [key for token_block, key in sequence_to_blockchain_keys(TPB, self.scope, self.tokens) if token_block]
        self.assertEqual(self._committed_keys(), hashed)

    def test_empty_keys_matches_nothing(self) -> None:
        result = self.tree.match_block_keys(self.scope, [])
        self.assertEqual(result.blocks, [])
        self.assertEqual(result.num_tokens, 0)


class TestMatchWithKeyHint(unittest.TestCase):
    """match_with_key_hint must be exactly equivalent to match() — the hint
    only replaces hashing with dict lookups + token verification."""

    def setUp(self) -> None:
        self.tree = BlockRadixTree(cast(LifeCycleRegistry, _EmptyLifeCycles()), tokens_per_block=TPB)
        self.scope = ReuseScope(lora_id=7, salt=11)
        self.tokens = [TokenId(t) for t in (1, 2, 3, 4, 5, 6)]
        root = self.tree.add_or_get_existing(self.scope)
        self.blocks: list[Block] = []
        prev: object = root
        for beg in range(0, len(self.tokens), TPB):
            block = Block(self.tokens[beg : beg + TPB], prev)
            self.blocks.append(block)
            prev = block
        self.hint = [block.key for block in self.blocks]

    def _assert_parity(self, tokens, hint, enable_partial_match=False) -> None:
        expected = self.tree.match(self.scope, tokens, enable_partial_match)
        result = self.tree.match_with_key_hint(self.scope, tokens, hint, enable_partial_match)
        self.assertEqual(result.blocks, expected.blocks)
        self.assertEqual(result.num_tokens, expected.num_tokens)

    def test_unchanged_prefix_full_parity(self) -> None:
        self._assert_parity(self.tokens, self.hint)
        result = self.tree.match_with_key_hint(self.scope, self.tokens, self.hint)
        self.assertEqual(result.num_tokens, len(self.tokens))

    def test_suffix_beyond_hint_is_hash_matched(self) -> None:
        # A later request already committed one more block; the hint only
        # covers the first three — the suffix must still match via hashing.
        extra = [TokenId(7), TokenId(8)]
        extra_block = Block(extra, self.blocks[-1])
        self._assert_parity(self.tokens + extra, self.hint)
        result = self.tree.match_with_key_hint(self.scope, self.tokens + extra, self.hint)
        self.assertEqual(result.blocks[-1], extra_block)
        self.assertEqual(result.num_tokens, len(self.tokens) + len(extra))

    def test_edited_history_is_detected_by_verification(self) -> None:
        # Block 1's content changed: the stale hint key still exists in the
        # tree, but verification against the new tokens must reject it.
        edited = list(self.tokens)
        edited[2] = TokenId(99)
        self._assert_parity(edited, self.hint)
        result = self.tree.match_with_key_hint(self.scope, edited, self.hint)
        self.assertEqual(result.blocks, self.blocks[:1])
        self.assertEqual(result.num_tokens, TPB)

    def test_truncated_history_stops_at_token_end(self) -> None:
        self._assert_parity(self.tokens[:3], self.hint)

    def test_partial_tail_via_suffix_walk(self) -> None:
        # Tokens end mid-block; the partial match runs in the suffix
        # continuation, after the verified prefix.
        tokens = self.tokens[:5]
        self._assert_parity(tokens, self.hint, enable_partial_match=True)
        result = self.tree.match_with_key_hint(self.scope, tokens, self.hint, True)
        self.assertEqual(result.num_tokens, 5)

    def test_removed_subtree_falls_back_to_hashing(self) -> None:
        remove_subtree(self.blocks[1])
        self._assert_parity(self.tokens, self.hint)
        result = self.tree.match_with_key_hint(self.scope, self.tokens, self.hint)
        self.assertEqual(result.blocks, self.blocks[:1])

    def test_empty_hint_equals_plain_match(self) -> None:
        self._assert_parity(self.tokens, [])

    def test_wrong_scope_matches_nothing(self) -> None:
        result = self.tree.match_with_key_hint(ReuseScope(lora_id=7, salt=12), self.tokens, self.hint)
        self.assertEqual(result.blocks, [])
        self.assertEqual(result.num_tokens, 0)


if __name__ == "__main__":
    unittest.main()
