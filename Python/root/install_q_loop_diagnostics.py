"""Install Q-loop logging in the DQN trainer without changing learning or actions.
Run from the project root: python install_q_loop_diagnostics_fixed.py
"""
from pathlib import Path

p = Path('core/ai/dqn/dqn_trainer.py')
s = p.read_text(encoding='utf-8')
assert 'Q_LOOP_DIAG_BEGIN' not in s, 'Q-loop diagnostics already installed'
anchor = '''                (
                    next_observations,
                    rewards,
                    terminated,
                    truncated,
                    next_info,
                ) = env.step(actions)'''
assert s.count(anchor) == 1, 'Unexpected trainer version: env.step block not found uniquely'
pre = '''                # Q_LOOP_DIAG_BEGIN: diagnostic logging only.
                loop_probe = {}
                for wi in range(num_workers):
                    streak = int(diagnostic_no_change_streak[wi])
                    if needs_reset[wi] or streak < 10 or (streak != 10 and streak % 25 != 0):
                        continue
                    with torch.no_grad():
                        state_tensor = torch.as_tensor(
                            observations[wi:wi+1], dtype=torch.float32,
                            device=self.agent.device,
                        )
                        q = self.agent.policy_net(state_tensor)[0].detach().cpu().numpy()
                    legal = np.flatnonzero(masks[wi])
                    best = legal[np.argsort(q[legal])[-5:][::-1]] if len(legal) else []
                    ai = int(actions[wi])
                    loop_probe[wi] = (streak, ai)
                    print(
                        f"Q_LOOP_PRE worker={wi} streak={streak} action={ai} "
                        f"selected_q={float(q[ai]):.5f} "
                        f"top5={[(int(j), round(float(q[j]), 5)) for j in best]} "
                        f"legal={len(legal)} updates={self.total_training_updates} "
                        f"epsilon={self.agent.epsilon:.5f}", flush=True,
                    )
                # Q_LOOP_DIAG_END
'''
s = s.replace(anchor, pre + anchor)
anchor2 = '                next_masks = next_info["action_mask"]'
assert s.count(anchor2) == 1, 'Unexpected trainer version: next mask anchor not found uniquely'
post = '''
                # Q_LOOP_DIAG_POST_BEGIN
                for wi, (streak, ai) in loop_probe.items():
                    print(
                        f"Q_LOOP_POST worker={wi} action={ai} "
                        f"unchanged={bool(np.array_equal(observations[wi], next_observations[wi]))} "
                        f"reward={float(rewards[wi]):.5f} "
                        f"terminated={bool(terminated[wi])} truncated={bool(truncated[wi])} "
                        f"mask_same={bool(np.array_equal(masks[wi], next_masks[wi]))}",
                        flush=True,
                    )
                # Q_LOOP_DIAG_POST_END'''
s = s.replace(anchor2, anchor2 + post)
compile(s, str(p), 'exec')
backup = p.with_name('dqn_trainer.py.before_q_diag')
if backup.exists():
    assert backup.read_text(encoding='utf-8') == p.read_text(encoding='utf-8'), (
        f'Existing backup differs from current trainer: {backup}; refusing to modify trainer'
    )
else:
    backup.write_text(p.read_text(encoding='utf-8'), encoding='utf-8')
p.write_text(s, encoding='utf-8')
print(f'Installed diagnostics in {p}. Original backup: {backup}')
