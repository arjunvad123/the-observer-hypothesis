"""
Dynamical System Generators

8 diverse dynamical systems providing training data for the CfC executor.
Each system has mathematically precise "interesting moments" stored as
boolean interest masks — the key improvement over Experiment 5's vague
"garden-path sentences" for the surprise probe.

Systems range from simple (damped sine) to chaotic (Lorenz, double pendulum)
to discrete (logistic map, step function), giving the executor diverse
temporal dynamics to learn.
"""

import numpy as np
import h5py
from pathlib import Path

from . import config


def _rk4_step(f, state, dt, *args):
    """Single RK4 integration step."""
    k1 = f(state, *args)
    k2 = f(state + 0.5 * dt * k1, *args)
    k3 = f(state + 0.5 * dt * k2, *args)
    k4 = f(state + dt * k3, *args)
    return state + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)


def _integrate(f, ic, length, dt, *args):
    """Integrate ODE with RK4. Returns (length, dim) trajectory."""
    trajectory = np.zeros((length, len(ic)))
    trajectory[0] = ic
    for t in range(1, length):
        trajectory[t] = _rk4_step(f, trajectory[t - 1], dt, *args)
        # Clamp to prevent divergence
        trajectory[t] = np.clip(trajectory[t], -1e6, 1e6)
    return trajectory


# ── System 1: Lorenz Attractor ────────────────────────────────────────

def generate_lorenz(n_traj, length, dt, seed=0):
    """Lorenz attractor. Native dim: 3. sigma=10, rho=28, beta=8/3.
    Interest mask: lobe switches (sign change in x-coordinate).
    """
    sigma, rho, beta = 10.0, 28.0, 8.0 / 3.0
    rng = np.random.RandomState(seed)

    def lorenz(state):
        x, y, z = state
        return np.array([sigma * (y - x), x * (rho - z) - y, x * y - beta * z])

    trajectories = np.zeros((n_traj, length, 3))
    interest_masks = np.zeros((n_traj, length), dtype=bool)

    for i in range(n_traj):
        ic = rng.randn(3) * 5 + np.array([1.0, 1.0, 25.0])
        traj = _integrate(lorenz, ic, length, dt)
        trajectories[i] = traj

        # Interest: lobe switches (x crosses zero)
        x = traj[:, 0]
        sign_changes = np.diff(np.sign(x)) != 0
        interest_masks[i, 1:] = sign_changes

    return trajectories, interest_masks


# ── System 2: Rossler Attractor ──────────────────────────────────────

def generate_rossler(n_traj, length, dt, seed=1):
    """Rossler attractor. Native dim: 3. a=0.2, b=0.2, c=5.7.
    Interest mask: z-spikes (z > threshold).
    """
    a, b, c = 0.2, 0.2, 5.7
    rng = np.random.RandomState(seed)

    def rossler(state):
        x, y, z = state
        return np.array([-(y + z), x + a * y, b + z * (x - c)])

    trajectories = np.zeros((n_traj, length, 3))
    interest_masks = np.zeros((n_traj, length), dtype=bool)

    for i in range(n_traj):
        ic = rng.randn(3) * 2 + np.array([1.0, 0.0, 0.0])
        traj = _integrate(rossler, ic, length, dt)
        trajectories[i] = traj

        # Interest: z-spikes (z exceeds 2 std above mean)
        z = traj[:, 2]
        threshold = np.mean(z) + 2 * np.std(z)
        interest_masks[i] = z > threshold

    return trajectories, interest_masks


# ── System 3: Double Pendulum ────────────────────────────────────────

def generate_double_pendulum(n_traj, length, dt, seed=2):
    """Double pendulum. Native dim: 4 (theta1, theta2, omega1, omega2).
    Interest mask: energy exchange events (rapid change in kinetic energy split).
    """
    g, L1, L2, m1, m2 = 9.81, 1.0, 1.0, 1.0, 1.0
    rng = np.random.RandomState(seed)

    def double_pend(state):
        t1, t2, w1, w2 = state
        delta = t2 - t1
        den1 = (m1 + m2) * L1 - m2 * L1 * np.cos(delta) ** 2
        den2 = (L2 / L1) * den1

        dt1 = w1
        dt2 = w2
        dw1 = (m2 * L1 * w1 ** 2 * np.sin(delta) * np.cos(delta) +
               m2 * g * np.sin(t2) * np.cos(delta) +
               m2 * L2 * w2 ** 2 * np.sin(delta) -
               (m1 + m2) * g * np.sin(t1)) / den1
        dw2 = (-m2 * L2 * w2 ** 2 * np.sin(delta) * np.cos(delta) +
               (m1 + m2) * g * np.sin(t1) * np.cos(delta) -
               (m1 + m2) * L1 * w1 ** 2 * np.sin(delta) -
               (m1 + m2) * g * np.sin(t2)) / den2

        return np.array([dt1, dt2, dw1, dw2])

    trajectories = np.zeros((n_traj, length, 4))
    interest_masks = np.zeros((n_traj, length), dtype=bool)

    for i in range(n_traj):
        # Random initial angles, small initial velocities
        ic = np.array([
            rng.uniform(np.pi / 4, 3 * np.pi / 4),
            rng.uniform(np.pi / 4, 3 * np.pi / 4),
            rng.randn() * 0.5,
            rng.randn() * 0.5,
        ])
        traj = _integrate(double_pend, ic, length, dt)
        trajectories[i] = traj

        # Interest: rapid energy exchange (large change in KE ratio)
        ke1 = 0.5 * m1 * (L1 * traj[:, 2]) ** 2
        ke2 = 0.5 * m2 * (L2 * traj[:, 3]) ** 2
        ke_total = ke1 + ke2 + 1e-10
        ke_ratio = ke1 / ke_total
        ratio_change = np.abs(np.diff(ke_ratio))
        threshold = np.mean(ratio_change) + 2 * np.std(ratio_change)
        interest_masks[i, 1:] = ratio_change > threshold

    return trajectories, interest_masks


# ── System 4: Coupled Oscillators ────────────────────────────────────

def generate_coupled_oscillators(n_traj, length, dt, seed=3):
    """Two coupled Duffing oscillators. Native dim: 4 (x1, v1, x2, v2).
    Coupling strength varies per trajectory.
    Interest mask: synchronization breaks.
    """
    rng = np.random.RandomState(seed)

    trajectories = np.zeros((n_traj, length, 4))
    interest_masks = np.zeros((n_traj, length), dtype=bool)

    for i in range(n_traj):
        # Coupling varies: weak (periodic) to strong (chaotic)
        coupling = rng.uniform(0.1, 2.0)
        alpha = 1.0
        beta = -1.0
        delta = 0.3

        def coupled_duffing(state, c=coupling):
            x1, v1, x2, v2 = state
            dx1 = v1
            dv1 = -delta * v1 + alpha * x1 + beta * x1 ** 3 + c * (x2 - x1)
            dx2 = v2
            dv2 = -delta * v2 + alpha * x2 + beta * x2 ** 3 + c * (x1 - x2)
            return np.array([dx1, dv1, dx2, dv2])

        ic = rng.randn(4) * 0.5
        traj = _integrate(coupled_duffing, ic, length, dt)
        trajectories[i] = traj

        # Interest: synchronization breaks (large |x1 - x2| after period of small)
        sync_error = np.abs(traj[:, 0] - traj[:, 2])
        # Rolling mean
        window = 20
        if length > window:
            rolling_mean = np.convolve(sync_error, np.ones(window) / window, mode='same')
            sync_change = np.abs(np.diff(rolling_mean))
            threshold = np.mean(sync_change) + 2 * np.std(sync_change)
            interest_masks[i, 1:] = sync_change > threshold

    return trajectories, interest_masks


# ── System 5: Van der Pol Oscillator ─────────────────────────────────

def generate_van_der_pol(n_traj, length, dt, seed=4):
    """Van der Pol oscillator. Native dim: 2. mu varies [0.5, 5].
    Interest mask: relaxation jumps (when |dx/dt| spikes).
    """
    rng = np.random.RandomState(seed)

    trajectories = np.zeros((n_traj, length, 2))
    interest_masks = np.zeros((n_traj, length), dtype=bool)

    for i in range(n_traj):
        mu = rng.uniform(0.5, 5.0)

        def vdp(state, mu_val=mu):
            x, v = state
            return np.array([v, mu_val * (1 - x ** 2) * v - x])

        ic = rng.randn(2) * 2
        traj = _integrate(vdp, ic, length, dt)
        trajectories[i] = traj

        # Interest: relaxation jumps (velocity spikes)
        v_abs = np.abs(traj[:, 1])
        threshold = np.mean(v_abs) + 2 * np.std(v_abs)
        interest_masks[i] = v_abs > threshold

    return trajectories, interest_masks


# ── System 6: Damped Sinusoidal ──────────────────────────────────────

def generate_damped_sine(n_traj, length, dt, seed=5):
    """Damped sinusoidal. Native dim: 1. Random freq/decay/phase.
    Interest mask: zero crossings.
    """
    rng = np.random.RandomState(seed)

    trajectories = np.zeros((n_traj, length, 1))
    interest_masks = np.zeros((n_traj, length), dtype=bool)

    for i in range(n_traj):
        freq = rng.uniform(0.5, 5.0)
        decay = rng.uniform(0.01, 0.1)
        phase = rng.uniform(0, 2 * np.pi)
        amplitude = rng.uniform(0.5, 3.0)

        t = np.arange(length) * dt
        signal = amplitude * np.exp(-decay * t) * np.sin(2 * np.pi * freq * t + phase)
        trajectories[i, :, 0] = signal

        # Interest: zero crossings
        sign_changes = np.diff(np.sign(signal)) != 0
        interest_masks[i, 1:] = sign_changes

    return trajectories, interest_masks


# ── System 7: Step Function ──────────────────────────────────────────

def generate_step_function(n_traj, length, dt, seed=6):
    """Random step functions with Gaussian noise. Native dim: 1.
    Interest mask: step transitions.
    """
    rng = np.random.RandomState(seed)

    trajectories = np.zeros((n_traj, length, 1))
    interest_masks = np.zeros((n_traj, length), dtype=bool)

    for i in range(n_traj):
        n_steps = rng.randint(3, 12)
        step_positions = sorted(rng.choice(range(10, length - 10), n_steps, replace=False))
        levels = rng.randn(n_steps + 1) * 2
        noise_std = rng.uniform(0.01, 0.1)

        signal = np.zeros(length)
        current_level = levels[0]
        step_idx = 0
        for t in range(length):
            if step_idx < n_steps and t >= step_positions[step_idx]:
                step_idx += 1
                current_level = levels[step_idx]
            signal[t] = current_level + rng.randn() * noise_std

        trajectories[i, :, 0] = signal

        # Interest: step transitions (within 3 timesteps of a step)
        for pos in step_positions:
            start = max(0, pos - 1)
            end = min(length, pos + 2)
            interest_masks[i, start:end] = True

    return trajectories, interest_masks


# ── System 8: Logistic Map ───────────────────────────────────────────

def generate_logistic_map(n_traj, length, dt, seed=7):
    """Logistic map sequences. Native dim: 1. r varies [2.5, 4.0].
    r < 3.57: periodic. r > 3.57: chaotic. r = 4: fully chaotic.
    Interest mask: period-doubling transitions.
    """
    rng = np.random.RandomState(seed)

    trajectories = np.zeros((n_traj, length, 1))
    interest_masks = np.zeros((n_traj, length), dtype=bool)

    for i in range(n_traj):
        # Slowly sweep r through period-doubling cascade
        r_start = rng.uniform(2.5, 3.2)
        r_end = rng.uniform(3.5, 4.0)
        r_values = np.linspace(r_start, r_end, length)

        x = rng.uniform(0.1, 0.9)
        sequence = np.zeros(length)
        sequence[0] = x

        for t in range(1, length):
            x = r_values[t] * x * (1 - x)
            x = np.clip(x, 0, 1)
            sequence[t] = x

        trajectories[i, :, 0] = sequence

        # Interest: period-doubling (large local variance change)
        window = 20
        if length > 2 * window:
            local_var = np.array([
                np.var(sequence[max(0, t - window):t + 1])
                for t in range(length)
            ])
            var_change = np.abs(np.diff(local_var))
            threshold = np.mean(var_change) + 2 * np.std(var_change)
            interest_masks[i, 1:] = var_change > threshold

    return trajectories, interest_masks


# ── Utilities ────────────────────────────────────────────────────────

def pad_to_system_dim(data, target_dim=None):
    """Zero-pad native dimensions to uniform SYSTEM_DIM.
    (n_traj, length, native_dim) -> (n_traj, length, target_dim)
    """
    target_dim = target_dim or config.SYSTEM_DIM
    n_traj, length, native_dim = data.shape
    if native_dim == target_dim:
        return data
    padded = np.zeros((n_traj, length, target_dim))
    padded[:, :, :native_dim] = data
    return padded


def normalize_trajectories(data):
    """Per-dimension normalization to zero mean, unit variance.
    Applied across all trajectories and timesteps.
    """
    # Reshape to (n_samples, dim)
    original_shape = data.shape
    flat = data.reshape(-1, original_shape[-1])
    mean = flat.mean(axis=0, keepdims=True)
    std = flat.std(axis=0, keepdims=True) + 1e-8
    normalized = (data - mean) / std
    return normalized, mean.squeeze(), std.squeeze()


# ── Main Generation ──────────────────────────────────────────────────

GENERATORS = {
    'lorenz': generate_lorenz,
    'rossler': generate_rossler,
    'double_pendulum': generate_double_pendulum,
    'coupled_oscillators': generate_coupled_oscillators,
    'van_der_pol': generate_van_der_pol,
    'damped_sine': generate_damped_sine,
    'step_function': generate_step_function,
    'logistic_map': generate_logistic_map,
}


def generate_all_systems():
    """Generate all 8 systems.

    Returns:
        dict with:
            trajectories: (800, 500, 8) padded states
            system_types: (800,) string labels
            interest_masks: (800, 500) boolean
            native_dims: dict mapping system_name -> int
    """
    n_traj = config.N_TRAJECTORIES_PER_SYSTEM
    length = config.TRAJECTORY_LENGTH
    dt = config.DT

    all_trajectories = []
    all_interest_masks = []
    all_system_types = []
    native_dims = {}

    for sys_name in config.SYSTEM_TYPES:
        print(f"  Generating {sys_name}...")
        gen_func = GENERATORS[sys_name]
        traj, masks = gen_func(n_traj, length, dt)

        native_dims[sys_name] = traj.shape[2]
        padded = pad_to_system_dim(traj)

        all_trajectories.append(padded)
        all_interest_masks.append(masks)
        all_system_types.extend([sys_name] * n_traj)

    trajectories = np.concatenate(all_trajectories, axis=0)
    interest_masks = np.concatenate(all_interest_masks, axis=0)
    system_types = np.array(all_system_types)

    # Normalize per dimension
    trajectories, norm_mean, norm_std = normalize_trajectories(trajectories)

    print(f"  Total trajectories: {trajectories.shape}")
    print(f"  Interest mask density: {interest_masks.mean():.4f}")
    for sys_name in config.SYSTEM_TYPES:
        sys_mask = system_types == sys_name
        density = interest_masks[sys_mask].mean()
        print(f"    {sys_name}: {density:.4f}")

    return {
        'trajectories': trajectories.astype(np.float32),
        'system_types': system_types,
        'interest_masks': interest_masks,
        'native_dims': native_dims,
        'norm_mean': norm_mean.astype(np.float32),
        'norm_std': norm_std.astype(np.float32),
    }


def save_system_data(data, path=None):
    """Save generated system data to HDF5."""
    path = path or str(config.SYSTEM_DATA_PATH)
    Path(path).parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(path, 'w') as f:
        f.create_dataset('trajectories', data=data['trajectories'],
                         compression='gzip', compression_opts=4)
        f.create_dataset('interest_masks', data=data['interest_masks'],
                         compression='gzip', compression_opts=4)

        # Store system types as fixed-length ASCII strings
        dt_str = h5py.string_dtype('ascii', 30)
        f.create_dataset('system_types', data=data['system_types'].astype('S30'),
                         dtype=dt_str)

        f.create_dataset('norm_mean', data=data['norm_mean'])
        f.create_dataset('norm_std', data=data['norm_std'])

        # Metadata
        f.attrs['n_trajectories'] = len(data['trajectories'])
        f.attrs['trajectory_length'] = config.TRAJECTORY_LENGTH
        f.attrs['system_dim'] = config.SYSTEM_DIM
        f.attrs['n_systems'] = config.N_SYSTEMS
        f.attrs['dt'] = config.DT

        # Native dims as JSON string
        import json
        f.attrs['native_dims'] = json.dumps(data['native_dims'])

    print(f"  Saved system data to {path}")


def load_system_data(path=None):
    """Load system data from HDF5."""
    path = path or str(config.SYSTEM_DATA_PATH)

    with h5py.File(path, 'r') as f:
        data = {
            'trajectories': f['trajectories'][:],
            'interest_masks': f['interest_masks'][:],
            'system_types': np.array([s.decode() if isinstance(s, bytes) else s
                                      for s in f['system_types'][:]]),
            'norm_mean': f['norm_mean'][:],
            'norm_std': f['norm_std'][:],
        }

    return data
