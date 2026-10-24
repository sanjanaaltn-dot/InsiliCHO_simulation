import os
import time
import numpy as np
import pandas as pd
from multiprocessing import Pool
from scipy.stats import qmc
from insilicho import run

N_MAIN = 300
N_EXTRA = 30

STATE_NAMES = ["Xv", "Xt", "Cglc", "Cgln", "Clac", "Camm",
               "Cmab", "Coxygen", "V", "pH"]
VAR_NAMES = ["F", "T", "mu", "mu_d", "q_glc", "q_gln",
             "q_lac", "q_amm", "q_mab", "Osmolarity"]
CONC = ["Cglc", "Cgln", "Clac", "Camm", "Cmab"]
HOURS = np.arange(0, 289, 1.0)  # 12 days, hourly


def simulate_batch(args):
    batch_id, split, feed, temp, seed = args
    try:
        model = run.GrowCHO({}, feed_fn=lambda t: feed, temp_fn=lambda t: temp)
        model.execute(plot=False,
                      initial_conditions={"V": "50 mL", "Xv": seed, "Xt": seed})
        r = model.full_result
        state = np.asarray(r.state)
        svars = np.asarray(r.state_vars)
        t = np.asarray(r.t)

        if abs(state[0, 0] - seed) > 1e-6 * seed:
            return batch_id, "seed_not_applied: first Xv = %g, asked for %g" % (state[0, 0], seed), None

        cols = {}
        for i, n in enumerate(STATE_NAMES):
            cols[n] = np.interp(HOURS, t, state[:, i])
        for i, n in enumerate(VAR_NAMES):
            cols[n] = np.interp(HOURS, t, svars[:, i])
        for n in CONC:
            cols[n] = np.clip(cols[n], 0, None)

        df = pd.DataFrame(cols)
        df.insert(0, "time_h", HOURS)
        df.insert(0, "seed_Xv", seed)
        df.insert(0, "temp_setpoint", temp)
        df.insert(0, "feed_rate", feed)
        df.insert(0, "split", split)
        df.insert(0, "batch_id", batch_id)
        return batch_id, "ok", df
    except Exception as e:
        return batch_id, "error: %r" % (e,), None


if __name__ == "__main__":
    t0 = time.time()

    # Latin hypercube sampling of feed rate, temperature, seeding density
    u = qmc.LatinHypercube(d=3, seed=42).random(N_MAIN)
    feeds = 0.001 + u[:, 0] * (0.006 - 0.001)
    temps = 35.0 + u[:, 1] * (37.5 - 35.0)
    seeds = 2e9 + u[:, 2] * (5e9 - 2e9)  # 2-5 million cells/mL
    jobs = [(i, "train", feeds[i], temps[i], seeds[i]) for i in range(N_MAIN)]

    # extrapolation batches
    if N_EXTRA > 0:
        v = qmc.LatinHypercube(d=3, seed=7).random(N_EXTRA)
        ef = 0.006 + v[:, 0] * (0.008 - 0.006)
        et = 37.5 + v[:, 1] * (38.5 - 37.5)
        es = 2e9 + v[:, 2] * (5e9 - 2e9)
        jobs += [(N_MAIN + i, "extrapolation", ef[i], et[i], es[i])
                 for i in range(N_EXTRA)]

    print("Running %d simulations..." % len(jobs))
    with Pool(processes=max(1, (os.cpu_count() or 2) - 1)) as pool:
        results = pool.map(simulate_batch, jobs)

    ok = [df for _, status, df in results if status == "ok"]
    bad = [(b, s) for b, s, _ in results if s != "ok"]
    print("%d ok, %d failed" % (len(ok), len(bad)))
    for b, s in bad[:3]:
        print("  batch", b, "->", s)
    if not ok:
        raise SystemExit("All batches failed.")

    traj = pd.concat(ok, ignore_index=True)
    traj["Xv_Mcells_mL"] = traj["Xv"] * 1e-9
    traj["Xt_Mcells_mL"] = traj["Xt"] * 1e-9

    # measurement noise
    rng = np.random.default_rng(0)
    n = len(traj)
    traj["T_meas"] = traj["T"] + rng.normal(0, 0.10, n)
    traj["F_meas"] = traj["F"] * (1 + rng.normal(0, 0.02, n))
    traj["pH_meas"] = traj["pH"] + rng.normal(0, 0.02, n)
    traj["DO_meas"] = traj["Coxygen"] * (1 + rng.normal(0, 0.02, n))

    # offline samples every 12 h
    assays = traj[traj["time_h"] % 12 == 0][
        ["batch_id", "split", "time_h", "feed_rate", "temp_setpoint", "seed_Xv",
         "Xv", "Xt", "Cglc", "Cgln", "Clac", "Camm", "Cmab"]].copy()
    for c in ["Xv", "Xt", "Cglc", "Cgln", "Clac", "Camm", "Cmab"]:
        assays[c] = assays[c] * np.exp(rng.normal(0, 0.05, len(assays)))

    # batch-level summary
    last = traj[traj["time_h"] == HOURS[-1]]
    batches = last[["batch_id", "split", "feed_rate", "temp_setpoint", "seed_Xv"]].copy()
    batches["final_titer_Cmab"] = last["Cmab"].values
    batches["dead_cells_at_harvest"] = (last["Xt"] - last["Xv"]).values
    batches["final_viability"] = (last["Xv"] / last["Xt"]).values

    traj.to_csv("cho_trajectories.csv", index=False)
    assays.to_csv("cho_assays.csv", index=False)
    batches.to_csv("cho_batches.csv", index=False)
    print("Saved cho_trajectories.csv, cho_assays.csv, cho_batches.csv")
    print("Took %.1f seconds" % (time.time() - t0))
    print(batches.describe().T[["min", "mean", "max"]])