import h5py
from os import path
import fever_bout_generator as mbg
import numpy as np
import matplotlib.pyplot as pl
import matplotlib as mpl
import seaborn as sns
import argparse
from scipy.stats import kstest, norm, rankdata
from typing import List, Tuple
import pandas as pd


def simulate(length_ms: int, min_temp: float, max_temp: float, bout_gen: mbg.SwimBoutGenerator):
    def temperature(y):
        nonlocal min_temp, max_temp, chamber_min
        global chamber_max
        return (y-chamber_min)/(chamber_max-chamber_min)*(max_temp-min_temp)+min_temp

    global chamber_max
    positions = np.full((length_ms, 3), np.nan)
    chamber_min = 0.0
    start_x = 0
    start_y = np.random.rand()*chamber_max
    start_heading = np.random.rand()*2*np.pi
    positions[0] = np.array([start_x, start_y, start_heading])
    time_ms = 0
    previous_delta_t = 0
    bouts = []
    while time_ms < length_ms:
        curr_temp = temperature(positions[time_ms][1])
        # draw a new swimbout
        bout = bout_gen.draw_next_bout(curr_temp, previous_delta_t, False, False)
        bout.T = curr_temp
        bout.dT = previous_delta_t
        bouts.append(bout)
        next_bout_time = time_ms + int(bout.pre_bout_ibi_ms) + 15  # +15 for 150 ms bout duration
        if next_bout_time > length_ms-1:
            positions[time_ms:] = positions[time_ms][None, :]
            break
        positions[time_ms:next_bout_time] = positions[time_ms][None, :]
        new_heading = positions[time_ms][2] + np.deg2rad(bout.bout_angle_deg)
        dx = np.cos(new_heading)*bout.bout_displacement_mm
        dy = np.sin(new_heading)*bout.bout_displacement_mm
        if positions[time_ms][1] + dy > chamber_max or positions[time_ms][1] + dy < chamber_min:
            new_heading += np.pi
            dx = np.cos(new_heading) * bout.bout_displacement_mm
            dy = np.sin(new_heading) * bout.bout_displacement_mm
        elif positions[time_ms][0] + dx > 25 or positions[time_ms][0] + dx < -25:
            new_heading += np.pi
            dx = np.cos(new_heading) * bout.bout_displacement_mm
            dy = np.sin(new_heading) * bout.bout_displacement_mm
        positions[next_bout_time] = np.array([positions[time_ms, 0]+dx, positions[time_ms, 1]+dy, new_heading])
        previous_delta_t = temperature(positions[next_bout_time][1]) - curr_temp
        time_ms = next_bout_time
    return positions, bouts


def ks_bootstrap_test_by_fish(sample1: List, sample2: List, nboot: int) -> Tuple[float, np.ndarray, float]:
    """
    Computes ks test statistic in cases where data come from different individuals and where variability is believed
    to be larger between individuals than within individuals
    :param sample1: List of data-samples with each element in the list being one individual fish
    :param sample2: List of data-samples with each element in the list being one individual fish
    :param nboot: The number of bootstrap resamples to perform
    :return:
        [0]: The p-value
        [1]: The test statistic of the joint bootstrap resamples
        [2]: The test statistic of the true comparison
    """
    true_ks_stat = kstest(np.hstack(sample1), np.hstack(sample2)).statistic
    combined = sample1 + sample2
    ix_combined = np.arange(len(combined)).astype(int)
    boot_stats = np.zeros(nboot)
    for i in range(nboot):
        ix1 = np.random.choice(ix_combined, len(sample1))
        ix2 = np.random.choice(ix_combined, len(sample2))
        s1 = np.hstack([combined[ix] for ix in ix1])
        s2 = np.hstack([combined[ix] for ix in ix2])
        boot_stats[i] = kstest(s1, s2).statistic
    # estimate p-value based on normal approximation if no elements in boot_stats are larger than true_ks_stat
    if np.sum(boot_stats >= true_ks_stat) == 0:
        print("p-value estimated through normal approximation")
        std = np.std(boot_stats)
        n_sigma = (true_ks_stat - np.mean(boot_stats)) / std
        p = 1 - norm.cdf(n_sigma)
    else:
        p = (1+np.sum(boot_stats >= true_ks_stat)) / (nboot+1)
    return p, boot_stats, true_ks_stat


def generate_stat_df(p_values: np.ndarray, names: np.ndarray):
    pv_ranks = rankdata(p_values, method='max')
    rank_ix = np.argsort(pv_ranks)
    fdrs = [0.05, 0.01, 0.001]
    data = {"Name": [], "p-value": [], "Rank": []} | {f"Critical at {q}": [] for q in fdrs}
    for ix in rank_ix:
        data["Name"].append(names[ix])
        data["p-value"].append(p_values[ix])
        data["Rank"].append(pv_ranks[ix])
        for q in fdrs:
            data[f"Critical at {q}"].append(pv_ranks[ix]/p_values.size*q)
    return pd.DataFrame(data)


if __name__ == '__main__':

    a_parser = argparse.ArgumentParser(prog="simulator.py",
                                       description="Runs Markov Model simulations on identified treatments")
    a_parser.add_argument("-f", "--file", help="Path to 3rd order Markov Model Fit", type=str, required=True)
    a_parser.add_argument("-c", "--control_name", help="Name of control treatment for mixed model analysis", type=str)
    a_parser.add_argument("-min", "--t_min", help="Minimum temperature in Celsius", type=float, default=20.0)
    a_parser.add_argument("-max", "--t_max", help="Maximal temperature in Celsius", type=float, default=30.0)
    a_parser.add_argument("-cl", "--chamber_length", help="Chamber length in mm", type=float, default=200.0)

    mpl.rcParams['pdf.fonttype'] = 42

    args = a_parser.parse_args()

    dfile_path = args.file
    control_name = args.control_name
    t_min = args.t_min
    t_max = args.t_max
    chamber_max = args.chamber_length

    n_simulations = 1_000

    plot_folder = path.split(dfile_path)[0]

    with h5py.File(dfile_path, "r") as f:
        all_temp_hists = {k: [] for k in f.keys()}
        all_temps = {k: [] for k in f.keys()}

    tbins = np.linspace(t_min,t_max, 100)
    tbc = np.diff(tbins)/2 + tbins[:-1]

    for condition in all_temp_hists:
        for i in range(n_simulations):
            print(f"Running iteration {i+1} on condition {condition}")
            this_bout_generator = mbg.SwimBoutGenerator(dfile_path, 0, 0, 2, 3, condition)
            sim_pos = simulate(1000*60*60 , min_temp=t_min, max_temp=t_max, bout_gen=this_bout_generator)[0]
            temperatures = (sim_pos[:, 1])/chamber_max*(t_max-t_min)+t_min
            all_temps[condition].append(temperatures[::2000].copy())  # only append snapshot every 2s to make significance tests below feasible (due to the autocorrelation in the position traces it is unlikely that this loses power
            h_temp = np.histogram(temperatures, tbins)[0].astype(float)
            h_temp /= np.sum(h_temp)
            all_temp_hists[condition].append(h_temp)
        all_temp_hists[condition] = np.vstack(all_temp_hists[condition])

    fig = pl.figure()
    for i, condition in enumerate(all_temp_hists):
        m = np.mean(np.cumsum(all_temp_hists[condition], axis=1), 0)
        e = np.std(np.cumsum(all_temp_hists[condition], axis=1), 0) / np.sqrt(all_temp_hists[condition].shape[0])
        pl.plot(tbc, m, color=f"C{i}", label=condition)
        pl.fill_between(tbc, m-e, m+e, color=f"C{i}", alpha=0.4)
    pl.legend()
    pl.xlabel("Temperature [C]")
    pl.ylabel("Cumulative proportion")
    sns.despine()
    fig.savefig(path.join(plot_folder, "Simulation_Cumulative_Occupancy.pdf"))

    fig = pl.figure()
    for i, condition in enumerate(all_temp_hists):
        m = np.mean(all_temp_hists[condition], 0)
        e = np.std(all_temp_hists[condition], 0) / np.sqrt(all_temp_hists[condition].shape[0])
        pl.plot(tbc, m, color=f"C{i}", label=condition)
        pl.fill_between(tbc, m-e, m+e, color=f"C{i}", alpha=0.4)
    pl.legend()
    pl.xlabel("Temperature [C]")
    pl.ylabel("Occupancy")
    sns.despine()
    fig.savefig(path.join(plot_folder, "Simulation_Occupancy.pdf"))

    # analyze significance of simulation differences save to csv
    names = []
    p_values = []
    cond_names = list(all_temps.keys())
    for i, cond1 in enumerate(cond_names):
        for j, cond2 in enumerate(cond_names[i+1:]):
            names.append(f"{cond1}-{cond2}")
            p_values.append(ks_bootstrap_test_by_fish(all_temps[cond1], all_temps[cond2], 1000)[0])
    df_sig_occupancy = generate_stat_df(np.hstack(p_values), np.hstack(names))
    df_sig_occupancy.to_csv(path.join(plot_folder, "stats_Simulation_Occupancy.csv"))

    # if a control condition is given run simulations versus that condition with mixed models
    mixed_model_temp_hists = {}
    mixed_model_temps = {}
    if control_name is not None:
        control_bout_gen = mbg.SwimBoutGenerator(dfile_path, 0, 0, 2, 3, control_name)
        for condition in all_temp_hists:
            if condition == control_name:
                continue
            # for each condition run two mixed simulations: 1 in which the emission models belong to the condition
            # but the transition model is the control model, the other in which the condition controls the transitions
            # while the control model controls the emissions
            transit_control = mbg.SwimBoutGenerator(dfile_path, 0, 0, 2, 3, condition)
            transit_control.swim_mode_transit = control_bout_gen.swim_mode_transit
            emission_control = mbg.SwimBoutGenerator(dfile_path, 0, 0, 2, 3, condition)
            emission_control.ibi_gamma = control_bout_gen.ibi_gamma
            emission_control.displacement_gamma = control_bout_gen.displacement_gamma
            emission_control.turn_glm_gmm = control_bout_gen.turn_glm_gmm
            transit_control_name = f"{condition}:emission only"
            emission_control_name = f"{condition}:transit only"
            for i in range(n_simulations):
                print(f"Running iteration {i + 1} on condition {condition} for model mixes")
                if transit_control_name not in mixed_model_temps:
                    mixed_model_temps[transit_control_name] = []
                    mixed_model_temp_hists[transit_control_name] = []
                if emission_control_name not in mixed_model_temps:
                    mixed_model_temps[emission_control_name] = []
                    mixed_model_temp_hists[emission_control_name] = []

                sim_pos = simulate(1000 * 60 * 60, min_temp=t_min, max_temp=t_max, bout_gen=transit_control)[0]
                temperatures = (sim_pos[:, 1]) / chamber_max * (t_max - t_min) + t_min
                mixed_model_temps[transit_control_name].append(temperatures[::2000].copy())
                h_temp = np.histogram(temperatures, tbins)[0].astype(float)
                h_temp /= np.sum(h_temp)
                mixed_model_temp_hists[transit_control_name].append(h_temp)

                sim_pos = simulate(1000 * 60 * 60, min_temp=t_min, max_temp=t_max, bout_gen=emission_control)[0]
                temperatures = (sim_pos[:, 1]) / chamber_max * (t_max - t_min) + t_min
                mixed_model_temps[emission_control_name].append(temperatures[::2000].copy())
                h_temp = np.histogram(temperatures, tbins)[0].astype(float)
                h_temp /= np.sum(h_temp)
                mixed_model_temp_hists[emission_control_name].append(h_temp)

        mixed_model_temp_hists = {k: np.vstack(mixed_model_temp_hists[k]) for k in mixed_model_temp_hists}

        fig = pl.figure()
        for i, condition in enumerate(mixed_model_temp_hists):
            m = np.mean(mixed_model_temp_hists[condition], 0)
            e = np.std(mixed_model_temp_hists[condition], 0) / np.sqrt(mixed_model_temp_hists[condition].shape[0])
            pl.plot(tbc, m, color=f"C{i}", label=condition)
            pl.fill_between(tbc, m-e, m+e, color=f"C{i}", alpha=0.4)
        # plot control
        m = np.mean(all_temp_hists[control_name], 0)
        e = np.std(all_temp_hists[control_name], 0) / np.sqrt(all_temp_hists[control_name].shape[0])
        pl.plot(tbc, m, color=f"k", label=control_name)
        pl.fill_between(tbc, m - e, m + e, color=f"k", alpha=0.4)
        # plot POLYIC
        m = np.mean(all_temp_hists["nac_POLYIC"], 0)
        e = np.std(all_temp_hists["nac_POLYIC"], 0) / np.sqrt(all_temp_hists["nac_POLYIC"].shape[0])
        pl.plot(tbc, m, "C3--", label="nac_POLYIC")
        pl.fill_between(tbc, m - e, m + e, color=f"C3", alpha=0.4)
        pl.legend()
        pl.xlabel("Temperature [C]")
        pl.ylabel("Occupancy")
        sns.despine()
        fig.savefig(path.join(plot_folder, "Simulation_MixedControls_Occupancy.pdf"))

        fig = pl.figure()
        for i, condition in enumerate(mixed_model_temp_hists):
            m = np.mean(np.cumsum(mixed_model_temp_hists[condition], axis=1), 0)
            e = np.std(np.cumsum(mixed_model_temp_hists[condition], axis=1), 0) / np.sqrt(mixed_model_temp_hists[condition].shape[0])
            pl.plot(tbc, m, color=f"C{i}", label=condition)
            pl.fill_between(tbc, m-e, m+e, color=f"C{i}", alpha=0.4)
        # plot control
        m = np.mean(np.cumsum(all_temp_hists[control_name], axis=1), 0)
        e = np.std(np.cumsum(all_temp_hists[control_name], axis=1), 0) / np.sqrt(all_temp_hists[control_name].shape[0])
        pl.plot(tbc, m, color=f"k", label=control_name)
        pl.fill_between(tbc, m - e, m + e, color=f"k", alpha=0.4)
        # plot POLYIC
        m = np.mean(np.cumsum(all_temp_hists["nac_POLYIC"], axis=1), 0)
        e = np.std(np.cumsum(all_temp_hists["nac_POLYIC"], axis=1), 0) / np.sqrt(all_temp_hists["nac_POLYIC"].shape[0])
        pl.plot(tbc, m, "C3--", label="nac_POLYIC")
        pl.fill_between(tbc, m - e, m + e, color=f"C3", alpha=0.4)
        pl.legend()
        pl.xlabel("Temperature [C]")
        pl.ylabel("Occupancy")
        sns.despine()
        fig.savefig(path.join(plot_folder, "Simulation_MixedControls_Cumulative_Occupancy.pdf"))

        # compute statistics - append the two experimental conditions
        mixed_model_temps[control_name] = all_temps[control_name]
        mixed_model_temps["nac_POLYIC"] = all_temps["nac_POLYIC"]
        names = []
        p_values = []
        cond_names = list(mixed_model_temps.keys())
        for i, cond1 in enumerate(cond_names):
            for j, cond2 in enumerate(cond_names[i+1:]):
                names.append(f"{cond1} vs. {cond2}")
                p_values.append(ks_bootstrap_test_by_fish(mixed_model_temps[cond1], mixed_model_temps[cond2], 1000)[0])
        df_sig_occupancy = generate_stat_df(np.hstack(p_values), np.hstack(names))
        df_sig_occupancy.to_csv(path.join(plot_folder, "stats_mixed_models_Occupancy.csv"))

        # compare the importance of different emission model components
        # Note: The followling lazily assumes that this will be run on the 20-30 gradient data but the plot code above
        # already does that anyway. In other words, the only non-control condition we expect is "nac_POLYIC"
        emission_test_temp_hists = {"IBI": [], "Displacement": [], "Turn": []}
        emission_test_temps = {"IBI": [], "Displacement": [], "Turn": []}
        for condition in all_temp_hists:
            if condition == control_name:
                continue
            ibi_only = mbg.SwimBoutGenerator(dfile_path, 0, 0, 2, 3, condition)
            ibi_only.swim_mode_transit = control_bout_gen.swim_mode_transit
            ibi_only.displacement_gamma = control_bout_gen.displacement_gamma
            ibi_only.turn_glm_gmm = control_bout_gen.turn_glm_gmm

            disp_only = mbg.SwimBoutGenerator(dfile_path, 0, 0, 2, 3, condition)
            disp_only.swim_mode_transit = control_bout_gen.swim_mode_transit
            disp_only.ibi_gamma = control_bout_gen.ibi_gamma
            disp_only.turn_glm_gmm = control_bout_gen.turn_glm_gmm

            turn_only = mbg.SwimBoutGenerator(dfile_path, 0, 0, 2, 3, condition)
            turn_only.swim_mode_transit = control_bout_gen.swim_mode_transit
            turn_only.ibi_gamma = control_bout_gen.ibi_gamma
            turn_only.displacement_gamma = control_bout_gen.displacement_gamma

            for i in range(n_simulations):
                print(f"Running iteration {i + 1} on condition {condition} for emission tests")
                sim_pos = simulate(1000 * 60 * 60, min_temp=t_min, max_temp=t_max, bout_gen=ibi_only)[0]
                temperatures = (sim_pos[:, 1]) / chamber_max * (t_max - t_min) + t_min
                emission_test_temps["IBI"].append(temperatures[::2000].copy())
                h_temp = np.histogram(temperatures, tbins)[0].astype(float)
                h_temp /= np.sum(h_temp)
                emission_test_temp_hists["IBI"].append(h_temp)

                sim_pos = simulate(1000 * 60 * 60, min_temp=t_min, max_temp=t_max, bout_gen=disp_only)[0]
                temperatures = (sim_pos[:, 1]) / chamber_max * (t_max - t_min) + t_min
                emission_test_temps["Displacement"].append(temperatures[::2000].copy())
                h_temp = np.histogram(temperatures, tbins)[0].astype(float)
                h_temp /= np.sum(h_temp)
                emission_test_temp_hists["Displacement"].append(h_temp)

                sim_pos = simulate(1000 * 60 * 60, min_temp=t_min, max_temp=t_max, bout_gen=turn_only)[0]
                temperatures = (sim_pos[:, 1]) / chamber_max * (t_max - t_min) + t_min
                emission_test_temps["Turn"].append(temperatures[::2000].copy())
                h_temp = np.histogram(temperatures, tbins)[0].astype(float)
                h_temp /= np.sum(h_temp)
                emission_test_temp_hists["Turn"].append(h_temp)

        emission_test_temp_hists = {k: np.vstack(emission_test_temp_hists[k]) for k in emission_test_temp_hists}

        fig = pl.figure()
        for i, condition in enumerate(emission_test_temp_hists):
            m = np.mean(emission_test_temp_hists[condition], 0)
            e = np.std(emission_test_temp_hists[condition], 0) / np.sqrt(emission_test_temp_hists[condition].shape[0])
            pl.plot(tbc, m, color=f"C{i}", label=condition)
            pl.fill_between(tbc, m-e, m+e, color=f"C{i}", alpha=0.4)
        # plot control
        m = np.mean(all_temp_hists[control_name], 0)
        e = np.std(all_temp_hists[control_name], 0) / np.sqrt(all_temp_hists[control_name].shape[0])
        pl.plot(tbc, m, color=f"k", label=control_name)
        pl.fill_between(tbc, m - e, m + e, color=f"k", alpha=0.4)
        # plot POLYIC
        m = np.mean(all_temp_hists["nac_POLYIC"], 0)
        e = np.std(all_temp_hists["nac_POLYIC"], 0) / np.sqrt(all_temp_hists["nac_POLYIC"].shape[0])
        pl.plot(tbc, m, "C3--", label="nac_POLYIC")
        pl.fill_between(tbc, m - e, m + e, color=f"C3", alpha=0.4)
        pl.legend()
        pl.xlabel("Temperature [C]")
        pl.ylabel("Occupancy")
        sns.despine()
        fig.savefig(path.join(plot_folder, "Simulation_EmissionTest_Occupancy.pdf"))

        # compute statistics for emission tests - append the two experimental conditions
        emission_test_temps[control_name] = all_temps[control_name]
        emission_test_temps["nac_POLYIC"] = all_temps["nac_POLYIC"]
        names = []
        p_values = []
        cond_names = list(emission_test_temps.keys())
        for i, cond1 in enumerate(cond_names):
            for j, cond2 in enumerate(cond_names[i+1:]):
                names.append(f"{cond1} vs. {cond2}")
                p_values.append(ks_bootstrap_test_by_fish(emission_test_temps[cond1], emission_test_temps[cond2], 1000)[0])
        df_sig_occupancy = generate_stat_df(np.hstack(p_values), np.hstack(names))
        df_sig_occupancy.to_csv(path.join(plot_folder, "stats_emission_model_tests_Occupancy.csv"))
    # end if control_name is not None
