import cmdstanpy as stan
import numpy as np
import matplotlib.pyplot as pl
import matplotlib as mpl
from os import path
import seaborn as sns
import pandas as pd
import argparse

import utilities as util
import os
import h5py
from typing import Dict, List


def fit_transition_model(train_traj: List, treatment: str) -> Dict[str, np.ndarray]:
    """
    Fit markov transition model of 3rd order
    :param train_traj: The training set trajectories
    :param treatment: The treatment name
    :return:
        [0]: Dictionary of parameter names and their draws
    """
    # train data
    train_modes = np.hstack([np.array(tj["State"]) for tj in train_traj]).astype(int)
    train_temp = np.hstack([np.array(tj["Temperature"]) for tj in train_traj])
    train_dtemp = np.hstack([np.array(tj["Prev Delta T"]) for tj in train_traj])
    train_traj_len = np.hstack([tj.shape[0] for tj in train_traj])

    data_file = f"fever_transition_{treatment}.data.json"
    data = {"T": train_modes.size,
            "M": train_traj_len.size,
            "N_POINTS": train_traj_len,
            "modes": train_modes,
            "temperatures": train_temp - 25,
            "d_temperatures": train_dtemp,
            "glm_order": 3}
    stan.write_stan_json(data_file, data)

    stan_file = path.join("models", "swim_mode_MM.stan")
    model = stan.CmdStanModel(stan_file=stan_file)
    fit = model.sample(data=data_file, iter_sampling=1000, chains=4)
    print("$$$$$$$$$$$$$$$$$$$$$$$$$$$")
    print(f"DIAGNOSTIC FOR TRANSITION MODEL ON {treatment}")
    print(fit.diagnose())
    print("$$$$$$$$$$$$$$$$$$$$$$$$$$$")

    # model results
    mm_orig = fit.stan_variable("mm_orig")
    mm_icept = fit.stan_variable("mm_icept")

    return {"mm_orig": mm_orig, "mm_icept": mm_icept}


def fit_displacement_model(train_traj: List, mode: int, treatment: str) -> Dict[str, np.ndarray]:
    """
    Fit displacement model on one swim mode
    :param train_traj: The training set trajectories
    :param mode: The swim mode on which the model should be fit
    :param treatment: The treatment name
    :return:
        [0]: Dictionary of parameter names and their draws
    """
    # train data
    train_modes = np.hstack([np.array(tj["State"]) for tj in train_traj]).astype(int)
    valid = train_modes == mode
    train_displace = np.hstack([np.array(tj["Displacement"]) for tj in train_traj])[valid]
    train_temp = np.hstack([np.array(tj["Temperature"]) for tj in train_traj])[valid]
    train_dtemp = np.hstack([np.array(tj["Prev Delta T"]) for tj in train_traj])[valid]
    train_prev_displace = np.hstack([np.array(tj["Prev displacement"]) for tj in train_traj])[valid]

    # fit background model
    data_file = f"fever_bg_displacement_{treatment}_{mode}.data.json"
    data = {"T": train_displace.size,
            "y": train_displace}
    stan.write_stan_json(data_file, data)
    stan_file = path.join("models", "fit_gamma.stan")
    model = stan.CmdStanModel(stan_file=stan_file)
    bg_fit = model.sample(data=data_file, iter_sampling=1000, chains=4)
    bg_alpha = np.mean(bg_fit.stan_variable("alpha"))
    bg_beta = np.mean(bg_fit.stan_variable("beta"))

    # fit displacement model
    data_file = f"fever_displacement_{treatment}_{mode}.data.json"
    data = {"T": train_displace.size,
            "y": train_displace,
            "temperatures": train_temp - 25,
            "d_temperatures": train_dtemp,
            "prev_displace": train_prev_displace-2,
            "beta": bg_beta}
    stan.write_stan_json(data_file, data)
    stan_file = path.join("models", "displacement_gamma_glm_2ndOrder.stan")
    model = stan.CmdStanModel(stan_file=stan_file)
    fit = model.sample(data=data_file, iter_sampling=1000, chains=4)
    print("$$$$$$$$$$$$$$$$$$$$$$$$$$$")
    print(f"DIAGNOSTIC FOR DISPLACEMENT MODEL ON {treatment}, mode {mode}")
    print(fit.diagnose())
    print("$$$$$$$$$$$$$$$$$$$$$$$$$$$")

    # model results
    return {"glm_mu_orig": fit.stan_variable("glm_mu_orig"),
            "glm_mu_icept": fit.stan_variable("glm_mu_icept"),
            "bg_beta": bg_beta,
            "bg_alpha": bg_alpha}


def fit_ibi_model(train_traj: List, mode: int, treatment: str) -> Dict[str, np.ndarray]:
    """
    Fit IBI model on one swim mode
    :param train_traj: The training set trajectories
    :param mode: The swim mode on which the model should be fit
    :param treatment: The treatment name
    :return:
        [0]: Dictionary of parameter names and their draws
    """
    # train data
    train_modes = np.hstack([np.array(tj["State"]) for tj in train_traj]).astype(int)
    valid = train_modes == mode
    train_ibi = np.hstack([np.array(tj["IBI"]) for tj in train_traj])[valid]
    train_temp = np.hstack([np.array(tj["Temperature"]) for tj in train_traj])[valid]
    train_dtemp = np.hstack([np.array(tj["Prev Delta T"]) for tj in train_traj])[valid]

    # fit background model
    data_file = f"fever_bg_ibi_{treatment}_{mode}.data.json"
    data = {"T": train_ibi.size,
            "y": train_ibi}
    stan.write_stan_json(data_file, data)
    stan_file = path.join("models", "fit_gamma.stan")
    model = stan.CmdStanModel(stan_file=stan_file)
    bg_fit = model.sample(data=data_file, iter_sampling=1000, chains=4)
    bg_alpha = np.mean(bg_fit.stan_variable("alpha"))
    bg_beta = np.mean(bg_fit.stan_variable("beta"))

    # fit ibi model
    data_file = f"fever_ibi_{treatment}_{mode}.data.json"
    data = {"T": train_ibi.size,
            "y": train_ibi,
            "temperatures": train_temp - 25,
            "d_temperatures": train_dtemp,
            "beta": bg_beta}
    stan.write_stan_json(data_file, data)
    stan_file = path.join("models", "ibi_gamma_glm_2ndOrder.stan")
    model = stan.CmdStanModel(stan_file=stan_file)
    fit = model.sample(data=data_file, iter_sampling=1000, chains=4)
    print("$$$$$$$$$$$$$$$$$$$$$$$$$$$")
    print(f"DIAGNOSTIC FOR IBI MODEL ON {treatment}, mode {mode}")
    print(fit.diagnose())
    print("$$$$$$$$$$$$$$$$$$$$$$$$$$$")

    # model results
    return {"glm_mu_orig": fit.stan_variable("glm_mu_orig"),
            "glm_mu_icept": fit.stan_variable("glm_mu_icept"),
            "bg_beta": bg_beta,
            "bg_alpha": bg_alpha}


def fit_turn_model(train_traj: List, mode: int, treatment: str) -> Dict[str, np.ndarray]:
    """
    Fit turn model on one swim mode
    :param train_traj: The training set trajectories
    :param mode: The swim mode on which the model should be fit
    :param treatment: The treatment name
    :return:
        [0]: Dictionary of parameter names and their draws
    """
    # train data
    train_modes = np.hstack([np.array(tj["State"]) for tj in train_traj]).astype(int)
    valid = train_modes == mode
    train_angles = np.rad2deg(np.hstack([np.array(tj["Angle change"]) for tj in train_traj]))[valid]
    train_temp = np.hstack([np.array(tj["Temperature"]) for tj in train_traj])[valid]
    train_dtemp = np.hstack([np.array(tj["Prev Delta T"]) for tj in train_traj])[valid]
    train_prev_angles = np.rad2deg(np.hstack([np.array(tj["Prev angle"]) for tj in train_traj]))[valid]
    # perform turn angle random flipping to even out handedness biases
    flip = np.random.rand(train_angles.size) > 0.5
    train_angles[flip] = -1 * train_angles[flip]
    train_prev_angles[flip] = -1 * train_prev_angles[flip]
    # Fit background GMM on all training data within the chosen mode to determine distributions
    data_file = f"fever_bg_turn_{treatment}_{mode}.data.json"
    data = {
        "T": train_angles.size,
        "y": train_angles
    }
    stan.write_stan_json(data_file, data)
    stan_file = path.join("models", "turn_GMM.stan")
    model = stan.CmdStanModel(stan_file=stan_file)
    gmm_fit = model.sample(data=data_file, iter_sampling=250)
    turn_mu = gmm_fit.stan_variable('turn_mu').mean()
    straight_std = gmm_fit.stan_variable('straight_std').mean()
    turn_var = gmm_fit.stan_variable('turn_var').mean()
    theta_mix = gmm_fit.stan_variable('theta_mix').mean(axis=0)
    gmm_avg_mix = np.array([theta_mix[0]*0.5 + theta_mix[2]*0.5, theta_mix[1], theta_mix[0]*0.5 + theta_mix[2]*0.5])
    turn_alpha = gmm_fit.stan_variable('turn_alpha').mean()
    turn_beta = gmm_fit.stan_variable('turn_beta').mean()

    # Fit Gamma-Gaussian GLM mixture model
    data_file = f"fever_turn_{treatment}_{mode}.data.json"
    data = {"T": train_angles.size,
            "y": train_angles,
            "temperatures": train_temp - 25,
            "d_temperatures": train_dtemp,
            "prev_angles": train_prev_angles,
            "turn_mu": turn_mu,
            "straight_std": straight_std,
            "turn_var": turn_var}
    stan.write_stan_json(data_file, data)
    stan_file = path.join("models", "turn_GLM_GMM_2ndOrder.stan")
    model = stan.CmdStanModel(stan_file=stan_file)
    fit = model.sample(data=data_file, iter_sampling=1000, chains=4)
    print("$$$$$$$$$$$$$$$$$$$$$$$$$$$")
    print(f"DIAGNOSTIC FOR TURN MODEL ON {treatment}, mode {mode}")
    print(fit.diagnose())
    print("$$$$$$$$$$$$$$$$$$$$$$$$$$$")

    # model results
    return{
        "straight_std": straight_std,
        "turn_alpha": turn_alpha,
        "turn_beta": turn_beta,
        "gmm_mix_orig": fit.stan_variable("gmm_mix_orig"),
        "gmm_mix_icept": fit.stan_variable("gmm_mix_icept"),
        "turn_avg_mix": gmm_avg_mix
    }


def fit_all_components(path_to_bouts: str) -> Dict[str, Dict]:
    """
    Fit all component models (transition and emission) on one experimental condition
    :param path_to_bouts: Path to the directory with all bout pickles
    :return: Dictionary of model types and associated model parameter draws
    """
    all_trajectories = util.load_all_trajectories(path_to_bouts, len_thresh=3)
    train_traj, test_traj = util.split_train_test(all_trajectories, test_every=5)
    fits = {
        "Transition": fit_transition_model(train_traj, path.split(path_to_bouts)[-1]),
    }
    fits |= {f"Displacement_{m}": fit_displacement_model(train_traj, m, path.split(path_to_bouts)[-1]) for m in [-1, 0, 1]}
    fits |= {f"IBI_{m}": fit_ibi_model(train_traj, m, path.split(path_to_bouts)[-1]) for m in [-1, 0, 1]}
    fits |= {f"Turn_{m}": fit_turn_model(train_traj, m, path.split(path_to_bouts)[-1]) for m in [-1, 0, 1]}
    return fits


def tm_from_glm(t: float, dt: float, glm_orig_coef: np.ndarray, glm_icept: np.ndarray) -> pd.DataFrame:
    """
    From temperature and delta-temp inputs uses the glm coefficients to compute the transition matrix
    :param t: The temperature at which to compute the transition matrix
    :param dt: The delta-temperature at which to compute the transition matrix
    :param glm_orig_coef: The glm coefficients of the Markov Model transition
    :param glm_icept: The glm intercepts of the Markov Model transition
    :return: The transition matrix
    """
    trans_mat = np.zeros((3, 3))
    lp = np.zeros(3)
    design = np.array([t, dt, t*dt, t**2, dt**2, t*t*dt, t*dt*dt, t**3, dt**3])
    # Transitions from reversal
    lp[0] = glm_orig_coef[0, :, 0] @ design + glm_icept[0, 0]
    lp[1] = glm_orig_coef[0, :, 1] @ design + glm_icept[0, 1]
    lp[2] = 0
    exp_lp = np.exp(lp)
    for k in range(3):
        trans_mat[0, k] = exp_lp[k] / np.sum(exp_lp)
    # Transitions from general
    lp[0] = glm_orig_coef[1, :, 0] @ design + glm_icept[1, 0]
    lp[1] = glm_orig_coef[1, :, 1] @ design + glm_icept[1, 1]
    lp[2] = 0
    exp_lp = np.exp(lp)
    for k in range(3):
        trans_mat[1, k] = exp_lp[k] / np.sum(exp_lp)
    # Transitions from persistent
    lp[0] = glm_orig_coef[2, :, 0] @ design + glm_icept[2, 0]
    lp[1] = glm_orig_coef[2, :, 1] @ design + glm_icept[2, 1]
    lp[2] = 0
    exp_lp = np.exp(lp)
    for k in range(3):
        trans_mat[2, k] = exp_lp[k] / np.sum(exp_lp)
    return pd.DataFrame(trans_mat, index=["R(o)", "X(o)", "P(o)"], columns=["R(i)", "X(i)", "P(i)"])


def compute_steady_state(coefs, icepts, all_temps, all_dtemps, for_stats=False):
    p_reversal = np.zeros((all_dtemps.size, all_temps.size))
    p_x = np.zeros((all_dtemps.size, all_temps.size))
    p_persist = np.zeros((all_dtemps.size, all_temps.size))
    n_samples = 1000 if for_stats else 100
    p_reversal_draws = np.zeros((n_samples, all_dtemps.size, all_temps.size))
    n_draws = coefs.shape[0]
    for i, temps in enumerate(all_temps):
        for j, dtemps in enumerate(all_dtemps):
            prob_sum = np.zeros(3)
            for s in range(n_samples):
                draw = np.random.randint(n_draws)
                tm = np.array(tm_from_glm(temps, dtemps, coefs[draw], icepts[draw]))
                tm_ss = np.linalg.matrix_power(tm, 100)
                p_reversal_draws[s, j, i] = tm_ss[0, 0]
                prob_sum += tm_ss[0]
            p_reversal[j, i] = prob_sum[0]/n_samples
            p_x[j, i] = prob_sum[1]/n_samples
            p_persist[j, i] = prob_sum[2]/n_samples
    p_reversal = pd.DataFrame(p_reversal, index=np.round(all_dtemps, 2), columns=np.round(all_temps+25, 2))
    p_x = pd.DataFrame(p_x, index=np.round(all_dtemps, 2), columns=np.round(all_temps+25, 2))
    p_persist = pd.DataFrame(p_persist, index=np.round(all_dtemps, 2), columns=np.round(all_temps+25, 2))

    if not for_stats:
        return p_reversal, p_x, p_persist
    else:
        return p_reversal, p_x, p_persist, p_reversal_draws


def compute_stimulus_valence(threshold=0.025):
    cooling = all_dtemps < -threshold
    heating = all_dtemps > threshold
    no_change = np.logical_not(np.logical_or(heating, cooling))
    valences = {
        "Heating": {k: np.zeros((all_temps.size, 3)) for k in ss_reversal},
        "Cooling": {k: np.zeros((all_temps.size, 3)) for k in ss_reversal}
    }
    for cnd in ss_reversal:
        avg = np.mean(ss_reversal[cnd], axis=0)
        lower = np.percentile(ss_reversal[cnd], 5, axis=0)
        upper = np.percentile(ss_reversal[cnd], 95, axis=0)
        for ix, temp in enumerate(all_temps):
            valences["Heating"][cnd][ix, 0] = np.mean(lower[no_change, ix]) / np.mean(lower[heating, ix])
            valences["Cooling"][cnd][ix, 0] = np.mean(lower[no_change, ix]) / np.mean(lower[cooling, ix])
            valences["Heating"][cnd][ix, 1] = np.mean(avg[no_change, ix]) / np.mean(avg[heating, ix])
            valences["Cooling"][cnd][ix, 1] = np.mean(avg[no_change, ix]) / np.mean(avg[cooling, ix])
            valences["Heating"][cnd][ix, 2] = np.mean(upper[no_change, ix]) / np.mean(upper[heating, ix])
            valences["Cooling"][cnd][ix, 2] = np.mean(upper[no_change, ix]) / np.mean(upper[cooling, ix])
    return valences


if __name__ == '__main__':
    a_parser = argparse.ArgumentParser(prog="fit_fever_models.py",
                                       description="Fits navigation models across different experimental conditions")
    a_parser.add_argument("-f", "--folder", help="Path to base-folder with treatment folders of "
                                                 "bout pickles for each condition", type=str, action=util.CheckArgs)

    mpl.rcParams['pdf.fonttype'] = 42

    args = a_parser.parse_args()

    base_folder = args.folder

    # find experimental subfolders
    conditions = [path.join(base_folder, f) for f in os.listdir(base_folder) if "plots" not in f and path.isdir(path.join(base_folder, f))]

    plot_dir = path.join(base_folder, "plots")
    if not path.exists(plot_dir):
        os.makedirs(plot_dir)

    if path.exists(path.join(plot_dir, "fit.h5")):
        all_dtemps = np.linspace(-0.1, 0.1, 111)
        all_temps = np.linspace(20 - 25, 30 - 25, 101)
        ss_reversal = {}
        gen_tmix = {}
        with h5py.File(path.join(plot_dir, "fit.h5"), "r") as dfile:
            for cond in dfile.keys():
                cond_grp = dfile[cond]
                mm_orig = cond_grp["Transition"]["mm_orig"][()]
                mm_icept = cond_grp["Transition"]["mm_icept"][()]
                gen_state_tmix = cond_grp["Turn_0"]["gmm_mix_orig"][()]
                p_rev, p_gen, p_pers, p_rev_draws = compute_steady_state(mm_orig, mm_icept, all_temps, all_dtemps, True)
                fig, axes = pl.subplots(ncols=3, figsize=(18, 4.8))
                sns.heatmap(p_rev, ax=axes[0], vmin=0, vmax=0.35)
                axes[0].set_title("Reversal")
                sns.heatmap(p_pers, ax=axes[1], vmin=0, vmax=0.7)
                axes[1].set_title("Persistent")
                sns.heatmap(p_gen, ax=axes[2], vmin=0, vmax=0.8)
                axes[2].set_title("General")
                fig.savefig(path.join(plot_dir, f"Steady_state_mode_probabilities_{path.split(cond)[-1]}.pdf"))
                gen_tmix[cond] = gen_state_tmix
                ss_reversal[cond] = p_rev_draws

        # plot valence of heating stimuli based on log ratio of reversal probability heating/no-context
        valence = compute_stimulus_valence(0.025)
        fig = pl.figure()
        for i, cond in enumerate(ss_reversal):
            pl.fill_between(all_temps+25, np.log2(valence["Heating"][cond][:, 0]),
                            np.log2(valence["Heating"][cond][:, 2]), color=f"C{i}", alpha=0.4)
            pl.plot(all_temps+25, np.log2(valence["Heating"][cond][:, 1]), f"C{i}", label=f"Heating {cond}")
            pl.fill_between(all_temps + 25, np.log2(valence["Cooling"][cond][:, 0]),
                            np.log2(valence["Cooling"][cond][:, 2]), color=f"C{i}", alpha=0.4)
            pl.plot(all_temps+25, np.log2(valence["Cooling"][cond][:, 1]), f"C{i}--", label=f"Cooling {cond}")
        pl.plot(all_temps+25, np.zeros(all_temps.size), 'k--')
        pl.xlabel("Temperature [C]")
        pl.ylabel("Valence")
        sns.despine()
        pl.legend()
        fig.savefig(path.join(plot_dir, "valence.pdf"))
        # determine robustness of valence differences between conditions to threshold changes - but we only do this for
        # the initial experiments
        if len(conditions) == 2:
            c_control = path.split(conditions[0])[-1]
            c_fever = path.split(conditions[1])[-1]
            dt_thresholds = np.linspace(0, 0.1)
            lrd_heating = np.full(dt_thresholds.size, np.nan)
            lrd_cooling = lrd_heating.copy()
            for i, th in enumerate(dt_thresholds):
                valence = compute_stimulus_valence(th)
                lrd_heating[i] = np.mean(np.log2(valence["Heating"][c_fever][:, 1])) - np.mean(np.log2(valence["Heating"][c_control][:, 1]))
                lrd_cooling[i] = np.mean(np.log2(valence["Cooling"][c_fever][:, 1])) - np.mean(np.log2(valence["Cooling"][c_control][:, 1]))
            fig = pl.figure()
            pl.plot(dt_thresholds, lrd_heating, "C3", label="Heating")
            pl.plot(dt_thresholds, lrd_cooling, "C1", label="Cooling")
            pl.plot([0.025, 0.025], [-0.1, 0.1], 'k--')
            pl.xlabel("Change threshold")
            pl.ylabel("Valence difference")
            pl.legend()
            sns.despine()
            fig.savefig(path.join(plot_dir, "valence_threshold_compare.pdf"))


        # For temp/delta-temp grid, at each point compute the derivative of the log-probability of making a turn in
        # the positive direction with respect to the turn angle input. Note, that the PA**2 input into the model
        # means that the derivative also depends on the turn angle itself. We will compute it around PA=0, i.e. a
        # linear expansion around 0
        # (looking at the parameters acting on PA*2, they are negative across states across treatments for
        # left and right turns. This means that for larger previous angles, the chance of making either a left or a
        # right turn decreases.)
        def p_maintain(gmm_params: np.ndarray, temp: float, dtemp: float, pa: float, n_draws) -> float:
            """
            For the given angle, compute the probability to produce a turn of the same angle rather than the opposite
            turn.
            :param gmm_params:
            :param temp:
            :param dtemp:
            :param pa:
            :param n_draws:
            :return:
            """
            assert gmm_params.shape[2] == 10
            assert pa > 0  # because of symmetrie we don't really need to consider negative turn angles
            design_left = np.array([temp, dtemp, temp*dtemp, temp**2, dtemp**2, -pa, -pa*temp, -pa*dtemp, -pa*temp*dtemp, pa**2])
            design_right = np.array([temp, dtemp, temp*dtemp, temp**2, dtemp**2, pa, pa*temp, pa*dtemp, pa*temp*dtemp, pa**2])
            retval = 0
            for i in range(n_draws):
                draw_ix = np.random.randint(gmm_params.shape[0])
                mix_left = gmm_params[draw_ix, 0, :]
                mix_right = gmm_params[draw_ix, 1, :]
                lp_same = np.dot(design_left, mix_left) + np.dot(design_right, mix_right)
                lp_flip = np.dot(design_left, mix_right) + np.dot(design_right, mix_left)
                e_same = np.exp(lp_same)
                e_flip = np.exp(lp_flip)
                p_mnt = e_same / (e_same + e_flip)
                retval += p_mnt
            return retval / n_draws

        pmnt = {cond: np.full((all_dtemps.size, all_temps.size), np.nan) for cond in gen_tmix}
        for cond in pmnt:
            for i, dt in enumerate(all_dtemps):
                for j, t in enumerate(all_temps):
                    pmnt[cond][i, j] = (p_maintain(gen_tmix[cond], t, dt, 30, 100))
        for cond in pmnt:
            dp_mnt = pd.DataFrame(pmnt[cond], index=np.round(all_dtemps, 2), columns=np.round(all_temps + 25, 2))
            fig = pl.figure()
            sns.heatmap(dp_mnt, center=0.5, vmin=0.4, vmax=0.6)
            fig.savefig(path.join(plot_dir, f"{cond}_GeneralState_pMaintain.pdf"))

    else:
        with h5py.File(path.join(plot_dir, "fit.h5"), "x") as dfile:
            for cond in conditions:
                cond_grp = dfile.create_group(path.split(cond)[-1])
                model_fits = fit_all_components(cond)
                for k in model_fits:
                    model_grp = cond_grp.create_group(k)
                    for kk in model_fits[k]:
                        model_grp.create_dataset(kk, data=model_fits[k][kk])
                all_dtemps = np.linspace(-0.1, 0.1, 101)
                all_temps = np.linspace(20 - 25, 30 - 25, 101)
                p_rev, p_gen, p_pers = compute_steady_state(model_fits["Transition"]["mm_orig"],
                                                            model_fits["Transition"]["mm_icept"],
                                                            all_temps, all_dtemps)

                fig, axes = pl.subplots(ncols=3, figsize=(18, 4.8))
                sns.heatmap(p_rev, ax=axes[0], vmin=0, vmax=0.35)
                axes[0].set_title("Reversal")
                sns.heatmap(p_pers, ax=axes[1], vmin=0, vmax=0.7)
                axes[1].set_title("Persistent")
                sns.heatmap(p_gen, ax=axes[2], vmin=0, vmax=0.8)
                axes[2].set_title("General")
                fig.savefig(path.join(plot_dir, f"Steady_state_mode_probabilities_{path.split(cond)[-1]}.pdf"))