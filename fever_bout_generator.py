"""
Module implementing a class that will generate swim bouts based on the swim-mode Markov Model
"""

from dataclasses import dataclass
from typing import Optional, Dict, List, Tuple
import numpy as np
import h5py


@dataclass
class Diagnostics:
    """Additional diagnostic data about the bout generation process"""
    prev_bout_angle_deg: float
    prev_bout_disp_mm: float
    prev_swim_mode: int
    temperature: float
    delta_temperature: float
    ibi_mu: float
    ibi_beta: float
    disp_mu: float
    disp_beta: float
    straight_std: float
    turn_alpha: float
    turn_beta: float
    turn_mixing_probs: np.ndarray
    transit_probs: np.ndarray


@dataclass
class SwimBout:
    """Class representing a generated swim bout"""
    pre_bout_ibi_ms: float
    bout_displacement_mm: float
    bout_angle_deg: float
    swim_mode: int
    # Optional diagnostics
    diagnostics: Optional[Diagnostics]


class SwimBoutGenerator:
    """Class generating swim bouts based on fit parameters of MarkovModel and emissions"""
    def __init__(self, data_file: str, starting_mode: int, prev_bout_angle: float, prev_bout_disp: float, order=3, treatment: Optional[str]=None):
        """
        Generate a new SwimBoutGenerator
        :param data_file: Path to file with fit data
        :param starting_mode: The initial swim mode of the fish
        :param prev_bout_angle: At the start the intended previous swim bout angle
        :param prev_bout_disp: At the start the intended previous swim bout displacement
        :param order: The order of the transition model
        :param treatment: If treatment specific models should be loaded, the name of the treatment
        """
        self.order = order
        with h5py.File(data_file, 'r') as f:
            if treatment is None:
                if order == 3:
                    self.swim_mode_transit = {"mm_icept": f["Swim_mode_transit"]["mm_icept"][()],
                                              "mm_orig": f["Swim_mode_transit"]["mm_orig"][()]}
                elif order == 0:
                    self.swim_mode_transit = {"mm_icept": f["Swim_mode_transit_Order_0"]["mm_icept"][()],
                                              "mm_orig": None}
                else:
                    self.swim_mode_transit = {"mm_icept": f[f"Swim_mode_transit_Order_{order}"]["mm_icept"][()],
                                              "mm_orig": f[f"Swim_mode_transit_Order_{order}"]["mm_orig"][()]}
                self.displacement_gamma = self._load_emission_params(f["Displacement_Gamma"])
                self.ibi_gamma = self._load_emission_params(f["IBI_Gamma"])
                self.turn_glm_gmm = self._load_emission_params(f["Turn_GLM_GMM"])
            else:
                self._load_treatment_params(f, treatment)
        self.previous_mode = starting_mode
        self.previous_bout_angle = prev_bout_angle
        self.previous_bout_disp = prev_bout_disp

    def _load_treatment_params(self, dfile: h5py.File, treatment: str):
        trt_grp = dfile[treatment]
        # Transition model
        self.swim_mode_transit = {"mm_icept": trt_grp["Transition"]["mm_icept"][()],
                                  "mm_orig": trt_grp["Transition"]["mm_orig"][()]}
        # Emission models
        params = {}
        for emit in ["Displacement", "IBI", "Turn"]:
            params[emit] = {}
            for mode in [-1, 0, 1]:
                mode_grp = trt_grp[f"{emit}_{mode}"]
                params[emit][mode] = {k: mode_grp[k][()] for k in mode_grp}
        self.displacement_gamma = params["Displacement"]
        self.ibi_gamma = params["IBI"]
        self.turn_glm_gmm = params["Turn"]

    @staticmethod
    def _load_emission_params(model_grp: h5py.Group) -> Dict:
        d = {}
        for mode in [-1, 0, 1]:
            d[mode] = {}
            for k in model_grp[f"Mode_{mode}"].keys():
                d[mode][k] = model_grp[f"Mode_{mode}"][k][()]
        return d

    @staticmethod
    def _select_parameter(draws: list[np.ndarray]) -> List:
        selector = np.random.randint(draws[0].shape[0])
        return [d[selector] for d in draws]

    @staticmethod
    def _design_matrix(t: float, dt: float, use3rd: bool, pa: Optional[float] = None) -> np.ndarray:
        if use3rd and pa is not None:
            return np.hstack([t, dt, t * dt, t ** 2, dt ** 2, pa, pa * t, pa * dt, pa * t * dt, pa ** 2])
        if use3rd:
            return np.array([t, dt, t * dt, t ** 2, dt ** 2, (t**2)*dt, t*(dt**2), t**3, dt**3])
        if pa is None:
            return np.array([t, dt, t * dt, t ** 2, dt ** 2])
        else:
            return np.hstack([t, dt, t * dt, t ** 2, dt ** 2, pa, pa * t, pa * dt, pa ** 2])

    def _draw_disp_ibi(self, mp: dict, t: float, dt: float, max_val: float, prev: Optional[float]) -> Tuple[float, float, float]:
        """
        Draw sample from either displacement or ibi distribution (since both are gamma models)
        :param mp: The mode parameter dictionary
        :param t: The standardized temperature
        :param dt: The delta-temperature of the previous bout
        :param max_val: The maximum allowed value since gamma distributions have very long tails unlike behavior
        :param prev: The previous displacement
        :return:
            [0]: a drawn displacement or ibi
            [1]: The distribution average
            [2]: The beta parameter
        """
        m_i, m_o = self._select_parameter([mp["glm_mu_icept"], mp["glm_mu_orig"]])
        beta = mp["bg_beta"]
        design = self._design_matrix(t, dt, False, prev)
        mu_glm = np.dot(design, m_o) + m_i
        mu = max_val/(1+np.exp(-mu_glm))
        out = np.inf
        retry_count = 0
        while out > max_val*1.5:
            out = np.random.gamma(shape=mu*beta, scale=1/beta)
            if retry_count > 10:
                out = max_val / 2
                break
        return out, mu, beta

    def draw_displacement(self, mode: int, t: float, dt: float) -> Tuple[float, float, float]:
        """
        Draw a swim displacement from the appropriate distribution
        :param mode: The swim mode
        :param t: The standardized temperature
        :param dt: The delta-temperature of the previous bout
        :return:
            [0]: a drawn bout displacement given the inputs
            [1]: The alpha parameter
            [2]: The beta parameter
        """
        mp = self.displacement_gamma[mode]
        # In the following we need to copy the standardization of the previous bout displacement - during
        # fitting the average displacement (~ 2mm) was subtracted to avoid having a model input that is
        # exclusively > 0
        return self._draw_disp_ibi(mp, t, dt, 10, self.previous_bout_disp-2)

    def draw_ibi(self, mode: int, t: float, dt: float) -> Tuple[float, float, float]:
        """
        Draw an interbout interval from the appropriate distribution
        :param mode: The swim mode
        :param t: The standardized temperature
        :param dt: The delta-temperature of the previous bout
        :return:
            [0]: a drawn bout ibi given the inputs
            [1]: The alpha parameter
            [2]: The beta parameter
        """
        mp = self.ibi_gamma[mode]
        return self._draw_disp_ibi(mp, t, dt, 3000, None)

    def draw_turn_angle(self, mode: int, t: float, dt: float) -> Tuple[float, float, float, float, np.ndarray]:
        """
        Draw a turn angle from the appropriate distribution
        :param mode: The swim mode
        :param t: The standardized temperature
        :param dt: The delta-temperature of the previous bout
        :return:
            [0]: a drawn bout turn angle given the inputs
            [1]: The standard deviation of the straight distribution
            [2]: The alpha parameter of the left and right distributions
            [3]: The beta parameter of the left and right distributions
            [4]: The mixing probabilities
        """
        mp = self.turn_glm_gmm[mode]
        mix_i, mix_o = self._select_parameter([mp["gmm_mix_icept"], mp["gmm_mix_orig"]])
        str_std = mp["straight_std"]
        t_alpha = mp["turn_alpha"]
        t_beta = mp["turn_beta"]
        design = self._design_matrix(t, dt, True, self.previous_bout_angle)
        lpmix1 = np.dot(design, mix_o[0, :]) + mix_i
        lpmix2 = 0
        lpmix3 = np.dot(design, mix_o[1, :]) + mix_i
        lse = np.log(np.exp(lpmix1) + np.exp(lpmix2) + np.exp(lpmix3))
        p = np.hstack([np.exp(lpmix1 - lse), np.exp(lpmix2 - lse), np.exp(lpmix3 - lse)])
        ta = np.inf
        retry_count = 0
        while np.abs(ta) > 150:
            mix = np.random.choice(a=3, p=p)
            if mix == 0:
                ta = -np.random.gamma(shape=t_alpha, scale=1/t_beta)
            elif mix == 1:
                ta = np.random.normal() * str_std
            else:
                ta = np.random.gamma(shape=t_alpha, scale=1/t_beta)
            retry_count += 1
            if retry_count > 10:
                ta = 75 if mix == 2 else -75
                break
        return ta, str_std, t_alpha, t_beta, p

    def draw_next_mode(self, t: float, dt: float) -> Tuple[int, np.ndarray]:
        """
        Draw the next swim mode
        :param t: The standardized temperature
        :param dt: The delta-temperature of the previous bout
        :return:
            [0]: The next swim mode
            [1]: The calculated transition matrix
        """
        pd = self.swim_mode_transit
        if self.order > 0:
            mm_i, mm_o = self._select_parameter([pd["mm_icept"], pd["mm_orig"]])
        else:
            mm_i = self._select_parameter([pd["mm_icept"]])[0]
            mm_o = np.zeros((3, 9, 2))
        trans_mat = np.zeros((3, 3))
        lp = np.zeros(3)
        design = self._design_matrix(t, dt, True)
        # If our model order is reduced, reduce the size of the design matrix to accomodate
        if self.order == 1:
            design = design[:2]
        elif self.order == 2:
            design = design[:5]
        # Transitions from reversal
        lp[0] = mm_o[0, :, 0] @ design + mm_i[0, 0]
        lp[1] = mm_o[0, :, 1] @ design + mm_i[0, 1]
        lp[2] = 0
        exp_lp = np.exp(lp)
        for k in range(3):
            trans_mat[0, k] = exp_lp[k] / np.sum(exp_lp)
        # Transitions from general
        lp[0] = mm_o[1, :, 0] @ design + mm_i[1, 0]
        lp[1] = mm_o[1, :, 1] @ design + mm_i[1, 1]
        lp[2] = 0
        exp_lp = np.exp(lp)
        for k in range(3):
            trans_mat[1, k] = exp_lp[k] / np.sum(exp_lp)
        # Transitions from persistent
        lp[0] = mm_o[2, :, 0] @ design + mm_i[2, 0]
        lp[1] = mm_o[2, :, 1] @ design + mm_i[2, 1]
        lp[2] = 0
        exp_lp = np.exp(lp)
        for k in range(3):
            trans_mat[2, k] = exp_lp[k] / np.sum(exp_lp)
        # our modes are -1, 0 and 1 which have to be translated back and forth btw indexes 0, 1, 2 in the transition
        # matrix
        p_transit = trans_mat[self.previous_mode+1]
        return np.random.choice(a=3, p=p_transit)-1, trans_mat

    def draw_next_bout(self, t: float, dt: float, force_mode_0: bool, diagnostic=False) -> SwimBout:
        """
        Draw the next bout and update internal information on previous mode and previous turn angle
        :param t: The current temperature in degree celsius
        :param dt: The delta-temperature of the previous bout
        :param force_mode_0: If true, no swim mode transitions will be performed
        :param diagnostic: If set to true, SwimBout will be returned with full generator information
        :return: Swim bout
        """
        # Transition to next swim mode
        if force_mode_0:
            new_mode = 0
            trans_mat = np.full((3, 3), np.nan)
        else:
            new_mode, trans_mat = self.draw_next_mode(t-25, dt)
        # Draw bout kinematics
        disp, disp_mu, disp_beta = self.draw_displacement(new_mode, t-25, dt)
        ibi, ibi_mu, ibi_beta = self.draw_ibi(new_mode, t-25, dt)
        angle, str_std, t_alpha, t_beta, mix_p = self.draw_turn_angle(new_mode, t-25, dt)
        if diagnostic:
            diag = Diagnostics(prev_bout_angle_deg=self.previous_bout_angle,
                               prev_bout_disp_mm=self.previous_bout_disp,
                               prev_swim_mode=self.previous_mode,
                               temperature=t,
                               delta_temperature=dt,
                               ibi_mu=ibi_mu,
                               ibi_beta=ibi_beta,
                               disp_mu=disp_mu,
                               disp_beta=disp_beta,
                               straight_std=str_std,
                               turn_alpha=t_alpha,
                               turn_beta=t_beta,
                               turn_mixing_probs=mix_p,
                               transit_probs=trans_mat)
        else:
            diag = None
        # Update internal information
        self.previous_mode = new_mode
        self.previous_bout_angle = angle
        self.previous_bout_disp = disp
        return SwimBout(ibi, disp, angle, new_mode, diag)
