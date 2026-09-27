# -*- coding: utf-8 -*-
"""
BACKEND — aucune dépendance à Streamlit. Prédit uniquement le montant (DthAmt), avec LightGBM seul.

Sections : 1. Constantes  2. Préparation des données (+ cohorte)  3. Strates : tranches personnalisables,
crédibilité, suggestion de fusion  4. Sélection dynamique (colonnes secondaires seulement)
5. Entraînement LightGBM, pli par pli  6. Tendance temporelle (appliquée APRÈS la prédiction)
7. Métriques (RMSE, déviance de Poisson, Gini, double lift)  8. Sauvegarde / chargement / prédiction
"""
import math
import pickle
import time
import warnings
from statistics import NormalDist

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

warnings.filterwarnings("ignore")

# ============================================================================================
# SECTION 1 — Constantes
# ============================================================================================
TARGET = "DthAmt"
REQUIRED = ["Year", "Sex", "Smoke", "PolTypeGrp", "Base", "Par", "Size", "IssueAge",
            "PolYear", "ExposCnt", "ExposAmt", "ExposAmt2", "ExpecCnt", "ExpecAmt"]
CRITIQUES = ["Year", "Size", "IssueAge", "PolYear", "ExposCnt", "ExposAmt", "ExpecCnt", "ExpecAmt"]

KEY = ["Sex", "Smoke", "PolGrp", "Base", "Par", "Size", "IssueAge", "IssueYear"]
GRP_MAP = {1: 1, 2: 2, 3: 3, 4: 3, 5: 3, 11: 3, 6: 6, 7: 6, 10: 6, 8: 8, 9: 9}
N_PLEIN = 3006

LIBELLES = {
    "Sex": {1: "Homme", 2: "Femme"},
    "Smoke": {1: "Fumeur", 2: "Non-fumeur", 3: "Inconnu"},
    "PolGrp": {1: "Vie entière", 2: "T100", 3: "Vie universelle", 6: "Temporaire 10/20 ans",
               8: "Autre temporaire", 9: "Autre"},
    "Size": {1: "0-10k", 2: "10-50k", 3: "50-100k", 4: "100-250k", 5: "250-500k",
             6: "500k-1m", 7: "1-2m", 8: "2m+"},
    "Par": {0: "Sans participation", 1: "Avec participation"},
    "Base": {0: "Avenant", 1: "Contrat de base"},
}

# Le socle : toujours inclus dans le modèle, jamais soumis à la sélection dynamique.
SOCLE = ["Sex", "Smoke", "PolYear", "AttdAge", "Size"]

# Colonnes secondaires : seules celles-ci passent par la sélection dynamique (section 4).
CANDIDATS_SECONDAIRES = {
    "PolGrp":      "Type de police regroupé",
    "Base":        "Contrat de base ou avenant",
    "Par":         "Avec ou sans participation",
    "IssueYear":   "Année d'émission (cohorte de souscription)",
    "log_qx_amt":  "Log du taux de mortalité implicite de la table",
    "log_avg_size":"Log du capital moyen par police",
    "cv2":         "Dispersion des capitaux dans la cellule",
    "log_expos":   "Log du nombre de polices exposées",
    "coh_ae_amt":  "Réel/attendu des autres lignes de la même cohorte",
    "coh_log_E":   "Poids de l'information derrière coh_ae_amt",
}

# Colonnes numériques qu'on peut découper soi-même en tranches, à l'étape « Strates ».
COLONNES_DECOUPABLES = ["AttdAge", "PolYear", "YearStart"]

# Fourchette de référence pour le taux de tendance (source : recherche ICA, voir conversation).
TAUX_TENDANCE_MIN = 0.010
TAUX_TENDANCE_MAX = 0.019
TAUX_TENDANCE_SOURCE = ("Fourchette raisonnable selon la recherche de l'ICA (CIA-MI-2024) pour un taux "
                        "d'amélioration de la mortalité à long terme : 1,0 % à 1,9 % par année.")


# ============================================================================================
# SECTION 2 — Préparation des données (+ cohorte)
# ============================================================================================
def lire_csv(fichier_bytes_ou_chemin, sep=None, decimal="."):
    import io
    buf = io.BytesIO(fichier_bytes_ou_chemin) if isinstance(fichier_bytes_ou_chemin, (bytes, bytearray)) else fichier_bytes_ou_chemin
    return pd.read_csv(buf, sep=sep, decimal=decimal, engine="python" if sep is None else "c")


def diagnostic_colonnes(df):
    return dict(manque=[c for c in REQUIRED if c not in df.columns], a_target=TARGET in df.columns,
               colonnes=list(df.columns))


def _lignes_invalides(d, cols):
    x = d[cols].apply(pd.to_numeric, errors="coerce").to_numpy(float)
    return ~np.isfinite(x).all(axis=1)


def nettoyer(df, resultats=None):
    rapport = []
    for nom, d, cols in (("df", df, REQUIRED + [TARGET]), ("resultats", resultats, REQUIRED)):
        if d is None:
            continue
        for c in cols:
            if c not in d.columns:
                continue
            x = pd.to_numeric(d[c], errors="coerce")
            bad = int((~np.isfinite(x.to_numpy(float))).sum())
            if bad:
                rapport.append(f"{nom} · {c} : {bad:,} valeurs manquantes/infinies".replace(",", " "))
    erreur_res = None
    if resultats is not None:
        cols_ok = [c for c in CRITIQUES if c in resultats.columns]
        bad_r = _lignes_invalides(resultats, cols_ok)
        if bad_r.any():
            erreur_res = (f"{int(bad_r.sum())} lignes de resultats ont une valeur manquante/infinie dans "
                         f"{cols_ok} : impossible de les prédire tant qu'elles ne sont pas corrigées.")
    bad = _lignes_invalides(df, [c for c in CRITIQUES + [TARGET] if c in df.columns])
    n_retirees = int(bad.sum())
    if n_retirees:
        rapport.append(f"ATTENTION : {n_retirees:,} lignes de df retirées de l'entraînement.".replace(",", " "))
        df = df.loc[~bad].copy()
    return df, rapport, erreur_res


def prep(d):
    """Ajoute toutes les colonnes calculées."""
    d = d.copy()
    d["Year"] = pd.to_numeric(d["Year"]).astype(int)
    d["YearStart"] = d["Year"] // 100 + 2000
    d["PolGrp"] = d["PolTypeGrp"].map(GRP_MAP).fillna(9).astype(int)
    if "AttdAge" not in d.columns:
        d["AttdAge"] = d["IssueAge"] + d["PolYear"] - 1
    d["IssueYear"] = d["YearStart"] - d["PolYear"]
    e = d["ExposCnt"].clip(lower=1e-9)
    a = d["ExposAmt"].clip(lower=1e-9)
    d["qx_amt"] = (d["ExpecAmt"] / a).clip(lower=1e-9)
    d["log_qx_amt"] = np.log(d["qx_amt"])
    d["log_avg_size"] = np.log(a / e)
    d["cv2"] = d["ExposAmt2"] * e / a ** 2
    d["log_expos"] = np.log(e)
    d["eff_n"] = a ** 2 / d["ExposAmt2"].clip(lower=1e-9)
    return d


def detecter_mode(df, resultats):
    return "melange" if resultats["Year"].isin(df["Year"]).mean() > 0.9 else "futur"


def plis(d, mode, n_splits, seed=42):
    if mode == "melange":
        return list(KFold(n_splits, shuffle=True, random_state=seed).split(d))
    ys = np.sort(d["YearStart"].unique())[-min(3, n_splits):]
    yr = d["YearStart"].to_numpy()
    return [(np.where(yr < v)[0], np.where(yr == v)[0]) for v in ys]


def pdev(y, p):
    y, p = np.asarray(y, float), np.clip(np.asarray(p, float), 1e-12, None)
    t = np.where(y > 0, y * np.log(np.where(y > 0, y, 1) / p), 0.0)
    return float(2 * np.sum(t - (y - p)) / len(y))


def rmse(y, p):
    return float(np.sqrt(np.mean((np.asarray(y, float) - np.asarray(p, float)) ** 2)))


def cohort_table(src, k=1.0):
    g = src.groupby(KEY, sort=False).agg(DA=("DthAmt", "sum"), EA=("ExpecAmt", "sum")).reset_index()
    k_amt = k * src["ExpecAmt"].sum() / max(len(src), 1)
    return g, k_amt


def cohort_from_table(g, k_amt, dst, k=1.0):
    m = dst[KEY].merge(g, on=KEY, how="left").fillna(0.0)
    return pd.DataFrame({"coh_ae_amt": ((m["DA"] + k_amt) / (m["EA"] + k_amt)).values,
                         "coh_log_E": np.log1p(m["EA"]).values}, index=dst.index)


def cohort_apply(src, dst, k=1.0):
    g, k_amt = cohort_table(src, k)
    return cohort_from_table(g, k_amt, dst, k)


def cohort_oof(src, n_splits=5, seed=0):
    out = pd.DataFrame(index=src.index, columns=["coh_ae_amt", "coh_log_E"], dtype=float)
    for tr, va in KFold(n_splits, shuffle=True, random_state=seed).split(src):
        out.iloc[va] = cohort_apply(src.iloc[tr], src.iloc[va]).values
    return out


def preparer(df, mode):
    """prep() + cohorte correctement rattachée (hors-pli en mode 'melange', neutre sinon)."""
    d = prep(df)
    if mode == "melange":
        d = pd.concat([d, cohort_oof(d)], axis=1)
    else:
        d["coh_ae_amt"], d["coh_log_E"] = 1.0, 0.0
    return d


# ============================================================================================
# SECTION 3 — Strates : tranches personnalisables, crédibilité, suggestion de fusion
# ============================================================================================
def niveau_pct(p):
    if p is None or not np.isfinite(p):
        return None
    if p < 0.33:
        return "bas"
    if p < 0.67:
        return "moyen"
    return "haut"


def colonnes_decoupables_disponibles(d):
    return [c for c in COLONNES_DECOUPABLES if c in d.columns]


def appliquer_decoupage(d, decoupage):
    """
    decoupage : dict optionnel {colonne: config}.
      - Pour une colonne numérique (AttdAge, PolYear, YearStart) : config = liste de bornes croissantes,
        ex. [0, 20, 40, 60, 80, 200] -> tranches "0-19", "20-39", ...
      - Pour Size (déjà en tranches 1-8) : config = dict {code_original: libellé_du_groupe},
        pour regrouper des tranches existantes, ex. {1: "0-50k", 2: "0-50k", 3: "50k+", ...}.
    Renvoie (d avec les colonnes *_grp ajoutées, mapping colonne_originale -> colonne_a_utiliser_dans_strate).
    """
    d = d.copy()
    correspondance = {}
    for col, config in (decoupage or {}).items():
        if col == "Size" or isinstance(config, dict):
            grp_col = col + "_grp"
            d[grp_col] = d[col].map(config).fillna(d[col].astype(str))
            correspondance[col] = grp_col
        else:
            bornes = sorted(config)
            labels = [f"{bornes[i]}-{bornes[i+1]-1}" if bornes[i+1] < 10 ** 6 else f"{bornes[i]}+"
                     for i in range(len(bornes) - 1)]
            grp_col = col + "_grp"
            d[grp_col] = pd.cut(d[col], bornes, labels=labels, right=False, include_lowest=True).astype(str)
            correspondance[col] = grp_col
    return d, correspondance


def suggestion_bornes(d, col, n_tranches=5):
    """Bornes de départ suggérées (quantiles), modifiables ensuite par l'utilisateur."""
    q = np.linspace(0, 1, n_tranches + 1)
    bornes = sorted(set(int(x) for x in d[col].quantile(q).to_numpy()))
    if bornes[0] > d[col].min():
        bornes[0] = int(d[col].min())
    bornes[-1] = 10 ** 6
    return bornes


def credibilite(deces, n_plein=N_PLEIN):
    return np.minimum(1.0, np.sqrt(np.asarray(deces, float) / n_plein))


def table_strates(d, strate, decoupage=None, n_plein=N_PLEIN):
    """d doit déjà être préparé (prep()/preparer()). `strate` : colonnes ORIGINALES choisies par l'utilisateur ;
    `decoupage` transforme certaines d'entre elles en tranches personnalisées avant le regroupement."""
    d2, correspondance = appliquer_decoupage(d, decoupage)
    cols_reel = [correspondance.get(c, c) for c in strate]
    cols = cols_reel or ["_tout"]
    d2 = d2.assign(_tout=0) if not strate else d2
    g = (d2.groupby(cols).agg(Lignes=("DthAmt", "size"), Deces=("DthCnt", "sum") if "DthCnt" in d.columns else ("DthAmt", "size"),
                              Attendu=("ExpecAmt", "sum"), Reel=("DthAmt", "sum")).reset_index())
    if "DthCnt" not in d.columns:
        g["Deces"] = 0.0
    g["Ratio"] = np.where(g["Attendu"] > 0, g["Reel"] / g["Attendu"].where(g["Attendu"] > 0, 1.0), 1.0)
    g["Z"] = credibilite(g["Deces"], n_plein)
    g["Facteur"] = g["Z"] * g["Ratio"]
    g["Poids"] = g["Attendu"] / g["Attendu"].sum()
    g["Fiabilite"] = g["Z"].map(lambda z: {"bas": "Faible", "moyen": "Moyenne", "haut": "Bonne"}.get(niveau_pct(z), "Inconnue"))
    for c_orig, c_reel in zip(strate, cols_reel):
        if c_reel == c_orig and c_orig in LIBELLES:       # pas de découpage perso : on garde le libellé standard
            g[c_orig + "_libelle"] = g[c_orig].map(LIBELLES[c_orig]).fillna(g[c_orig].astype(str))
    g.attrs["cols_reel"] = cols_reel
    g.attrs["strate_orig"] = list(strate)
    return g.sort_values("Deces", ascending=False).reset_index(drop=True)


def appliquer_facteur(tab, strate, d, decoupage=None):
    d2, correspondance = appliquer_decoupage(d, decoupage)
    cols = [correspondance.get(c, c) for c in strate] or ["_tout"]
    d2 = d2.assign(_tout=0) if not strate else d2
    m = d2[cols].merge(tab[cols + ["Facteur"]], on=cols, how="left")
    inconnu = m["Facteur"].isna().to_numpy()
    f = m["Facteur"].fillna(0.0).to_numpy(float)
    return d["ExpecAmt"].to_numpy(float) * f, f, inconnu


def suggestion_fusion(tab, seuil_z=1.5):
    """
    Compare les strates voisines le long de la DERNIÈRE colonne de la strate (supposée ordonnée :
    une tranche numérique découpée, ou Size). Pour chaque groupe de strates qui ne diffèrent que sur
    cette dernière colonne, propose une fusion entre voisines quand l'écart de ratio est petit par
    rapport à l'incertitude statistique attendue (approximation Poisson : erreur type ~ ratio/racine(décès)).
    Renvoie un DataFrame, vide si la strate n'a pas au moins 2 niveaux sur sa dernière colonne.
    """
    cols_reel = tab.attrs.get("cols_reel", [])
    if len(cols_reel) < 1:
        return pd.DataFrame()
    col_ord, cols_fixes = cols_reel[-1], cols_reel[:-1]
    lignes = []
    groupes = tab.groupby(cols_fixes) if cols_fixes else [(None, tab)]
    for _, g in groupes:
        g = g.sort_values(col_ord).reset_index(drop=True)
        for i in range(len(g) - 1):
            a, b = g.iloc[i], g.iloc[i + 1]
            se_a = a["Ratio"] / max(np.sqrt(a["Deces"]), 1e-6)
            se_b = b["Ratio"] / max(np.sqrt(b["Deces"]), 1e-6)
            ecart = abs(a["Ratio"] - b["Ratio"])
            se_comb = np.sqrt(se_a ** 2 + se_b ** 2)
            fusion_reco = ecart < seuil_z * se_comb
            deces_f, att_f, reel_f = a["Deces"] + b["Deces"], a["Attendu"] + b["Attendu"], a["Reel"] + b["Reel"]
            z_f = float(credibilite(deces_f))
            ratio_f = reel_f / att_f if att_f > 0 else 1.0
            lignes.append({**{c: a[c] for c in cols_fixes}, f"{col_ord} (A)": a[col_ord], f"{col_ord} (B)": b[col_ord],
                          "Ratio A": a["Ratio"], "Ratio B": b["Ratio"], "Écart": ecart, "Écart attendu (bruit)": se_comb,
                          "Z fusionné": z_f, "Facteur fusionné": z_f * ratio_f, "Recommandation": "Fusionner" if fusion_reco else "Garder séparées"})
    return pd.DataFrame(lignes)


# ============================================================================================
# SECTION 4 — Sélection dynamique (colonnes SECONDAIRES seulement ; le socle est toujours inclus)
# ============================================================================================
def _glm_ratio_oof(d, cols, plis_):
    from sklearn.linear_model import PoissonRegressor
    from sklearn.compose import ColumnTransformer
    from sklearn.preprocessing import OneHotEncoder, SplineTransformer
    from sklearn.pipeline import Pipeline
    num = [c for c in cols if c in ("AttdAge", "PolYear", "IssueAge", "YearStart", "log_qx_amt",
                                    "log_avg_size", "log_expos", "coh_log_E", "coh_ae_amt", "cv2")]
    cat = [c for c in cols if c not in num]
    o = np.ones(len(d))
    if not cols:
        return o
    blocs = []
    if cat:
        blocs.append(("cat", OneHotEncoder(handle_unknown="ignore"), cat))
    if num:
        blocs.append(("num", SplineTransformer(n_knots=5, degree=3, extrapolation="constant"), num))
    E, y = d["ExpecAmt"].to_numpy(float), d["DthAmt"].to_numpy(float)
    for tr, va in plis_:
        pipe = Pipeline([("prep", ColumnTransformer(blocs, sparse_threshold=1.0)), ("glm", PoissonRegressor(alpha=1e-4, max_iter=400))])
        pipe.fit(d.iloc[tr][cols], y[tr] / np.clip(E[tr], 1e-9, None), glm__sample_weight=E[tr])
        o[va] = np.clip(pipe.predict(d.iloc[va][cols]), 1e-3, 1e2)
    return o


def selectionner_colonnes(d, mode, seuil_gain, n_splits=3, callback=None):
    """Forward selection SUR LES COLONNES SECONDAIRES SEULEMENT (le socle est ajouté systématiquement
    au moment de l'entraînement, jamais testé ici). d doit déjà inclure les colonnes de cohorte (preparer())."""
    candidats = [c for c in CANDIDATS_SECONDAIRES if c in d.columns]
    plis_ = plis(d, mode, n_splits)
    y, E = d["DthAmt"].to_numpy(float), d["ExpecAmt"].to_numpy(float)
    base_cols = list(SOCLE)
    base_dev = pdev(y, E * _glm_ratio_oof(d, base_cols, plis_))     # référence = socle seul

    retenues, restantes, historique = [], list(candidats), []
    dev_actuelle = base_dev
    while restantes:
        gains = {}
        for c in restantes:
            o = _glm_ratio_oof(d, base_cols + retenues + [c], plis_)
            gains[c] = pdev(y, E * o)
        meilleure = min(gains, key=gains.get)
        gain_relatif = 1 - gains[meilleure] / dev_actuelle
        etape = dict(colonne=meilleure, deviance=gains[meilleure], gain=gain_relatif, retenue=gain_relatif >= seuil_gain)
        historique.append(etape)
        if callback:
            callback(etape)
        if gain_relatif < seuil_gain:
            break
        retenues.append(meilleure)
        restantes.remove(meilleure)
        dev_actuelle = gains[meilleure]
    return retenues, historique, base_dev


# ============================================================================================
# SECTION 5 — Entraînement LightGBM, pli par pli (interruptible)
# ============================================================================================
PARAMS_LGB = dict(learning_rate=0.05, num_leaves=31, min_data_in_leaf=50, feature_fraction=0.9,
                  bagging_fraction=0.9, bagging_freq=1, lambda_l2=10.0, verbose=-1)


def _mapping(d, cols):
    m = {}
    for c in cols:
        if c in d.columns and not pd.api.types.is_numeric_dtype(d[c]):
            m[c] = {v: i for i, v in enumerate(sorted(d[c].astype(str).unique()))}
    return m


def _X(d, cols, mapping):
    X = d[cols].copy() if cols else pd.DataFrame(index=d.index)
    for c, mp in mapping.items():
        X[c] = d[c].astype(str).map(mp).fillna(-1)
    return X.astype(float)


def _point_depart(d, F):
    return d["ExpecAmt"].to_numpy(float) * (np.ones(len(d)) if F is None else np.maximum(F, 1e-3))


def etat_initial(d, cols, mode, n_splits, params, avec_facteur, strate, decoupage):
    return dict(d=d, cols=cols, mode=mode, n_splits=n_splits, params=params, avec_facteur=avec_facteur,
               strate=strate, decoupage=decoupage, plis=plis(d, mode, n_splits), pli_courant=0,
               oof=np.full(len(d), np.nan), modeles=[], termine=False, interrompu=False)


def entrainer_un_pli(etat, seed=0):
    import lightgbm as lgb
    k = etat["pli_courant"]
    tr, va = etat["plis"][k]
    d, cols = etat["d"], etat["cols"]
    d_tr, d_va = d.iloc[tr], d.iloc[va]
    F_tr = F_va = None
    if etat["avec_facteur"]:
        tab_tr = table_strates(d_tr, etat["strate"], etat["decoupage"])
        F_tr = appliquer_facteur(tab_tr, etat["strate"], d_tr, etat["decoupage"])[1]
        F_va = appliquer_facteur(tab_tr, etat["strate"], d_va, etat["decoupage"])[1]
    base_tr, base_va = _point_depart(d_tr, F_tr), _point_depart(d_va, F_va)
    mapping = _mapping(d_tr, cols)
    Xtr, Xva = _X(d_tr, cols, mapping), _X(d_va, cols, mapping)
    ytr, yva = d_tr["DthAmt"].to_numpy(float), d_va["DthAmt"].to_numpy(float)
    ztr, zva = ytr / np.clip(base_tr, 1e-9, None), yva / np.clip(base_va, 1e-9, None)
    p = {k: v for k, v in etat["params"].items() if k != "max_rounds"}
    p = dict(p, objective="poisson", seed=seed)
    t0 = time.time()
    dtr_lgb = lgb.Dataset(Xtr, ztr, weight=base_tr)
    dva_lgb = lgb.Dataset(Xva, zva, weight=base_va, reference=dtr_lgb)
    b = lgb.train(p, dtr_lgb, etat["params"].get("max_rounds", 2000), valid_sets=[dva_lgb],
                  callbacks=[lgb.early_stopping(100, verbose=False)])
    p_va = base_va * np.exp(b.predict(Xva, raw_score=True, num_iteration=b.best_iteration))
    dt = time.time() - t0
    etat["oof"][va] = p_va
    etat["modeles"].append(dict(booster=b.model_to_string(num_iteration=b.best_iteration), mapping=mapping))
    etat["pli_courant"] += 1
    if etat["pli_courant"] >= len(etat["plis"]):
        etat["termine"] = True
    return dict(pli=k + 1, sur=len(etat["plis"]), temps=dt, deviance_pli=pdev(yva, p_va),
               deviance_table_pli=pdev(yva, d_va["ExpecAmt"].to_numpy(float)), n_arbres=b.best_iteration)


def resume_hors_pli(etat):
    cov = ~np.isnan(etat["oof"])
    d = etat["d"]
    y, e = d["DthAmt"].to_numpy(float)[cov], d["ExpecAmt"].to_numpy(float)[cov]
    p = etat["oof"][cov]
    return dict(n=int(cov.sum()), y=y, e=e, p=p, deviance_table=pdev(y, e), deviance_modele=pdev(y, p),
               rmse_table=rmse(y, e), rmse_modele=rmse(y, p), somme_pred_reel=float(p.sum() / y.sum()))


def finaliser(etat, seed=0):
    import lightgbm as lgb
    d = etat["d"]
    F = None
    if etat["avec_facteur"]:
        tab_all = table_strates(d, etat["strate"], etat["decoupage"])
        F = appliquer_facteur(tab_all, etat["strate"], d, etat["decoupage"])[1]
        etat["tab_facteur_finale"] = tab_all
    if etat["mode"] == "melange" and any(c in etat["cols"] for c in ("coh_ae_amt", "coh_log_E")):
        etat["coh_g"], etat["coh_k_amt"] = cohort_table(d)   # table complète (pas hors-pli) pour predire()
    base = _point_depart(d, F)
    mapping = _mapping(d, etat["cols"])
    X = _X(d, etat["cols"], mapping)
    y = d["DthAmt"].to_numpy(float)
    z = y / np.clip(base, 1e-9, None)
    n_rounds = max(50, etat.get("_n_rounds_moyen", 200))
    p = {k: v for k, v in etat["params"].items() if k != "max_rounds"}
    p = dict(p, objective="poisson", seed=seed)
    b = lgb.train(p, lgb.Dataset(X, z, weight=base), n_rounds)
    etat["modele_final"] = dict(booster=b.model_to_string(), mapping=mapping)
    return etat["modele_final"]


# ============================================================================================
# SECTION 6 — Tendance temporelle (appliquée APRÈS la prédiction du modèle, jamais en entrée)
# ============================================================================================
def appliquer_tendance(pred, n_annees, taux):
    """pred déjà calculée par le modèle ; n_annees = écart entre l'année de la ligne et la dernière
    année de df ; taux = taux annuel d'amélioration (ex. 0.012 pour 1,2 %)."""
    n_annees = np.asarray(n_annees, float)
    return np.asarray(pred, float) * (1 - taux) ** np.clip(n_annees, 0, None)


def calculer_n_annees(resultats, derniere_annee_df):
    ys = pd.to_numeric(resultats["Year"]).astype(int) // 100 + 2000
    return (ys - derniere_annee_df).to_numpy(float)


# ============================================================================================
# SECTION 7 — Métriques : RMSE, déviance de Poisson, Gini, double lift chart
# ============================================================================================
def gini_normalise(y, p, poids=None):
    """Indice de Gini normalisé (0 = aucune discrimination, 1 = discrimination parfaite).
    On trie par prédiction croissante, on trace la part cumulée du réel, et on compare à la
    courbe obtenue en triant directement par le réel (le meilleur Gini possible)."""
    y, p = np.asarray(y, float), np.asarray(p, float)
    w = np.ones(len(y)) if poids is None else np.asarray(poids, float)

    _trapz = getattr(np, "trapezoid", None) or np.trapz   # np.trapz retire en numpy >= 2.0

    def lorenz_aire(ordre):
        yo, wo = y[ordre], w[ordre]
        cum_w = np.cumsum(wo) / wo.sum()
        cum_y = np.cumsum(yo * wo) / (yo * wo).sum()
        return float(_trapz(cum_y, cum_w))

    aire_modele = lorenz_aire(np.argsort(p))
    aire_parfaite = lorenz_aire(np.argsort(y))
    gini_modele = 1 - 2 * aire_modele
    gini_parfait = 1 - 2 * aire_parfaite
    return gini_modele / gini_parfait if gini_parfait > 1e-9 else float("nan")


def double_lift(y, p_modele, p_reference, n_quantiles=10):
    """Trie les lignes par le RAPPORT modèle/référence, regroupe en quantiles de poids égal (pondérés
    par la référence), et compare la moyenne réelle à la moyenne de chaque prédiction dans chaque
    quantile. Sert à voir lequel des deux colle le mieux à la réalité selon la tranche de risque."""
    y, pm, pr = np.asarray(y, float), np.asarray(p_modele, float), np.asarray(p_reference, float)
    ratio = pm / np.clip(pr, 1e-9, None)
    ordre = np.argsort(ratio)
    y, pm, pr = y[ordre], pm[ordre], pr[ordre]
    poids = np.clip(pr, 1e-9, None)
    cum = np.cumsum(poids) / poids.sum()
    q = np.clip((cum * n_quantiles).astype(int), 0, n_quantiles - 1)
    df = pd.DataFrame({"quantile": q, "y": y, "pm": pm, "pr": pr})
    g = df.groupby("quantile").agg(Réel=("y", "mean"), Modèle=("pm", "mean"), Référence=("pr", "mean"), Lignes=("y", "size"))
    g.index = [f"Q{i+1}" for i in g.index]
    g.index.name = "Quantile"
    return g.reset_index()


# ============================================================================================
# SECTION 8 — Sauvegarde, chargement, prédiction sur resultats
# ============================================================================================
def construire_bundle(etat, taux_tendance=None, derniere_annee_df=None):
    return dict(cols=etat["cols"], mode=etat["mode"], avec_facteur=etat["avec_facteur"], strate=etat["strate"],
               decoupage=etat["decoupage"], tab_facteur=etat.get("tab_facteur_finale"), modele=etat["modele_final"],
               coh_g=etat.get("coh_g"), coh_k_amt=etat.get("coh_k_amt"),
               resume=resume_hors_pli(etat), taux_tendance=taux_tendance, derniere_annee_df=derniere_annee_df)


def bundle_to_bytes(bundle):
    b = dict(bundle)
    b.pop("resume", None)          # contient des tableaux numpy volumineux, inutiles pour re-prédire
    return pickle.dumps(b)


def bundle_from_bytes(b):
    return pickle.loads(b)


def _predict_lgb(modele, d, cols, F):
    import lightgbm as lgb
    base = _point_depart(d, F)
    X = _X(d, cols, modele["mapping"])
    booster = lgb.Booster(model_str=modele["booster"])
    return base * np.exp(booster.predict(X, raw_score=True))


def predire(bundle, resultats, appliquer_la_tendance=True):
    res = prep(resultats).reset_index(drop=True)
    if bundle.get("coh_g") is not None:
        res = pd.concat([res, cohort_from_table(bundle["coh_g"], bundle["coh_k_amt"], res)], axis=1)
    else:
        res["coh_ae_amt"], res["coh_log_E"] = 1.0, 0.0    # mode 'futur', ou cohorte non utilisée
    F = None
    if bundle["avec_facteur"]:
        F = appliquer_facteur(bundle["tab_facteur"], bundle["strate"], res, bundle["decoupage"])[1]
    amt = _predict_lgb(bundle["modele"], res, bundle["cols"], F)
    if appliquer_la_tendance and bundle.get("taux_tendance") is not None and bundle.get("derniere_annee_df") is not None:
        n = calculer_n_annees(resultats, bundle["derniere_annee_df"])
        amt = appliquer_tendance(amt, n, bundle["taux_tendance"])
    sub = resultats.copy()
    sub["DthAmt"] = np.clip(amt, 0, None)
    return sub

























































# -*- coding: utf-8 -*-
"""
FRONTEND (Streamlit) — ne contient AUCUN calcul : tout est délégué à backend.py.
Pipeline : Données -> Strates & crédibilité -> Facteur -> Colonnes -> Entraînement -> Résultats.

Lancement : streamlit run app.py
"""
import hashlib
import importlib
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent))
import backend as b
importlib.reload(b)   # Streamlit ne relit pas les fichiers importés tout seul : voir conversation précédente

assert hasattr(b, "niveau_pct") and hasattr(b, "gini_normalise"), (
    f"Le fichier backend.py chargé ({b.__file__}) est une VERSION PÉRIMÉE. "
    "Remplace-le par le dernier backend.py fourni, puis supprime le dossier __pycache__.")

st.set_page_config(page_title="Assistant mortalité — montant des décès", page_icon="🧮", layout="wide")

CSS_PATH = Path(__file__).parent / "style.css"
if CSS_PATH.exists():
    st.markdown(f"<style>{CSS_PATH.read_text(encoding='utf-8')}</style>", unsafe_allow_html=True)

NOM_DEFI = "Défi Données · Modélisation actuarielle"
NOM_EQUIPE = "Équipe 7"

ETAPES = [
    ("📂", "Données", "Glisser-déposer et nettoyer"),
    ("🧩", "Strates & crédibilité", "Tranches, fusion, crédibilité"),
    ("⚖️", "Facteur", "Point de départ ou non"),
    ("🎯", "Colonnes", "Socle fixe + sélection dynamique"),
    ("🚀", "Entraînement", "LightGBM, pli par pli"),
    ("🏆", "Résultats", "Métriques, tendance, téléchargement"),
]


# ============================================================================================
# État
# ============================================================================================
def init_etat():
    defauts = dict(etape=0, df=None, df_nom=None, strate=[], sans_strate=False, strate_confirmee=False,
                   decoupage={}, tab_strate=None, avec_facteur=None, seuil_gain=0.01, n_splits=5,
                   rapide=False, learning_rate=0.05, max_rounds=2000, cols_secondaires=None,
                   historique_selection=None, train=None, train_signature=None, bundle=None,
                   taux_tendance=0.012, resultats_bytes=None, resultats_nom=None)
    for k, v in defauts.items():
        st.session_state.setdefault(k, v)


init_etat()


def aller(delta):
    st.session_state.etape = max(0, min(len(ETAPES) - 1, st.session_state.etape + delta))


def signature_actuelle():
    s = (tuple(st.session_state.cols_secondaires or []), st.session_state.n_splits, st.session_state.avec_facteur,
        tuple(st.session_state.strate), str(st.session_state.decoupage), st.session_state.rapide,
        st.session_state.learning_rate, st.session_state.max_rounds)
    return hashlib.md5(str(s).encode()).hexdigest()


def badge(symbole, texte, niveau=None):
    classe = {"✓": "badge-ok", "⚠": "badge-warn", "ℹ": "badge-info"}[symbole]
    pastille = ""
    if niveau in ("bas", "moyen", "haut"):
        label = {"bas": "Faible", "moyen": "Moyenne", "haut": "Bonne"}[niveau]
        pastille = f"<span class='niveau-pill niveau-{niveau}'>● Fiabilité {label}</span>"
    st.markdown(f"<div class='ligne-message'><span class='badge {classe}'>{symbole}</span> {texte} {pastille}</div>",
               unsafe_allow_html=True)


def _couleur_cellule(niveau):
    couleurs = {"bas": ("#fdeceb", "#a83226"), "moyen": ("#fdf3dc", "#8a6300"), "haut": ("#e6f6ec", "#116639")}
    fond, texte = couleurs.get(niveau, ("", ""))
    return f"background-color:{fond}; color:{texte}; font-weight:700;" if fond else ""


def colorer_fiabilite(val):
    return _couleur_cellule(b.niveau_pct(val))


def kicker(icone, titre, description):
    st.markdown(f"""<div class="kicker-page"><div class="icone">{icone}</div>
        <div class="textes"><p class="titre">{titre}</p><p class="desc">{description}</p></div></div>""",
               unsafe_allow_html=True)


def pied_navigation(peut_avancer=True, texte_suivant="Suivant →"):
    st.markdown("<div class='pied-page'>", unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        if st.session_state.etape > 0 and st.button("← Précédent", use_container_width=True, key=f"prec_{st.session_state.etape}"):
            aller(-1)
            st.rerun()
    with c2:
        if st.button(texte_suivant, type="primary", disabled=not peut_avancer, use_container_width=True, key=f"suiv_{st.session_state.etape}"):
            aller(1)
            st.rerun()
    st.markdown("</div>", unsafe_allow_html=True)


# ============================================================================================
# En-tête + barre latérale
# ============================================================================================
st.markdown(f"""
<div class="entete-marque">
    <div class="gauche"><div class="puce">🧮</div>
        <div><h1>Assistant de modélisation — montant des décès</h1>
        <p class="sous-titre">LightGBM, point de départ par crédibilité, tendance temporelle appliquée après coup</p></div>
    </div>
    <div class="droite"><div class="nom-defi">{NOM_DEFI}</div><div class="nom-equipe">{NOM_EQUIPE}</div></div>
</div>""", unsafe_allow_html=True)

with st.sidebar:
    st.markdown("<div class='sb-logo'><div class='puce'>🌲</div><div class='txt'>Parcours du modèle</div></div>", unsafe_allow_html=True)
    for i, (icone, nom, _) in enumerate(ETAPES):
        cls = "active" if i == st.session_state.etape else ("faite" if i < st.session_state.etape else "")
        num = "✓" if i < st.session_state.etape else str(i + 1)
        st.markdown(f"<div class='sb-etape {cls}'><div class='num'>{num}</div><div>{icone} {nom}</div></div>", unsafe_allow_html=True)
    lignes_resume = []
    if st.session_state.df is not None:
        lignes_resume.append(("Lignes", f"{len(st.session_state.df):,}".replace(",", " ")))
    if st.session_state.strate:
        lignes_resume.append(("Strate", ", ".join(st.session_state.strate)))
    elif st.session_state.sans_strate:
        lignes_resume.append(("Strate", "aucune"))
    if st.session_state.cols_secondaires is not None:
        lignes_resume.append(("Colonnes", str(len(b.SOCLE) + len(st.session_state.cols_secondaires))))
    if st.session_state.bundle is not None:
        lignes_resume.append(("Déviance", f"{st.session_state.bundle['resume']['deviance_modele']:,.0f}".replace(",", " ")))
    if lignes_resume:
        html_resume = "".join(f"<div class='ligne'><span>{k}</span><b>{v}</b></div>" for k, v in lignes_resume)
        st.markdown(f"<div class='sb-resume'>{html_resume}</div>", unsafe_allow_html=True)

etape = st.session_state.etape


# ============================================================================================
# ÉTAPE 0 — Données
# ============================================================================================
if etape == 0:
    kicker("📂", "Charger et voir les données", "Glisse-dépose le fichier df, on nettoie et on vérifie tout de suite")
    st.markdown("<div class='carte'>", unsafe_allow_html=True)
    col1, col2 = st.columns([2, 1])
    with col1:
        fichier = st.file_uploader("Glisse-dépose le CSV de df ici", type=["csv"])
    with col2:
        sep = st.selectbox("Séparateur", ["auto", ",", ";", "\t"], index=0)
        decimal = st.selectbox("Décimale", [".", ","], index=0)

    if fichier is not None:
        try:
            df = b.lire_csv(fichier.getvalue(), sep=None if sep == "auto" else sep, decimal=decimal)
            st.session_state.df, st.session_state.df_nom = df, fichier.name
        except Exception as e:
            st.error(f"Impossible de lire ce fichier : {e}")
            st.session_state.df = None

    df = st.session_state.df
    if df is not None:
        diag = b.diagnostic_colonnes(df)
        st.success(f"« {st.session_state.df_nom} » chargé : {len(df):,} lignes, {len(df.columns)} colonnes".replace(",", " "))
        if diag["manque"]:
            st.error(f"Colonnes obligatoires manquantes : {diag['manque']}")
        else:
            dfc, rapport, _ = b.nettoyer(df, None)
            if dfc is not df:
                st.session_state.df = dfc
            if rapport:
                for r in rapport:
                    badge("⚠", r)
            else:
                badge("✓", "Aucune valeur manquante ou infinie détectée.")
            st.markdown("**Aperçu**")
            st.dataframe(st.session_state.df.head(50), height=260, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)
    peut_avancer = st.session_state.df is not None and not b.diagnostic_colonnes(st.session_state.df)["manque"]
    pied_navigation(peut_avancer)


# ============================================================================================
# ÉTAPE 1 — Strates & crédibilité (tranches personnalisables, confirmation, suggestion de fusion)
# ============================================================================================
elif etape == 1:
    df = st.session_state.df
    d = b.prep(df)
    colonnes_possibles = [c for c in ["Sex", "Smoke", "PolGrp", "Base", "Par", "Size", "AttdAge", "PolYear"] if c in d.columns]

    kicker("🧩", "Strates & crédibilité", "Choisis un découpage, ajuste les tranches, puis confirme")
    st.markdown("<div class='carte'><h3>Colonnes de la strate</h3>", unsafe_allow_html=True)
    sans = st.checkbox("Continuer sans strate (ne pas travailler avec le facteur)", value=st.session_state.sans_strate)
    if sans != st.session_state.sans_strate:
        st.session_state.sans_strate = sans
        st.session_state.strate_confirmee = False

    choix_libre = st.multiselect("Colonnes de la strate", colonnes_possibles, default=st.session_state.strate,
                                 disabled=sans, help="Choisis tes colonnes, ajuste leurs tranches si besoin, puis confirme.")
    if choix_libre != st.session_state.strate:
        st.session_state.strate = choix_libre
        st.session_state.strate_confirmee = False
        st.session_state.decoupage = {k: v for k, v in st.session_state.decoupage.items() if k in choix_libre}

    decoupables = [c for c in choix_libre if c in b.colonnes_decoupables_disponibles(d)]
    if decoupables and not sans:
        st.markdown("**Tranches personnalisées** (facultatif — laisse vide pour garder les valeurs telles quelles)")
        for col in decoupables:
            c1, c2 = st.columns([3, 1])
            with c1:
                sugg = ", ".join(str(x) for x in b.suggestion_bornes(d, col))
                txt = st.text_input(f"Bornes pour {col} (ex. 0, 20, 40, 60, 80)", value="", placeholder=f"suggestion : {sugg}", key=f"bornes_{col}")
            with c2:
                st.caption(" ")
                if st.button(f"Réinitialiser {col}", key=f"reset_{col}"):
                    txt = ""
                    st.session_state[f"bornes_{col}"] = ""
            if txt.strip():
                try:
                    bornes = sorted(int(x.strip()) for x in txt.split(","))
                    if st.session_state.decoupage.get(col) != bornes:
                        st.session_state.decoupage[col] = bornes
                        st.session_state.strate_confirmee = False
                except ValueError:
                    st.warning(f"Bornes invalides pour {col} : utilise des nombres séparés par des virgules.")
            elif col in st.session_state.decoupage:
                del st.session_state.decoupage[col]
                st.session_state.strate_confirmee = False
    if "Size" in choix_libre and not sans:
        st.markdown("**Regroupement des tranches de capital (`Size`)** (facultatif)")
        tailles = sorted(d["Size"].dropna().unique().tolist())
        libelles = {t: b.LIBELLES.get("Size", {}).get(t, str(t)) for t in tailles}
        groupe_actuel = st.session_state.decoupage.get("Size", {t: libelles[t] for t in tailles})
        cols_g = st.columns(len(tailles))
        nouveau_groupe = {}
        for i, t in enumerate(tailles):
            with cols_g[i]:
                nouveau_groupe[t] = st.text_input(libelles[t], value=str(groupe_actuel.get(t, libelles[t])), key=f"size_grp_{t}", label_visibility="visible")
        if nouveau_groupe != groupe_actuel:
            st.session_state.decoupage["Size"] = nouveau_groupe
            st.session_state.strate_confirmee = False

    c_conf, c_msg = st.columns([1, 3])
    with c_conf:
        if st.button("✅ Confirmer cette strate", disabled=sans or not choix_libre, use_container_width=True):
            st.session_state.strate_confirmee = True
            st.rerun()
    with c_msg:
        if sans:
            st.caption("Aucune strate : le facteur ne sera pas utilisé.")
        elif st.session_state.strate_confirmee:
            st.caption(f"✔ Strate confirmée : `{st.session_state.strate}`. Change la sélection ou les tranches puis reconfirme pour la modifier.")
        elif choix_libre:
            st.caption("Sélection en attente de confirmation — le tableau ci-dessous n'apparaîtra qu'après le clic.")
        else:
            st.caption("Sélectionne au moins une colonne, ou coche « continuer sans strate ».")
    st.markdown("</div>", unsafe_allow_html=True)

    strate = st.session_state.strate
    if not sans and st.session_state.strate_confirmee and strate:
        tab = b.table_strates(d, strate, st.session_state.decoupage)
        st.session_state.tab_strate = tab
        z100 = tab["Z"] >= 1.0
        st.markdown("<div class='carte'><h3>Crédibilité des strates</h3>", unsafe_allow_html=True)
        m1, m2, m3 = st.columns(3)
        m1.markdown(f"<div class='metrique-box'><div class='val'>{len(tab)}</div><div class='lab'>strates</div></div>", unsafe_allow_html=True)
        m2.markdown(f"<div class='metrique-box'><div class='val'>{int(z100.sum())}</div><div class='lab'>à 100 % de crédibilité</div></div>", unsafe_allow_html=True)
        m3.markdown(f"<div class='metrique-box'><div class='val'>{tab.loc[z100, 'Poids'].sum():.0%}</div><div class='lab'>des prestations couvertes</div></div>", unsafe_allow_html=True)
        st.write("")
        st.caption("🟢 Vert = strate fiable. 🟡 Jaune = fiabilité moyenne. 🔴 Rouge = peu de décès observés.")
        cols_reel = tab.attrs.get("cols_reel", strate)
        cols_aff = cols_reel + ["Lignes", "Deces", "Attendu", "Reel", "Ratio", "Z", "Fiabilite", "Facteur", "Poids"]
        cols_aff = [c for c in cols_aff if c in tab.columns]
        style = tab[cols_aff].style.format({"Attendu": "{:,.0f}", "Reel": "{:,.0f}", "Ratio": "{:.3f}", "Z": "{:.0%}",
                                            "Facteur": "{:.3f}", "Poids": "{:.1%}", "Deces": "{:,.0f}"})
        if "Z" in cols_aff:
            style = style.map(colorer_fiabilite, subset=["Z"])
        st.dataframe(style, use_container_width=True, height=300)
        st.markdown("</div>", unsafe_allow_html=True)

        fusion = b.suggestion_fusion(tab)
        if len(fusion):
            st.markdown("<div class='carte'><h3>🔗 Suggestion de fusion entre strates voisines</h3>", unsafe_allow_html=True)
            st.caption("Paires de strates voisines dont l'écart de ratio est petit par rapport au bruit statistique attendu — "
                      "des candidates à regrouper, à toi de juger si la logique métier le justifie.")
            n_reco = int((fusion["Recommandation"] == "Fusionner").sum())
            st.caption(f"{n_reco} paire(s) sur {len(fusion)} recommandée(s) pour fusion.")
            st.dataframe(fusion.style.format({c: "{:.3f}" for c in fusion.columns if fusion[c].dtype == float}),
                        use_container_width=True, height=min(300, 40 + 35 * len(fusion)))
            st.markdown("</div>", unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        if st.session_state.etape > 0 and st.button("← Précédent", use_container_width=True, key="prec_1"):
            aller(-1)
            st.rerun()
    with c2:
        peut = sans or (st.session_state.strate_confirmee and bool(strate))
        if st.button("Suivant →", type="primary", disabled=not peut, use_container_width=True, key="suiv_1"):
            aller(1)
            st.rerun()


# ============================================================================================
# ÉTAPE 2 — Facteur
# ============================================================================================
elif etape == 2:
    kicker("⚖️", "Facteur", "Décide s'il sert de point de départ au modèle")
    st.markdown("<div class='carte'>", unsafe_allow_html=True)
    if st.session_state.sans_strate or not st.session_state.strate:
        st.info("Aucune strate choisie à l'étape précédente : le facteur n'est pas utilisé.")
        st.session_state.avec_facteur = False
    else:
        st.write(f"Strate retenue : `{st.session_state.strate}`. Le facteur peut servir de **point de départ** "
                "au modèle (celui-ci n'apprend plus que la correction qu'il reste à faire), ou être ignoré.")
        choix = st.radio("Choix", ["Utiliser le facteur comme point de départ", "Ne pas l'utiliser (repartir de la table seule)"],
                         index=0 if st.session_state.avec_facteur is not False else 1)
        st.session_state.avec_facteur = choix.startswith("Utiliser")
    st.markdown("</div>", unsafe_allow_html=True)
    pied_navigation()


# ============================================================================================
# ÉTAPE 3 — Colonnes : socle fixe + sélection dynamique (colonnes secondaires seulement)
# ============================================================================================
elif etape == 3:
    kicker("🎯", "Colonnes du modèle", "Un socle toujours inclus, une sélection automatique pour le reste")
    st.markdown("<div class='carte'><h3>Socle (toujours inclus)</h3>", unsafe_allow_html=True)
    st.write("Ces variables sont des facteurs de risque reconnus : elles entrent toujours dans le modèle, "
            "sans passer par un calcul de gain statistique.")
    st.markdown("".join(f"<span class='tag-col'>{c}</span>" for c in b.SOCLE), unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div class='carte'><h3>Colonnes secondaires — sélection dynamique</h3>", unsafe_allow_html=True)
    st.caption("💡 En clair : on essaie les colonnes une par une et on ne garde que celles qui améliorent "
              "vraiment les prédictions, par rapport au socle seul.")
    st.session_state.n_splits = st.number_input("Nombre de plis (validation croisée)", 2, 10, st.session_state.n_splits)
    st.session_state.seuil_gain = st.number_input("Seuil de gain relatif (une colonne est gardée si elle réduit la déviance d'au moins ce %)",
                                                  min_value=0.0001, max_value=0.05, value=st.session_state.seuil_gain, step=0.0005, format="%.4f")
    if st.button("Lancer la sélection dynamique", type="primary"):
        mode = b.detecter_mode(st.session_state.df, st.session_state.df)   # mode réel connu seulement avec resultats ; par défaut mélange ici
        mode = "melange"
        d = b.preparer(st.session_state.df, mode)
        zone = st.empty()
        historique = []

        def callback(etape_dict):
            historique.append(etape_dict)
            with zone.container():
                for e in historique:
                    s = "✓" if e["retenue"] else "⚠"
                    badge(s, f"{e['colonne']} : gain {e['gain']:.2%} (déviance {e['deviance']:,.0f})".replace(",", " "))

        with st.spinner("Sélection en cours..."):
            cols_sec, hist, base_dev = b.selectionner_colonnes(d, mode, st.session_state.seuil_gain,
                                                                n_splits=2 if st.session_state.rapide else int(st.session_state.n_splits), callback=callback)
        st.session_state.cols_secondaires = cols_sec
        st.session_state.historique_selection = hist
    if st.session_state.cols_secondaires is not None:
        st.success(f"Colonnes secondaires retenues : {st.session_state.cols_secondaires or '(aucune)'}")
        st.markdown("**Colonnes finales du modèle :**")
        st.markdown("".join(f"<span class='tag-col'>{c}</span>" for c in b.SOCLE + st.session_state.cols_secondaires), unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)
    pied_navigation(st.session_state.cols_secondaires is not None)


# ============================================================================================
# ÉTAPE 4 — Entraînement LightGBM, pli par pli
# ============================================================================================
elif etape == 4:
    kicker("🚀", "Entraînement", "Un pli à la fois pour pouvoir t'arrêter, ou tous d'un coup")
    st.markdown("<div class='carte'>", unsafe_allow_html=True)
    mode = "melange"   # affiné une fois resultats chargé, à l'étape Résultats
    d = b.preparer(st.session_state.df, mode)
    cols_final = list(b.SOCLE) + list(st.session_state.cols_secondaires or [])

    sig = signature_actuelle()
    if st.session_state.train is None or st.session_state.train_signature != sig:
        params = dict(learning_rate=st.session_state.learning_rate, max_rounds=200 if st.session_state.rapide else st.session_state.max_rounds)
        st.session_state.train = b.etat_initial(d, cols_final, mode, 2 if st.session_state.rapide else st.session_state.n_splits,
                                                params, st.session_state.avec_facteur, st.session_state.strate, st.session_state.decoupage)
        st.session_state.train_signature = sig
        st.session_state.bundle = None
        st.info("Réglages pris en compte : entraînement (ré)initialisé.")

    etat = st.session_state.train
    fait, total = etat["pli_courant"], len(etat["plis"])
    st.progress(fait / total if total else 0, text=f"Pli {fait} / {total}")

    c1, c2, c3 = st.columns(3)
    if c1.button("▶ Entraîner un pli", disabled=etat["termine"], use_container_width=True):
        with st.spinner("Entraînement du pli en cours..."):
            r = b.entrainer_un_pli(etat)
        st.toast(f"Pli {r['pli']}/{r['sur']} terminé en {r['temps']:.1f}s — déviance {r['deviance_pli']:,.0f}".replace(",", " "))
        st.rerun()
    if c2.button("⏭ Entraîner tous les plis restants", disabled=etat["termine"], use_container_width=True,
                help="Non interruptible une fois lancé — préfère 'un pli' si tu veux pouvoir t'arrêter en cours de route."):
        barre = st.progress(0.0)
        while not etat["termine"]:
            b.entrainer_un_pli(etat)
            barre.progress(etat["pli_courant"] / total)
        st.rerun()
    if c3.button("⏹ Interrompre / arrêter ici", disabled=fait == 0, use_container_width=True):
        etat["termine"] = True
        etat["interrompu"] = fait < total
        st.rerun()

    if fait > 0:
        lignes = []
        for i, (tr, va) in enumerate(etat["plis"][:fait]):
            y_va = d.iloc[va]["DthAmt"].to_numpy(float)
            p_va = etat["oof"][va]
            e_va = d.iloc[va]["ExpecAmt"].to_numpy(float)
            lignes.append(dict(Pli=i + 1, Déviance_table=b.pdev(y_va, e_va), Déviance_modèle=b.pdev(y_va, p_va)))
        histo = pd.DataFrame(lignes)
        st.markdown("**Déviance de Poisson par pli**")
        st.line_chart(histo.set_index("Pli")[["Déviance_table", "Déviance_modèle"]])
        resume = b.resume_hors_pli(etat)
        m1, m2, m3 = st.columns(3)
        m1.markdown(f"<div class='metrique-box'><div class='val'>{resume['deviance_modele']:,.0f}</div><div class='lab'>déviance (hors-pli)</div></div>".replace(",", " "), unsafe_allow_html=True)
        gain = 1 - resume['deviance_modele'] / resume['deviance_table']
        m2.markdown(f"<div class='metrique-box'><div class='val'>{gain:.1%}</div><div class='lab'>gain vs table</div></div>", unsafe_allow_html=True)
        m3.markdown(f"<div class='metrique-box'><div class='val'>{resume['somme_pred_reel']:.3f}</div><div class='lab'>somme prédite / réelle</div></div>", unsafe_allow_html=True)

    if etat["termine"] and st.session_state.bundle is None:
        if etat.get("interrompu"):
            st.warning(f"Entraînement arrêté après {fait}/{total} plis : l'estimation hors-pli est moins fiable.")
        if st.button("✅ Finaliser le modèle (entraîne sur toutes les données)", type="primary"):
            with st.spinner("Entraînement final..."):
                b.finaliser(etat)
                st.session_state.bundle = b.construire_bundle(etat)
            st.rerun()
    st.markdown("</div>", unsafe_allow_html=True)
    pied_navigation(st.session_state.bundle is not None, texte_suivant="Voir les résultats →")


# ============================================================================================
# ÉTAPE 5 — Résultats : métriques, tendance temporelle, téléchargement, prédiction
# ============================================================================================
elif etape == 5:
    bundle = st.session_state.bundle
    kicker("🏆", "Résultats", "Métriques, tendance temporelle et téléchargement")
    st.markdown("<div class='carte'>", unsafe_allow_html=True)
    if bundle is None:
        st.warning("Aucun modèle finalisé. Reviens à l'étape Entraînement.")
    else:
        r = bundle["resume"]
        st.caption("💡 En clair : le RMSE et la déviance mesurent l'écart entre prédictions et réalité (plus petit "
                  "= mieux). Le Gini mesure la capacité à bien classer les risques du plus faible au plus élevé "
                  "(proche de 1 = très bon classement).")
        m1, m2, m3, m4 = st.columns(4)
        m1.markdown(f"<div class='metrique-box'><div class='val'>{r['rmse_modele']:,.0f}</div><div class='lab'>RMSE</div></div>".replace(",", " "), unsafe_allow_html=True)
        m2.markdown(f"<div class='metrique-box'><div class='val'>{r['deviance_modele']:,.0f}</div><div class='lab'>déviance Poisson</div></div>".replace(",", " "), unsafe_allow_html=True)
        gini_m = b.gini_normalise(r["y"], r["p"])
        gini_t = b.gini_normalise(r["y"], r["e"])
        m3.markdown(f"<div class='metrique-box'><div class='val'>{gini_m:.3f}</div><div class='lab'>Gini (table : {gini_t:.3f})</div></div>", unsafe_allow_html=True)
        m4.markdown(f"<div class='metrique-box'><div class='val'>{1 - r['deviance_modele'] / r['deviance_table']:.1%}</div><div class='lab'>gain vs table</div></div>", unsafe_allow_html=True)

        st.markdown("**Double lift chart** — modèle contre table de référence, par tranche de risque")
        dl = b.double_lift(r["y"], r["p"], r["e"])
        st.line_chart(dl.set_index("Quantile")[["Réel", "Modèle", "Référence"]])
        st.dataframe(dl.style.format({"Réel": "{:,.0f}", "Modèle": "{:,.0f}", "Référence": "{:,.0f}"}), use_container_width=True)

        st.write(f"**Colonnes du modèle :** {bundle['cols']}")
        st.write(f"**Facteur comme point de départ :** {'oui — strate ' + str(bundle['strate']) if bundle['avec_facteur'] else 'non'}")
    st.markdown("</div>", unsafe_allow_html=True)

    if bundle is not None:
        st.markdown("<div class='carte'><h3>📅 Tendance temporelle</h3>", unsafe_allow_html=True)
        st.caption(b.TAUX_TENDANCE_SOURCE)
        st.session_state.taux_tendance = st.slider("Taux d'amélioration annuel appliqué APRÈS la prédiction (uniquement "
                                                    "sur les années futures à df)", 0.0, 0.02, st.session_state.taux_tendance,
                                                   step=0.001, format="%.3f")
        derniere = int(pd.to_numeric(st.session_state.df["Year"]).astype(int).max() // 100 + 2000)
        exemples = pd.DataFrame({"Écart (années)": [1, 3, 5, 10]})
        exemples["Facteur appliqué"] = [(1 - st.session_state.taux_tendance) ** n for n in exemples["Écart (années)"]]
        st.dataframe(exemples.style.format({"Facteur appliqué": "{:.4f}"}), use_container_width=True, hide_index=True)
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("<div class='carte'><h3>💾 Modèle et prédiction</h3>", unsafe_allow_html=True)
        bundle_final = dict(bundle, taux_tendance=st.session_state.taux_tendance, derniere_annee_df=derniere)
        st.download_button("💾 Télécharger le modèle entraîné (.pkl)", data=b.bundle_to_bytes(bundle_final),
                           file_name="modele_montant.pkl", mime="application/octet-stream", use_container_width=True)
        st.markdown("<hr class='separateur'>", unsafe_allow_html=True)
        st.markdown("**Prédire sur un fichier `resultats` (mêmes colonnes que df, sans DthAmt)**")
        fichier_res = st.file_uploader("Glisse-dépose le CSV de resultats", type=["csv"], key="upload_resultats")
        if fichier_res is not None:
            try:
                res = b.lire_csv(fichier_res.getvalue())
                _, _, err = b.nettoyer(st.session_state.df, res)
                if err:
                    st.error(err)
                else:
                    mode_reel = b.detecter_mode(st.session_state.df, res)
                    st.caption(f"Mode détecté : {mode_reel}" + (" — la tendance temporelle s'appliquera." if mode_reel == "futur" else " — la tendance ne s'applique pas (années déjà connues)."))
                    sub = b.predire(bundle_final, res)
                    st.dataframe(sub.head(30), use_container_width=True)
                    st.download_button("💾 Télécharger la soumission (CSV)", data=sub.to_csv(index=False).encode("utf-8"),
                                       file_name="soumission_montant.csv", mime="text/csv", use_container_width=True)
            except Exception as e:
                st.error(f"Erreur : {e}")
        st.markdown("</div>", unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        if st.session_state.etape > 0 and st.button("← Précédent", use_container_width=True):
            aller(-1)
            st.rerun()
    with c2:
        if st.button("🔄 Recommencer (nouveau df)", use_container_width=True):
            for k in list(st.session_state.keys()):
                del st.session_state[k]
            st.rerun()








































/* ============================================================================================
   Assistant de modélisation — palette forêt & or, ambiance "atelier de données" premium
   ============================================================================================ */
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@600;700;800&family=Inter:wght@400;500;600;700&display=swap');

:root {
    --vert-950: #06231a;
    --vert-900: #0b3d2e;
    --vert-700: #0f6e4f;
    --vert-500: #17a06e;
    --vert-100: #e8f5ee;
    --or-500: #d4af37;
    --or-300: #f0dc9e;
    --encre: #142621;
    --gris-600: #6b7a75;
    --gris-300: #c9d6d0;
    --carte: #ffffff;
    --bordure: #e1ebe5;
    --ombre: 0 10px 30px rgba(6, 35, 26, 0.08);
    --ombre-hover: 0 14px 38px rgba(6, 35, 26, 0.14);
    --rayon: 18px;
}

html, body, [class*="css"] { font-family: 'Inter', -apple-system, sans-serif; color: var(--encre); }
h1, h2, h3, .titre-page, .entete-marque .nom-defi { font-family: 'Poppins', sans-serif; }

.stApp {
    background:
        radial-gradient(circle at 8% 0%, rgba(23, 160, 110, 0.10), transparent 40%),
        radial-gradient(circle at 100% 12%, rgba(212, 175, 55, 0.10), transparent 35%),
        linear-gradient(180deg, #f7fbf8 0%, #eef5f0 100%);
}
[data-testid="stAppViewContainer"] > .main { padding-top: 0 !important; }
[data-testid="stHeader"] { background: transparent; }

/* ---------------------------------------------------------------------------- en-tête / marque */
.entete-marque {
    display: flex; align-items: center; justify-content: space-between; gap: 20px;
    padding: 22px 30px; margin: 8px 0 22px 0; border-radius: 22px; color: white;
    background: linear-gradient(120deg, var(--vert-950) 0%, var(--vert-900) 45%, var(--vert-700) 100%);
    box-shadow: var(--ombre-hover);
    position: relative; overflow: hidden;
}
.entete-marque::after {
    content: ""; position: absolute; right: -60px; top: -60px; width: 220px; height: 220px;
    background: radial-gradient(circle, rgba(212,175,55,0.35), transparent 65%);
}
.entete-marque .gauche { display: flex; align-items: center; gap: 16px; z-index: 1; }
.entete-marque .puce {
    font-size: 2.1rem; width: 58px; height: 58px; display: flex; align-items: center; justify-content: center;
    background: rgba(255,255,255,0.12); border-radius: 16px; border: 1px solid rgba(240,220,158,0.4);
}
.entete-marque h1 { margin: 0; font-size: 1.3rem; font-weight: 700; letter-spacing: 0.2px; }
.entete-marque .sous-titre { margin: 3px 0 0 0; font-size: 0.85rem; opacity: 0.85; font-weight: 400; }
.entete-marque .droite { text-align: right; z-index: 1; }
.entete-marque .nom-defi {
    font-size: 0.78rem; font-weight: 700; letter-spacing: 1.4px; text-transform: uppercase;
    color: var(--or-300);
}
.entete-marque .nom-equipe {
    margin-top: 2px; font-size: 0.78rem; color: rgba(255,255,255,0.75); font-weight: 500;
}

/* ---------------------------------------------------------------------------- kicker de page (remplace les titres repetes) */
.kicker-page { display: flex; align-items: center; gap: 12px; margin: 4px 0 18px 0; }
.kicker-page .icone {
    width: 46px; height: 46px; min-width: 46px; border-radius: 14px; display: flex; align-items: center;
    justify-content: center; font-size: 1.4rem; background: linear-gradient(135deg, var(--vert-100), #ffffff);
    border: 1px solid var(--bordure); box-shadow: var(--ombre);
}
.kicker-page .textes .titre { font-size: 1.15rem; font-weight: 700; color: var(--vert-900); margin: 0; }
.kicker-page .textes .desc { font-size: 0.85rem; color: var(--gris-600); margin: 1px 0 0 0; }

/* ---------------------------------------------------------------------------- cartes */
.carte {
    background: var(--carte); border: 1px solid var(--bordure); border-radius: var(--rayon);
    padding: 22px 24px; margin-bottom: 16px; box-shadow: var(--ombre);
    transition: box-shadow 0.25s ease, transform 0.25s ease;
}
.carte:hover { box-shadow: var(--ombre-hover); }
.carte h3 { margin-top: 0; color: var(--vert-900); font-size: 1.02rem; }

/* ---------------------------------------------------------------------------- badges de messages */
.badge {
    display: inline-flex; align-items: center; gap: 6px; padding: 3px 11px; border-radius: 999px;
    font-size: 0.76rem; font-weight: 700; margin: 2px 0;
}
.badge-ok { background: #e2f6ea; color: #167c3e; }
.badge-warn { background: #fdf0dc; color: #9a5f13; }
.badge-info { background: #e7f0fd; color: #1f5aad; }
.ligne-message {
    display: flex; gap: 9px; align-items: flex-start; padding: 6px 4px; font-size: 0.87rem;
    color: var(--encre); border-bottom: 1px dashed var(--bordure);
}
.ligne-message:last-child { border-bottom: none; }
.niveau-pill {
    display: inline-flex; align-items: center; gap: 5px; padding: 2px 10px; border-radius: 999px;
    font-size: 0.72rem; font-weight: 700; margin-left: 6px; white-space: nowrap;
}
.niveau-bas { background: #fdeceb; color: #a83226; }
.niveau-moyen { background: #fdf3dc; color: #8a6300; }
.niveau-haut { background: #e6f6ec; color: #116639; }

/* ---------------------------------------------------------------------------- metriques */
.metrique-box {
    background: linear-gradient(160deg, var(--vert-100), #ffffff); border-radius: 16px; padding: 14px 16px;
    text-align: center; border: 1px solid var(--vert-500); box-shadow: var(--ombre);
    transition: transform 0.2s ease;
}
.metrique-box:hover { transform: translateY(-2px); }
.metrique-box .val { font-size: 1.55rem; font-weight: 800; color: var(--vert-900); font-family: 'Poppins', sans-serif; }
.metrique-box .lab { font-size: 0.68rem; color: var(--gris-600); text-transform: uppercase; letter-spacing: 0.5px; margin-top: 2px; }

/* ---------------------------------------------------------------------------- boutons */
div.stButton > button {
    border-radius: 12px; font-weight: 600; font-family: 'Inter', sans-serif; padding: 0.55em 1.1em;
    border: 1.5px solid var(--vert-500); color: var(--vert-900); background: white;
    transition: all 0.18s ease; box-shadow: 0 1px 3px rgba(6,35,26,0.06);
}
div.stButton > button:hover:not(:disabled) {
    border-color: var(--vert-900); background: var(--vert-100); transform: translateY(-1px);
    box-shadow: 0 6px 14px rgba(15,110,79,0.18);
}
div.stButton > button[kind="primary"] {
    background: linear-gradient(120deg, var(--vert-700), var(--vert-500)); border-color: var(--vert-900); color: white;
}
div.stButton > button[kind="primary"]:hover:not(:disabled) {
    background: linear-gradient(120deg, var(--vert-900), var(--vert-700)); box-shadow: 0 8px 20px rgba(15,110,79,0.32);
}
div.stButton > button:disabled { opacity: 0.4; }
div.stDownloadButton > button {
    border-radius: 12px; font-weight: 700; background: linear-gradient(120deg, var(--or-500), #e9c766);
    border: none; color: var(--vert-950);
}

/* ---------------------------------------------------------------------------- barre de navigation bas de page */
.pied-page { margin-top: 6px; padding-top: 14px; border-top: 1px solid var(--bordure); }

/* ---------------------------------------------------------------------------- barre latérale : stepper vertical */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, var(--vert-950), var(--vert-900) 60%, var(--vert-950));
}
section[data-testid="stSidebar"] * { color: #eef5f0; }
.sb-logo { display: flex; align-items: center; gap: 10px; padding: 4px 2px 14px 2px; border-bottom: 1px solid rgba(255,255,255,0.12); margin-bottom: 14px; }
.sb-logo .puce { font-size: 1.6rem; }
.sb-logo .txt { font-weight: 700; font-size: 0.95rem; font-family: 'Poppins', sans-serif; }
.sb-etape {
    display: flex; align-items: center; gap: 11px; padding: 9px 10px; border-radius: 12px; margin-bottom: 4px;
    font-size: 0.82rem; font-weight: 600; color: rgba(238,245,240,0.55); transition: all 0.15s ease;
}
.sb-etape .num {
    width: 24px; height: 24px; min-width: 24px; border-radius: 50%; display: flex; align-items: center;
    justify-content: center; font-size: 0.72rem; font-weight: 800; background: rgba(255,255,255,0.08);
    border: 1px solid rgba(255,255,255,0.18);
}
.sb-etape.faite { color: rgba(238,245,240,0.9); }
.sb-etape.faite .num { background: var(--or-500); color: var(--vert-950); border-color: var(--or-500); }
.sb-etape.active {
    color: white; background: rgba(255,255,255,0.10); box-shadow: inset 3px 0 0 var(--or-500);
}
.sb-etape.active .num { background: var(--vert-500); border-color: var(--vert-500); color: white; }
.sb-resume {
    margin-top: 18px; padding: 14px; border-radius: 14px; background: rgba(255,255,255,0.06);
    border: 1px solid rgba(255,255,255,0.12); font-size: 0.78rem;
}
.sb-resume .ligne { display: flex; justify-content: space-between; padding: 3px 0; color: rgba(238,245,240,0.85); }
.sb-resume .ligne b { color: var(--or-300); font-weight: 700; }

/* ---------------------------------------------------------------------------- divers */
.tag-col { display: inline-block; background: var(--vert-100); border: 1px solid var(--bordure); border-radius: 8px;
    padding: 3px 10px; margin: 3px; font-size: 0.78rem; color: var(--vert-900); font-weight: 600; }
hr.separateur { border: none; border-top: 1px dashed var(--bordure); margin: 20px 0; }
.stTabs [data-baseweb="tab-list"] { gap: 6px; }
.stTabs [data-baseweb="tab"] {
    background: var(--vert-100); border-radius: 10px 10px 0 0; font-weight: 600; color: var(--vert-900);
}
[data-testid="stFileUploaderDropzone"] {
    border-radius: 16px !important; border: 2px dashed var(--vert-500) !important; background: var(--vert-100) !important;
}

