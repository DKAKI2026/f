# -*- coding: utf-8 -*-
"""
BACKEND — aucune dépendance à Streamlit. Prédit DthCnt ET DthAmt, avec LightGBM.
Validation croisée : 5 plis x 3 graines par cible (30 modèles au total) — le même rythme de calcul
que le tout premier script de cette conversation (onlyLightGBM.py), reconstitué à la demande.

Le même facteur de crédibilité (calculé sur le MONTANT) sert de point de départ aux deux cibles :
  point de départ DthCnt  = ExpecCnt x facteur
  point de départ DthAmt  = ExpecAmt x facteur

Sections : 1. Constantes  2. Préparation des données (+ cohorte)  3. Strates : tranches personnalisables,
crédibilité, les 5 critères  4. Sélection des colonnes secondaires par importance LightGBM
5. Entraînement croisé, pli par pli (interruptible) — DEUX cibles à chaque pli
6. Tendance temporelle (taux fixe)  7. Métriques (RMSE, déviance de Poisson, Gini, résidus par strate)
8. Sauvegarde / chargement / prédiction
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
TARGETS = ["DthCnt", "DthAmt"]
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

SOCLE = ["Sex", "Smoke", "PolYear", "AttdAge", "Size"]

CANDIDATS_SECONDAIRES = {
    "PolGrp":       "Type de police regroupé (fusionne les catégories qui n'existent que sur certaines années)",
    "Base":         "Contrat de base ou avenant",
    "Par":          "Avec ou sans participation",
    "IssueYear":    "Année d'émission (identifie la cohorte de souscription)",
    "log_qx_amt":   "Log du taux de mortalité implicite de la table (ExpecAmt / ExposAmt)",
    "log_avg_size": "Log du capital moyen par police (ExposAmt / ExposCnt)",
    "cv2":          "Dispersion des capitaux dans la cellule (1 = tous identiques)",
    "log_expos":    "Log du nombre de polices exposées",
    "coh_ae_amt":   "Réel/attendu des AUTRES lignes de la même cohorte (mêmes assurés, autres années)",
    "coh_log_E":    "Poids (log) de l'information disponible derrière coh_ae_amt",
}

COLONNES_DECOUPABLES = ["AttdAge", "PolYear", "YearStart"]

TAUX_TENDANCE = 0.013
TAUX_TENDANCE_SOURCE = ("Taux fixé à 1,3 % par année — le taux ultime publié par l'ICA (étude CIA-MI-2024) "
                        "pour les âges 40 à 90 ans. Appliqué uniquement sur les lignes portant sur une "
                        "année postérieure à la dernière année de df, jamais donné en entrée au modèle.")

# Régularisation par défaut (voir conversation : sans elle, un entraînement long peut déraper).
PARAMS_LGB_DEFAUT = dict(num_leaves=31, min_data_in_leaf=50, lambda_l2=10.0,
                         feature_fraction=0.9, bagging_fraction=0.9, bagging_freq=1, verbose=-1)
RAW_SCORE_MAX = 12  # borne du score log avant exponentiation (garde-fou anti-explosion numérique)


# ============================================================================================
# SECTION 2 — Préparation des données (+ cohorte)
# ============================================================================================
def lire_csv(fichier_bytes_ou_chemin, sep=None, decimal="."):
    import io
    buf = io.BytesIO(fichier_bytes_ou_chemin) if isinstance(fichier_bytes_ou_chemin, (bytes, bytearray)) else fichier_bytes_ou_chemin
    return pd.read_csv(buf, sep=sep, decimal=decimal, engine="python" if sep is None else "c")


def diagnostic_colonnes(df):
    return dict(manque=[c for c in REQUIRED if c not in df.columns],
               a_target=all(t in df.columns for t in TARGETS), colonnes=list(df.columns))


def _lignes_invalides(d, cols):
    x = d[cols].apply(pd.to_numeric, errors="coerce").to_numpy(float)
    return ~np.isfinite(x).all(axis=1)


def nettoyer(df, resultats=None):
    rapport = []
    for nom, d, cols in (("df", df, REQUIRED + TARGETS), ("resultats", resultats, REQUIRED)):
        if d is None:
            continue
        for c in cols:
            if c not in d.columns:
                continue
            x = pd.to_numeric(d[c], errors="coerce")
            bad = int((~np.isfinite(x.to_numpy(float))).sum())
            if bad:
                rapport.append(f"{nom} · {c} : {bad:,} valeurs manquantes/infinies".replace(",", " "))
            d[c] = x   # conversion RÉELLE : sinon une colonne lue comme texte reste du texte malgré la vérification
    erreur_res = None
    if resultats is not None:
        cols_ok = [c for c in CRITIQUES if c in resultats.columns]
        bad_r = _lignes_invalides(resultats, cols_ok)
        if bad_r.any():
            erreur_res = (f"{int(bad_r.sum())} lignes de resultats ont une valeur manquante/infinie dans "
                         f"{cols_ok} : impossible de les prédire tant qu'elles ne sont pas corrigées.")
    bad = _lignes_invalides(df, [c for c in CRITIQUES + TARGETS if c in df.columns])
    n_retirees = int(bad.sum())
    if n_retirees:
        rapport.append(f"ATTENTION : {n_retirees:,} lignes de df retirées de l'entraînement.".replace(",", " "))
        df = df.loc[~bad].copy()
    return df, rapport, erreur_res


def prep(d):
    d = d.copy()
    for c in ("IssueAge", "PolYear", "ExposCnt", "ExposAmt", "ExposAmt2", "ExpecCnt", "ExpecAmt"):
        if c in d.columns:
            d[c] = pd.to_numeric(d[c], errors="coerce")   # sécurité : nettoyer() a dû le faire, mais on ne suppose rien
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


def cohort_table(src, k=1.0):
    g = src.groupby(KEY, sort=False).agg(DA=("DthAmt", "sum"), EA=("ExpecAmt", "sum")).reset_index()
    k_amt = k * src["ExpecAmt"].sum() / max(len(src), 1)
    return g, k_amt


def cohort_from_table(g, k_amt, dst, k=1.0):
    m = dst[KEY].merge(g, on=KEY, how="left").fillna(0.0)
    return pd.DataFrame({"coh_ae_amt": ((m["DA"] + k_amt) / (m["EA"] + k_amt)).values,
                         "coh_log_E": np.log1p(m["EA"]).values}, index=dst.index)


def cohort_oof(src, n_splits=5, seed=0):
    out = pd.DataFrame(index=src.index, columns=["coh_ae_amt", "coh_log_E"], dtype=float)
    for tr, va in KFold(n_splits, shuffle=True, random_state=seed).split(src):
        g, k_amt = cohort_table(src.iloc[tr])
        out.iloc[va] = cohort_from_table(g, k_amt, src.iloc[va]).values
    return out


def preparer(df, mode):
    d = prep(df)
    if mode == "melange":
        d = pd.concat([d, cohort_oof(d)], axis=1)
    else:
        d["coh_ae_amt"], d["coh_log_E"] = 1.0, 0.0
    return d


def pdev(y, p):
    """Déviance de Poisson MOYENNE (divisée par le nombre de lignes, jamais une somme brute)."""
    y, p = np.asarray(y, float), np.clip(np.asarray(p, float), 1e-12, None)
    t = np.where(y > 0, y * np.log(np.where(y > 0, y, 1) / p), 0.0)
    return float(2 * np.sum(t - (y - p)) / len(y))


def rmse(y, p):
    return float(np.sqrt(np.mean((np.asarray(y, float) - np.asarray(p, float)) ** 2)))


# ============================================================================================
# SECTION 3 — Strates : tranches personnalisables, crédibilité (MONTANT), les 5 critères
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


def suggestion_bornes(d, col, n_tranches=5):
    q = np.linspace(0, 1, n_tranches + 1)
    bornes = sorted(set(int(x) for x in d[col].quantile(q).to_numpy()))
    if bornes[0] > d[col].min():
        bornes[0] = int(d[col].min())
    bornes[-1] = 10 ** 6
    return bornes


def appliquer_decoupage(d, decoupage):
    d = d.copy()
    correspondance = {}
    for col, config in (decoupage or {}).items():
        if col == "Size" or isinstance(config, dict):
            defaut = LIBELLES.get(col, {})
            propre = {k: (str(v).strip() if str(v).strip() else str(defaut.get(k, k))) for k, v in config.items()}
            grp_col = col + "_grp"
            d[grp_col] = d[col].map(propre).fillna(d[col].map(defaut)).fillna(d[col].astype(str))
            correspondance[col] = grp_col
        else:
            bornes = sorted(config)
            labels = [f"{bornes[i]}-{bornes[i+1]-1}" if bornes[i+1] < 10 ** 6 else f"{bornes[i]}+"
                     for i in range(len(bornes) - 1)]
            grp_col = col + "_grp"
            d[grp_col] = pd.cut(d[col], bornes, labels=labels, right=False, include_lowest=True).astype(str)
            correspondance[col] = grp_col
    return d, correspondance


def credibilite(deces, n_plein=N_PLEIN):
    return np.minimum(1.0, np.sqrt(np.asarray(deces, float) / n_plein))


def table_strates(d, strate, decoupage=None, n_plein=N_PLEIN):
    """Facteur calculé sur le MONTANT (Reel/Attendu = DthAmt/ExpecAmt) ; sert de point de départ
    partagé pour les deux cibles (voir _point_depart)."""
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
        if c_reel == c_orig and c_orig in LIBELLES:
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
    return f, inconnu


def criteres_credibilite(d, strate, tab, decoupage=None, n_plein=N_PLEIN):
    """Les 5 critères, chacun {titre, lignes: [(symbole, texte, niveau_ou_None)]}. Langage simple."""
    d2, correspondance = appliquer_decoupage(d, decoupage)
    cols = [correspondance.get(c, c) for c in strate] or ["_tout"]
    d2 = d2.assign(_tout=0) if not strate else d2
    blocs = []

    def bloc(titre, fn):
        lignes = []
        try:
            fn(lignes)
        except Exception as e:
            lignes.append(("⚠", f"Vérification impossible ({type(e).__name__}: {e})", None))
        blocs.append({"titre": titre, "lignes": lignes})

    def _fmt(x, nd=0):
        return f"{x:,.{nd}f}".replace(",", " ")

    def racine(lignes):
        z = np.minimum(1.0, np.sqrt(tab["Deces"] / n_plein))
        ecart = float(np.abs(z - tab["Z"]).max())
        lignes.append(("✓" if ecart < 1e-9 else "⚠",
                       f"Règle utilisée : en dessous de {_fmt(n_plein)} décès, on ne fait pas encore pleinement "
                       "confiance à une strate — plus elle a de décès, plus son facteur compte, jusqu'à ce seuil.", None))
        pleines = tab["Z"] >= 1.0
        part = float(tab.loc[pleines, "Poids"].sum())
        lignes.append(("ℹ", f"{int(pleines.sum())} strate(s) sur {len(tab)} ont assez de décès pour être pleinement "
                       f"fiables (100 % de crédibilité), et elles représentent {part:.0%} du montant total attendu.", niveau_pct(part)))

    def intervalle(lignes):
        z_def = float(math.erf(0.03 * math.sqrt(n_plein) / math.sqrt(2)))
        lignes.append(("✓" if abs(z_def - 0.90) < 0.02 else "⚠",
                       f"Le seuil de {_fmt(n_plein)} décès correspond à une confiance d'environ 90 % que l'écart "
                       "avec la réalité reste faible — une autre façon d'arriver au même seuil que la règle simple.", None))
        if "ExposAmt2" not in d.columns or "ExposCnt" not in d.columns:
            lignes.append(("ℹ", "Les colonnes nécessaires pour affiner ce calcul au montant sont absentes : "
                           "cette vérification est ignorée.", None))
            return
        a = d2.groupby(cols).agg(E=("ExposCnt", "sum"), A=("ExposAmt", "sum"), A2=("ExposAmt2", "sum")).reset_index()
        m = tab[cols].merge(a, on=cols, how="left")
        cv2 = np.clip(np.where(m["A"] > 0, m["A2"] * m["E"] / m["A"] ** 2, 1.0), 1.0, None)
        seuil = n_plein * cv2
        z_ajuste = np.minimum(1.0, np.sqrt(tab["Deces"].to_numpy(float) / seuil))
        moins = (tab["Z"].to_numpy(float) - z_ajuste) > 0.10
        w = tab["Poids"].to_numpy(float)
        part = float(w[moins].sum())
        lignes.append(("⚠" if moins.any() else "✓",
                       f"{int(moins.sum())} strate(s), représentant {part:.0%} du montant attendu, ont des capitaux "
                       "très inégaux (quelques grosses polices dominent) : leur crédibilité est peut-être surestimée "
                       "si on ne compte que le nombre de décès.", niveau_pct(1 - part) if moins.any() else "haut"))

    def whitney(lignes):
        K = n_plein / 4
        lignes.append(("✓", f"Cette formule donne presque le même résultat que la règle simple : à mi-chemin du "
                       f"seuil ({_fmt(K)} décès), la confiance atteint déjà 50 %.", None))
        z_w = tab["Deces"] / (tab["Deces"] + K)
        ecart = float(np.average(np.abs(tab["Z"] - z_w), weights=tab["Poids"]))
        lignes.append(("ℹ", f"L'écart moyen entre les deux méthodes, pondéré par le poids de chaque strate, "
                       f"est de {ecart:.1%} : les deux règles donnent des résultats proches.", niveau_pct(1 - ecart)))

    def buhlmann(lignes):
        x = d2[d2["ExpecAmt"] > 0]
        g = x.groupby(cols, sort=False).ngroup().to_numpy()
        r, n = int(g.max()) + 1, len(x)
        if r < 2 or n <= r:
            lignes.append(("⚠", "Pas assez de strates différentes pour appliquer cette méthode ici.", None))
            return
        m = x["ExpecAmt"].to_numpy(float)
        X = (x["DthAmt"] / x["ExpecAmt"]).to_numpy(float)
        mj = np.bincount(g, weights=m)
        Xj = np.bincount(g, weights=m * X) / mj
        mtot = mj.sum()
        Xbar = float((mj * Xj).sum() / mtot)
        v = float(np.bincount(g, weights=m * (X - Xj[g]) ** 2).sum() / (n - r))
        a = float(((mj * (Xj - Xbar) ** 2).sum() - (r - 1) * v) / (mtot - (mj ** 2).sum() / mtot))
        if not (a > 0 and np.isfinite(a) and np.isfinite(v)):
            lignes.append(("⚠", "Les strates ne se distinguent pas assez les unes des autres pour que cette méthode "
                           "apporte quelque chose de plus ici.", None))
            return
        k = v / a
        z_bs = mj / (mj + k)
        moy = float(np.average(z_bs, weights=mj / mtot))
        lignes.append(("✓", "Cette méthode, plus savante, calcule elle-même le seuil de confiance à partir de tes "
                       "données, au lieu d'utiliser un seuil fixe comme les 3 006 décès de la règle simple.", None))
        lignes.append(("ℹ", f"Avec cette méthode, la confiance moyenne (pondérée) dans tes strates est de {moy:.0%} "
                       f"(contre {np.average(tab['Z'], weights=tab['Poids']):.0%} avec la règle simple).", niveau_pct(moy)))

    def bayes(lignes):
        if "ExpecCnt" not in d.columns:
            lignes.append(("ℹ", "La colonne du nombre de décès attendus est absente : cette vérification est ignorée.", None))
            return
        a = d2.groupby(cols).agg(E=("ExpecCnt", "sum")).reset_index()
        m = tab[cols].merge(a, on=cols, how="left")
        E = m["E"].to_numpy(float)
        D = tab["Deces"].to_numpy(float)
        ok = np.isfinite(E) & (E > 0)
        E, D, w = E[ok], D[ok], tab["Poids"].to_numpy(float)[ok]
        if len(E) < 2:
            lignes.append(("⚠", "Pas assez de strates différentes pour appliquer cette méthode ici.", None))
            return
        mu = D.sum() / E.sum()
        R = D / E
        tau2 = ((E * (R - mu) ** 2).sum() - mu * (len(E) - 1)) / (E.sum() - (E ** 2).sum() / E.sum())
        if not (tau2 > 0 and np.isfinite(tau2)):
            lignes.append(("⚠", "Les strates ne se distinguent pas assez les unes des autres pour que cette méthode "
                           "apporte quelque chose de plus ici.", None))
            return
        beta = mu / tau2
        Z = E / (E + beta)
        moy = float(np.average(Z, weights=w))
        lignes.append(("✓", "Cette méthode part d'une hypothèse de départ raisonnable (le niveau moyen de tout le "
                       "portefeuille), puis la corrige progressivement à mesure que les données d'une strate "
                       "s'accumulent.", None))
        lignes.append(("ℹ", f"Confiance moyenne (pondérée) obtenue avec cette méthode : {moy:.0%}.", niveau_pct(moy)))

    bloc("Fluctuation limitée, racine carrée", racine)
    bloc("Fluctuation limitée, intervalle de confiance", intervalle)
    bloc("Formule de Whitney N/(N+K)", whitney)
    bloc("Bühlmann / Bühlmann-Straub", buhlmann)
    bloc("Bayésienne (Poisson-Gamma)", bayes)
    return blocs


def residus_par_strate(d, y_cnt, p_cnt, y_amt, p_amt, strate, decoupage=None):
    d2, correspondance = appliquer_decoupage(d, decoupage)
    cols = [correspondance.get(c, c) for c in strate] or ["_tout"]
    d2 = d2.assign(_tout=0) if not strate else d2
    t = d2[cols].copy()
    t["_reel_cnt"], t["_predit_cnt"] = np.asarray(y_cnt, float), np.asarray(p_cnt, float)
    t["_reel_amt"], t["_predit_amt"] = np.asarray(y_amt, float), np.asarray(p_amt, float)
    g = t.groupby(cols).agg(Lignes=("_reel_amt", "size"), Reel_Cnt=("_reel_cnt", "sum"), Predit_Cnt=("_predit_cnt", "sum"),
                            Reel_Amt=("_reel_amt", "sum"), Predit_Amt=("_predit_amt", "sum")).reset_index()
    g["Ecart_Cnt"] = g["Reel_Cnt"] - g["Predit_Cnt"]
    g["Ecart_Amt"] = g["Reel_Amt"] - g["Predit_Amt"]
    g["Ecart_Amt_pct"] = np.where(g["Reel_Amt"] > 0, g["Ecart_Amt"] / g["Reel_Amt"], 0.0)
    return g.sort_values("Lignes", ascending=False).reset_index(drop=True)


# ============================================================================================
# SECTION 4 — Sélection des colonnes secondaires par IMPORTANCE LightGBM (rapide, une seule fois,
# sur la cible DthAmt — les colonnes retenues servent ensuite aux DEUX cibles).
# ============================================================================================
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


def _point_depart(d, F, colonne_expo):
    base = d[colonne_expo].to_numpy(float)
    return base * (np.ones(len(d)) if F is None else np.maximum(F, 1e-3))


def _fit_lgb(X, z, w, params, valid=None, seed=0):
    import lightgbm as lgb
    p = {k: v for k, v in params.items() if k != "max_rounds"}
    p = dict(PARAMS_LGB_DEFAUT, **p, objective="poisson", seed=seed)
    dtr = lgb.Dataset(X, z, weight=w)
    if valid is not None:
        Xv, zv, wv = valid
        dva = lgb.Dataset(Xv, zv, weight=wv, reference=dtr)
        return lgb.train(p, dtr, params.get("max_rounds", 2000), valid_sets=[dva],
                         callbacks=[lgb.early_stopping(100, verbose=False)])
    return lgb.train(p, dtr, params.get("max_rounds", 300))


def _predire_borne(booster, X, base, n_iteration=None):
    score = np.clip(booster.predict(X, raw_score=True, num_iteration=n_iteration), -RAW_SCORE_MAX, RAW_SCORE_MAX)
    return base * np.exp(score)


def importance_colonnes(d, params, seed=0):
    """Entraîne UN LightGBM (cible DthAmt, sans validation croisée : juste pour classer les colonnes)
    avec le socle + toutes les colonnes secondaires disponibles, et renvoie l'importance de chacune."""
    candidats = [c for c in CANDIDATS_SECONDAIRES if c in d.columns]
    cols = list(SOCLE) + candidats
    mapping = _mapping(d, cols)
    X = _X(d, cols, mapping)
    base = d["ExpecAmt"].to_numpy(float)
    z = d["DthAmt"].to_numpy(float) / np.clip(base, 1e-9, None)
    b = _fit_lgb(X, z, base, dict(params, max_rounds=params.get("max_rounds", 300)), seed=seed)
    imp = b.feature_importance(importance_type="gain")
    total = imp.sum() or 1.0
    t = pd.DataFrame({"Colonne": cols, "Importance (%)": 100 * imp / total})
    t["Groupe"] = np.where(t["Colonne"].isin(SOCLE), "Socle", "Secondaire")
    t["Définition"] = t["Colonne"].map(CANDIDATS_SECONDAIRES).fillna("Variable du socle")
    return t.sort_values("Importance (%)", ascending=False).reset_index(drop=True)


def selectionner_par_importance(d, seuil_pct, params, seed=0):
    t = importance_colonnes(d, params, seed)
    t_sec = t[t["Groupe"] == "Secondaire"]
    retenues = t_sec.loc[t_sec["Importance (%)"] >= seuil_pct, "Colonne"].tolist()
    return retenues, t


# ============================================================================================
# SECTION 5 — Entraînement croisé, pli par pli (interruptible) — DEUX CIBLES à chaque pli
# ============================================================================================
def etat_initial(d, cols, mode, n_splits, seeds, params, avec_facteur, strate, decoupage):
    return dict(d=d, cols=cols, mode=mode, n_splits=n_splits, seeds=list(seeds), params=params,
               avec_facteur=avec_facteur, strate=strate, decoupage=decoupage,
               plis=plis(d, mode, n_splits), pli_courant=0,
               oof_cnt=np.full(len(d), np.nan), oof_amt=np.full(len(d), np.nan),
               modeles_cnt=[], modeles_amt=[], mapping=None, termine=False, interrompu=False)


def entrainer_un_pli(etat):
    """Un pli = TOUTES les graines, POUR LES DEUX CIBLES (DthCnt et DthAmt) sur ce pli. Renvoie un
    résumé (pour affichage) ; met etat à jour en place."""
    k = etat["pli_courant"]
    tr, va = etat["plis"][k]
    d, cols = etat["d"], etat["cols"]
    d_tr, d_va = d.iloc[tr], d.iloc[va]
    F_tr = F_va = None
    if etat["avec_facteur"]:
        tab_tr = table_strates(d_tr, etat["strate"], etat["decoupage"])
        F_tr = appliquer_facteur(tab_tr, etat["strate"], d_tr, etat["decoupage"])[0]
        F_va = appliquer_facteur(tab_tr, etat["strate"], d_va, etat["decoupage"])[0]
    mapping = etat["mapping"] or _mapping(d_tr, cols)
    etat["mapping"] = mapping
    Xtr, Xva = _X(d_tr, cols, mapping), _X(d_va, cols, mapping)

    t0 = time.time()
    resultats_seeds = {}
    for cible, colonne_expo in (("cnt", "ExpecCnt"), ("amt", "ExpecAmt")):
        base_tr = _point_depart(d_tr, F_tr, colonne_expo)
        base_va = _point_depart(d_va, F_va, colonne_expo)
        y_tr = d_tr[TARGETS[0] if cible == "cnt" else TARGETS[1]].to_numpy(float)
        y_va = d_va[TARGETS[0] if cible == "cnt" else TARGETS[1]].to_numpy(float)
        z_tr, z_va = y_tr / np.clip(base_tr, 1e-9, None), y_va / np.clip(base_va, 1e-9, None)
        p_va_moy, meilleures_iters = np.zeros(len(va)), []
        for s in etat["seeds"]:
            b = _fit_lgb(Xtr, z_tr, base_tr, etat["params"], valid=(Xva, z_va, base_va), seed=s)
            p_va_moy += _predire_borne(b, Xva, base_va, b.best_iteration) / len(etat["seeds"])
            meilleures_iters.append(b.best_iteration)
            etat[f"modeles_{cible}"].append(dict(booster=b.model_to_string(num_iteration=b.best_iteration)))
        resultats_seeds[cible] = (p_va_moy, y_va, meilleures_iters)
    dt = time.time() - t0

    etat["oof_cnt"][va], etat["oof_amt"][va] = resultats_seeds["cnt"][0], resultats_seeds["amt"][0]
    etat["pli_courant"] += 1
    if etat["pli_courant"] >= len(etat["plis"]):
        etat["termine"] = True
    return dict(pli=k + 1, sur=len(etat["plis"]), temps=dt,
               deviance_cnt=pdev(resultats_seeds["cnt"][1], resultats_seeds["cnt"][0]),
               deviance_amt=pdev(resultats_seeds["amt"][1], resultats_seeds["amt"][0]),
               iters_cnt=resultats_seeds["cnt"][2], iters_amt=resultats_seeds["amt"][2])


def resume_hors_pli(etat):
    cov = ~np.isnan(etat["oof_amt"])
    d = etat["d"]
    y_cnt, p_cnt = d["DthCnt"].to_numpy(float)[cov], etat["oof_cnt"][cov]
    y_amt, p_amt = d["DthAmt"].to_numpy(float)[cov], etat["oof_amt"][cov]
    e_cnt, e_amt = d["ExpecCnt"].to_numpy(float)[cov], d["ExpecAmt"].to_numpy(float)[cov]
    return dict(n=int(cov.sum()), cov=cov,
               y_cnt=y_cnt, p_cnt=p_cnt, e_cnt=e_cnt, y_amt=y_amt, p_amt=p_amt, e_amt=e_amt,
               deviance_cnt=pdev(y_cnt, p_cnt), deviance_amt=pdev(y_amt, p_amt),
               deviance_cnt_table=pdev(y_cnt, e_cnt), deviance_amt_table=pdev(y_amt, e_amt),
               rmse_cnt=rmse(y_cnt, p_cnt), rmse_amt=rmse(y_amt, p_amt),
               somme_pred_reel_cnt=float(p_cnt.sum() / max(y_cnt.sum(), 1e-9)),
               somme_pred_reel_amt=float(p_amt.sum() / y_amt.sum()))


# ============================================================================================
# SECTION 6 — Tendance temporelle (taux FIXE, appliquée après la prédiction, sur les DEUX cibles)
# ============================================================================================
def appliquer_tendance(pred, n_annees, taux=TAUX_TENDANCE):
    n_annees = np.asarray(n_annees, float)
    return np.asarray(pred, float) * (1 - taux) ** np.clip(n_annees, 0, None)


def calculer_n_annees(resultats, derniere_annee_df):
    ys = pd.to_numeric(resultats["Year"]).astype(int) // 100 + 2000
    return (ys - derniere_annee_df).to_numpy(float)


# ============================================================================================
# SECTION 7 — Métriques
# ============================================================================================
def gini_normalise(y, p, poids=None):
    y, p = np.asarray(y, float), np.asarray(p, float)
    w = np.ones(len(y)) if poids is None else np.asarray(poids, float)
    _trapz = getattr(np, "trapezoid", None) or np.trapz

    def lorenz_aire(ordre):
        yo, wo = y[ordre], w[ordre]
        cum_w = np.cumsum(wo) / wo.sum()
        cum_y = np.cumsum(yo * wo) / (yo * wo).sum()
        return float(_trapz(cum_y, cum_w))

    gini_modele = 1 - 2 * lorenz_aire(np.argsort(p))
    gini_parfait = 1 - 2 * lorenz_aire(np.argsort(y))
    return gini_modele / gini_parfait if gini_parfait > 1e-9 else float("nan")


# ============================================================================================
# SECTION 8 — Sauvegarde, chargement, prédiction sur resultats
# ============================================================================================
def construire_bundle(etat, derniere_annee_df):
    return dict(cols=etat["cols"], mapping=etat["mapping"], mode=etat["mode"], avec_facteur=etat["avec_facteur"],
               strate=etat["strate"], decoupage=etat["decoupage"],
               tab_facteur=table_strates(etat["d"], etat["strate"], etat["decoupage"]) if etat["avec_facteur"] else None,
               modeles_cnt=etat["modeles_cnt"], modeles_amt=etat["modeles_amt"],
               coh_g=(cohort_table(etat["d"])[0] if etat["mode"] == "melange" and any(c in etat["cols"] for c in ("coh_ae_amt", "coh_log_E")) else None),
               coh_k_amt=(cohort_table(etat["d"])[1] if etat["mode"] == "melange" and any(c in etat["cols"] for c in ("coh_ae_amt", "coh_log_E")) else None),
               resume=resume_hors_pli(etat), derniere_annee_df=derniere_annee_df)


def bundle_to_bytes(bundle):
    b = dict(bundle)
    b["resume"] = {k: v for k, v in b["resume"].items() if k not in
                  ("y_cnt", "p_cnt", "e_cnt", "y_amt", "p_amt", "e_amt", "cov")}
    return pickle.dumps(b)


def bundle_from_bytes(raw):
    return pickle.loads(raw)


def predire(bundle, resultats, appliquer_la_tendance=True):
    res = prep(resultats).reset_index(drop=True)
    if bundle.get("coh_g") is not None:
        res = pd.concat([res, cohort_from_table(bundle["coh_g"], bundle["coh_k_amt"], res)], axis=1)
    else:
        res["coh_ae_amt"], res["coh_log_E"] = 1.0, 0.0
    F = None
    if bundle["avec_facteur"]:
        F = appliquer_facteur(bundle["tab_facteur"], bundle["strate"], res, bundle["decoupage"])[0]
    import lightgbm as lgb
    X = _X(res, bundle["cols"], bundle["mapping"])
    sub = resultats.copy()
    for cible, colonne_expo, cible_out, modeles in (("cnt", "ExpecCnt", "DthCnt", bundle["modeles_cnt"]),
                                                     ("amt", "ExpecAmt", "DthAmt", bundle["modeles_amt"])):
        base = _point_depart(res, F, colonne_expo)
        pred = np.mean([_predire_borne(lgb.Booster(model_str=m["booster"]), X, base) for m in modeles], axis=0)
        if appliquer_la_tendance:
            n = calculer_n_annees(resultats, bundle["derniere_annee_df"])
            pred = appliquer_tendance(pred, n)
        sub[cible_out] = np.clip(pred, 0, None)
    return sub



































# -*- coding: utf-8 -*-
"""
FRONTEND (Streamlit) — ne contient AUCUN calcul : tout est délégué à backend.py.
Pipeline : Données -> Strates & crédibilité -> Facteur -> Colonnes -> Entraînement -> Résultats.
Prédit DthCnt ET DthAmt. Validation croisée 5 plis x 3 graines par cible (30 modèles), pli par pli,
interruptible — le même rythme de calcul que le tout premier script de cette conversation.

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
importlib.reload(b)

assert hasattr(b, "entrainer_un_pli") and hasattr(b, "TARGETS"), (
    f"Le fichier backend.py chargé ({b.__file__}) est une VERSION PÉRIMÉE. "
    "Remplace-le par le dernier backend.py fourni, puis supprime le dossier __pycache__.")

st.set_page_config(page_title="Assistant mortalité — décès et montant", page_icon="🧮", layout="wide")

CSS_PATH = Path(__file__).parent / "style.css"
if CSS_PATH.exists():
    st.markdown(f"<style>{CSS_PATH.read_text(encoding='utf-8')}</style>", unsafe_allow_html=True)

NOM_DEFI = "Défi Données · Modélisation actuarielle"
NOM_EQUIPE = "Équipe 7"

ETAPES = [
    ("📂", "Données", "Glisser-déposer et nettoyer"),
    ("🧩", "Strates & crédibilité", "Tranches, crédibilité"),
    ("⚖️", "Facteur", "Point de départ pour les 2 cibles"),
    ("🎯", "Colonnes", "Socle fixe + sélection"),
    ("🚀", "Entraînement", "5 plis x 3 graines, 2 cibles"),
    ("🏆", "Résultats", "Métriques, tendance, téléchargement"),
]


# ============================================================================================
# État
# ============================================================================================
def init_etat():
    defauts = dict(etape=0, df=None, df_nom=None, strate=[], sans_strate=False, strate_confirmee=False,
                   decoupage={}, tab_strate=None, avec_facteur=None, methode_cols="importance", seuil_pct=1.0,
                   cols_secondaires=None, tab_importance=None, learning_rate=0.05, max_rounds=2000,
                   n_splits=5, n_seeds=3, rapide=False, train=None, train_signature=None, bundle=None)
    for k, v in defauts.items():
        st.session_state.setdefault(k, v)


init_etat()


def aller(delta):
    st.session_state.etape = max(0, min(len(ETAPES) - 1, st.session_state.etape + delta))


def signature_actuelle():
    s = (tuple(st.session_state.cols_secondaires or []), st.session_state.avec_facteur, tuple(st.session_state.strate),
        str(st.session_state.decoupage), st.session_state.rapide, st.session_state.learning_rate,
        st.session_state.max_rounds, st.session_state.n_splits, st.session_state.n_seeds)
    return hashlib.md5(str(s).encode()).hexdigest()


def badge(symbole, texte, niveau=None):
    classe = {"✓": "badge-ok", "⚠": "badge-warn", "ℹ": "badge-info"}[symbole]
    pastille = ""
    if niveau in ("bas", "moyen", "haut"):
        label = {"bas": "Faible", "moyen": "Moyenne", "haut": "Bonne"}[niveau]
        pastille = f"<span class='niveau-pill niveau-{niveau}'>● Fiabilité {label}</span>"
    st.markdown(f"<div class='ligne-message'><span class='badge {classe}'>{symbole}</span> {texte} {pastille}</div>", unsafe_allow_html=True)


def _couleur_cellule(niveau):
    couleurs = {"bas": ("#fdeceb", "#a83226"), "moyen": ("#fdf3dc", "#8a6300"), "haut": ("#e6f6ec", "#116639")}
    fond, texte = couleurs.get(niveau, ("", ""))
    return f"background-color:{fond}; color:{texte}; font-weight:700;" if fond else ""


def colorer_fiabilite(val):
    return _couleur_cellule(b.niveau_pct(val))


def kicker(icone, titre, description):
    st.markdown(f"""<div class="kicker-page"><div class="icone">{icone}</div>
        <div class="textes"><p class="titre">{titre}</p><p class="desc">{description}</p></div></div>""", unsafe_allow_html=True)


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
        <div><h1>Assistant de modélisation — nombre et montant des décès</h1>
        <p class="sous-titre">LightGBM, validation croisée 5 plis x 3 graines, un facteur partagé comme point de départ</p></div>
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
    if st.session_state.train is not None:
        lignes_resume.append(("Plis faits", f"{st.session_state.train['pli_courant']} / {len(st.session_state.train['plis'])}"))
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
# ÉTAPE 1 — Strates & crédibilité (tranches personnalisables, confirmation, les 5 critères)
# ============================================================================================
elif etape == 1:
    df = st.session_state.df
    d = b.prep(df)
    colonnes_possibles = [c for c in ["Sex", "Smoke", "PolGrp", "Base", "Par", "Size", "AttdAge", "PolYear"] if c in d.columns]

    kicker("🧩", "Strates & crédibilité", "Choisis un découpage, ajuste les tranches, puis confirme")
    st.markdown("<div class='carte'><h3>Colonnes de la strate</h3>", unsafe_allow_html=True)
    st.caption("Le facteur est calculé sur le MONTANT (réel/attendu). Il sert ensuite de point de départ partagé aux deux cibles.")
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
            sugg = ", ".join(str(x) for x in b.suggestion_bornes(d, col))
            txt = st.text_input(f"Bornes pour {col} (ex. 0, 20, 40, 60, 80)", value="", placeholder=f"suggestion : {sugg}", key=f"bornes_{col}")
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
        st.markdown("**Regroupement des tranches de capital (`Size`)** — mets le même texte à des tranches pour les fusionner ; laisse vide pour garder le libellé d'origine")
        tailles = sorted(d["Size"].dropna().unique().tolist())
        libelles = {t: b.LIBELLES.get("Size", {}).get(t, str(t)) for t in tailles}
        groupe_actuel = st.session_state.decoupage.get("Size", {})
        cols_g = st.columns(len(tailles))
        nouveau_groupe = {}
        for i, t in enumerate(tailles):
            with cols_g[i]:
                nouveau_groupe[t] = st.text_input(libelles[t], value=str(groupe_actuel.get(t, "")), key=f"size_grp_{t}")
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
        cols_aff = [c for c in cols_reel + ["Lignes", "Deces", "Attendu", "Reel", "Ratio", "Z", "Fiabilite", "Facteur", "Poids"] if c in tab.columns]
        style = tab[cols_aff].style.format({"Attendu": "{:,.2f}", "Reel": "{:,.2f}", "Ratio": "{:.2f}", "Z": "{:.0%}",
                                            "Facteur": "{:.2f}", "Poids": "{:.1%}", "Deces": "{:,.2f}"})
        if "Z" in cols_aff:
            style = style.map(colorer_fiabilite, subset=["Z"])
        st.dataframe(style, use_container_width=True, height=300)
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("<div class='carte'><h3>🔎 Autres critères de crédibilité (à titre informatif)</h3>", unsafe_allow_html=True)
        st.caption("Ce sont d'autres façons de calculer la même idée : à quel point peut-on se fier aux données "
                  "observées plutôt qu'à la table de référence. Elles n'influencent pas le facteur ci-dessus.")
        blocs = b.criteres_credibilite(d, strate, tab, st.session_state.decoupage)
        onglets = st.tabs([bl["titre"] for bl in blocs])
        for onglet, bl in zip(onglets, blocs):
            with onglet:
                for s, t, n in bl["lignes"]:
                    badge(s, t, n)
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
# ÉTAPE 2 — Facteur (partagé entre les deux cibles)
# ============================================================================================
elif etape == 2:
    kicker("⚖️", "Facteur", "Un seul facteur (montant), point de départ pour les deux cibles")
    st.markdown("<div class='carte'>", unsafe_allow_html=True)
    if st.session_state.sans_strate or not st.session_state.strate:
        st.info("Aucune strate choisie à l'étape précédente : le facteur n'est pas utilisé.")
        st.session_state.avec_facteur = False
    else:
        st.write(f"Strate retenue : `{st.session_state.strate}`. Le même facteur (calculé sur le montant) peut "
                "servir de **point de départ** aux deux cibles : `DthCnt` part de `ExpecCnt x facteur`, `DthAmt` "
                "part de `ExpecAmt x facteur`. Le modèle n'apprend alors que la correction qu'il reste à faire.")
        choix = st.radio("Choix", ["Utiliser le facteur comme point de départ", "Ne pas l'utiliser (repartir de la table seule)"],
                         index=0 if st.session_state.avec_facteur is not False else 1)
        st.session_state.avec_facteur = choix.startswith("Utiliser")
    st.markdown("</div>", unsafe_allow_html=True)
    pied_navigation()


# ============================================================================================
# ÉTAPE 3 — Colonnes : socle fixe + sélection (manuelle ou par importance LightGBM)
# ============================================================================================
elif etape == 3:
    kicker("🎯", "Colonnes du modèle", "Un socle toujours inclus, une sélection manuelle ou automatique pour le reste — utilisé pour les DEUX cibles")
    st.markdown("<div class='carte'><h3>Socle (toujours inclus)</h3>", unsafe_allow_html=True)
    st.write("Ces variables sont des facteurs de risque reconnus : elles entrent toujours dans le modèle, "
            "sans passer par une sélection.")
    st.markdown("".join(f"<span class='tag-col'>{c}</span>" for c in b.SOCLE), unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div class='carte'><h3>Colonnes secondaires disponibles</h3>", unsafe_allow_html=True)
    tab_def = pd.DataFrame({"Colonne": list(b.CANDIDATS_SECONDAIRES), "Définition": list(b.CANDIDATS_SECONDAIRES.values())})
    st.dataframe(tab_def, use_container_width=True, hide_index=True)
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div class='carte'><h3>Méthode de sélection</h3>", unsafe_allow_html=True)
    methode = st.radio("Méthode", ["Sélection automatique (importance du modèle)", "Sélection manuelle"],
                       index=0 if st.session_state.methode_cols == "importance" else 1)
    st.session_state.methode_cols = "importance" if methode.startswith("Sélection automatique") else "manuelle"
    st.session_state.rapide = st.checkbox("Mode rapide (moins de plis/graines/arbres — pour tester que tout fonctionne avant l'exécution complète)", value=st.session_state.rapide)

    if st.session_state.methode_cols == "manuelle":
        defaut = [c for c in (st.session_state.cols_secondaires or []) if c in b.CANDIDATS_SECONDAIRES]
        cols_sec = st.multiselect("Colonnes secondaires à inclure", list(b.CANDIDATS_SECONDAIRES), default=defaut)
        st.session_state.cols_secondaires = cols_sec
    else:
        st.caption("💡 En clair : on entraîne un modèle une fois (cible montant, sans validation croisée — juste "
                  "pour classer les colonnes), on regarde combien chacune compte pour lui, et on ne garde que "
                  "celles au-dessus du seuil. Les colonnes retenues serviront ensuite aux DEUX cibles.")
        st.session_state.seuil_pct = st.number_input("Seuil d'importance (% du total)", min_value=0.1, max_value=20.0,
                                                      value=st.session_state.seuil_pct, step=0.5)
        if st.button("Calculer l'importance et sélectionner", type="primary"):
            mode = "melange"
            d = b.preparer(st.session_state.df, mode)
            params = dict(learning_rate=st.session_state.learning_rate, max_rounds=100 if st.session_state.rapide else 300)
            with st.spinner("Calcul en cours..."):
                cols_sec, tab_imp = b.selectionner_par_importance(d, st.session_state.seuil_pct, params)
            st.session_state.cols_secondaires = cols_sec
            st.session_state.tab_importance = tab_imp
        if st.session_state.tab_importance is not None:
            t = st.session_state.tab_importance.copy()
            t["Retenue"] = np.where(t["Groupe"] == "Socle", "—", np.where(t["Colonne"].isin(st.session_state.cols_secondaires or []), "✓", ""))
            st.dataframe(t.style.format({"Importance (%)": "{:.2f}"}), use_container_width=True, hide_index=True)

    if st.session_state.cols_secondaires is not None:
        st.success(f"Colonnes secondaires retenues : {st.session_state.cols_secondaires or '(aucune)'}")
    st.markdown("</div>", unsafe_allow_html=True)
    pied_navigation(st.session_state.cols_secondaires is not None)


# ============================================================================================
# ÉTAPE 4 — Entraînement croisé, pli par pli (interruptible) — DEUX CIBLES
# ============================================================================================
elif etape == 4:
    kicker("🚀", "Entraînement", "Validation croisée, pli par pli — un pli entraîne les deux cibles")
    st.markdown("<div class='carte'>", unsafe_allow_html=True)
    mode = "melange"
    d = b.preparer(st.session_state.df, mode)
    cols_final = list(b.SOCLE) + list(st.session_state.cols_secondaires or [])

    c1, c2 = st.columns(2)
    st.session_state.n_splits = c1.number_input("Nombre de plis", 2, 10, st.session_state.n_splits, disabled=st.session_state.rapide)
    st.session_state.n_seeds = c2.number_input("Nombre de graines par pli", 1, 5, st.session_state.n_seeds, disabled=st.session_state.rapide)
    c3, c4 = st.columns(2)
    st.session_state.learning_rate = c3.slider("Taux d'apprentissage", 0.01, 0.3, st.session_state.learning_rate, disabled=st.session_state.rapide)
    st.session_state.max_rounds = c4.number_input("Nombre maximal d'arbres (arrêt anticipé actif)", 100, 3000, st.session_state.max_rounds, step=100, disabled=st.session_state.rapide)
    n_splits_reel = 2 if st.session_state.rapide else st.session_state.n_splits
    n_seeds_reel = 1 if st.session_state.rapide else st.session_state.n_seeds
    st.caption(f"Total prévu : {n_splits_reel} plis x {n_seeds_reel} graine(s) x 2 cibles = "
              f"{n_splits_reel * n_seeds_reel * 2} modèles entraînés.")

    sig = signature_actuelle()
    if st.session_state.train is None or st.session_state.train_signature != sig:
        params = dict(learning_rate=st.session_state.learning_rate, max_rounds=200 if st.session_state.rapide else st.session_state.max_rounds)
        seeds = list(range(n_seeds_reel))
        st.session_state.train = b.etat_initial(d, cols_final, mode, n_splits_reel, seeds, params,
                                                st.session_state.avec_facteur, st.session_state.strate, st.session_state.decoupage)
        st.session_state.train_signature = sig
        st.session_state.bundle = None
        st.info("Réglages pris en compte : entraînement (ré)initialisé.")

    etat = st.session_state.train
    fait, total = etat["pli_courant"], len(etat["plis"])
    st.progress(fait / total if total else 0, text=f"Pli {fait} / {total}")

    c1, c2, c3 = st.columns(3)
    if c1.button("▶ Entraîner un pli", disabled=etat["termine"], use_container_width=True):
        with st.spinner(f"Entraînement du pli ({len(etat['seeds'])} graines x 2 cibles)..."):
            r = b.entrainer_un_pli(etat)
        st.toast(f"Pli {r['pli']}/{r['sur']} en {r['temps']:.0f}s — dev. cnt {r['deviance_cnt']:.2f}, dev. amt {r['deviance_amt']:,.0f}".replace(",", " "))
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
        resume = b.resume_hors_pli(etat)
        m1, m2 = st.columns(2)
        m1.markdown(f"<div class='metrique-box'><div class='val'>{resume['deviance_cnt']:.2f}</div><div class='lab'>déviance nombre (hors-pli)</div></div>", unsafe_allow_html=True)
        m2.markdown(f"<div class='metrique-box'><div class='val'>{resume['deviance_amt']:,.2f}</div><div class='lab'>déviance montant (hors-pli)</div></div>".replace(",", " "), unsafe_allow_html=True)

    if etat["termine"] and st.session_state.bundle is None:
        if etat.get("interrompu"):
            st.warning(f"Entraînement arrêté après {fait}/{total} plis : l'estimation hors-pli est moins fiable, "
                      "et chaque cible a moins de {len(etat['seeds'])} modèle(s) par pli manquant.")
        derniere = int(d["YearStart"].max())
        st.session_state.bundle = b.construire_bundle(etat, derniere)
        st.success("Entraînement terminé — bundle prêt.")
    st.markdown("</div>", unsafe_allow_html=True)
    pied_navigation(st.session_state.bundle is not None, texte_suivant="Voir les résultats →")


# ============================================================================================
# ÉTAPE 5 — Résultats : métriques (2 cibles), résidus par strate, tendance, téléchargement
# ============================================================================================
elif etape == 5:
    bundle = st.session_state.bundle
    kicker("🏆", "Résultats", "Métriques pour les deux cibles, résidus par strate et téléchargement")
    st.markdown("<div class='carte'>", unsafe_allow_html=True)
    if bundle is None:
        st.warning("Aucun modèle entraîné. Reviens à l'étape Entraînement.")
    else:
        r = bundle["resume"]
        st.caption("💡 En clair : le RMSE et la déviance mesurent l'écart entre prédictions et réalité (plus petit "
                  "= mieux). Le Gini mesure la capacité à bien classer les risques du plus faible au plus élevé. "
                  "Toutes ces métriques viennent de prédictions hors-pli (jamais vues à l'entraînement).")

        st.markdown("**Nombre de décès (DthCnt)**")
        m1, m2, m3 = st.columns(3)
        m1.markdown(f"<div class='metrique-box'><div class='val'>{r['rmse_cnt']:.2f}</div><div class='lab'>RMSE</div></div>", unsafe_allow_html=True)
        m2.markdown(f"<div class='metrique-box'><div class='val'>{r['deviance_cnt']:.2f}</div><div class='lab'>déviance Poisson</div></div>", unsafe_allow_html=True)
        gini_cnt = b.gini_normalise(r["y_cnt"], r["p_cnt"])
        m3.markdown(f"<div class='metrique-box'><div class='val'>{gini_cnt:.2f}</div><div class='lab'>Gini</div></div>", unsafe_allow_html=True)

        st.markdown("**Montant des décès (DthAmt)**")
        m1, m2, m3 = st.columns(3)
        m1.markdown(f"<div class='metrique-box'><div class='val'>{r['rmse_amt']:,.2f}</div><div class='lab'>RMSE</div></div>".replace(",", " "), unsafe_allow_html=True)
        m2.markdown(f"<div class='metrique-box'><div class='val'>{r['deviance_amt']:,.2f}</div><div class='lab'>déviance Poisson</div></div>".replace(",", " "), unsafe_allow_html=True)
        gini_amt = b.gini_normalise(r["y_amt"], r["p_amt"])
        m3.markdown(f"<div class='metrique-box'><div class='val'>{gini_amt:.2f}</div><div class='lab'>Gini</div></div>", unsafe_allow_html=True)

        st.markdown("**Résidus par strate** (écart entre réel et prédit, pour les deux cibles)")
        if st.session_state.strate and not st.session_state.sans_strate:
            d = b.preparer(st.session_state.df, "melange")
            res_strate = b.residus_par_strate(d, r["y_cnt"], r["p_cnt"], r["y_amt"], r["p_amt"], st.session_state.strate, st.session_state.decoupage)
            st.dataframe(res_strate.style.format({"Reel_Cnt": "{:,.2f}", "Predit_Cnt": "{:,.2f}", "Ecart_Cnt": "{:,.2f}",
                                                  "Reel_Amt": "{:,.2f}", "Predit_Amt": "{:,.2f}", "Ecart_Amt": "{:,.2f}", "Ecart_Amt_pct": "{:.2%}"}),
                        use_container_width=True, height=300)
        else:
            st.caption("Aucune strate choisie à l'étape 2 : pas de tableau de résidus par strate.")

        st.write(f"**Colonnes du modèle :** {bundle['cols']}")
        st.write(f"**Facteur comme point de départ (partagé) :** {'oui — strate ' + str(bundle['strate']) if bundle['avec_facteur'] else 'non'}")
        st.write(f"**Modèles conservés :** {len(bundle['modeles_cnt'])} pour le nombre, {len(bundle['modeles_amt'])} pour le montant (moyenne à la prédiction)")
    st.markdown("</div>", unsafe_allow_html=True)

    if bundle is not None:
        st.markdown("<div class='carte'><h3>📅 Tendance temporelle</h3>", unsafe_allow_html=True)
        st.caption(b.TAUX_TENDANCE_SOURCE + " Appliquée aux deux cibles.")
        exemples = pd.DataFrame({"Écart (années)": [1, 3, 5, 10]})
        exemples["Facteur appliqué"] = [(1 - b.TAUX_TENDANCE) ** n for n in exemples["Écart (années)"]]
        st.dataframe(exemples.style.format({"Facteur appliqué": "{:.4f}"}), use_container_width=True, hide_index=True)
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("<div class='carte'><h3>💾 Modèle et prédiction</h3>", unsafe_allow_html=True)
        st.download_button("💾 Télécharger le modèle entraîné (.pkl)", data=b.bundle_to_bytes(bundle),
                           file_name="modele_deces.pkl", mime="application/octet-stream", use_container_width=True)
        st.markdown("<hr class='separateur'>", unsafe_allow_html=True)
        st.markdown("**Prédire sur un fichier `resultats` (mêmes colonnes que df, sans DthCnt ni DthAmt)**")
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
                    sub = b.predire(bundle, res)
                    st.dataframe(sub.head(30), use_container_width=True)
                    st.download_button("💾 Télécharger la soumission (CSV, DthCnt + DthAmt)", data=sub.to_csv(index=False).encode("utf-8"),
                                       file_name="soumission.csv", mime="text/csv", use_container_width=True)
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

