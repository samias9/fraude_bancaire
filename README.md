## 🔍 Détection de fraude sur transactions bancaires

> Système de détection des fraudes avec **74% de précision** et **63% de rappel**, développé sur le dataset IEEE‑CIS Fraud Detection (≈590K transactions).

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-1.7+-green.svg)](https://xgboost.readthedocs.io/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

![Feature Importance](visualisations/feature_importance.png)

---

### Sommaire
- **Contexte et objectif**
- **Résultats clés**
- **Données** (téléchargement et placement)
- **Prérequis & installation**
- **Reproduire l’expérience**
- **Structure du dépôt**
- **Pipeline de modélisation**
- **Méthodologie**
- **Hyperparamètres**
- **Visualisations**
- **Limites & prochaines étapes**
- **Contact & licence**

---

## 📌 Contexte et objectif

Dans le contexte bancaire, la détection précoce des fraudes est cruciale. Ce projet propose un modèle de classification qui **détecte les transactions frauduleuses** malgré un **déséquilibre de classes très marqué** (≈96.5% légitimes / 3.5% fraudes).

**Objectif principal** : maximiser le rappel (détecter un maximum de fraudes) tout en maintenant une précision acceptable (limiter les fausses alertes).

---

## 🎯 Résultats clés

| Métrique | Valeur | Impact business |
|----------|--------|-----------------|
| **Précision (fraude)** | **74%** | 7–8 vraies fraudes sur 10 alertes |
| **Rappel (fraude)** | **63%** | ≈ 2/3 des fraudes réelles détectées |
| **F1-Score** | **0.68** | Bon compromis précision/rappel |
| **Fausses alertes** | **920** | −88% vs baseline Random Forest |
| **Exactitude globale** | **98%** | Très élevée |

**Gain vs baseline** : +133% de rappel (27% → 63%).

---

## 📊 Données

Le dataset n’est pas inclus dans ce dépôt. Pour reproduire :
1. Téléchargez le dataset Kaggle `IEEE‑CIS Fraud Detection` : `https://www.kaggle.com/c/ieee-fraud-detection`
2. Placez tous les fichiers dans le dossier `data/` à la racine du projet.

---

## 🧰 Prérequis & installation

- Python 3.8+
- Environnement virtuel recommandé

```bash
python -m venv .venv
source .venv/bin/activate  # sous macOS/Linux
python -m pip install -U pip
pip install pandas numpy scikit-learn xgboost imbalanced-learn matplotlib seaborn jupyter ipykernel
```

---

## ▶️ Reproduire l’expérience

1. Préparer les données (voir section Données) dans `data/`.
2. Lancer Jupyter et ouvrir les notebooks clés :
   - `Exploration_et_nettoyage_des_données_fraudes.ipynb`
   - `Visualisation_et_modeling.ipynb`

```bash
jupyter lab  # ou: jupyter notebook
```

---

## 🗂️ Structure du dépôt

```text
Fraude_bancaire/
├─ data/
│  ├─ train_transaction.csv, train_identity.csv, ...
├─ Exploration_et_nettoyage_des_données_fraudes.ipynb
├─ Visualisation_et_modeling.ipynb
├─ df_train_encoded.pkl
├─ df_train_reduced.pkl
├─ label_mappings.json
└─ README.md
```

---

## 🛠️ Technologies utilisées

- **Python 3.8+** : langage principal
- **Pandas / NumPy** : manipulation des données
- **Scikit‑learn** : preprocessing, métriques, train/validation split
- **XGBoost** : modèle retenu (surpasse la baseline Random Forest)
- **Imbalanced‑learn (SMOTE)** : gestion du déséquilibre de classes
- **Matplotlib / Seaborn** : visualisations

---

## 📊 Pipeline de modélisation

1. Données brutes (≈590K transactions)
2. Feature engineering & preprocessing
3. Split train/validation (80/20) avec stratification
4. Rééquilibrage via **SMOTE** (ratio ≈ 0.5)
5. Entraînement **XGBoost** avec `scale_pos_weight="balanced"`
6. Optimisation du seuil de décision (0.4)
7. Évaluation finale : précision 74% / rappel 63%

---

## 🚀 Méthodologie

### 1) Exploration des données (EDA)
- Analyse de ≈590 540 transactions, >100 features
- Déséquilibre identifié : ≈3.5% de fraudes
- Corrélations : A8, A10, A7 parmi les plus prédictives

### 2) Preprocessing
- Traitement des valeurs manquantes
- Sélection de variables guidée par corrélations/importance
- Réduction dimensionnelle (≈100 features conservées)

### 3) Gestion du déséquilibre
- **SMOTE** (oversampling synthétique, ratio ≈0.5)
- **Pondération de classe** : `scale_pos_weight` (XGBoost)
- **Ajustement du seuil** : 0.4 (vs 0.5 par défaut)

### 4) Modélisation
- Baseline Random Forest : rappel 71%, précision 28%
- **XGBoost (choisi)** : rappel 63%, précision 74% ✅

---

## ⚙️ Hyperparamètres XGBoost

```python
{
    'n_estimators': 200,
    'max_depth': 8,
    'learning_rate': 0.1,
    'scale_pos_weight': 2.0,
    'eval_metric': 'aucpr'
}
```

---

## 📈 Visualisations

- Importance des variables (ex. `visualisations/feature_importance.png`)
- Courbe précision–rappel et trade‑off seuil
- Matrice de confusion (validation) :

|                | Prédit légitime | Prédit fraude |
|----------------|-----------------|---------------|
| **Réel légitime** | 113 056         | 919           |
| **Réel fraude**   | 1 529           | 2 604         |

---

## 💡 Insights techniques

**Pourquoi XGBoost > Random Forest ?**
- Meilleure optimisation par gradient boosting
- Régularisation intégrée (moins d’overfitting)
- Plus robuste sur données déséquilibrées

**Impact du seuil de décision**
- 0.5 (défaut) : rappel 45%, précision 80%
- **0.4 (optimal)** : rappel 63%, précision 74% ✅
- 0.3 : rappel 71%, précision 62%

---

🎓 Compétences Démontrées

✅ Machine Learning : Classification supervisée, gestion du déséquilibre
✅ Feature Engineering : Sélection, création, analyse de corrélations
✅ Data Preprocessing : Nettoyage, transformation, SMOTE
✅ Model Evaluation : Métriques adaptées (Precision/Recall/F1), matrices de confusion
✅ Hyperparameter Tuning : Optimisation de seuils, comparaison de modèles
✅ Python : Pandas, Scikit-learn, XGBoost, Matplotlib
✅ Business Understanding : Trade-off coûts fraudes vs fausses alertes

---

## 🤝 Crédits

- Dataset Kaggle `IEEE‑CIS Fraud Detection` (`https://www.kaggle.com/c/ieee-fraud-detection`)

---

## 📞 Contact & licence

Samia CARCHAF  •  Email : `samia.carchaf@gmail.com`

LinkedIn : `https://www.linkedin.com/in/samia-carchaf-ia/`

Portfolio : `https://carchaf-portfolio.netlify.app/`

Licence : MIT