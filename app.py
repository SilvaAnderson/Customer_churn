from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


DATA_FILES = [
    "WA_Fn-UseC_-Telco-Customer-Churn.csv",
    "WA_Fn-UseC_-Telco-Customer-Churn.xls",
]
CHURN_THRESHOLD = 0.15
FN_COST = 5.0
FP_COST = 1.0
RECALL_TARGET = 0.75
NUMERIC_FEATURES = ["tenure", "MonthlyCharges", "TotalCharges"]


def resolve_data_path() -> str:
    for file_name in DATA_FILES:
        if Path(file_name).exists():
            return file_name
    raise FileNotFoundError("Arquivo de dados nao encontrado no diretorio do projeto.")


@st.cache_data
def load_data() -> pd.DataFrame:
    # Cache de dados puros para acelerar EDA sem retrain do modelo.
    df = pd.read_csv(resolve_data_path())
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df["TotalCharges"] = df["TotalCharges"].fillna(df["TotalCharges"].median())
    df["SeniorCitizen"] = df["SeniorCitizen"].map({0: "No", 1: "Yes"}).astype("object")
    return df


@st.cache_resource
def load_and_train_model() -> dict:
    # Treina uma unica vez: pipeline + calibracao por sigmoid para probabilidade confiavel.
    df = load_data().copy()
    model_df = df.drop(columns=["customerID"]).copy()
    model_df["Churn"] = model_df["Churn"].map({"Yes": 1, "No": 0})

    X = model_df.drop(columns=["Churn"])
    y = model_df["Churn"]
    categorical_features = [c for c in X.columns if c not in NUMERIC_FEATURES]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    preprocess = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERIC_FEATURES),
            ("cat", OneHotEncoder(drop="first", handle_unknown="ignore"), categorical_features),
        ]
    )
    base_pipeline = Pipeline(
        steps=[
            ("preprocess", preprocess),
            ("model", LogisticRegression(max_iter=1000, random_state=42)),
        ]
    )

    calibrated_model = CalibratedClassifierCV(estimator=base_pipeline, method="sigmoid", cv=5)
    calibrated_model.fit(X_train, y_train)

    test_proba = calibrated_model.predict_proba(X_test)[:, 1]
    test_pred = (test_proba >= CHURN_THRESHOLD).astype(int)
    cm = confusion_matrix(y_test, test_pred)

    # Usa media dos coeficientes dos folds para leitura executiva de importancia.
    coef_stack = []
    feature_names = None
    for clf in calibrated_model.calibrated_classifiers_:
        est = clf.estimator
        fold_preprocess = est.named_steps["preprocess"]
        fold_model = est.named_steps["model"]
        if feature_names is None:
            feature_names = fold_preprocess.get_feature_names_out()
        coef_stack.append(fold_model.coef_[0])

    mean_coef = np.mean(np.vstack(coef_stack), axis=0)
    importance_df = pd.DataFrame(
        {
            "feature": feature_names,
            "coefficient": mean_coef,
            "abs_coefficient": np.abs(mean_coef),
        }
    ).sort_values("abs_coefficient", ascending=False)

    train_defaults = {}
    for col in X.columns:
        if pd.api.types.is_numeric_dtype(X[col]):
            train_defaults[col] = float(X_train[col].median())
        else:
            train_defaults[col] = X_train[col].mode().iloc[0]

    cat_options = {col: sorted(X[col].dropna().unique().tolist()) for col in categorical_features}
    num_limits = {
        col: {
            "min": float(X[col].min()),
            "max": float(X[col].max()),
            "step": 1.0 if col == "tenure" else 0.1,
        }
        for col in NUMERIC_FEATURES
    }

    return {
        "model": calibrated_model,
        "X_test": X_test,
        "y_test": y_test,
        "test_proba": test_proba,
        "test_pred": test_pred,
        "confusion_matrix": cm,
        "importance_df": importance_df,
        "defaults": train_defaults,
        "cat_options": cat_options,
        "num_limits": num_limits,
        "feature_order": X.columns.tolist(),
    }


def clean_feature_name(name: str) -> str:
    n = name.replace("num__", "").replace("cat__", "")
    if "_" in n:
        left, right = n.split("_", 1)
        return f"{left} = {right}"
    return n


def render_eda_page(df: pd.DataFrame):
    st.header("📊 Visao Geral e Analise")

    churn_rate = (df["Churn"].eq("Yes").mean()) * 100
    total_customers = len(df)
    avg_monthly = df["MonthlyCharges"].mean()
    avg_tenure = df["tenure"].mean()

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Total de Clientes", f"{total_customers:,}".replace(",", "."))
    k2.metric("Taxa de Churn", f"{churn_rate:.2f}%")
    k3.metric("Ticket Medio Mensal", f"${avg_monthly:.2f}")
    k4.metric("Tenure Medio", f"{avg_tenure:.1f} meses")

    c1, c2 = st.columns(2)
    with c1:
        churn_dist = df["Churn"].value_counts().reset_index()
        churn_dist.columns = ["Churn", "Clientes"]
        fig_donut = px.pie(
            churn_dist,
            names="Churn",
            values="Clientes",
            hole=0.55,
            color="Churn",
            color_discrete_map={"Yes": "#d62728", "No": "#1f77b4"},
            title="Distribuicao do Churn",
        )
        st.plotly_chart(fig_donut, use_container_width=True)

    with c2:
        fig_hist = px.histogram(
            df,
            x="tenure",
            color="Churn",
            barmode="overlay",
            nbins=30,
            opacity=0.7,
            color_discrete_map={"Yes": "#d62728", "No": "#1f77b4"},
            title="Distribuicao de Tenure por Churn",
        )
        st.plotly_chart(fig_hist, use_container_width=True)

    c3, c4 = st.columns(2)
    with c3:
        fig_box = px.box(
            df,
            x="Churn",
            y="MonthlyCharges",
            color="Churn",
            color_discrete_map={"Yes": "#d62728", "No": "#1f77b4"},
            title="MonthlyCharges vs Churn",
        )
        st.plotly_chart(fig_box, use_container_width=True)

    with c4:
        st.markdown("### Drivers de Churn por Segmento")
        cols = ["Contract", "InternetService", "PaymentMethod"]
        for col in cols:
            churn_by_col = (
                df.groupby(col, as_index=False)["Churn"]
                .apply(lambda s: (s == "Yes").mean() * 100)
                .rename(columns={"Churn": "ChurnRate"})
                .sort_values("ChurnRate", ascending=False)
            )
            fig_bar = px.bar(
                churn_by_col,
                x=col,
                y="ChurnRate",
                color="ChurnRate",
                color_continuous_scale="Reds",
                title=f"Taxa de Churn por {col}",
                labels={"ChurnRate": "Taxa de Churn (%)"},
            )
            fig_bar.update_layout(coloraxis_showscale=False)
            st.plotly_chart(fig_bar, use_container_width=True)


def render_model_page(artifacts: dict):
    st.header("🤖 Desempenho do Modelo e Regras")
    st.markdown(
        """
Modelo escolhido: Regressao Logistica com calibracao de probabilidade via sigmoid.

Regra de negocio aplicada: threshold fixo em 0.15.

- Custo de falso negativo = 5x custo de falso positivo
- Alvo operacional de recall = 75%
- Decisao: probabilidade >= 0.15 indica risco de churn
"""
    )

    importance_df = artifacts["importance_df"].copy()
    top_imp = importance_df.head(20).copy()
    top_imp["feature_clean"] = top_imp["feature"].apply(clean_feature_name)
    top_imp["direction"] = np.where(top_imp["coefficient"] >= 0, "Aumenta risco", "Reduz risco")

    fig_imp = px.bar(
        top_imp.sort_values("coefficient"),
        x="coefficient",
        y="feature_clean",
        orientation="h",
        color="direction",
        color_discrete_map={"Aumenta risco": "#d62728", "Reduz risco": "#1f77b4"},
        title="Importancia das Variaveis (coeficientes)",
        labels={"coefficient": "Coeficiente", "feature_clean": "Feature"},
    )
    st.plotly_chart(fig_imp, use_container_width=True)

    st.dataframe(
        top_imp[["feature_clean", "coefficient", "abs_coefficient", "direction"]].rename(
            columns={
                "feature_clean": "Feature",
                "coefficient": "Coeficiente",
                "abs_coefficient": "|Coeficiente|",
                "direction": "Direcao",
            }
        ),
        use_container_width=True,
    )

    st.info(
        "Threshold 0.15 prioriza recall para identificar churners cedo. "
        "Mesmo com mais falsos positivos, reduzimos perda de clientes de alto valor."
    )

    cm = artifacts["confusion_matrix"]
    cm_df = pd.DataFrame(
        cm,
        index=["Real: No Churn", "Real: Churn"],
        columns=["Predito: No Churn", "Predito: Churn"],
    )
    fig_cm = px.imshow(
        cm_df,
        text_auto=True,
        color_continuous_scale="Blues",
        title="Matriz de Confusao (threshold = 0.15)",
    )
    st.plotly_chart(fig_cm, use_container_width=True)


def render_simulator_page(artifacts: dict):
    st.header("🔮 Simulador de Churn")
    st.markdown("Preencha os dados de um cliente e clique em Prever Risco.")

    model = artifacts["model"]
    defaults = artifacts["defaults"]
    cat_options = artifacts["cat_options"]
    num_limits = artifacts["num_limits"]
    feature_order = artifacts["feature_order"]

    with st.form("sim_form"):
        c1, c2, c3 = st.columns(3)
        user_input = {}

        with c1:
            user_input["gender"] = st.selectbox("Gender", cat_options["gender"], index=cat_options["gender"].index(defaults["gender"]))
            user_input["SeniorCitizen"] = st.selectbox("SeniorCitizen", cat_options["SeniorCitizen"], index=cat_options["SeniorCitizen"].index(defaults["SeniorCitizen"]))
            user_input["Partner"] = st.selectbox("Partner", cat_options["Partner"], index=cat_options["Partner"].index(defaults["Partner"]))
            user_input["Dependents"] = st.selectbox("Dependents", cat_options["Dependents"], index=cat_options["Dependents"].index(defaults["Dependents"]))
            user_input["tenure"] = st.number_input(
                "Tenure (months)",
                min_value=num_limits["tenure"]["min"],
                max_value=num_limits["tenure"]["max"],
                value=float(defaults["tenure"]),
                step=num_limits["tenure"]["step"],
            )
            user_input["PhoneService"] = st.selectbox("PhoneService", cat_options["PhoneService"], index=cat_options["PhoneService"].index(defaults["PhoneService"]))

        with c2:
            user_input["MultipleLines"] = st.selectbox("MultipleLines", cat_options["MultipleLines"], index=cat_options["MultipleLines"].index(defaults["MultipleLines"]))
            user_input["InternetService"] = st.selectbox("InternetService", cat_options["InternetService"], index=cat_options["InternetService"].index(defaults["InternetService"]))
            user_input["OnlineSecurity"] = st.selectbox("OnlineSecurity", cat_options["OnlineSecurity"], index=cat_options["OnlineSecurity"].index(defaults["OnlineSecurity"]))
            user_input["OnlineBackup"] = st.selectbox("OnlineBackup", cat_options["OnlineBackup"], index=cat_options["OnlineBackup"].index(defaults["OnlineBackup"]))
            user_input["DeviceProtection"] = st.selectbox("DeviceProtection", cat_options["DeviceProtection"], index=cat_options["DeviceProtection"].index(defaults["DeviceProtection"]))
            user_input["TechSupport"] = st.selectbox("TechSupport", cat_options["TechSupport"], index=cat_options["TechSupport"].index(defaults["TechSupport"]))

        with c3:
            user_input["StreamingTV"] = st.selectbox("StreamingTV", cat_options["StreamingTV"], index=cat_options["StreamingTV"].index(defaults["StreamingTV"]))
            user_input["StreamingMovies"] = st.selectbox("StreamingMovies", cat_options["StreamingMovies"], index=cat_options["StreamingMovies"].index(defaults["StreamingMovies"]))
            user_input["Contract"] = st.selectbox("Contract", cat_options["Contract"], index=cat_options["Contract"].index(defaults["Contract"]))
            user_input["PaperlessBilling"] = st.selectbox("PaperlessBilling", cat_options["PaperlessBilling"], index=cat_options["PaperlessBilling"].index(defaults["PaperlessBilling"]))
            user_input["PaymentMethod"] = st.selectbox("PaymentMethod", cat_options["PaymentMethod"], index=cat_options["PaymentMethod"].index(defaults["PaymentMethod"]))
            user_input["MonthlyCharges"] = st.number_input(
                "MonthlyCharges",
                min_value=num_limits["MonthlyCharges"]["min"],
                max_value=num_limits["MonthlyCharges"]["max"],
                value=float(defaults["MonthlyCharges"]),
                step=num_limits["MonthlyCharges"]["step"],
            )
            user_input["TotalCharges"] = st.number_input(
                "TotalCharges",
                min_value=num_limits["TotalCharges"]["min"],
                max_value=num_limits["TotalCharges"]["max"],
                value=float(defaults["TotalCharges"]),
                step=num_limits["TotalCharges"]["step"],
            )

        submitted = st.form_submit_button("Prever Risco")

    if not submitted:
        return

    input_df = pd.DataFrame([user_input])[feature_order]
    churn_proba = float(model.predict_proba(input_df)[:, 1][0])

    st.subheader("Probabilidade calibrada de churn")
    st.progress(min(max(churn_proba, 0.0), 1.0))
    st.caption(f"Probabilidade estimada: {churn_proba:.2%}")

    gauge = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=churn_proba * 100,
            number={"suffix": "%"},
            title={"text": "Risco de Churn"},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": "#d62728" if churn_proba >= CHURN_THRESHOLD else "#1f77b4"},
                "steps": [
                    {"range": [0, CHURN_THRESHOLD * 100], "color": "#dbeafe"},
                    {"range": [CHURN_THRESHOLD * 100, 100], "color": "#fee2e2"},
                ],
                "threshold": {"line": {"color": "black", "width": 3}, "value": CHURN_THRESHOLD * 100},
            },
        )
    )
    gauge.update_layout(height=280, margin=dict(l=30, r=30, t=40, b=20))
    st.plotly_chart(gauge, use_container_width=True)

    if churn_proba >= CHURN_THRESHOLD:
        st.error("ALTO RISCO DE CHURN: Acao de retencao recomendada!")
    else:
        st.success("Cliente Retido: Baixo Risco.")

    # Explicabilidade local com contribuicoes lineares no espaco transformado.
    fold_estimator = model.calibrated_classifiers_[0].estimator
    preprocess = fold_estimator.named_steps["preprocess"]
    transformed = preprocess.transform(input_df)
    if hasattr(transformed, "toarray"):
        transformed = transformed.toarray()
    transformed = transformed[0]

    coef_stack = np.vstack([clf.estimator.named_steps["model"].coef_[0] for clf in model.calibrated_classifiers_])
    mean_coef = coef_stack.mean(axis=0)
    feature_names = preprocess.get_feature_names_out()

    contrib_df = pd.DataFrame(
        {
            "Feature": [clean_feature_name(n) for n in feature_names],
            "Contribuicao": transformed * mean_coef,
        }
    )
    contrib_df = contrib_df[np.abs(contrib_df["Contribuicao"]) > 1e-6].copy()
    contrib_df["Impacto"] = np.where(contrib_df["Contribuicao"] > 0, "Aumenta risco", "Reduz risco")
    contrib_df = contrib_df.reindex(contrib_df["Contribuicao"].abs().sort_values(ascending=False).index).head(10)

    with st.expander("Ver features que mais influenciaram esta previsao"):
        if contrib_df.empty:
            st.write("Nenhuma contribuicao relevante encontrada para este perfil.")
        else:
            st.dataframe(contrib_df, use_container_width=True)


def main():
    st.set_page_config(page_title="Customer Churn Intelligence", layout="wide")
    st.title("Customer Churn Intelligence Dashboard")
    st.caption("Regressao Logistica Calibrada | Threshold fixo de decisao = 0.15")

    df = load_data()
    artifacts = load_and_train_model()

    with st.sidebar:
        st.header("Navegacao")
        page = st.radio(
            "Selecione a pagina",
            [
                "📊 Visao Geral e Analise",
                "🤖 Desempenho do Modelo e Regras",
                "🔮 Simulador de Churn",
            ],
        )
        st.markdown("---")
        st.markdown("### Regra de Decisao")
        st.metric("Threshold Operacional", f"{CHURN_THRESHOLD:.2f}")
        st.caption(
            f"FN:FP = {int(FN_COST)}:{int(FP_COST)} | Recall alvo: {int(RECALL_TARGET * 100)}%"
        )

    if page == "📊 Visao Geral e Analise":
        render_eda_page(df)
    elif page == "🤖 Desempenho do Modelo e Regras":
        render_model_page(artifacts)
    else:
        render_simulator_page(artifacts)


if __name__ == "__main__":
    main()
