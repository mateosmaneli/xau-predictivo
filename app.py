"""XAU Predictivo 3.1 — educational research only."""
import io
from datetime import datetime, timezone
import numpy as np
import pandas as pd
import requests
import streamlit as st
import yfinance as yf
from sklearn.base import clone
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

st.set_page_config(page_title="XAU Predictivo 3.1", layout="wide")
st.title("XAU Predictivo 3.1 · Laboratorio cuantitativo")
st.warning("INVESTIGACIÓN / PAPER TRADING. GC=F es un futuro continuo, NO XAU/USD spot ni el CFD de Revolut. Sin costes CFD ni datos punto-en-el-tiempo auditados. No utilizar como señal de inversión.")
SERIES = {"DGS10":"Bono nominal EE. UU. 10 años (%)", "DFII10":"Bono real EE. UU. 10 años (%)", "DTWEXBGS":"Dólar amplio ponderado por comercio", "VIXCLS":"VIX (volatilidad acciones)"}

@st.cache_data(ttl=3600, show_spinner=False)
def load_data():
    raw = yf.download("GC=F", period="10y", interval="1d", auto_adjust=False, progress=False, threads=False)
    if raw.empty: raise RuntimeError("Yahoo Finance no devolvió GC=F")
    close = raw["Close"]
    if isinstance(close, pd.DataFrame): close = close.iloc[:, 0]
    close = pd.to_numeric(close, errors="coerce")
    close.index = pd.to_datetime(close.index).tz_localize(None).normalize()
    df = close.rename("gold").to_frame().dropna().sort_index()
    df = df.loc[~df.index.duplicated(keep="last")]
    if (df.gold <= 0).any(): raise ValueError("Precios de oro no positivos")
    status=[]
    for code, label in SERIES.items():
        try:
            r=requests.get(f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={code}", timeout=(4, 7), headers={"User-Agent":"xau-predictivo/3.0 educational research"})
            r.raise_for_status()
            s=pd.read_csv(io.StringIO(r.text))
            if "observation_date" in s.columns: datecol="observation_date"
            elif "DATE" in s.columns: datecol="DATE"
            elif "date" in s.columns: datecol="date"
            else: datecol=s.columns[0]
            s[datecol]=pd.to_datetime(s[datecol], errors="coerce")
            s[code]=pd.to_numeric(s[code], errors="coerce")
            series=s.dropna(subset=[datecol]).set_index(datecol)[code].sort_index()
            series=series.loc[~series.index.duplicated(keep="last")]
            df=df.join(series,how="left")
            last=series.last_valid_index()
            status.append({"Fuente":"FRED", "Serie":code, "Descripción":label,"Última fecha":str(last.date()) if last is not None else "Sin datos", "Observaciones":int(series.notna().sum()), "Estado":"OK" if last is not None else "SIN DATOS"})
        except Exception as exc:
            df[code]=np.nan
            status.append({"Fuente":"FRED", "Serie":code,"Descripción":label,"Última fecha":"—", "Observaciones":0,"Estado":"ERROR: "+str(exc)[:85]})
    return df, pd.DataFrame(status)

def make_features(data, include_macro=False):
    f=pd.DataFrame(index=data.index)
    gold=data.gold
    daily=np.log(gold/gold.shift(1))
    for n in (1,3,5,10,20): f[f"oro_retorno_{n}"]=np.log(gold/gold.shift(n))
    for n in (5,10,20,60): f[f"oro_dist_media_{n}"]=gold/gold.rolling(n).mean()-1
    for n in (5,10,20): f[f"oro_vol_{n}"]=daily.rolling(n).std()
    # Conservative two-GOLD-session publication lag; not a substitute for actual historical release vintages.
    if include_macro:
        for code in SERIES:
            s=data[code].ffill().shift(2)
            f[f"{code}_nivel"]=s
            f[f"{code}_cambio_5"]=s-s.shift(5)
    return f.replace([np.inf,-np.inf],np.nan)

def models():
    return {
        "Ridge":make_pipeline(StandardScaler(),Ridge(alpha=10.0)),
        "Random Forest":RandomForestRegressor(n_estimators=130,max_depth=5,min_samples_leaf=12,random_state=42,n_jobs=2),
        "XGBoost":XGBRegressor(n_estimators=170,max_depth=3,learning_rate=.035,subsample=.8,colsample_bytree=.8,reg_lambda=12,objective="reg:squarederror",random_state=42,n_jobs=2)
    }

def backtest(table, horizon, n_folds=4):
    """Expanding window, nonoverlapping test folds, purge h rows before each test."""
    names=list(models())
    n=len(table)
    initial=max(450,int(n*.55))
    test_size=(n-initial)//n_folds
    if test_size < 45: raise ValueError("No hay observaciones suficientes para cuatro ventanas de prueba")
    rows=[]; preds=[]
    features=[c for c in table.columns if c!="target"]
    for k in range(n_folds):
        test_start=initial+k*test_size
        test_end=initial+(k+1)*test_size if k<n_folds-1 else n
        # Last training target must end strictly before first test prediction date.
        train=table.iloc[:max(0,test_start-horizon)]
        test=table.iloc[test_start:test_end]
        if len(train)<300 or test.empty: continue
        true=test.target.to_numpy()
        for name, template in models().items():
            m=clone(template)
            m.fit(train[features],train.target)
            prediction=m.predict(test[features])
            for date, actual, estimated in zip(test.index,true,prediction):
                preds.append({"fecha":date,"modelo":name,"real":float(actual),"predicho":float(estimated),"fold":k+1})
        for date, actual in zip(test.index,true):
            preds.append({"fecha":date,"modelo":"Sin cambio","real":float(actual),"predicho":0.0,"fold":k+1})
    if not preds: raise ValueError("No se pudieron generar ventanas de validación")
    p=pd.DataFrame(preds)
    for name, part in p.groupby("modelo"):
        actual=part.real.to_numpy(); estimate=part.predicho.to_numpy()
        active=np.abs(estimate)>1e-12
        hit=float(np.mean(np.sign(actual[active])==np.sign(estimate[active]))*100) if active.any() else np.nan
        rows.append({"Modelo":name,"MAE (%)":100*mean_absolute_error(actual,estimate),"RMSE (%)":100*np.sqrt(mean_squared_error(actual,estimate)),"Acierto direccional (%)":hit,"Observaciones":len(part)})
    return pd.DataFrame(rows).sort_values("MAE (%)"),p

with st.sidebar:
    st.header("Configuración")
    horizon=st.selectbox("Horizonte (sesiones)",[1,3,5],index=2)
    st.caption("Datos gratuitos. Entrenamiento solo cuando pulsas calcular; el resultado puede tardar.")

try:
    with st.spinner("Descargando cotizaciones y datos macroeconómicos…"):
        data, status=load_data()
except Exception as exc:
    st.error(f"Error en fuente de oro: {exc}"); st.stop()

last_date=data.index[-1]
age=(pd.Timestamp.now(tz="UTC").tz_localize(None).normalize()-last_date).days
c1,c2,c3=st.columns(3)
c1.metric("Último cierre GC=F",f"{data.gold.iloc[-1]:,.2f} USD")
c2.metric("Fecha último cierre",str(last_date.date()))
c3.metric("Antigüedad (días naturales)",age)
if age>4:
    st.error("DATOS DEL ORO ANTIGUOS: no se publicará predicción actual. Revisar la fuente y los festivos.")
elif age>2:
    st.warning("Último cierre con más de dos días naturales de antigüedad. Revisar festivos y horarios.")

st.subheader("1. Auditoría de fuentes")
st.dataframe(status,hide_index=True,use_container_width=True)
st.caption("FRED: fecha de observación, NO hora real de publicación. DTWEXBGS no es DXY. Las publicaciones macro se retrasan dos sesiones del oro de forma conservadora, pero esto NO garantiza disponibilidad histórica punto-en-el-tiempo.")
st.line_chart(data.gold.tail(250))
missing=[x for x in SERIES if data[x].notna().sum()<400]
if missing:
    st.warning("Datos macroeconómicos insuficientes: " + ", ".join(missing) + ". Se utilizará el modelo técnico; no se imputarán variables macroeconómicas inexistentes.")
    include_macro=False
else:
    include_macro=st.radio("Modalidad de análisis", [False, True], format_func=lambda x: "Multivariable (oro + macro)" if x else "Técnico (solo oro)", horizontal=True, index=1)
    if include_macro:
        st.caption("La modalidad macro utiliza un retraso aproximado de 2 sesiones; no garantiza reconstrucción histórica exacta de publicación.")
mode="macro" if include_macro else "tecnico"

X=make_features(data, include_macro=include_macro)
y=np.log(data.gold.shift(-horizon)/data.gold).rename("target")
table=X.join(y).dropna()
forecast_X=X.dropna()
if len(table)<650 or forecast_X.empty:
    st.error(f"Historia insuficiente: {len(table)} filas completas (mínimo 650).")
    st.stop()

st.subheader("2. Validación temporal walk-forward")
st.info("Modalidad activa: " + ("Multivariable" if include_macro else "Técnica") + ". Las estimaciones son experimentales y no constituyen señales de trading.")
st.write(f"Observaciones entrenables: **{len(table):,}** · Variables predictoras: **{len(X.columns)}** · Horizonte: **{horizon} sesiones**.")
st.caption("Cuatro pruebas cronológicas de ventana expansiva. Se excluyen h observaciones antes de cada prueba para evitar que los objetivos de entrenamiento alcancen el periodo de test. Hiperparámetros prefijados, sin optimización sobre el test.")
if st.button("Calcular modelos y evaluación",type="primary"):
    with st.spinner("Entrenando y evaluando Ridge, Random Forest y XGBoost…"):
        try:
            metrics, history=backtest(table,horizon)
            latest=forecast_X.iloc[[-1]]
            forecasts=[]
            for name,template in models().items():
                m=clone(template)
                m.fit(table[X.columns],table.target)
                val=float(m.predict(latest[X.columns])[0])
                forecasts.append({"Modelo":name,"Retorno previsto (%)":100*val,"Precio proyectado GC=F (USD)":float(data.gold.loc[latest.index[-1]]*np.exp(val))})
            forecasts=pd.DataFrame(forecasts)
            st.session_state["evaluation"]={"metrics":metrics,"history":history,"forecasts":forecasts,"horizon":horizon,"date":str(latest.index[-1].date()),"mode":mode}
        except Exception as exc:
            st.error(f"Fallo de entrenamiento: {exc}")

out=st.session_state.get("evaluation")
if out and out["horizon"]==horizon and out["date"]==str(forecast_X.index[-1].date()) and out.get("mode")==mode:
    metrics=out["metrics"]
    st.dataframe(metrics.style.format({"MAE (%)":"{:.3f}","RMSE (%)":"{:.3f}","Acierto direccional (%)":"{:.1f}"},na_rep="No aplica"),hide_index=True,use_container_width=True)
    baseline=float(metrics.loc[metrics.Modelo=="Sin cambio","MAE (%)"].iloc[0])
    eligible=metrics[(metrics.Modelo!="Sin cambio")&(metrics["MAE (%)"]<baseline)]
    if eligible.empty:
        st.warning("Ningún algoritmo supera el baseline sin cambio en MAE. No se selecciona modelo ganador.")
    else:
        winner=eligible.sort_values("MAE (%)").iloc[0]
        st.success(f"Menor MAE entre los modelos que superan el baseline: {winner['Modelo']} ({winner['MAE (%)']:.3f}% frente a {baseline:.3f}%). Esto no demuestra rentabilidad.")
    st.subheader("3. Proyecciones experimentales")
    if age>4:
        st.error("Predicciones actuales bloqueadas por antigüedad de la cotización.")
    else:
        st.dataframe(out["forecasts"].style.format({"Retorno previsto (%)":"{:+.3f}","Precio proyectado GC=F (USD)":"{:,.2f}"}),hide_index=True,use_container_width=True)
    st.download_button("Descargar métricas CSV",metrics.to_csv(index=False).encode(),"metricas_xau_v3.csv","text/csv")
    st.download_button("Descargar predicciones históricas CSV",out["history"].to_csv(index=False).encode(),"walk_forward_xau_v3.csv","text/csv")
    st.caption("Las predicciones históricas de test se generan con entrenamiento anterior a cada bloque. La proyección actual se reentrena con todo el histórico etiquetado disponible.")
else:
    st.info("Pulsa «Calcular modelos y evaluación» para generar métricas y proyecciones.")

st.subheader("4. Limitaciones que impiden su uso operativo")
st.markdown("""- **Activo:** GC=F (futuro continuo) no es el XAU/USD spot ni el CFD de Revolut; pueden existir diferencias por vencimientos y ajustes.\n- **Publicación macro:** el retraso fijo de dos sesiones no reemplaza datos con marcas temporales y versiones históricas (ALFRED).\n- **Selección:** se comparan métricas fuera de muestra, pero todavía faltan estabilidad por régimen, calibración y costes.\n- **Costes CFD:** sin spread real, financiación overnight, deslizamiento ni apalancamiento; no existe backtest económico neto.\n- **Actualización:** Streamlit consulta datos al abrir; no registra automáticamente una predicción inmutable cada día.\n- **Señales:** deliberadamente no se generan órdenes de compra/venta ni probabilidades de éxito no calibradas.""")
