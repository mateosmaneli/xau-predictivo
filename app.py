"""XAU Predictivo 3.2 — investigación, no señales de inversión."""
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

st.set_page_config(page_title="XAU Predictivo 3.2", layout="wide")
st.title("XAU Predictivo 3.2 · Comparación técnica y macro")
st.warning("INVESTIGACIÓN / PAPER TRADING. GC=F no es XAU/USD spot ni el CFD de Revolut. Datos históricos no auditados punto-en-el-tiempo. No usar como señal de inversión.")

# No se confunden sustitutos de mercado con las series oficiales de FRED.
SOURCES = {
    "DGS10": ("Tipo nominal 10 años (%)", "^TNX", "Rentabilidad nominal 10 años Yahoo (^TNX); escala a verificar"),
    "DFII10": ("Tipo real 10 años (%)", None, "Sin sustituto equivalente gratuito verificado"),
    "DTWEXBGS": ("Dólar amplio FRED", "DX-Y.NYB", "Índice DXY Yahoo; NO equivalente a DTWEXBGS"),
    "VIXCLS": ("VIX", "^VIX", "VIX Yahoo, proveedor alternativo"),
}

def normalize_close(raw):
    if raw is None or raw.empty: return pd.Series(dtype=float)
    v = raw["Close"]
    if isinstance(v, pd.DataFrame): v = v.iloc[:, 0]
    v = pd.to_numeric(v, errors="coerce")
    idx = pd.to_datetime(v.index)
    if idx.tz is not None: idx = idx.tz_convert("UTC").tz_localize(None)
    v.index = idx.normalize()
    return v[~v.index.duplicated(keep="last")].sort_index().dropna()

@st.cache_data(ttl=3600, show_spinner=False)
def yahoo(ticker):
    raw = yf.download(ticker, period="10y", interval="1d", auto_adjust=False, progress=False, threads=False, timeout=12)
    return normalize_close(raw)

@st.cache_data(ttl=21600, show_spinner=False)
def fred(code):
    url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={code}"
    r = requests.get(url, timeout=(4, 9), headers={"User-Agent": "Mozilla/5.0 XAUResearch/3.2"})
    r.raise_for_status()
    data = pd.read_csv(io.StringIO(r.text))
    dc = "observation_date" if "observation_date" in data else ("DATE" if "DATE" in data else data.columns[0])
    if code not in data: raise ValueError("CSV sin columna " + code)
    dates = pd.to_datetime(data[dc], errors="coerce")
    s = pd.Series(pd.to_numeric(data[code], errors="coerce").to_numpy(), index=dates, name=code)
    return s[~s.index.isna()].sort_index().dropna()

@st.cache_data(ttl=3600, show_spinner=False)
def fetch_all():
    gold = yahoo("GC=F")
    if len(gold) < 700: raise RuntimeError("Histórico GC=F insuficiente")
    if (gold <= 0).any(): raise ValueError("Precio no positivo")
    frame = gold.rename("gold").to_frame()
    status = []
    for code, (description, fallback, note) in SOURCES.items():
        series = pd.Series(dtype=float)
        provider = "FRED"
        error = ""
        try:
            series = fred(code)
            if len(series) < 400: raise ValueError("Menos de 400 observaciones")
        except Exception as e:
            error = f"{type(e).__name__}: {str(e)[:85]}"
            if fallback:
                try:
                    series = yahoo(fallback)
                    if len(series) < 400: raise ValueError("Histórico alternativo insuficiente")
                    provider = f"Yahoo ({fallback})"
                except Exception as e2:
                    error += f"; Yahoo: {type(e2).__name__}"
                    series = pd.Series(dtype=float)
            else:
                series = pd.Series(dtype=float)
        # ^TNX suele venir expresado en porcentaje; no se equipara sin validación de escala.
        col = f"macro_{code}"
        frame[col] = series.reindex(frame.index)
        status.append({"Indicador": code, "Proveedor utilizado": provider if not series.empty else "No disponible", "Última fecha": str(series.index[-1].date()) if not series.empty else "—", "Observaciones": int(series.notna().sum()), "Nota": note if provider != "FRED" else "Serie oficial FRED", "Error FRED": error})
    return frame, pd.DataFrame(status)

def features(data, use_macro):
    f = pd.DataFrame(index=data.index)
    g = data.gold
    r = np.log(g/g.shift(1))
    for n in (1,3,5,10,20): f[f"oro_ret_{n}"] = np.log(g/g.shift(n))
    for n in (5,10,20,60): f[f"oro_media_{n}"] = g/g.rolling(n).mean()-1
    for n in (5,10,20): f[f"oro_vol_{n}"] = r.rolling(n).std()
    if use_macro:
        for code in use_macro:
            # Retraso de 2 sesiones, conservador pero NO prueba point-in-time.
            s = data[f"macro_{code}"].ffill().shift(2)
            f[f"{code}_nivel"] = s
            f[f"{code}_cambio5"] = s-s.shift(5)
    return f.replace([np.inf,-np.inf],np.nan)

def algorithms():
    return {
        "Ridge": make_pipeline(StandardScaler(), Ridge(alpha=10)),
        "Random Forest": RandomForestRegressor(n_estimators=130, max_depth=5, min_samples_leaf=12, random_state=42, n_jobs=2),
        "XGBoost": XGBRegressor(n_estimators=170, max_depth=3, learning_rate=.035, subsample=.8, colsample_bytree=.8, reg_lambda=12, objective="reg:squarederror", random_state=42, n_jobs=2),
    }

def evaluate(table, h):
    n=len(table); start=max(450,int(n*.55)); step=(n-start)//4
    if step < 45: raise ValueError("Histórico insuficiente para validación")
    cols=[c for c in table if c!="target"]
    pred=[]
    for k in range(4):
        a=start+k*step; b=start+(k+1)*step if k<3 else n
        train=table.iloc[:a-h]; test=table.iloc[a:b]
        if len(train)<300 or test.empty: continue
        for name, model in algorithms().items():
            m=clone(model).fit(train[cols],train.target)
            values=m.predict(test[cols])
            pred.extend({"fecha":d,"fold":k+1,"modelo":name,"real":float(y),"predicho":float(z)} for d,y,z in zip(test.index,test.target,values))
        pred.extend({"fecha":d,"fold":k+1,"modelo":"Sin cambio","real":float(y),"predicho":0.0} for d,y in zip(test.index,test.target))
    if not pred: raise ValueError("No se pudieron construir ventanas")
    p=pd.DataFrame(pred)
    metrics=[]
    for name, x in p.groupby("modelo"):
        a=x.real.to_numpy(); z=x.predicho.to_numpy()
        active=np.abs(z)>1e-12
        metrics.append({"Modelo":name,"MAE (%)":100*mean_absolute_error(a,z),"RMSE (%)":100*np.sqrt(mean_squared_error(a,z)),"Acierto direccional (%)":float(100*np.mean(np.sign(a[active])==np.sign(z[active]))) if active.any() else np.nan,"Observaciones":len(x)})
    return pd.DataFrame(metrics).sort_values("MAE (%)"),p

with st.sidebar:
    st.header("Configuración")
    horizon=st.selectbox("Horizonte (sesiones)",[1,3,5],index=2)
    st.caption("La descarga puede tardar; el entrenamiento comienza al pulsar el botón.")

try:
    with st.spinner("Descargando datos con límites de espera…"):
        data,status=fetch_all()
except Exception as e:
    st.error(f"No se ha podido descargar el oro: {type(e).__name__}: {e}")
    st.stop()

last=data.index[-1]
age=(pd.Timestamp.now(tz="UTC").tz_localize(None).normalize()-last).days
c1,c2,c3=st.columns(3)
c1.metric("Último dato GC=F",f"{data.gold.iloc[-1]:,.2f} USD")
c2.metric("Fecha",str(last.date()))
c3.metric("Antigüedad (días)",age)
st.caption("La fecha más reciente puede corresponder a una sesión aún abierta; no está verificada como cierre consolidado.")
st.subheader("1. Auditoría de fuentes")
st.dataframe(status,hide_index=True,use_container_width=True)
st.line_chart(data.gold.tail(250))

# Solo se habilitan indicadores con cobertura adecuada, sin confundirlos con equivalentes exactos.
usable=[]
for code in SOURCES:
    col=f"macro_{code}"
    available=data[col].ffill().shift(2)
    if available.notna().sum()>=700 and available.iloc[-1:].notna().all(): usable.append(code)

st.subheader("2. Modalidad y variables")
choice=st.radio("Modelo que deseas evaluar",["Técnico (solo oro)","Técnico + indicadores externos disponibles"],horizontal=True)
selected=usable if choice.startswith("Técnico +") else []
if choice.startswith("Técnico +") and not selected:
    st.warning("No hay indicadores externos utilizables. Se mantiene el modelo técnico.")
st.write("Indicadores externos incorporados:", ", ".join(selected) if selected else "Ninguno")
if selected:
    st.warning("Algunas series son sustitutos Yahoo, NO equivalentes a las series FRED. Retrasar dos sesiones no reconstruye las fechas reales de publicación.")

X=features(data,selected)
y=np.log(data.gold.shift(-horizon)/data.gold).rename("target")
table=X.join(y).dropna()
latest=X.dropna()
if len(table)<650 or latest.empty:
    st.error(f"No hay suficiente histórico completo ({len(table)} observaciones). Prueba el modo técnico.")
    st.stop()

st.subheader("3. Validación temporal walk-forward")
st.write(f"Observaciones: **{len(table):,}** · Variables: **{len(X.columns)}** · Horizonte: **{horizon}** sesiones.")
st.caption("Comparación por ventanas expansivas, con exclusión de h sesiones. La elección del modo o de indicadores después de observar métricas puede introducir sesgo de selección.")
key=f"{horizon}-{'-'.join(selected)}-{str(last.date())}"
if st.button("Calcular modelos y evaluación",type="primary"):
    try:
        with st.spinner("Evaluando modelos…"):
            metrics,history=evaluate(table,horizon)
            forecast=[]
            for name,model in algorithms().items():
                fitted=clone(model).fit(table[X.columns],table.target)
                value=float(fitted.predict(latest.iloc[[-1]][X.columns])[0])
                forecast.append({"Modelo":name,"Retorno previsto (%)":100*value,"Precio proyectado GC=F (USD)":float(data.gold.loc[latest.index[-1]]*np.exp(value))})
            st.session_state["result"]={"key":key,"metrics":metrics,"history":history,"forecast":pd.DataFrame(forecast)}
    except Exception as e:
        st.error(f"Error al entrenar: {type(e).__name__}: {e}")

out=st.session_state.get("result")
if out and out["key"]==key:
    m=out["metrics"]
    st.dataframe(m.style.format({"MAE (%)":"{:.3f}","RMSE (%)":"{:.3f}","Acierto direccional (%)":"{:.1f}"},na_rep="—"),hide_index=True,use_container_width=True)
    baseline=float(m.loc[m.Modelo=="Sin cambio","MAE (%)"].iloc[0])
    eligible=m[(m.Modelo!="Sin cambio")&(m["MAE (%)"]<baseline)]
    if eligible.empty: st.warning("Ningún modelo mejora el MAE del modelo sin cambio.")
    else: st.info(f"Menor MAE que la referencia: {eligible.iloc[0]['Modelo']}. No demuestra significación estadística ni rentabilidad.")
    st.subheader("4. Proyecciones experimentales")
    if age>4:
        st.error("Precio antiguo: se bloquean las proyecciones actuales.")
    else:
        st.dataframe(out["forecast"].style.format({"Retorno previsto (%)":"{:+.3f}","Precio proyectado GC=F (USD)":"{:,.2f}"}),hide_index=True,use_container_width=True)
    st.download_button("Descargar métricas",m.to_csv(index=False).encode(),"metricas_3_2.csv","text/csv")
    st.download_button("Descargar backtest",out["history"].to_csv(index=False).encode(),"backtest_3_2.csv","text/csv")
else:
    st.info("Pulsa «Calcular modelos y evaluación» para obtener resultados.")

st.subheader("5. Limitaciones")
st.markdown("""- **No es XAU/USD spot:** GC=F puede diferir del CFD Revolut.
- **Macroeconomía:** los indicadores alternativos no sustituyen exactamente a los oficiales; falta auditar fechas reales de publicación.
- **Validación:** sin pruebas de significación ni backtest de rentabilidad neta de costes CFD.
- **Persistencia:** las predicciones no se guardan automáticamente cada día.
- **No ejecutar operaciones** basadas únicamente en estas estimaciones.""")
