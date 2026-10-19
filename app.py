import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf
import plotly.graph_objects as go
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

st.set_page_config(page_title='XAU/USD Predictivo', layout='wide')
st.title('XAU/USD · Laboratorio predictivo')
st.caption('Investigación cuantitativa educativa · No son señales de inversión ni cotizaciones ejecutables de Revolut.')

@st.cache_data(ttl=3600)
def get_data(ticker):
    raw = yf.download(ticker, period='10y', interval='1d', auto_adjust=True, progress=False)
    if raw.empty:
        raise ValueError('No se han recibido cotizaciones.')
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)
    raw = raw[['Close']].dropna().copy()
    raw.index = pd.to_datetime(raw.index).tz_localize(None) if raw.index.tz is not None else pd.to_datetime(raw.index)
    return raw

def build_features(data):
    df = data.copy()
    p = df['Close']
    r = np.log(p / p.shift(1))
    for n in (1, 2, 3, 5, 10, 20):
        df[f'ret_{n}'] = np.log(p / p.shift(n))
    for n in (5, 10, 20, 60):
        df[f'ma_ratio_{n}'] = p / p.rolling(n).mean() - 1
    for n in (5, 10, 20):
        df[f'vol_{n}'] = r.rolling(n).std()
    delta = p.diff()
    up = delta.clip(lower=0).ewm(alpha=1/14, adjust=False).mean()
    down = (-delta.clip(upper=0)).ewm(alpha=1/14, adjust=False).mean()
    df['rsi14'] = 100 - 100 / (1 + up / down.replace(0, np.nan))
    return df.replace([np.inf, -np.inf], np.nan)

def walk_forward(df, features, horizon, min_train=500, step=20):
    # At prediction time t, labels require Close[t+h]. Training labels must be known by t.
    targets = np.log(df['Close'].shift(-horizon) / df['Close'])
    valid_x = df[features].notna().all(axis=1)
    eligible = np.where(valid_x.to_numpy())[0]
    preds, actuals, dates, baselines = [], [], [], []
    for j in range(min_train + horizon, len(eligible) - horizon, step):
        test_pos = eligible[j]
        train_pos = eligible[:j]
        train_pos = train_pos[train_pos + horizon <= test_pos]
        train_pos = train_pos[targets.iloc[train_pos].notna().to_numpy()]
        if len(train_pos) < min_train:
            continue
        model = make_pipeline(SimpleImputer(strategy='median'), StandardScaler(), Ridge(alpha=20.0))
        model.fit(df[features].iloc[train_pos], targets.iloc[train_pos])
        preds.append(float(model.predict(df[features].iloc[[test_pos]])[0]))
        actuals.append(float(targets.iloc[test_pos]))
        dates.append(df.index[test_pos])
        baselines.append(0.0)
    return pd.DataFrame({'pred': preds, 'real': actuals, 'base': baselines}, index=dates)

st.sidebar.header('Configuración')
source = st.sidebar.selectbox('Fuente de datos (proxy)', ['GC=F (futuros oro)', 'GLD (ETF oro)'])
ticker = 'GC=F' if source.startswith('GC=F') else 'GLD'
horizon = st.sidebar.selectbox('Horizonte (sesiones)', [1, 3, 5], index=2)
threshold = st.sidebar.slider('Umbral de señal (% estimado)', 0.0, 2.0, 0.5, 0.1) / 100
st.sidebar.caption('Los datos de Yahoo Finance son indicativos y pueden estar retrasados. GC=F y GLD NO son el CFD XAU/USD de Revolut.')

try:
    prices = get_data(ticker)
    features_df = build_features(prices)
    features = [c for c in features_df.columns if c != 'Close']
    clean = features_df.dropna(subset=features)
    if len(clean) < 750:
        st.error('No hay suficientes observaciones para validar el modelo.')
        st.stop()
    # Latest prediction uses only current and prior closes, assuming daily bar has finished.
    last = clean.iloc[[-1]]
    targets = np.log(clean['Close'].shift(-horizon) / clean['Close'])
    known = np.arange(0, len(clean) - horizon)
    model = make_pipeline(SimpleImputer(strategy='median'), StandardScaler(), Ridge(alpha=20.0))
    model.fit(clean[features].iloc[known], targets.iloc[known])
    estimate = float(model.predict(last[features])[0])
    estimate_pct = 100 * np.expm1(estimate)
    latest_price = float(last['Close'].iloc[0])
    target_price = latest_price * np.exp(estimate)
    direction = 'Alcista' if estimate > threshold else ('Bajista' if estimate < -threshold else 'Neutral')
    a, b, c = st.columns(3)
    a.metric('Último cierre proxy', f'{latest_price:,.2f} USD')
    b.metric(f'Variación prevista ({horizon} sesiones)', f'{estimate_pct:+.2f}%')
    c.metric('Precio proyectado proxy', f'{target_price:,.2f} USD')
    st.info(f'Señal de investigación: **{direction}** · Fecha último dato: {last.index[-1].date()} · Instrumento: {ticker}')
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=prices.index[-180:], y=prices['Close'].iloc[-180:], name='Cierre observado'))
    fig.update_layout(title='Últimos 180 cierres disponibles', xaxis_title='Fecha', yaxis_title='USD', height=380)
    st.plotly_chart(fig, use_container_width=True)
    st.subheader('Validación walk-forward (Ridge)')
    with st.spinner('Evaluando predicciones históricas...'):
        evaluation = walk_forward(clean, features, horizon)
    if len(evaluation) < 10:
        st.warning('Muestra de validación insuficiente.')
    else:
        mae = mean_absolute_error(evaluation['real'], evaluation['pred']) * 100
        base_mae = mean_absolute_error(evaluation['real'], evaluation['base']) * 100
        accuracy = (np.sign(evaluation['pred']) == np.sign(evaluation['real'])).mean() * 100
        v1, v2, v3 = st.columns(3)
        v1.metric('MAE modelo (puntos log-%)', f'{mae:.2f}')
        v2.metric('MAE referencia sin cambio', f'{base_mae:.2f}')
        v3.metric('Acierto direccional', f'{accuracy:.1f}%')
        st.caption(f'{len(evaluation)} predicciones fuera de muestra, separadas por 20 sesiones. Horizontes solapados pueden afectar a la independencia estadística.')
        st.line_chart(evaluation[['pred', 'real']] * 100)
        export = evaluation.copy()
        export.index.name = 'fecha_prediccion'
        st.download_button('Descargar validación CSV', export.to_csv().encode('utf-8'), file_name=f'validacion_{ticker.replace("=", "")}_{horizon}d.csv', mime='text/csv')
    st.warning('No se incluyen spreads, swaps, comisiones, ejecución ni apalancamiento de Revolut. No operar con estas señales sin validar costes y riesgo.')
    with st.expander('Limitaciones metodológicas'):
        st.write('Los futuros GC=F y el ETF GLD son aproximaciones, no cotizaciones XAU/USD CFD. El modelo usa exclusivamente precio histórico, sin variables macroeconómicas. La actualización depende del proveedor de datos y del cierre de la vela diaria. Las proyecciones no son probabilidades calibradas. El backtest emplea un único modelo Ridge y no constituye una simulación completa de operaciones.')
except Exception as e:
    st.error(f'No fue posible obtener o procesar datos: {e}')
    st.info('Prueba la otra fuente en el menú lateral o inténtalo más tarde.')
