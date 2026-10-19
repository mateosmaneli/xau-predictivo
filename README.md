# XAU/USD Predictivo — prototipo 1.0

Aplicación educativa de previsión a 1, 3 y 5 sesiones usando precios diarios de Yahoo Finance como **proxy** (futuros GC=F o ETF GLD). **No es el CFD XAU/USD de Revolut**.

## Publicar gratis
1. Crea un repositorio público nuevo en GitHub.
2. Sube `app.py` y `requirements.txt` en la raíz.
3. En https://share.streamlit.io/ inicia sesión con GitHub, selecciona repositorio y rama, e indica `app.py` como archivo principal.
4. Pulsa Deploy y abre la URL que se genere.

## Ejecutar localmente
`pip install -r requirements.txt`
`streamlit run app.py`

## Alcance
- Variables técnicas calculadas con precios disponibles al cierre.
- Ridge regularizado y evaluación walk-forward con separación temporal de etiquetas.
- Predicción porcentual y comparación de MAE con predicción sin cambio.
- Exportación CSV de predicciones históricas evaluadas.

## Pendiente para versión 2
Fuente spot XAU/USD licenciada y fiable, cotejo de cotizaciones Revolut, persistencia diaria de pronósticos, costes CFD, bandas de incertidumbre calibradas, benchmarks adicionales y paper trading.

**Advertencia:** El sistema no emite señales de inversión validadas. El apalancamiento puede causar pérdidas rápidas.
