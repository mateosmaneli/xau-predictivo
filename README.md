# XAU Predictivo 3.2

Aplicación Streamlit de investigación cuantitativa sobre GC=F, que NO es XAU/USD spot ni el CFD de Revolut.

## Mejoras
- Descarga FRED con tiempos de espera y alternativas Yahoo para DGS10 (^TNX), DTWEXBGS (DX-Y.NYB, NO equivalente) y VIXCLS (^VIX). DFII10 no tiene sustituto equivalente validado y no se inventa.
- Modo técnico siempre disponible y modo con indicadores externos cuando hay cobertura suficiente.
- Ridge, Random Forest, XGBoost y referencia sin cambio, validación temporal y descarga CSV.

## Publicar
Subir `app.py`, `requirements.txt` y `README.md` a la raíz de `mateosmaneli/xau-predictivo`, reemplazando los existentes. Streamlit actualiza la app tras el commit.

## Advertencias
La conexión a Yahoo/FRED no está garantizada. No se ha validado aquí una descarga en vivo. La versión no certifica información punto-en-el-tiempo, no calcula costes CFD, no registra predicciones diarias persistentes y no produce señales de inversión operables.
