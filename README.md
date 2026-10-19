# XAU Predictivo 3.0 — laboratorio de investigación

Panel Streamlit gratuito para estudiar predicciones de retornos del oro a 1, 3 y 5 sesiones.

## Datos
- Yahoo Finance mediante `yfinance`: `GC=F` (futuros continuos del oro, **NO** spot XAU/USD ni CFD de Revolut).
- FRED CSV: `DGS10`, `DFII10`, `DTWEXBGS`, `VIXCLS`.
- Variables de tendencia, retornos y volatilidad derivados de GC=F.
- Las variables FRED se retrasan 2 sesiones, pero **NO** es una reconstrucción point-in-time certificada. La cobertura histórica y actualizaciones pueden cambiar.

## Algoritmos
Baseline de retorno cero, Ridge, Random Forest y XGBoost. Cuatro pruebas walk-forward de ventana expansiva con exclusión de `h` observaciones anteriores al test. Métricas MAE, RMSE y acierto direccional. Se considera mejor solo si mejora el MAE del baseline; **no** se ha probado rentabilidad neta de CFD.

## Publicación
1. Subir `app.py`, `requirements.txt` y `README.md` a la **raíz** del repositorio GitHub (reemplazar los antiguos).
2. En Streamlit Community Cloud, usar el repositorio y `app.py` como archivo principal. La app ya desplegada debería actualizarse tras el commit.
3. Abrir la aplicación y pulsar «Calcular modelos y evaluación».

## Límites
El proyecto no ejecuta órdenes. Los datos pueden retrasarse, fallar o contener revisiones. No hay cálculo de costes CFD, garantías de precio ejecutable ni automatización diaria persistente. Para avanzar hacia un sistema operable hay que verificar cotizaciones spot y reconstrucción temporal de publicaciones macro, registrar predicciones inmutables y simular costes reales.
