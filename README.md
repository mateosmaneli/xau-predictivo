# XAU Predictivo 3.1 — Corrección de disponibilidad

Corrige la detención completa del panel cuando fallan las cuatro fuentes macroeconómicas FRED. En ausencia de datos macroeconómicos suficientes se habilita el modo técnico (solo GC=F). Si FRED funciona, se puede seleccionar modo técnico o multivariable.

## Instalación en Streamlit
Sustituir `app.py`, `requirements.txt` y `README.md` en la raíz del repositorio de GitHub. Streamlit reconstruirá la aplicación automáticamente. Seleccionar horizonte y pulsar «Calcular modelos y evaluación».

## Limitaciones
Los datos del oro son futuros GC=F, no XAU/USD spot ni precios ejecutables de CFD Revolut. El modelo no incluye costes ni verificación punto-en-el-tiempo de los datos macroeconómicos. La modalidad técnica no usa variables macroeconómicas. La predicción no está validada para operaciones reales.
