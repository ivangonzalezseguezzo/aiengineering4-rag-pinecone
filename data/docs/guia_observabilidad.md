---
titulo: Guía de observabilidad y SLOs
categoria: observabilidad
etiquetas: [observabilidad, slo, error-budget, alertas, logs, trazas, opentelemetry]
---

# Guía de observabilidad y SLOs

Esta guía define cómo instrumentan sus servicios los equipos de Órbita Pagos y cómo se mide su confiabilidad. La regla general es que ningún servicio llega a producción sin métricas, logs estructurados y trazas distribuidas.

## Stack de observabilidad

- **Trazas:** instrumentación con OpenTelemetry y almacenamiento en Grafana Tempo.
- **Métricas:** exportadas en formato Prometheus y almacenadas en Grafana Mimir.
- **Logs:** enviados a Grafana Loki.
- **Tableros y alertas:** Grafana, con las alertas críticas ruteadas a Sirena.

Todos los servicios usan la librería interna `orbita-telemetry`, que configura los tres pilares con los nombres y etiquetas estándar. No se permite instrumentar "a mano" con otra convención.

## Las cuatro señales doradas

Cada servicio publica un tablero con las cuatro señales doradas:

1. **Latencia:** percentiles p50, p95 y p99, separando las solicitudes exitosas de las fallidas.
2. **Tráfico:** solicitudes por segundo por endpoint.
3. **Errores:** tasa de respuestas 5xx y de errores de negocio.
4. **Saturación:** uso de CPU, memoria, conexiones a la base de datos y lag de consumidores de Kafka.

## SLOs de los servicios tier 1

| Servicio | Disponibilidad mensual | Latencia |
|---|---|---|
| pagos-api | 99,95 % | p95 menor a 800 ms |
| autorizaciones | 99,9 % | p95 menor a 1.500 ms |
| antifraude | 99,9 % | p99 menor a 300 ms |
| billetera | 99,95 % | p95 menor a 500 ms |

La disponibilidad se mide como el porcentaje de solicitudes válidas que no terminan en error 5xx ni superan el timeout.

## Presupuesto de error (error budget)

El presupuesto de error es la cantidad de fallas que el SLO permite en el mes. Con un SLO de 99,95 %, el presupuesto es de unos 21 minutos de indisponibilidad mensual. La política de uso es:

- Si en la primera mitad del mes se consumió más del 50 % del presupuesto, el equipo prioriza trabajo de confiabilidad sobre funcionalidades nuevas.
- Si el presupuesto se agota, el servicio entra en congelamiento de funcionalidades: solo se despliegan correcciones y mejoras de confiabilidad hasta que el presupuesto se recupere.

## Alertas por tasa de consumo (burn rate)

Las alertas de SLO no se disparan por un umbral fijo sino por la velocidad con la que se consume el presupuesto:

| Ventana | Tasa de consumo | Acción |
|---|---|---|
| 1 hora | 14,4 veces la tasa sostenible | Página a la guardia |
| 6 horas | 6 veces la tasa sostenible | Página a la guardia |
| 3 días | 1 vez la tasa sostenible | Ticket para el equipo |

Una tasa de 14,4 durante una hora consume el 2 % del presupuesto mensual: es una falla que no puede esperar.

## Logs estructurados

Los logs se emiten en JSON, una línea por evento, con estos campos obligatorios: `timestamp`, `level`, `service`, `trace_id`, `span_id` y, cuando aplica, `merchant_id`. Los logs no contienen datos personales ni datos de tarjeta.

La retención es de 30 días en almacenamiento rápido y de 1 año en almacenamiento frío. Los logs de nivel DEBUG no se envían desde producción salvo que se active temporalmente un flag de diagnóstico, que se apaga solo a las 2 horas.

## Trazas distribuidas

El `trace_id` se propaga en todas las llamadas gRPC, en las solicitudes HTTP (header `traceparent`) y en los headers de los mensajes de Kafka. El muestreo por defecto es del 10 % de las trazas, pero toda traza que contenga un error se conserva completa (tail sampling).
