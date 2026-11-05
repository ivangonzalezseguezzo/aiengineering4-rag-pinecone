---
titulo: Arquitectura de la plataforma de cobros
categoria: arquitectura
etiquetas: [arquitectura, kubernetes, kafka, microservicios, multi-region]
---

# Arquitectura de la plataforma de cobros

Este documento describe la arquitectura de referencia de la plataforma de cobros de Órbita Pagos: qué servicios la componen, cómo se comunican y cómo se distribuyen entre regiones.

## Vista general

La plataforma corre en Kubernetes administrado (Amazon EKS). La región primaria es us-east-1 y la región de contingencia es sa-east-1 (São Paulo). El tráfico externo entra por un balanceador de carga global, pasa por el API gateway y llega a los servicios de dominio.

Los servicios de dominio principales son:

| Servicio | Lenguaje | Tier | Responsabilidad |
|---|---|---|---|
| api-gateway | Go | 1 | Autenticación, rate limiting y ruteo de las APIs públicas. |
| pagos-api | Go | 1 | Orquesta el ciclo de vida de un cobro, desde la creación hasta la liquidación. |
| autorizaciones | Java | 1 | Se comunica con los adquirentes y las redes de tarjetas para autorizar cada cobro. |
| antifraude | Python | 1 | Calcula el puntaje de riesgo de cada transacción con un modelo de machine learning. |
| billetera | Go | 1 | Administra los saldos de las cuentas virtuales de los usuarios. |
| conciliacion | Python | 2 | Cruza los archivos de liquidación de los adquirentes contra los cobros registrados. |
| notificaciones | Node.js | 3 | Envía webhooks a comercios y correos a usuarios. |

## Comunicación entre servicios

La comunicación sincrónica interna usa gRPC sobre mTLS. Se reserva para operaciones donde quien llama necesita la respuesta para continuar, por ejemplo la consulta de riesgo durante una autorización.

Todo lo demás se resuelve de forma asincrónica con Apache Kafka (Amazon MSK). Los tópicos principales son `pagos.creados`, `pagos.autorizados` y `pagos.liquidados`, con una retención de 7 días. Los servicios publican eventos con el patrón outbox: el evento se guarda en la misma transacción de base de datos que el cambio de estado y un proceso aparte lo publica en Kafka. Así nunca queda un cobro registrado sin su evento, ni un evento sin su cobro.

Los consumidores deben ser idempotentes, porque Kafka garantiza entrega "al menos una vez" y un mismo evento puede llegar dos veces.

## El camino de un cobro

1. El comercio envía `POST /v1/payments` con su `Idempotency-Key`.
2. El api-gateway valida el token OAuth, aplica el rate limit y deriva la solicitud a pagos-api.
3. pagos-api consulta a antifraude por gRPC. Si antifraude no responde en 300 milisegundos, pagos-api aplica las reglas estáticas de respaldo y marca la transacción para revisión posterior.
4. pagos-api pide la autorización al servicio autorizaciones, que elige el adquirente según el tipo de tarjeta y la disponibilidad de cada uno.
5. Con la respuesta del adquirente, pagos-api registra el resultado y publica `pagos.autorizados`.
6. notificaciones consume el evento y envía el webhook al comercio.

## Datos y estado

Cada servicio es dueño de su propia base de datos PostgreSQL 16 (Amazon Aurora) y ningún otro servicio accede a ella directamente: la integración se hace por API o por eventos. Redis se usa para las claves de idempotencia, los contadores de rate limiting y la caché de configuración de comercios. Los reportes y los archivos de liquidación se guardan en Amazon S3.

## Alta disponibilidad y contingencia

Dentro de la región primaria, cada servicio tier 1 corre con al menos 3 réplicas repartidas en 3 zonas de disponibilidad. Entre regiones, el esquema es activo-pasivo: sa-east-1 mantiene réplicas de las bases de datos y una copia reducida de los servicios lista para escalar.

El objetivo de recuperación ante la pérdida total de la región primaria es de 30 minutos (RTO), con una pérdida máxima de datos de 5 minutos (RPO). La conmutación entre regiones se ensaya dos veces por año, en un simulacro planificado con el área de Confiabilidad.

## Decisiones de arquitectura

Las decisiones relevantes se registran como ADR (Architecture Decision Records) en el repositorio `orbita/adr`. Cada ADR describe el contexto, las alternativas evaluadas, la decisión y sus consecuencias. Un cambio que contradice un ADR vigente requiere un ADR nuevo que lo reemplace.
