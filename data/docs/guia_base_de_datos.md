---
titulo: Guía de operación de bases de datos
categoria: datos
etiquetas: [postgresql, aurora, backups, replicas, pgbouncer, migraciones]
---

# Guía de operación de bases de datos

Esta guía reúne las prácticas para operar las bases de datos PostgreSQL de Órbita Pagos. La administra el equipo de Plataforma de Datos, pero los equipos dueños de cada servicio son responsables de su esquema, sus consultas y sus migraciones.

## Motor y topología

Todas las bases transaccionales usan PostgreSQL 16 sobre Amazon Aurora. Cada servicio tiene su propio clúster con una instancia primaria (escrituras) y al menos dos réplicas de lectura en zonas de disponibilidad distintas. Ningún servicio comparte base de datos con otro.

## Backups y recuperación

- La recuperación a un punto en el tiempo (PITR) está habilitada con una ventana de 35 días.
- Se toman snapshots diarios que se conservan 30 días.
- Se toman snapshots mensuales que se conservan 12 meses y se copian a la región de contingencia.

Para la base de transacciones de pagos-api, el objetivo es perder como máximo 5 minutos de datos (RPO) y recuperar el servicio en menos de 30 minutos (RTO). Cada trimestre se ensaya una restauración completa en un entorno aislado y se mide el tiempo real de recuperación.

## Réplicas de lectura

Las consultas que toleran datos apenas desactualizados (reportes, listados, búsquedas) van a las réplicas. El atraso (lag) máximo tolerado es de 30 segundos. Si una réplica supera ese atraso, la librería de acceso a datos registra el error `DB-1105`, la saca del pool de lectura y deriva las consultas críticas al primario hasta que se recupere.

Las lecturas que tienen que reflejar una escritura recién hecha (por ejemplo, consultar un cobro inmediatamente después de crearlo) siempre van al primario.

## Pool de conexiones

Los servicios se conectan a través de PgBouncer en modo transacción. Cada pod abre como máximo 20 conexiones al pool. Si un servicio agota el pool, las solicitudes esperan una conexión hasta 5 segundos y después fallan con el error `DB-2001`.

Agotar el pool casi nunca se resuelve agrandándolo: la causa suele ser una consulta lenta o una transacción que queda abierta mientras se llama a otro servicio. Una transacción nunca debe incluir llamadas de red.

## Límites de tiempo

- `statement_timeout` de 5 segundos para las consultas de las aplicaciones.
- `lock_timeout` de 3 segundos para las migraciones, para que una migración no bloquee la tabla indefinidamente.
- `idle_in_transaction_session_timeout` de 60 segundos, para cortar transacciones olvidadas.

Las consultas de más de 500 milisegundos se registran como lentas y el equipo dueño las revisa cada semana.

## Migraciones de esquema

Las migraciones se versionan en el repositorio del servicio y las aplica el pipeline de despliegue, nunca una persona a mano. Siguen el patrón expand/contract descrito en la política de despliegues. Además:

- Los índices nuevos se crean con `CREATE INDEX CONCURRENTLY` para no bloquear escrituras.
- Agregar una columna con valor por defecto no volátil es seguro; agregar una restricción `NOT NULL` a una tabla grande se hace en dos pasos, con una restricción `CHECK ... NOT VALID` primero.
- Toda migración tiene su script de reversión probado.

## Retención de datos

Los registros de transacciones se conservan 10 años por obligaciones regulatorias. Los datos personales de usuarios que cierran su cuenta se anonimizan a los 5 años del cierre, salvo los que deban conservarse por una obligación legal.
