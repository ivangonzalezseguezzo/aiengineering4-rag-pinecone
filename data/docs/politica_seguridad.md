---
titulo: Política de seguridad de la información
categoria: seguridad
etiquetas: [seguridad, secretos, vault, mfa, vulnerabilidades, pci-dss]
---

# Política de seguridad de la información

Esta política establece los controles mínimos de seguridad para los sistemas de Órbita Pagos. Su cumplimiento se audita cada año como parte de la certificación PCI DSS y es obligatorio para todo el personal y los proveedores con acceso a sistemas productivos.

## Gestión de secretos

Los secretos (contraseñas, API keys, certificados y claves de cifrado) se guardan exclusivamente en HashiCorp Vault. Está prohibido guardarlos en el código, en variables de entorno de los repositorios, en tickets o en mensajes de chat.

- Los secretos estáticos de producción se rotan cada 90 días.
- Las credenciales de bases de datos son dinámicas: Vault las genera por servicio con una vigencia de 24 horas.
- Las API keys que se entregan a comercios vencen al año y pueden rotarse antes a pedido del comercio.

Todos los repositorios tienen un hook de pre-commit con gitleaks que bloquea commits con patrones de secretos.

## Qué hacer si se filtra un secreto

Si un secreto queda expuesto (por ejemplo, se subió a un repositorio o se pegó en un chat), se trata como un incidente SEV1 de seguridad:

1. Revocar el secreto de inmediato en Vault, sin esperar a confirmar si fue usado.
2. Generar uno nuevo y desplegarlo en los servicios que lo consumen.
3. Revisar los logs de acceso del período de exposición en busca de uso indebido.
4. Eliminar el secreto del historial del repositorio.
5. Notificar al equipo de Seguridad, que evalúa si corresponde una notificación regulatoria dentro de las 72 horas.

## Acceso a producción

- Nadie tiene acceso permanente a producción. El acceso se pide "justo a tiempo" a través del bastión Puente, con una justificación y la aprobación de una segunda persona del equipo.
- Cada acceso dura como máximo 4 horas y toda la sesión queda grabada.
- Para entrar se exige autenticación multifactor con llave física FIDO2. No se aceptan códigos por SMS.
- Los permisos se revisan cada trimestre: el engineering manager confirma o revoca cada acceso de su equipo.

## Principio de mínimo privilegio

Cada servicio corre con una identidad propia y solo con los permisos que necesita. Un servicio que solo lee de una cola no puede escribir en ella, y ningún servicio tiene permisos de administrador sobre la cuenta de nube.

## Gestión de vulnerabilidades

Las imágenes de contenedores y las dependencias se escanean en cada build y una vez por día en producción. Los plazos máximos de corrección según la severidad son:

| Severidad | Plazo de corrección |
|---|---|
| Crítica | 7 días |
| Alta | 30 días |
| Media | 90 días |
| Baja | Se evalúa en la planificación trimestral |

Una vulnerabilidad crítica explotable desde internet se corrige en menos de 24 horas y se gestiona como incidente.

## Datos de tarjetas

El entorno que procesa datos de tarjetas (CDE) está segmentado del resto de la red. Fuera del CDE solo circulan tokens: el número de tarjeta se tokeniza en el primer contacto y ningún otro servicio ve el PAN real. Los logs de todos los servicios pasan por un filtro que redacta secuencias con formato de número de tarjeta.
