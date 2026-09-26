# Domino RL 🁫

Enseñarle a una IA a jugar **dominó cubano de 9** (doble-9, 55 fichas, parejas)
usando **Reinforcement Learning** — y de paso, aprender RL desde cero.

> Estado: Fases 1–2 listas (simulador, entorno Gymnasium, baselines, evaluación
> y script de entrenamiento). El entrenamiento real (Fase 5) es el siguiente paso.

## La idea en 30 segundos

En *supervised learning* entrenas con ejemplos etiquetados. En *reinforcement
learning* no hay etiquetas: hay un **agente** que juega, un **entorno** (este
simulador) que le dice el **estado** del juego, y una **recompensa** (+1 ganar,
−1 perder) al final de cada mano. El agente juega millones de manos contra sí
mismo (*self-play*) y ajusta su **política** (qué jugada elegir en cada estado)
para maximizar la recompensa. Sin GPU: la red es diminuta y el simulador es
Python puro — la capa gratuita de Colab sobra.

## Estructura

```
src/domino_rl/
  game.py       Motor del dominó cubano doble-9 (reglas, reparto, tantos)
  env.py        Entorno estilo Gymnasium: obs[143] (incluye historial público
                de fichas jugadas — el conteo debe emerger, no hardcodearse),
                111 acciones con máscara, reward sparse por defecto (bitter lesson)
  baselines.py  Agentes de referencia: random y greedy (heurístico)
  evaluate.py   Torneos entre políticas: win-rate, puntos, dominó vs tranca
  train.py      Entrenamiento PPO con self-play (CPU, sin GPU)
  demo/         Mesa digital: humanos (móvil) vs IA —
                `PYTHONPATH=src python demo/server.py`, abrir
                http://<ip>:8000 en cada teléfono
tests/          Tests del motor (TDD: se escribieron antes que el motor)
PLAN.md         El plan paso a paso completo (Fases 0–8)
```

## Uso rápido

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[train]"

# 1. Ver que el simulador funciona: torneo greedy vs random
python -m domino_rl.evaluate --rounds 200 --a greedy --b random

# 2. Entrenar (CPU; en Colab gratis funciona igual)
python -m domino_rl.train --timesteps 2000000 --n-envs 8 --out checkpoints

# 3. Medir al campeón contra el heurístico
python -m domino_rl.evaluate --a ppo:checkpoints/final.zip --b greedy --rounds 500
```

## Reglas implementadas

Doble-9 (55 fichas), 4 jugadores × 10 fichas (15 quedan en el pozo), sale el
doble más alto en mano (obligado), por parejas (0&2 vs 1&3) o todos contra
todos (`--mode individual`), gana la mano por dominó o por tranca, tantos =
pips restantes del contrario. Detalles y decisiones de diseño en `PLAN.md`.

## Roadmap

Ver **[PLAN.md](PLAN.md)**: 9 fases desde "no sé nada de RL" hasta una IA que
te gana al dominó, con qué aprender en cada fase y cómo saber que vas bien.
